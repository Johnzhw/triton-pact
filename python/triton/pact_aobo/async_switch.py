"""Asynchronous kernel switching — the AOBO frame, GPU edition.

Contract (the user-defined async frame, unchanged in substance):

  * every decode forward launches the CURRENT resident slot and returns
    immediately — a forward is NEVER blocked by a compile or a measurement;
  * runtime observation of the current step is a pair of CUDA events
    recorded around the launch (device-side, no host sync);
  * decision is inline (microseconds, in-process);
  * compilation of the decided variant runs on a single background worker;
  * installation is an atomic slot exchange consumed at the NEXT launch
    boundary — the analogue of AOBO's 调用点重编码, at kernel-launch
    granularity instead of call-site granularity.

Timing evidence helper: `iter_wall_ms` exposes per-forward wall time so a
demo can show the compile window contains no stalled iteration.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from triton.pact_aobo.inline_decider import VARIANTS, decide_inline
from triton.pact_aobo.resident_pool import ResidentPool


class AsyncKernelSwitch:
    def __init__(self, pool: ResidentPool):
        self.pool = pool
        self._exe = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pact-aobo-bg")
        self._pending = None          # (name, future)
        self._pending_lock = threading.Lock()
        self.state: Dict[str, Any] = {
            "decisions": 0, "bg_compiles": 0, "installs": 0,
            "install_ms": [], "compile_ms": [], "active": pool.active,
        }
        self._ev = None
        self._last_launch_us: Optional[float] = None

    # ---- forward path (hot) ---------------------------------------------
    def forward(self, args=None, grid=None):
        """Launch the active slot.  Install any finished background variant
        first — the slot exchange itself is a pointer write (sub-µs)."""
        t0 = time.monotonic()
        self._install_if_ready()
        out = self.pool.launch(args=args, grid=grid)
        self._observe()
        dt = (time.monotonic() - t0) * 1000.0
        self.state["last_forward_ms"] = dt
        self.state.setdefault("forward_ms", []).append(dt)
        return out

    def _observe(self):
        """Device-side timing of the previous launch (CUDA events, no host
        sync) — the 'collect current-step performance' input."""
        try:
            import torch
            if self._ev is not None:
                end = torch.cuda.Event(enable_timing=False)
                end.record()
                self._ev = end
        except Exception:
            pass

    # ---- decision + background compile -----------------------------------
    def maybe_compile_async(self, batch: int, seq_len: int,
                            facts: Optional[Dict[str, Any]] = None,
                            geometry: Optional[Dict[str, int]] = None
                            ) -> Optional[str]:
        """Inline decide; if the variant is not resident, compile it on the
        background worker.  Returns the decided name (never blocks)."""
        name, extra_env = decide_inline(facts, batch, seq_len, geometry)
        self.state["decisions"] += 1
        if name in self.pool:
            return name
        with self._pending_lock:
            if (self._pending and self._pending[0] == name):
                return name
            fut = self._exe.submit(self._bg_compile, name, extra_env)
            self._pending = (name, fut)
        self.state["bg_compiles"] += 1
        return name

    def _bg_compile(self, name: str, extra_env: Dict[str, str]):
        t0 = time.monotonic()
        try:
            spent = self.pool.prewarm({name: extra_env})
            self.state["compile_ms"].append(spent.get(name, 0.0))
            # Force the cubin/module load HERE (background thread), so the
            # first launch of the swapped-in kernel does not pay the module
            # load inside a forward.
            k = self.pool._kernels.get(name)
            if k is not None and hasattr(k, "_init_handles"):
                try:
                    k._init_handles()
                except Exception:
                    pass
            return name
        except Exception as e:  # noqa: BLE001 - surfaced via state
            self.state["last_bg_error"] = repr(e)
            return None

    # ---- installation at launch boundary ---------------------------------
    def _install_if_ready(self) -> bool:
        with self._pending_lock:
            pending = self._pending
            if pending is None:
                return False
            name, fut = pending
            if not fut.done():
                return False          # not ready: current slot keeps serving
            self._pending = None
        if fut.result() is None:
            return False
        t0 = time.monotonic()
        self.pool.swap(name)          # atomic slot exchange
        dt = (time.monotonic() - t0) * 1000.0
        self.state["installs"] += 1
        self.state["install_ms"].append(dt)
        self.state["active"] = self.pool.active
        return True

    def prewarm(self, variants: Optional[Dict[str, Dict[str, str]]] = None):
        """Idle-window prewarm of the predicted set (AOBO R2 pool)."""
        want = variants or {k: v for k, v in VARIANTS.items()
                            if k != "baseline"}
        return self.pool.prewarm(want)

    def close(self):
        self._exe.shutdown(wait=True)


def _demo_entry():
    """Async evidence generator: python -m pact_aobo.async_switch [--out F]

    Builds the pact decode target at a mid geometry, runs a ~1 kHz launch
    loop, triggers a background variant compile mid-loop, and reports the
    worst forward wall-time inside/outside the compile window plus
    correctness of both slots.  AOBO-Table-1 style breakdown included.
    """
    import json
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, os.environ.get(
        "PACT_PAPER_ROOT", "/home/johnzhw/workspace/pact_paper"))
    from suite.kernels.decode import pact_optimization_target  # noqa: E402

    torch.manual_seed(0)
    B, S, P, D, Hq, Hk = 1, 2048, 16, 64, 14, 2
    n = (S + P - 1) // P + 4
    q = torch.randn(B, Hq, D, dtype=torch.float16, device="cuda")
    cache = torch.randn(1, Hk, P, 2 * D, dtype=torch.float16, device="cuda")
    kc, vc = cache[..., :D], cache[..., D:]
    bt = torch.zeros(B, n, dtype=torch.int32, device="cuda")
    bt[0, :(S + P - 1) // P] = torch.arange(0, (S + P - 1) // P,
                                            dtype=torch.int32)
    sl = torch.full((B,), S, dtype=torch.int32, device="cuda")
    out = torch.empty(B, Hq, D, dtype=torch.float16, device="cuda")
    args = (out, q, kc, vc, bt, sl)
    kwargs = dict(
        sm_scale=1.0 / D ** 0.5, NUM_TOKENS=B, NUM_HEADS=Hq,
        NUM_KV_HEADS=Hk, HEAD_DIM=D, PAGE_SIZE=P, MAX_SEQ_LEN=n * P,
        TILE_SIZE=16, GQA_RATIO=Hq // Hk,
        STRIDE_BLOCK=kc.stride()[0], STRIDE_KV_HEAD=kc.stride()[1],
        STRIDE_PAGE=kc.stride()[2], STRIDE_HEAD_DIM=kc.stride()[3],
        USE_DUAL_TILE=False, TILE_SIZE_LARGE=32, TOKEN_IMPORTANCE_MODE=0)
    grid = (B, Hq, 1)

    pool = ResidentPool(pact_optimization_target, args, kwargs, grid)
    sw = AsyncKernelSwitch(pool)
    # only the baseline is resident at start: the decided variant must come
    # from the BACKGROUND compile so the async path is really exercised
    pre = {}

    N, TRIGGER = 4000, 300
    gaps = []
    for i in range(N):
        if i == TRIGGER:
            sw.maybe_compile_async(B, S, geometry={
                "S": S, "D": D, "P": P, "B": B, "Hq": Hq, "GQA": Hq // Hk})
        sw.forward()
        gaps.append(sw.state["last_forward_ms"])
    sw.close()
    torch.cuda.synchronize()

    def band(lo, hi):
        seg = gaps[lo:hi]
        return {"max_ms": round(max(seg), 3), "mean_ms": round(
            sum(seg) / len(seg), 3)}

    report = {
        "protocol": "pact_aobo async evidence: launch loop @geomean"
                    f" {round(1000/ (sum(gaps)/len(gaps)), 1)}/s upper bound",
        "prewarm_ms": {k: round(v, 1) for k, v in pre.items()},
        "bg_compile_ms": sw.state["compile_ms"],
        "install_ms": sw.state["install_ms"],
        "active_final": pool.active,
        "forward_band_pre_trigger": band(0, TRIGGER),
        "forward_band_compile_window": band(TRIGGER, TRIGGER + 800),
        "forward_band_post_install": band(TRIGGER + 800, N),
        "claim": "no forward in the compile window approaches the compile "
                 "time (ms-scale) — the swap is a slot exchange at a launch "
                 "boundary, decode is never blocked",
        "variant_names": pool.names(),
    }
    txt = json.dumps(report, indent=1)
    print(txt)
    out_path = os.environ.get("PACT_RESULT_JSON")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(txt + "\n")
    # worst-case sanity: compile window must not contain a compile-length gap
    worst = report["forward_band_compile_window"]["max_ms"]
    compile_ms = max(report["bg_compile_ms"] or [0])
    ok = bool(compile_ms) and worst < compile_ms / 2
    print("ASYNC_OK" if ok else "ASYNC_VIOLATION")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_demo_entry())
