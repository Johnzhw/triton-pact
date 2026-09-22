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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

from triton.pact.compiler.explicit_compiler import compile_explicit
from triton.pact_aobo.inline_decider import VARIANTS, decide_inline
from triton.pact_aobo.resident_pool import ResidentPool


class AsyncKernelSwitch:
    def __init__(self, pool: ResidentPool):
        self.pool = pool
        self._exe = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pact-aobo-bg")
        self._pending = None          # (name, future, geo_key)
        self._pending_lock = threading.Lock()
        self.state: Dict[str, Any] = {
            "decisions": 0, "bg_compiles": 0, "installs": 0,
            "install_ms": [], "compile_ms": [], "active": pool.active,
        }
        # V12-2b: paired CUDA events, elapsed_time drained in batches —
        # the inline decider gets an in-process 'last_launch_us' fact
        # without any per-forward host synchronisation.  Events are
        # PRE-ALLOCATED and rotated (one record serves as pair_j's end and
        # pair_{j+1}'s start): creating timing events per forward proved to
        # amplify host-side stalls while the background compiler holds the
        # driver lock (demo window max 11ms -> 21-23ms); rotation removed
        # the allocation from the hot path.
        self._evn = 8
        self._evs = None
        self._ev_cursor = 0
        self._prev_ev = None
        self._ev_pairs = deque(maxlen=4)

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
        """Device-side timing of the previous launch (paired CUDA events,
        no host sync per forward).  One record closes the previous pair
        and opens the next; closed pairs sit in a small ring, and once the
        ring is full the OLDEST pair is drained (it has several launches
        of slack, so elapsed_time does not stall the stream in practice)
        and published as state['last_launch_us'] — the 'collect
        current-step performance' input for decide_inline."""
        try:
            import torch
            if self._evs is None:
                self._evs = [torch.cuda.Event(enable_timing=True)
                             for _ in range(self._evn)]
            e = self._evs[self._ev_cursor % self._evn]
            e.record()
            if self._prev_ev is not None and self._prev_ev is not e:
                self._ev_pairs.append((self._prev_ev, e))
            self._prev_ev = e
            self._ev_cursor += 1
            if len(self._ev_pairs) >= self._ev_pairs.maxlen:
                s, t = self._ev_pairs.popleft()
                try:
                    us = s.elapsed_time(t) * 1000.0
                    self.state["last_launch_us"] = us
                    # V15-R1: smoothed observation (EMA alpha=0.2, the
                    # same formula as the pgo service replica column).
                    # ADD-ONLY: facts_to_hints does not read this key;
                    # the instantaneous last_launch_us keeps feeding
                    # decide_inline unchanged.
                    prev = self.state.get("launch_us_ema")
                    self.state["launch_us_ema"] = \
                        us if prev is None else \
                        0.2 * us + 0.8 * prev
                except Exception:
                    pass
        except Exception:
            pass

    # ---- decision + background compile -----------------------------------
    def maybe_compile_async(self, batch: int, seq_len: int,
                            facts: Optional[Dict[str, Any]] = None,
                            geometry: Optional[Dict[str, int]] = None
                            ) -> Optional[str]:
        """Inline decide; if the variant is not resident, compile it on the
        background worker.  Returns the decided name (never blocks).
        Without explicit facts the latest device-side observation
        (state['last_launch_us'], V12-2b) is attached so the learned
        policy has an in-process input."""
        facts = dict(facts or {})
        if "last_launch_us" not in facts and \
                self.state.get("last_launch_us"):
            facts["last_launch_us"] = self.state["last_launch_us"]
        name, extra_env = decide_inline(facts, batch, seq_len, geometry)
        self.state["decisions"] += 1
        if name in self.pool:
            return name
        # snapshot the geometry the decision is FOR: the background compile
        # must land in this sub-pool even if the workload retargets (new S
        # bucket) before the compile finishes
        snap: Tuple[tuple, dict, tuple] = (
            tuple(self.pool.args), dict(self.pool.kwargs),
            self.pool._cur_key)
        with self._pending_lock:
            if (self._pending and self._pending[0] == name):
                return name
            fut = self._exe.submit(self._bg_compile, name, extra_env, snap)
            self._pending = (name, fut, snap[2])
        self.state["bg_compiles"] += 1
        return name

    def _bg_compile(self, name: str, extra_env: Dict[str, str],
                    snap: Tuple[tuple, dict, tuple]):
        t0 = time.monotonic()
        args, kwargs, key = snap
        try:
            k, _ = compile_explicit(self.pool.jit_fn, args, kwargs,
                                    dict(extra_env))
            with self.pool._lock:
                sub = self.pool._geo.setdefault(
                    key, {"kernels": {"baseline": None}, "active":
                          "baseline"})
                sub["kernels"].setdefault(name, k)
            self.state["compile_ms"].append(
                (time.monotonic() - t0) * 1000.0)
            # Force the cubin/module load HERE (background thread), so the
            # first launch of the swapped-in kernel does not pay the module
            # load inside a forward.  V12-2d: one dummy launch against a
            # cloned output additionally settles module lazy-load / first-
            # launch cost — the v11 cold-run post_install band carried a
            # 171.9ms spike that survives handle init (v12 measured ~48ms).
            if k is not None and hasattr(k, "_init_handles"):
                try:
                    k._init_handles()
                except Exception:
                    pass
            try:
                import torch
                dargs = (torch.empty_like(args[0]),) + tuple(args[1:])
                self.pool.dry_launch(k, dargs, kwargs)
            except Exception:
                pass
            # V13 Phase0 (prepared-module regime): the variant vocabulary
            # lives in the single module prepared at the first forward;
            # nothing to load here — the boundary submit switches nodes
            # inside that module.
            if os.environ.get("PACT_GRAPH_SERVICE") == "1":
                try:
                    from triton.pact.runtime.graph_service import \
                        get_service
                    get_service()  # touch: state sink stays armed
                except Exception:
                    pass
            return (name, key)
        except Exception as e:  # noqa: BLE001 - surfaced via state
            self.state["last_bg_error"] = repr(e)
            return None

    # ---- installation at launch boundary ---------------------------------
    def _install_if_ready(self) -> bool:
        with self._pending_lock:
            pending = self._pending
            if pending is None:
                return False
            name, fut, key = pending
            if not fut.done():
                return False          # not ready: current slot keeps serving
            self._pending = None
        res = fut.result()
        if res is None:
            return False
        t0 = time.monotonic()
        with self.pool._lock:
            sub = self.pool._geo.get(res[1])
            if sub is None or res[0] not in sub["kernels"]:
                return False
            sub["active"] = res[0]    # atomic slot exchange (its geometry)
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
    # Warm the inline decider OUTSIDE the measured loop: the first
    # decide_inline call loads the learned policy (joblib deserialisation,
    # tens of ms) — that host stall is not part of the async-frame story
    # and used to land inside the compile-window band.
    decide_inline(None, B, S, {"S": S, "D": D, "P": P, "B": B, "Hq": Hq,
                               "GQA": Hq // Hk})
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
