"""V13 Phase1 (plan D2): a dedicated compile subprocess for the async
PGO frame.

Why: the V12 async frame moved the decide/compile/measure cycle off the
forward path, but the cycle still ran INSIDE the EngineCore process on
the frame's background worker.  The MLIR/ptxas section holds the GIL for
~279ms per cycle (measured, async_journey full window) and long
「background compile + main-thread high-frequency launch」runs terminated
MLIR threads twice (core dumps, demo_async_journey 4000-forward long
runs).  Moving the section into its own process removes both.

Shape of the fix (PACT_COMPILE_SUBPROC=1, default off = bit-for-bit v12):
  * ONE resident worker process (spawn context — no fork, no
    shared_memory, clean exit; the v11 resource_tracker lesson).
  * The main process sends a JSON-able job spec: the jit kernel's import
    path, a SHAPE-EXACT tensor description of the compile inputs
    (torch.empty_strided rebuilds identical strides), the env preset and
    the measurement iters.
  * The worker compiles base+candidate (triton's on-disk cache makes the
    artifacts available to the EngineCore afterwards), measures both
    with CUDA events on its own context, and replies with the numbers
    plus timing.  The EngineCore's background thread then re-runs
    compile_candidate — a disk-cache hit, milliseconds — to obtain the
    local CompiledKernel object for the boundary install.
  * Pipe I/O blocks the background thread only; the serving threads and
    the replay/forward path never touch this module.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional


def _describe(args) -> list:
    """JSON-able, shape-exact description of a compile-args tuple."""
    out = []
    for a in args:
        if hasattr(a, "shape") and hasattr(a, "stride"):
            out.append({"t": "tensor", "shape": list(a.shape),
                        "strides": list(a.stride()),
                        "dtype": str(a.dtype)})
        elif isinstance(a, (int, float)):
            out.append({"t": "scalar", "v": a})
        else:
            raise ValueError(f"cannot describe arg {type(a)}")
    return out


def _rebuild(desc: list):
    import torch
    args = []
    for d in desc:
        if d["t"] == "tensor":
            dtype = getattr(torch, d["dtype"].split(".")[-1])
            args.append(torch.empty_strided(
                tuple(d["shape"]), tuple(d["strides"]), dtype=dtype,
                device="cuda"))
        else:
            args.append(d["v"])
    return tuple(args)


def _measure(jit_fn, kernel, bound, grid, iters: int) -> float:
    import torch
    samples = []
    kernel[grid](*bound.values())
    torch.cuda.synchronize()
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel[grid](*bound.values())
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def worker_loop(in_file, out_file) -> None:  # pragma: no cover - subprocess
    """Line-protocol loop: one JSON spec per line, one JSON result per
    line.  Serves as `python -m triton.pact.controller.compile_worker
    --fd N` (no multiprocessing spawn — that mechanism re-imports the
    parent's __main__, which would re-run an unguarded demo/cell script
    inside the worker)."""
    import torch
    torch.zeros(1, device="cuda")  # settle the CUDA context first
    from triton.pact.compiler.explicit_compiler import compile_explicit
    try:
        while True:
            line = in_file.readline()
            if not line:
                break
            spec = json.loads(line)
            if spec is None:
                break
            t0 = time.monotonic()
            try:
                mod = importlib.import_module(spec["jit_module"])
                jit_fn = getattr(mod, spec["jit_attr"])
                args = _rebuild(spec["args_desc"])
                kwargs = dict(spec["kwargs"])
                grid = tuple(spec["grid"])
                # compile ONLY — no cuModuleLoad, no CUDA events: those
                # take the driver lock and freeze the serving process's
                # launches across the process boundary (measured ~278ms).
                # MLIR/ptxas is user-space CPU work, which is exactly the
                # GIL section this worker exists to remove.  The EngineCore
                # re-runs compile_explicit as a disk-cache hit and does the
                # G4 measurements on its own background thread (the V12
                # regime, window worst 0.5ms).
                for env in (spec.get("baseline_env")
                            or {"PACT_ENABLE": "0"},
                            spec["extra_env"]
                            or {"PACT_ENABLE": "1"}):
                    compile_explicit(jit_fn, args, kwargs, dict(env),
                                     spec.get("options_override") or None)
                out_file.write(json.dumps({
                    "status": "ok",
                    "worker_ms": round(
                        (time.monotonic() - t0) * 1000.0, 1),
                }) + "\n")
            except Exception as e:  # noqa: BLE001 - loop survives
                out_file.write(json.dumps({
                    "status": "error", "error": repr(e)[:300],
                    "worker_ms": round(
                        (time.monotonic() - t0) * 1000.0, 1)}) + "\n")
            out_file.flush()
    finally:
        try:
            out_file.close()
        except Exception:
            pass


def _cli_main() -> None:  # pragma: no cover - subprocess entry
    worker_loop(sys.stdin, sys.stdout)


class CompileSubprocess:
    """Resident compile worker (`python -m triton.pact.controller.
    compile_worker`): specs on stdin, one JSON result line per job on
    stdout — no pass_fds so posix_spawn stays available (a fork of the
    multi-GB EngineCore freezes every thread for ~270ms, measured).
    One instance per EngineCore; jobs run strictly on the async frame's
    background thread.  warm_start() boots it in an untimed window."""

    def __init__(self):
        self._proc = None
        self._rfile = None
        self._lock = threading.Lock()
        self.jobs = 0
        self.last_error = None
        self.last_worker_ms = None

    def _ensure(self):
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._rfile
            env = dict(os.environ)
            # the worker must import the jit kernel's module (e.g.
            # suite.kernels.decode lives on the parent's sys.path, not on
            # the environment's PYTHONPATH)
            env["PYTHONPATH"] = os.pathsep.join(sys.path) + os.pathsep + \
                (env.get("PYTHONPATH") or "")
            self._proc = subprocess.Popen(
                [sys.executable, "-m",
                 "triton.pact.controller.compile_worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd(), env=env)
            self._rfile = self._proc.stdout
            return self._rfile

    def warm_start(self) -> bool:
        """Boot the worker in an untimed window (first bridge init); the
        interpreter+torch bootstrap is seconds and must never land in a
        serving window."""
        try:
            self._ensure()
            return True
        except Exception as e:  # noqa: BLE001 - lazy start fallback
            self.last_error = repr(e)[:200]
            return False

    def run(self, spec: Dict[str, Any],
            timeout: float = 180.0,
            _boot_timeout: float = 120.0) -> Optional[Dict[str, Any]]:
        """Blocking call for the background worker thread; the pipe wait
        does not hold the GIL, so serving threads keep running.  The
        first call also absorbs the worker's interpreter+torch+CUDA
        bootstrap (seconds) — still off the serving path."""
        rf = self._ensure()
        try:
            self._proc.stdin.write(json.dumps(spec).encode() + b"\n")
            self._proc.stdin.flush()
            deadline = time.monotonic() + timeout + _boot_timeout
            while True:
                remain = deadline - time.monotonic()
                if remain <= 0 or not self._rfile.readable():
                    self.last_error = "worker timeout"
                    return None
                # line-buffered read with a deadline: poll the pipe
                import select
                _, _, _ = select.select([self._rfile], [], [], min(remain, 5.0))
                line = self._rfile.readline()
                if line:
                    break
            resp = json.loads(line)
            self.jobs += 1
            self.last_worker_ms = resp.get("worker_ms")
            return resp
        except Exception as e:  # noqa: BLE001 - worker may have died
            self.last_error = repr(e)[:300]
            try:
                if self._proc is not None:
                    self._proc.kill()
            except Exception:
                pass
            self._proc = None
            self._rfile = None
            return None

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.stdin.write(b"null\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=5)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
            self._proc = None
            self._rfile = None


_SINGLETON: Optional[CompileSubprocess] = None
_SINGLETON_LOCK = threading.Lock()


def get_compile_subprocess() -> CompileSubprocess:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = CompileSubprocess()
        return _SINGLETON


def run_compile(jit_fn, args, kwargs, grid, extra_env=None,
                options=None, timeout: float = 180.0) -> bool:
    """Convenience wrapper for CompilerService compile callbacks (bridge
    and demo): ship one compile job to the worker and return True on
    success.  The caller's own compile_explicit afterwards is a disk-cache
    hit.  Raises on worker failure so callers can fall back to local
    compilation."""
    svc = get_compile_subprocess()
    spec = {
        "jit_module": getattr(getattr(jit_fn, "fn", None), "__module__",
                              None) or type(jit_fn).__module__,
        "jit_attr": getattr(getattr(jit_fn, "fn", None), "__name__",
                            None) or "kernel",
        "args_desc": _describe(args),
        "kwargs": dict(kwargs),
        "grid": list(grid),
        "extra_env": dict(extra_env or {"PACT_ENABLE": "1"}),
        "options_override": options,
    }
    r = svc.run(spec, timeout=timeout)
    if r is None or r.get("status") != "ok":
        raise RuntimeError(
            f"compile worker: {svc.last_error or (r or {}).get('error')}")
    return True


if __name__ == "__main__":  # pragma: no cover - subprocess entry
    _cli_main()
