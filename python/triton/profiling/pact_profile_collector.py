"""PACT PGO collector.

Compiles an instrumented TTGIR variant of the paged attention target through
Triton's TTGIR override mechanism, runs it under Proton's instrumentation
backend (KPerfIR/Proton dialect), and derives numeric facts for P6/P11.

Fallback note: if TRITON_KERNEL_OVERRIDE + instrumentation is unavailable,
the same collector can be pointed at cupti periodic_flushing / pcsampling
traces; only the parser changes.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TEST2 = Path("/home/johnzhw/workspace/test_2")
TRITON_PY = Path("/home/johnzhw/workspace/triton/python")
CACHE_ROOT = Path(os.environ.get("PACT_PGO_CACHE_ROOT",
                                 Path.home() / ".triton"))
DEFAULT_ENV = {
    "PACT_ENABLE": "1",
    "PACT_ENABLE_PAGE_TRANSFORM": "1",
    "PACT_ENABLE_PAGE_LOCAL_ANALYSIS": "1",
    "PACT_ENABLE_AXISINFO_OVERRIDE": "1",
    "PACT_ENABLE_AUTO_NUM_STAGES": "1",
}


def _make_runner(code: str, extra_env: Dict[str, str]) -> str:
    env_lines = "\n".join(f"os.environ['{k}'] = {v!r}" for k, v in extra_env.items())
    return f"""
import os, sys, torch
sys.path.insert(0, {str(TEST2)!r})
sys.path.insert(0, {str(TRITON_PY)!r})
{env_lines}
{code}
"""


class PactProfileCollector:
    def __init__(self, cache_root: Optional[Path] = None,
                 extra_env: Optional[Dict[str, str]] = None,
                 shape: Optional[Tuple[int, int, int, int, int, int]] = None):
        self.cache_root = Path(cache_root or CACHE_ROOT)
        self.cache_dir = self.cache_root / "cache-pgo"
        self.dump_dir = self.cache_root / "dump-pgo"
        self.override_dir = self.cache_root / "override-pgo"
        self.extra_env = dict(extra_env or {})
        # (B, S, P, D, Hq, Hk)
        self.shape = tuple(shape or (1, 256, 64, 64, 8, 2))

    def _base_env(self) -> Dict[str, str]:
        env = {
            **os.environ,
            **DEFAULT_ENV,
            **self.extra_env,
            "TRITON_CACHE_DIR": str(self.cache_dir),
            "TRITON_DUMP_DIR": str(self.dump_dir),
            "TRITON_OVERRIDE_DIR": str(self.override_dir),
        }
        return env

    def _run_subprocess(self, script: str, env: Dict[str, str],
                        timeout: int = 300, cwd: Optional[str] = None) -> Tuple[int, str, str]:
        r = subprocess.run(["python", "-c", script], capture_output=True,
                           text=True, timeout=timeout, env=env, cwd=cwd)
        return r.returncode, r.stdout, r.stderr

    def _target_args_code(self) -> str:
        B, S, P, D, Hq, Hk = self.shape
        return f"""
torch.manual_seed(42)
B,S,P,D,Hq,Hk = {B},{S},{P},{D},{Hq},{Hk}
n=(S+P-1)//P+B*4
q=torch.randn(B,Hq,D,dtype=torch.float16,device='cuda')
kc=torch.randn(n,Hk,P,D,dtype=torch.float16,device='cuda')
vc=torch.randn(n,Hk,P,D,dtype=torch.float16,device='cuda')
bt=torch.zeros(B,n,dtype=torch.int32,device='cuda')
for b in range(B):
    bt[b,:(S+P-1)//P]=torch.arange(b*((S+P-1)//P),(b+1)*((S+P-1)//P),dtype=torch.int32)
sl=torch.full((B,),S,dtype=torch.int32,device='cuda')
"""

    def dump_ttgir(self) -> Tuple[Path, str]:
        """Compile the target once with TRITON_KERNEL_DUMP and return TTGIR path + key."""
        shutil.rmtree(self.dump_dir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        code = self._target_args_code() + """
from kernels.pact_optimization_target import run_pact_target
out = run_pact_target(q, kc, vc, bt, sl, page_size=P)
"""
        env = self._base_env()
        env["TRITON_KERNEL_DUMP"] = "1"
        rc, out, err = self._run_subprocess(_make_runner(code, {}), env)
        if rc != 0:
            raise RuntimeError(f"TTGIR dump failed: {err[-2000:]}")
        keydirs = sorted(self.dump_dir.glob("*"), key=lambda p: p.stat().st_mtime,
                         reverse=True)
        if not keydirs:
            raise RuntimeError(f"no dump dirs under {self.dump_dir}")
        keydir = keydirs[0]
        ttgir = keydir / "pact_optimization_target.ttgir"
        if not ttgir.exists():
            raise RuntimeError(f"missing {ttgir}")
        return ttgir, keydir.name

    def collect(self, steps: int = 20, trace_path: Optional[Path] = None) -> Dict[str, Any]:
        from triton.profiler.hooks.pact_instrumentation import instrument_ttgir_text

        base_ttgir, key = self.dump_ttgir()
        instrumented = instrument_ttgir_text(base_ttgir.read_text())

        override_key_dir = self.override_dir / key
        shutil.rmtree(override_key_dir, ignore_errors=True)
        override_key_dir.mkdir(parents=True, exist_ok=True)
        (override_key_dir / "pact_optimization_target.ttgir").write_text(instrumented)

        trace_path = Path(trace_path or (self.cache_root / "pact_profile.chrome_trace"))
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if trace_path.exists():
            trace_path.unlink()

        code = self._target_args_code() + f"""
from triton.profiler import start, finalize
from triton.profiler import mode
from kernels.pact_optimization_target import run_pact_target
s = start({str(trace_path.with_suffix(''))!r}, data='trace', backend='instrumentation',
          mode=mode.Default(optimizations='clock32,time_shift'))
torch.cuda.synchronize()
start_ev = torch.cuda.Event(enable_timing=True); end_ev = torch.cuda.Event(enable_timing=True)
start_ev.record()
for _ in range({steps}):
    out = run_pact_target(q, kc, vc, bt, sl, page_size=P)
end_ev.record(); torch.cuda.synchronize()
latency_us = start_ev.elapsed_time(end_ev) * 1000 / {steps}
finalize(s, output_format='chrome_trace')
print('LATENCY_US:' + str(latency_us))
"""
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        env = self._base_env()
        env["TRITON_KERNEL_OVERRIDE"] = "1"
        rc, out, err = self._run_subprocess(_make_runner(code, {}), env,
                                            timeout=600, cwd=str(trace_path.parent))
        if rc != 0:
            raise RuntimeError(f"instrumented run failed: {err[-2000:]}")

        latency_us = None
        for line in out.splitlines():
            if line.startswith("LATENCY_US:"):
                latency_us = float(line.split(":", 1)[1])
        if not trace_path.exists():
            raise RuntimeError(f"trace missing: {trace_path}")

        B, S, P, D, Hq, Hk = self.shape
        facts = parse_instrument_trace(trace_path, expected_ctas=B * Hq,
                                       num_warps=4, steps=steps)
        if latency_us is not None:
            facts["latency_us"] = latency_us
        return facts


class PactCuptiFallbackCollector(PactProfileCollector):
    """Fallback path: cupti periodic_flushing + optional pcsampling.

    Kernel durations are real hardware timestamps; trip count is not directly
    observable here, so measured_iterations is omitted and P6 falls back to its
    theory-only default.  PCSampling parsing is added when the sampled session
    is available.
    """

    def collect(self, steps: int = 20, trace_path: Optional[Path] = None) -> Dict[str, Any]:
        trace_path = Path(trace_path or (self.cache_root / "pact_cupti.chrome_trace"))
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        code = self._target_args_code() + f"""
from triton.profiler import start, finalize
from kernels.pact_optimization_target import run_pact_target
s = start({str(trace_path.with_suffix(''))!r}, context='shadow', backend='cupti',
          mode='periodic_flushing:format=chrome_trace')
torch.cuda.synchronize()
start_ev = torch.cuda.Event(enable_timing=True); end_ev = torch.cuda.Event(enable_timing=True)
start_ev.record()
for _ in range({steps}):
    out = run_pact_target(q, kc, vc, bt, sl, page_size=P)
end_ev.record(); torch.cuda.synchronize()
latency_us = start_ev.elapsed_time(end_ev) * 1000 / {steps}
finalize(s, output_format='chrome_trace')
print('LATENCY_US:' + str(latency_us))
"""
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        env = self._base_env()
        rc, out, err = self._run_subprocess(_make_runner(code, {}), env,
                                            timeout=600, cwd=str(trace_path.parent))
        if rc != 0:
            raise RuntimeError(f"cupti fallback failed: {err[-2000:]}")
        facts: Dict[str, Any] = {}
        for line in out.splitlines():
            if line.startswith("LATENCY_US:"):
                facts["latency_us"] = float(line.split(":", 1)[1])
        return facts


def parse_instrument_trace(trace_path: Path, expected_ctas: int,
                           num_warps: int, steps: int = 1) -> Dict[str, Any]:
    with open(trace_path) as f:
        data = json.load(f)
    events = [e for e in data.get("traceEvents", []) if e.get("ph") == "X"]
    if not events:
        return {}

    def stats(prefix: str):
        evs = [e for e in events if e.get("name", "").startswith(prefix)]
        warps = len(set((e.get("pid"), e.get("tid")) for e in evs))
        cycles = [e.get("dur", 0.0) * 1000.0 for e in evs]  # us at 1 GHz
        return evs, warps, cycles

    load_evs, load_warps, load_cycles = stats("pact.load")
    copy_evs, copy_warps, copy_cycles = stats("pact.async_copy")
    wait_evs, wait_warps, wait_cycles = stats("pact.async_wait")
    compute_evs, compute_warps, compute_cycles = stats("pact.compute")

    facts: Dict[str, Any] = {}
    if load_evs and load_warps:
        # Same scope executes once per warp per iteration.
        first_scope = load_evs[0]["name"]
        first_count = sum(1 for e in load_evs if e["name"] == first_scope)
        facts["measured_iterations"] = int(round(
            first_count / max(load_warps, 1) / max(steps, 1)))

    all_warps = len(set((e.get("pid"), e.get("tid")) for e in events))
    expected_warps = expected_ctas * num_warps
    if expected_warps:
        ratio = min(1.0, all_warps / expected_warps)
        facts["active_warp_ratio_permille"] = int(round(ratio * 1000))

    if copy_cycles and wait_cycles:
        avg_copy = sum(copy_cycles) / len(copy_cycles)
        avg_wait = sum(wait_cycles) / len(wait_cycles)
        if avg_copy > 0:
            benefit = max(0.0, min(1.0, 1.0 - avg_wait / avg_copy))
            facts["pipeline_overlap_benefit_permille"] = int(round(benefit * 1000))
            facts["async_copy_cycles"] = avg_copy
            facts["async_wait_cycles"] = avg_wait

    if load_cycles:
        facts["load_cycles_avg"] = sum(load_cycles) / len(load_cycles)
    if compute_cycles:
        facts["compute_cycles_avg"] = sum(compute_cycles) / len(compute_cycles)
    return facts
