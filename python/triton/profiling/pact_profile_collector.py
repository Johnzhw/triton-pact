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

from triton.profiling.pact_profile_db import default_cache_root

TEST2 = Path(os.environ.get("PACT_TEST2_DIR", "/home/johnzhw/workspace/test_2"))
TRITON_PY = Path(os.environ.get("PACT_TRITON_PY",
                                "/home/johnzhw/workspace/triton/python"))
CACHE_ROOT = default_cache_root()
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

    def dump_ttgir(self) -> Tuple[Path, str, int, str]:
        """Compile the target once with TRITON_KERNEL_DUMP and return TTGIR
        path + key + real register count (regs_per_thread producer)."""
        shutil.rmtree(self.dump_dir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        B, S, P, D, Hq, Hk = self.shape
        code = self._target_args_code() + f"""
from kernels.pact_optimization_target import run_pact_target
from kernels.pact_optimization_target import pact_optimization_target
out = run_pact_target(q, kc, vc, bt, sl, page_size=P)
# regs_per_thread producer: warmup() returns the CompiledKernel whose n_regs
# comes from the binary metadata loaded by driver.active.utils.load_binary.
out_ref = torch.empty_like(q)
regs = 0
regs_err = ''
try:
    max_seq_len = bt.shape[1] * P
    gqa_ratio = Hq // Hk
    kernel = pact_optimization_target.warmup(
        out_ref, q, kc, vc, bt, sl,
        sm_scale=1.0 / (D ** 0.5), NUM_TOKENS=B, NUM_HEADS=Hq,
        NUM_KV_HEADS=Hk, HEAD_DIM=D, PAGE_SIZE=P, MAX_SEQ_LEN=max_seq_len,
        TILE_SIZE=16, GQA_RATIO=gqa_ratio,
        STRIDE_BLOCK=Hk * P * D, STRIDE_KV_HEAD=P * D, STRIDE_PAGE=D,
        STRIDE_HEAD_DIM=1, USE_DUAL_TILE=False, TILE_SIZE_LARGE=32,
        TOKEN_IMPORTANCE_MODE=0, grid=({B}, {Hq}))
    regs = int(getattr(kernel, 'n_regs', 0) or 0)
except Exception as e:
    regs_err = str(e)[:200]
print('N_REGS:' + str(regs))
if regs_err:
    print('N_REGS_UNAVAILABLE:' + repr(regs_err))
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
        regs = 0
        regs_err = ""
        for line in out.splitlines():
            if line.startswith("N_REGS:"):
                regs = int(line.split(":", 1)[1])
            elif line.startswith("N_REGS_UNAVAILABLE:"):
                regs_err = line.split(":", 1)[1].strip().strip("'")
        return ttgir, keydir.name, regs, regs_err

    def collect(self, steps: int = 20, trace_path: Optional[Path] = None) -> Dict[str, Any]:
        from triton.profiler.hooks.pact_instrumentation import instrument_ttgir_text

        base_ttgir, key, regs, regs_err = self.dump_ttgir()
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
s = start({str(trace_path.with_suffix(''))!r}, data='trace', backend='instrumentation',
          mode=mode.Default(optimizations='clock32,time_shift',
                            buffer_type='global', buffer_size=65536))
# Import the kernel only after proton has registered its instrumentation
# dialects in the current backend context.
from kernels.pact_optimization_target import run_pact_target
torch.cuda.synchronize()
# B3 fix: warm up outside any timed region so the first JIT compilation of the
# instrumented override is excluded from the reported step time.
out = run_pact_target(q, kc, vc, bt, sl, page_size=P)
torch.cuda.synchronize()
start_ev = torch.cuda.Event(enable_timing=True); end_ev = torch.cuda.Event(enable_timing=True)
start_ev.record()
for _ in range({steps}):
    out = run_pact_target(q, kc, vc, bt, sl, page_size=P)
end_ev.record(); torch.cuda.synchronize()
host_wall_step_us = start_ev.elapsed_time(end_ev) * 1000 / {steps}
print('HOST_WALL_STEP_US:' + str(host_wall_step_us))
# Kernel-only median must be measured while the instrumentation hook is still
# registered (finalize unregisters it and a post-finalize cache miss would
# otherwise re-parse the proton.record override without the proton dialect).
import statistics as _stat
klat=[]
for _ in range(min({steps}, 10)):
    st=torch.cuda.Event(enable_timing=True); en=torch.cuda.Event(enable_timing=True)
    st.record(); out = run_pact_target(q, kc, vc, bt, sl, page_size=P); en.record()
    torch.cuda.synchronize(); klat.append(st.elapsed_time(en)*1000.0)
print('KERNEL_MEDIAN_US:' + str(_stat.median(klat)))
finalize(s, output_format='chrome_trace')
"""
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        env = self._base_env()
        env["TRITON_KERNEL_OVERRIDE"] = "1"
        rc, out, err = self._run_subprocess(_make_runner(code, {}), env,
                                            timeout=600, cwd=str(trace_path.parent))
        if rc != 0:
            raise RuntimeError(f"instrumented run failed: {err[-2000:]}")

        host_wall_step_us = None
        kernel_median_us = None
        for line in out.splitlines():
            if line.startswith("HOST_WALL_STEP_US:"):
                host_wall_step_us = float(line.split(":", 1)[1])
            elif line.startswith("KERNEL_MEDIAN_US:"):
                kernel_median_us = float(line.split(":", 1)[1])
        if not trace_path.exists():
            raise RuntimeError(f"trace missing: {trace_path}")

        B, S, P, D, Hq, Hk = self.shape
        # The trace contains: 1 warmup launch (needed so the instrumented
        # override is compiled before the timed loop), the timed `steps` loop,
        # and min(steps,10) kernel-median launches.  Divide the per-scope event
        # count by the total number of launches.
        total_launches = steps + 1 + min(steps, 10)
        facts = parse_instrument_trace(trace_path, expected_ctas=B * Hq,
                                       num_warps=4, steps=total_launches)
        if host_wall_step_us is not None:
            facts["host_wall_step_us"] = host_wall_step_us
        if kernel_median_us is not None:
            facts["kernel_median_us"] = kernel_median_us
        if regs > 0:
            facts["regs_per_thread"] = regs
        elif regs_err:
            facts["regs_unavailable"] = regs_err
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
        backend = "roctracer" if os.environ.get("PACT_AMD_ARCH") else "cupti"
        code = self._target_args_code() + f"""
from triton.profiler import start, finalize
from kernels.pact_optimization_target import run_pact_target
s = start({str(trace_path.with_suffix(''))!r}, context='shadow', backend={backend!r},
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
            raise RuntimeError(f"{backend} fallback failed: {err[-2000:]}")
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
        return evs, warps

    load_evs, load_warps = stats("pact.load")

    facts: Dict[str, Any] = {}
    if load_evs and load_warps:
        # Same scope executes once per warp per iteration.
        first_scope = load_evs[0]["name"]
        first_count = sum(1 for e in load_evs if e["name"] == first_scope)
        facts["measured_iterations"] = int(round(
            first_count / max(load_warps, 1) / max(steps, 1)))

    # B2 honesty: this is a participation ratio (unique (pid,tid) observed in
    # any instrumented scope), NOT hardware occupancy.  It must never be
    # published under the pact.pgo.active_warp_ratio_permille hint.  A separate
    # CUPTI probe may provide the real hardware number.
    all_warps = len(set((e.get("pid"), e.get("tid")) for e in events))
    expected_warps = expected_ctas * num_warps
    if expected_warps:
        ratio = min(1.0, all_warps / expected_warps)
        facts["warp_participation_permille"] = int(round(ratio * 1000))

    return facts
