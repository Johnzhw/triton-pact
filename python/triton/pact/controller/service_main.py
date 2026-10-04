"""Compiler-service process: profile replica + decide family + one JIT + shm."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from triton.pact.compiler.explicit_compiler import compile_explicit, kernel_to_blob
from triton.pact.decider.family_table import FamilyTable
from triton.pact.decider.learned_policy import decide
from triton.pact.ipc.shm_manager import ShmManager
from triton.pact.ipc.socket_server import PactSocketServer
from triton.pact.profiler.cupti_collector import (collect_or_unavailable,
                                                   cupti_stats)
from triton.pact.profiler.replica_probe import replica_median_us


class CompilerService:
    def __init__(self, compile_fn: Optional[Callable] = None,
                 replica_launch: Optional[Callable] = None,
                 socket_path: str = "/tmp/pact_compiler.sock"):
        self.compile_fn = compile_fn
        self.replica_launch = replica_launch
        self.shm = ShmManager()
        self.table = FamilyTable.load()
        self.server = PactSocketServer(socket_path, self.handle)
        self.socket_path = socket_path
        # V21-C1/F2: per-bucket replica EMA (the old single global
        # _replica_ema fed every bucket's client baseline a cross-bucket
        # average) + the facts_only served counter (BR-17: counted,
        # never silent).
        self._replica_ema_by_bucket: Dict[str, float] = {}
        self._facts_only_served = 0
        # V13 Phase1: warm the CUPTI availability probe HERE (untimed
        # init window).  cuptiProfilerInitialize holds the driver/loader
        # lock for ~300ms; doing it on the first RPC froze the serving
        # threads' kernel launches (thread-stack-caught in the
        # 4000-forward run).  The cached result makes every later RPC
        # lock-free; RPC responses stay value-identical.
        try:
            collect_or_unavailable()
        except Exception:
            pass

    def _bucket_key(self, req: Dict[str, Any]) -> str:
        """V21-C1/F2: the per-bucket replica-EMA key.  New clients echo
        their sniffed bucket; an old client derives it from the workload
        pair (same bucket_bs grammar as the trigger path)."""
        b = req.get("bucket")
        if isinstance(b, str) and b:
            return b
        wl = req.get("workload") or {}
        try:
            from triton.pact.runtime.workload_sniffer import bucket_bs
            return repr(bucket_bs(int(wl.get("B") or 1),
                                  int(wl.get("S") or 0)))
        except Exception:
            return "unk"

    def _collect_facts(self, req: Dict[str, Any],
                       want_cupti: bool) -> Dict[str, Any]:
        """Replica median + per-bucket EMA (+ CUPTI counters when both
        the caller wants them AND PACT_CUPTI_PROFILING=1).  Shared by the
        profile_and_compile path (identical fields to V20) and the new
        facts_only observation path."""
        facts: Dict[str, Any] = {}
        if self.replica_launch is not None:
            try:
                facts.update(replica_median_us(self.replica_launch, iters=3))
                # V15-R1: smoothed column alongside the instantaneous
                # median (EMA alpha=0.2, same formula as the aobo
                # _observe mirror).  ADD-ONLY: no existing key or
                # decision input changes.  V21-C1/F2: the chain is
                # PER-BUCKET now.
                m = facts.get("replica_median_us")
                if isinstance(m, (int, float)):
                    key = self._bucket_key(req)
                    prev = self._replica_ema_by_bucket.get(key)
                    ema = m if prev is None else 0.2 * m + 0.8 * prev
                    self._replica_ema_by_bucket[key] = ema
                    facts["replica_median_us_ema"] = round(ema, 3)
                    facts["ema_bucket"] = key
            except Exception as e:
                facts["replica_error"] = str(e)
        if want_cupti:
            facts.update(collect_or_unavailable(
                launch_fn=self.replica_launch if os.environ.get(
                    "PACT_CUPTI_PROFILING") == "1" else None))
        return facts

    def handle(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if req.get("type") == "release_shm":
            self.shm.release(req.get("shm_name", ""))
            return {"msg_id": req.get("msg_id"), "type": "released"}
        if req.get("type") == "facts_only":
            # V21-C1 (F1 fix): lightweight observation RPC — facts only,
            # no decide, no compile, no shm.  Keeps the client's drift
            # state machines fed on STATIONARY done buckets (the F1
            # starvation chain).  BR-17: the served count is visible
            # state, never silent.
            t0 = time.monotonic()
            self._facts_only_served += 1
            facts = self._collect_facts(
                req, want_cupti=bool(req.get("want_cupti")))
            # V22 1-2 (K1): surface the CUPTI mutex counters on the
            # observation path (ADD-ONLY; the profile_and_compile
            # response field set is V20-frozen bit-for-bit).
            return {"msg_id": req.get("msg_id"), "type": "facts",
                    "bucket": self._bucket_key(req),
                    "facts": {k: v for k, v in facts.items() if k != "lib"},
                    "cupti_stats": cupti_stats(),
                    "profile_time_ms": (time.monotonic() - t0) * 1000.0}
        if req.get("type") != "profile_and_compile":
            return {"msg_id": req.get("msg_id"), "type": "error",
                    "reason": f"unknown type {req.get('type')}"}
        t0 = time.monotonic()
        facts = self._collect_facts(req, want_cupti=True)
        cfg = req.get("current_config") or {}
        try:
            n_regs = int(cfg.get("n_regs") or facts.get("regs_per_thread") or 0)
        except (TypeError, ValueError):
            n_regs = 0
        if n_regs > 0:
            facts["regs_per_thread"] = n_regs
        profile_ms = (time.monotonic() - t0) * 1000.0
        wl = req.get("workload") or {}
        batch = int(wl.get("B") or 1)
        seq = int(wl.get("S") or 0)
        geometry = req.get("geometry") or {}
        decision = decide(facts, batch, seq, table=self.table,
                          cfg=geometry if geometry.get("S") else None)
        t1 = time.monotonic()
        shm_name = ""
        shm_size = 0
        header: Dict[str, Any] = {}
        if self.compile_fn is not None:
            kernel = self.compile_fn(decision["extra_env"],
                                     decision.get("options_override") or {})
            header, cubin = kernel_to_blob(
                kernel, extra_header={"family": decision["family"]})
            shm_name = f"pact_k_{req.get('msg_id', 'x')}_{os.getpid()}"
            blob = self.shm.create(shm_name, header, cubin)
            shm_size = blob.shm_size
        compile_ms = (time.monotonic() - t1) * 1000.0
        return {
            "msg_id": req.get("msg_id"),
            "type": "kernel_ready",
            "bucket": self._bucket_key(req),   # V21-C1/F2: client EMA key
            "shm_name": shm_name,
            "shm_size": shm_size,
            "family": decision["family"],
            "new_config": {
                "warps": int(decision["extra_env"].get("PACT_OVERRIDE_WARPS") or
                             header.get("num_warps") or 4),
                # V10-P0 (V10-5): prefer the loop decision the compiler actually
                # wrote (P6's pact_optimal_num_stages); header num_stages is the
                # OPTIONS value and only remains as a legacy fallback.
                "stages": int(decision["extra_env"].get("PACT_OVERRIDE_STAGES") or
                              header.get("pact_loop_stages") or
                              header.get("num_stages") or 3),
                "V": int(os.environ.get("PACT_OVERRIDE_V") or 0),
            },
            "extra_env": decision["extra_env"],
            "options_override": decision["options_override"],
            "counter_adjusted": decision.get("counter_adjusted"),
            "facts": {k: v for k, v in facts.items() if k != "lib"},
            "profile_time_ms": profile_ms,
            "compile_time_ms": compile_ms,
        }

    def start(self):
        self.server.start()

    def stop(self):
        self.server.stop()
        self.shm.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/pact_compiler.sock")
    args = ap.parse_args()
    svc = CompilerService(socket_path=args.socket)
    svc.start()
    print(json.dumps({"listening": args.socket}), flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        svc.stop()


if __name__ == "__main__":
    main()
