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
from triton.pact.decider.online_decider import decide
from triton.pact.ipc.shm_manager import ShmManager
from triton.pact.ipc.socket_server import PactSocketServer
from triton.pact.profiler.cupti_collector import collect_or_unavailable
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

    def handle(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if req.get("type") == "release_shm":
            self.shm.release(req.get("shm_name", ""))
            return {"msg_id": req.get("msg_id"), "type": "released"}
        if req.get("type") != "profile_and_compile":
            return {"msg_id": req.get("msg_id"), "type": "error",
                    "reason": f"unknown type {req.get('type')}"}
        t0 = time.monotonic()
        facts: Dict[str, Any] = {}
        if self.replica_launch is not None:
            try:
                facts.update(replica_median_us(self.replica_launch, iters=3))
            except Exception as e:
                facts["replica_error"] = str(e)
        facts.update(collect_or_unavailable(
            launch_fn=self.replica_launch if os.environ.get(
                "PACT_CUPTI_PROFILING") == "1" else None))
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
        decision = decide(facts, batch, seq, table=self.table)
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
            "shm_name": shm_name,
            "shm_size": shm_size,
            "family": decision["family"],
            "new_config": {
                "warps": int(decision["extra_env"].get("PACT_OVERRIDE_WARPS") or
                             header.get("num_warps") or 4),
                "stages": int(decision["extra_env"].get("PACT_OVERRIDE_STAGES") or
                              header.get("num_stages") or 3),
                "V": int(os.environ.get("PACT_OVERRIDE_V") or 0),
            },
            "extra_env": decision["extra_env"],
            "options_override": decision["options_override"],
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
