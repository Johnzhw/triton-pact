"""Inference-process helper: sniff (B,S), request a compile, G4-swap."""
from __future__ import annotations

import itertools
import os
import time
from typing import Any, Dict, Optional, Tuple

from triton.pact.compiler.explicit_compiler import compile_explicit
from triton.pact.ipc.socket_client import PactSocketClient
from triton.pact.runtime.hot_swapper import HotSwapper
from triton.pact.runtime.workload_sniffer import should_trigger


_MSG_IDS = itertools.count(1)


class InferenceSession:
    def __init__(self, swapper: HotSwapper, socket_path: str = "/tmp/pact_compiler.sock"):
        self.swapper = swapper
        self.client = PactSocketClient(socket_path, timeout=30.0)
        self.prev_bucket: Optional[Tuple[str, str]] = None
        self.done_buckets = set()
        self.in_flight = False
        self.launch_count = 0

    def maybe_request(self, batch: int, seq_len: int,
                      current_config: Optional[Dict] = None,
                      geometry: Optional[Dict] = None) -> Optional[Dict]:
        fire, bucket = should_trigger(self.prev_bucket, batch, seq_len)
        if bucket in self.done_buckets:
            return None
        if not fire or self.in_flight:
            return None
        self.in_flight = True
        n_regs = 0
        cur = self.swapper.current
        try:
            if getattr(cur, "n_regs", None) in (None, 0) and hasattr(cur, "_init_handles"):
                cur._init_handles()
            n_regs = int(getattr(cur, "n_regs", 0) or 0)
        except Exception:
            n_regs = 0
        cfg = dict(current_config or {"warps": 4, "stages": 3, "V": 0})
        if n_regs > 0:
            cfg["n_regs"] = n_regs
        msg = {
            "msg_id": f"req_{next(_MSG_IDS)}",
            "type": "profile_and_compile",
            "kernel_key": getattr(self.swapper.jit_fn, "__name__", "kernel"),
            "workload": {"B": batch, "S": seq_len},
            "geometry": geometry or {},
            "current_config": cfg,
        }
        resp = self.client.try_request(msg, timeout=120.0)
        self.in_flight = False
        if resp is None or resp.get("type") != "kernel_ready":
            return resp
        self.prev_bucket = bucket
        self.done_buckets.add(bucket)
        extra_env = resp.get("extra_env") or {"PACT_ENABLE": "1"}
        options = resp.get("options_override") or None
        t_compile = time.monotonic()
        kernel = self.swapper.compile_candidate(extra_env, options)
        compile_ms = (time.monotonic() - t_compile) * 1000.0
        cache_hit = compile_ms < 50.0
        plan = self.swapper.g4_install(kernel)
        plan["cache_hit"] = cache_hit
        plan["local_compile_ms"] = compile_ms
        plan["service"] = {k: resp[k] for k in
                           ("family", "new_config", "profile_time_ms",
                            "compile_time_ms", "shm_name") if k in resp}
        ov = (float(resp.get("profile_time_ms") or 0) +
              float(resp.get("compile_time_ms") or 0) +
              float(plan.get("measure_time_ms") or 0))
        plan["overhead_ms"] = ov
        if resp.get("shm_name"):
            self.client.try_request(
                {"msg_id": msg["msg_id"], "type": "release_shm",
                 "shm_name": resp["shm_name"]}, timeout=2.0)
        return plan

    def launch(self):
        self.launch_count += 1
        return self.swapper.launch()
