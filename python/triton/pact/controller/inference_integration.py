"""Inference-process helper: sniff (B,S), request a compile, G4-swap."""
from __future__ import annotations

import itertools
import os
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
        self.in_flight = False
        self.launch_count = 0

    def maybe_request(self, batch: int, seq_len: int,
                      current_config: Optional[Dict] = None) -> Optional[Dict]:
        fire, bucket = should_trigger(self.prev_bucket, batch, seq_len)
        if not fire or self.in_flight:
            return None
        self.in_flight = True
        msg = {
            "msg_id": f"req_{next(_MSG_IDS)}",
            "type": "profile_and_compile",
            "kernel_key": getattr(self.swapper.jit_fn, "__name__", "kernel"),
            "workload": {"B": batch, "S": seq_len},
            "current_config": current_config or {"warps": 4, "stages": 3, "V": 0},
        }
        resp = self.client.try_request(msg, timeout=120.0)
        self.in_flight = False
        if resp is None or resp.get("type") != "kernel_ready":
            return resp
        self.prev_bucket = bucket
        extra_env = resp.get("extra_env") or {"PACT_ENABLE": "1"}
        options = resp.get("options_override") or None
        kernel = self.swapper.compile_candidate(extra_env, options)
        plan = self.swapper.g4_install(kernel)
        plan["service"] = {k: resp[k] for k in
                           ("family", "new_config", "profile_time_ms",
                            "compile_time_ms", "shm_name") if k in resp}
        if resp.get("shm_name"):
            self.client.try_request(
                {"msg_id": msg["msg_id"], "type": "release_shm",
                 "shm_name": resp["shm_name"]}, timeout=2.0)
        return plan

    def launch(self):
        self.launch_count += 1
        return self.swapper.launch()
