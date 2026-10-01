"""Inference-process helper: sniff (B,S), request a compile, G4-swap."""
from __future__ import annotations

import itertools
import os
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Dict, Optional, Tuple

from triton.pact.compiler.explicit_compiler import compile_explicit
from triton.pact.ipc.socket_client import PactSocketClient
from triton.pact.runtime.hot_swapper import HotSwapper
from triton.pact.runtime.workload_sniffer import should_trigger


_MSG_IDS = itertools.count(1)

# BR-17 (V18 G-1): swallowed exceptions on the decision/measurement path
# stay swallowed (the bridge must never crash the host) but become
# COUNTABLE.  A silent _init_handles failure feeds measure_kernel a
# handle-less kernel and the resulting gain_percent silently steers the
# swap decision -- invisible without this counter (BR-8 _SWALLOWED
# precedent).  Read it from probes/bridge state, never changed behavior.
_SWALLOWED: Dict[str, Any] = {"init_handles": 0, "init_handles_last": ""}


def _note_swallowed(site: str, exc: BaseException) -> None:
    _SWALLOWED[site] = int(_SWALLOWED.get(site, 0)) + 1
    _SWALLOWED[f"{site}_last"] = repr(exc)[:120]


def _g4_iters(default: int = 10) -> int:
    return int(os.environ.get("PACT_G4_ITERS", str(default)))


class InferenceSession:
    def __init__(self, swapper: HotSwapper, socket_path: str = "/tmp/pact_compiler.sock"):
        self.swapper = swapper
        self.client = PactSocketClient(socket_path, timeout=30.0)
        self.prev_bucket: Optional[Tuple[str, str]] = None
        self.done_buckets = set()
        self.in_flight = False
        self.launch_count = 0
        # V16-T6 L1 trigger (PROTON_SURVEY_V16 §4): cuda-event EMA
        # deviation re-arms an L0-done bucket.  OFF unless
        # PACT_TRIGGER_L1=1; tau/k/cooldown tunable.  The EMA itself has
        # been computed service-side since v15 (replica_median_us_ema,
        # add-only, unconsumed) -- this is its first consumer.
        self._l1_enabled = os.environ.get("PACT_TRIGGER_L1") == "1"
        self._l1_tau = float(os.environ.get("PACT_TRIGGER_L1_TAU") or 0.20)
        self._l1_k = int(os.environ.get("PACT_TRIGGER_L1_K") or 3)
        self._l1_cooldown = float(
            os.environ.get("PACT_TRIGGER_L1_COOLDOWN") or 30.0)
        self._ema_baseline: Dict[str, float] = {}
        self._l1_streak = 0
        self._l1_last_s = 0.0
        # V17 S0-1 ①: maybe_request runs on BOTH the inline decode thread
        # and the async background worker, so the L1 read-modify-write
        # sequence races (streak double-count / baseline tear).  R2 §2.4
        # allows single-writer OR lock; the lock keeps the v9 sync
        # contract untouched (single-writer would have to own the
        # maybe_request dispatch itself).
        self._l1_lock = threading.Lock()
        # V12-P1 (PACT_ASYNC_PGO=1): async frame state — a single daemon
        # background worker owns the decide/compile/measure cycle; the swap
        # is consumed at a launch boundary.  All-zero until the env is set.
        self._bg_thread: Optional[threading.Thread] = None
        self._bg_queue: Optional[queue.Queue] = None
        self._async_pending: Optional[Future] = None
        self.async_state: Dict[str, Any] = {
            "submits": 0, "installs": 0, "install_ms": [],
            "last_plan": None, "last_error": None,
        }
        if os.environ.get("PACT_COMPILE_SUBPROC") == "1":
            # V13 Phase1: boot the compile worker in THIS untimed init
            # window — a cold Popen during the first background cycle
            # forks the multi-GB process and freezes every thread for
            # ~270ms (measured; posix_spawn avoids it only when the
            # process already exists).
            try:
                from triton.pact.controller.compile_worker import \
                    get_compile_subprocess
                get_compile_subprocess().warm_start()
            except Exception as e:  # BR-17: counted, not silent
                _SWALLOWED["warm_start"] = _SWALLOWED.get("warm_start", 0) + 1
                _SWALLOWED["last_warm_start_err"] = str(e)[:100]

    def step(self, batch: int, seq_len: int,
             current_config: Optional[Dict] = None,
             geometry: Optional[Dict] = None) -> Optional[Dict]:
        """V10-4 scheduler-step entry: run the decide/compile/G4 cycle at a
        capture-safe point chosen by the caller (e.g. a scheduler step or a
        workload-change boundary) instead of inline inside the attention
        forward. Same contract and dedup state as maybe_request; returns the
        plan dict or None."""
        return self.maybe_request(batch, seq_len,
                                   current_config=current_config,
                                   geometry=geometry)

    def _l1_observe(self, resp: Optional[Dict]) -> None:
        """Track the service-reported replica EMA: first settled value per
        bucket becomes the baseline; consecutive over-tau deviations build
        the streak that later re-arms the bucket.  V17 S0-1: the whole
        read-modify-write runs under _l1_lock (two caller threads)."""
        if not self._l1_enabled or not isinstance(resp, dict):
            return
        facts = (resp.get("facts") or {}) if "facts" in resp else \
            ((resp.get("service") or {}).get("facts") or {})
        ema = facts.get("replica_median_us_ema")
        if not isinstance(ema, (int, float)) or ema <= 0:
            return
        with self._l1_lock:
            key = f"{self.prev_bucket}"
            base = self._ema_baseline.get(key)
            if base is None:
                self._ema_baseline[key] = float(ema)
                self._l1_streak = 0
                return
            if abs(ema - base) / base > self._l1_tau:
                self._l1_streak += 1
            else:
                self._l1_streak = 0
                # slow baseline drift-in when healthy (ema of emas)
                self._ema_baseline[key] = 0.7 * base + 0.3 * float(ema)

    def _l1_ready(self) -> bool:
        """Streak >= k AND cooldown elapsed since the last L1 re-arm
        (check-and-consume under _l1_lock)."""
        with self._l1_lock:
            if not self._l1_enabled or self._l1_streak < self._l1_k:
                return False
            now = time.monotonic()
            if now - self._l1_last_s < self._l1_cooldown:
                return False
            self._l1_last_s = now
            self._l1_streak = 0     # re-arm consumes the streak
            return True

    def maybe_request(self, batch: int, seq_len: int,
                      current_config: Optional[Dict] = None,
                      geometry: Optional[Dict] = None) -> Optional[Dict]:
        fire, bucket = should_trigger(self.prev_bucket, batch, seq_len)
        l1 = False
        if bucket in self.done_buckets:
            l1 = self._l1_ready()   # L1 re-arm of an L0-done bucket
            if not l1:
                return None
        if not (fire or l1) or self.in_flight:
            return None
        self.in_flight = True
        n_regs = 0
        cur = self.swapper.current
        try:
            if getattr(cur, "n_regs", None) in (None, 0) and hasattr(cur, "_init_handles"):
                cur._init_handles()
            n_regs = int(getattr(cur, "n_regs", 0) or 0)
        except Exception:  # BR-17: counted
            _SWALLOWED["n_regs_init"] = _SWALLOWED.get("n_regs_init", 0) + 1
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
        self._l1_observe(resp)
        if resp is None or resp.get("type") != "kernel_ready":
            return resp
        self.prev_bucket = bucket
        self.done_buckets.add(bucket)
        # V11-6a R2 fast path: a variant prewarmed into the resident pool
        # (idle-window compile, offline-validated mapping) switches by slot
        # exchange — no compile, no G4 timing on the critical path.  The
        # plan dict records the pool hit and the R4 segments.
        fam = resp.get("family")
        if fam and getattr(self.swapper, "pool_contains", None) and \
                self.swapper.pool_contains(fam):
            import time as _t
            t0 = _t.monotonic()
            hit = self.swapper.swap_from_pool(fam)
            plan = {
                "swapped": bool(hit), "rolled_back": False,
                "gain_percent": None, "cache_hit": True, "pool_hit": fam,
                "slot_swap_ms": (_t.monotonic() - t0) * 1000.0,
                "local_compile_ms": 0.0, "measure_time_ms": 0.0,
                "service": {k: resp[k] for k in
                            ("family", "new_config", "profile_time_ms",
                             "compile_time_ms", "shm_name") if k in resp},
            }
            plan["overhead_ms"] = (float(resp.get("profile_time_ms") or 0)
                                   + plan["slot_swap_ms"])
            if resp.get("shm_name"):
                self.client.try_request(
                    {"msg_id": msg["msg_id"], "type": "release_shm",
                     "shm_name": resp["shm_name"]}, timeout=2.0)
            return plan
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
        if os.environ.get("PACT_GRAPH_SERVICE") == "1" and \
                plan.get("swapped"):
            # V13 Phase0: under a captured graph the slot exchange above is
            # invisible to replay — offer the merged module and queue the
            # node retarget here (this sync path runs on the lazy-arm
            # background thread, never inside a forward/replay).
            self._gs_offer(kernel, resp.get("family"))
            plan["graph_retarget"] = self._gs_submit(
                kernel, resp.get("family"))
        if resp.get("shm_name"):
            self.client.try_request(
                {"msg_id": msg["msg_id"], "type": "release_shm",
                 "shm_name": resp["shm_name"]}, timeout=2.0)
        return plan

    # ---- V12-P1: async frame, PACT_ASYNC_PGO=1 ----------------------------
    # Protocol ported from the aobo tree's AsyncKernelSwitch: every forward
    # launches the current slot and returns; the decide/compile/measure cycle
    # runs on one background worker; installation is an atomic slot exchange
    # consumed at the next launch boundary (an R3 safe point).  Unlike the
    # aobo in-process inline decider, the decision itself still goes through
    # the out-of-process compiler service — the pgo production semantics.
    def _submit_bg(self, fn) -> Future:
        if self._bg_thread is None:
            self._bg_queue = queue.Queue()
            self._bg_thread = threading.Thread(
                target=self._bg_worker, name="pact-async-pgo", daemon=True)
            self._bg_thread.start()
        fut: Future = Future()
        self._bg_queue.put((fn, fut))
        return fut

    def _bg_worker(self):
        while True:
            fn, fut = self._bg_queue.get()
            if fn is None:
                return
            try:
                fut.set_result(fn())
            except BaseException as e:  # noqa: BLE001 - surfaced via install
                fut.set_exception(e)

    def _release_shm(self, msg: Dict, resp: Dict):
        if resp.get("shm_name"):
            try:
                self.client.try_request(
                    {"msg_id": msg["msg_id"], "type": "release_shm",
                     "shm_name": resp["shm_name"]}, timeout=2.0)
            except Exception:  # BR-17: counted
                _SWALLOWED["release_shm"] = _SWALLOWED.get("release_shm", 0) + 1

    # ---- V13 Phase0: in-graph retarget (async frame stays intact) --------
    def _gs(self):
        """The in-graph retarget service, or None when not armed.  Under a
        captured graph the real install is the node SetParams; the slot
        exchange below stays as the eager-path semantics."""
        if os.environ.get("PACT_GRAPH_SERVICE") != "1":
            return None
        try:
            from triton.pact.runtime.graph_service import get_service
            return get_service()
        except Exception:  # BR-17: counted
            _SWALLOWED["gs_import"] = _SWALLOWED.get("gs_import", 0) + 1
            return None

    def _gs_offer(self, kernel, variant) -> bool:
        """V13 Phase0 (full-vocab era): offering degenerated to a readiness
        check.  V16-T4 lazy vocab: the misses are real now -- compile +
        load the variant on demand as its own single-entry cubin
        (graph_service.offer; ms..s, we run on background frames only)
        and report readiness once loaded."""
        svc = self._gs()
        if svc is None or kernel is None or not variant:
            return False
        try:
            from triton.pact.runtime.graph_service import FAMILY_ALIAS
            v = FAMILY_ALIAS.get(str(variant), str(variant))
            name = getattr(self, "_gs_jit_name", None) or \
                str(kernel.metadata.name)
            ok = svc.has_variant(name, v)
            if not ok and hasattr(svc, "offer"):
                ok = svc.offer(name, v)
            if ok:
                self._gs_jit_name = name
            return ok
        except Exception as e:  # BR-17: counted
            _SWALLOWED["gs_offer"] = _SWALLOWED.get("gs_offer", 0) + 1
            _SWALLOWED["last_gs_offer_err"] = str(e)[:100]
            return False

    def _gs_submit(self, kernel, variant) -> bool:
        """Boundary-safe (us-scale): queue the retarget, consumed at the
        next replay boundary."""
        svc = self._gs()
        name = getattr(self, "_gs_jit_name", None)
        if svc is None or kernel is None or not variant or not name:
            return False
        try:
            return svc.submit_retarget(name, str(variant))
        except Exception as e:  # BR-17: counted
            _SWALLOWED["gs_submit"] = _SWALLOWED.get("gs_submit", 0) + 1
            _SWALLOWED["last_gs_submit_err"] = str(e)[:100]
            return False

    def _gs_rollback(self) -> None:
        svc = self._gs()
        name = getattr(self, "_gs_jit_name", None)
        if svc is not None and name:
            try:
                svc.rollback(name)
            except Exception:  # BR-17: counted
                _SWALLOWED["gs_rollback"] = _SWALLOWED.get("gs_rollback", 0) + 1

    def decide_async(self, batch: int, seq_len: int,
                     current_config: Optional[Dict] = None,
                     geometry: Optional[Dict] = None) -> Optional[Dict]:
        """Submit the decide/compile/measure cycle to the background worker
        and return immediately — a forward is never blocked by it.  The slot
        swap is deferred to the next launch boundary (install_at_boundary).
        While a submitted cycle is unconsumed, further triggers are skipped
        (bucket dedup still applies once the cycle lands)."""
        fire, bucket = should_trigger(self.prev_bucket, batch, seq_len)
        l1 = False
        if bucket in self.done_buckets:
            l1 = self._l1_ready()   # V16-T6 L1 re-arm
            if not l1:
                return None
        if not (fire or l1) or self.in_flight:
            return None
        self.in_flight = True
        fut = self._submit_bg(
            lambda: self._bg_cycle(batch, seq_len, current_config,
                                   geometry, bucket))
        self._async_pending = fut
        self.async_state["submits"] += 1
        return {"submitted": True,
                "bucket": [str(x) for x in bucket] if bucket else None}

    def _bg_cycle(self, batch: int, seq_len: int,
                  current_config: Optional[Dict], geometry: Optional[Dict],
                  bucket) -> Dict[str, Any]:
        """maybe_request's body minus the slot swap: service RPC decide, R2
        pool hit or local compile + handle load, then the G4 base/candidate
        measurements — all off the forward path.  The swap itself is a
        launch-boundary pointer exchange (install_at_boundary)."""
        msg: Optional[Dict] = None
        resp: Optional[Dict] = None
        try:
            n_regs = 0
            cur = self.swapper.current
            try:
                if getattr(cur, "n_regs", None) in (None, 0) and \
                        hasattr(cur, "_init_handles"):
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
                "kernel_key": getattr(self.swapper.jit_fn, "__name__",
                                      "kernel"),
                "workload": {"B": batch, "S": seq_len},
                "geometry": geometry or {},
                "current_config": cfg,
            }
            resp = self.client.try_request(msg, timeout=120.0)
            self._l1_observe(resp)
            if resp is None or resp.get("type") != "kernel_ready":
                return {"kind": "no_kernel",
                        "resp_type": (resp or {}).get("type", "none")}
            self.prev_bucket = bucket
            self.done_buckets.add(bucket)
            fam = resp.get("family")
            if fam and getattr(self.swapper, "pool_contains", None) and \
                    self.swapper.pool_contains(fam):
                if os.environ.get("PACT_GRAPH_SERVICE") == "1":
                    # graph-mode installs retarget the captured node — the
                    # pooled kernel must reach the merged module first
                    self._gs_offer(self.swapper.pool_kernel(fam), fam)
                return {"kind": "pool", "family": fam, "resp": resp,
                        "msg": msg}
            extra_env = resp.get("extra_env") or {"PACT_ENABLE": "1"}
            options = resp.get("options_override") or None
            if os.environ.get("PACT_COMPILE_SUBPROC") == "1":
                # V13 Phase1: the MLIR/ptxas section runs in the dedicated
                # subprocess (user-space CPU — no GIL, no driver lock, no
                # MLIR thread to terminate under long concurrency).  The
                # local compile_candidate below is then a disk-cache hit
                # and the G4 pair measurement runs here on the background
                # worker (the V12 regime, window worst 0.5ms).
                r = self._subproc_compile(extra_env, options)
                if r is None or r.get("status") != "ok":
                    return {"kind": "error",
                            "error": f"subproc compile: "
                                     f"{getattr(self, '_csub_err', None)}",
                            "msg": msg, "resp": resp}
                t_c = time.monotonic()
                kernel = self.swapper.compile_candidate(extra_env, options)
                compile_ms = (time.monotonic() - t_c) * 1000.0
                if getattr(kernel, "_init_handles", None) and not getattr(
                        kernel, "function", None):
                    try:
                        kernel._init_handles()
                    except Exception as e:  # noqa: BLE001
                        _note_swallowed("init_handles", e)
                iters = _g4_iters()
                base_us = self.swapper.measure_kernel(
                    self.swapper.slots[0], iters)
                cand_us = self.swapper.measure_kernel(kernel, iters)
                t_meas = time.monotonic()
                return {
                    "kind": "kernel", "kernel": kernel, "resp": resp,
                    "msg": msg, "base_us": base_us, "cand_us": cand_us,
                    "gain_percent": 100.0 * (base_us - cand_us) /
                    max(base_us, 1e-6),
                    "compile_ms": compile_ms,
                    "measure_ms": (t_meas - t_c - compile_ms / 1000.0)
                    * 1000.0,
                    "subproc": True,
                    "worker_ms": r.get("worker_ms"),
                }
            t_compile = time.monotonic()
            kernel = self.swapper.compile_candidate(extra_env, options)
            compile_ms = (time.monotonic() - t_compile) * 1000.0
            # Force the module load HERE (background thread) so the first
            # post-install forward does not pay it inside the serving path.
            if getattr(kernel, "_init_handles", None) and not getattr(
                    kernel, "function", None):
                try:
                    kernel._init_handles()
                except Exception as e:  # noqa: BLE001
                    _note_swallowed("init_handles", e)
            iters = _g4_iters()
            base_us = self.swapper.measure_kernel(self.swapper.slots[0],
                                                  iters)
            cand_us = self.swapper.measure_kernel(kernel, iters)
            if os.environ.get("PACT_GRAPH_SERVICE") == "1":
                # link+load the variant into the merged module while still
                # on the background worker (never inside a forward/replay)
                self._gs_offer(kernel, fam)
            t_meas = time.monotonic()
            return {
                "kind": "kernel", "kernel": kernel, "resp": resp, "msg": msg,
                "base_us": base_us, "cand_us": cand_us,
                "gain_percent": 100.0 * (base_us - cand_us) /
                max(base_us, 1e-6),
                "compile_ms": compile_ms,
                "measure_ms": (t_meas - t_compile - compile_ms / 1000.0)
                * 1000.0,
            }
        except Exception as e:  # noqa: BLE001 - drained by install_at_boundary
            return {"kind": "error", "error": repr(e),
                    "msg": msg, "resp": resp}
        finally:
            self.in_flight = False

    def _subproc_compile(self, extra_env, options) -> Optional[Dict]:
        """V13 Phase1 (PACT_COMPILE_SUBPROC=1): ship the compile+measure
        section to the resident worker process.  Called on the frame's
        background thread only — pipe waits do not hold the GIL."""
        try:
            from triton.pact.controller.compile_worker import (
                _describe, get_compile_subprocess)
            svc = get_compile_subprocess()
            self._csub = svc
            jit_fn = self.swapper.jit_fn
            spec = {
                "jit_module": getattr(getattr(jit_fn, "fn", None),
                                      "__module__", None)
                or type(jit_fn).__module__,
                "jit_attr": getattr(getattr(jit_fn, "fn", None),
                                    "__name__", None) or "kernel",
                "args_desc": _describe(self.swapper.args),
                "kwargs": dict(self.swapper.kwargs),
                "grid": list(self.swapper.grid),
                "extra_env": dict(extra_env),
                "baseline_env": {"PACT_ENABLE": "0"},
                "options_override": options,
                "iters": _g4_iters(),
            }
            r = svc.run(spec)
            if r is None:
                self._csub_err = svc.last_error
            return r
        except Exception as e:  # noqa: BLE001 - drained by install
            self._csub_err = repr(e)[:200]
            return None

    def install_at_boundary(self) -> Optional[Dict]:
        """Consume a finished background cycle at a launch boundary (R3 safe
        point): the swap is a slot-pointer exchange — sub-µs on the forward
        path.  Failed cycles are drained here too, so the trigger can retry
        on a later forward (matching the v11 maybe_request retry semantics).
        The G4 post-check runs on the background worker: an async rollback
        keeps the boundary µs-scale while preserving the rollback contract."""
        fut = self._async_pending
        if fut is None or not fut.done():
            return None
        self._async_pending = None
        payload = fut.result()
        st = self.async_state
        kind = payload.get("kind")
        if kind == "pool":
            resp = payload["resp"]
            t0 = time.monotonic()
            hit = self.swapper.swap_from_pool(payload["family"])
            dt = (time.monotonic() - t0) * 1000.0
            plan = {
                "swapped": bool(hit), "rolled_back": False,
                "gain_percent": None, "cache_hit": True,
                "pool_hit": payload["family"], "slot_swap_ms": dt,
                "local_compile_ms": 0.0, "measure_time_ms": 0.0,
                "async": True,
                "service": {k: resp[k] for k in
                            ("family", "new_config", "profile_time_ms",
                             "compile_time_ms", "shm_name") if k in resp},
            }
            plan["overhead_ms"] = (
                float(resp.get("profile_time_ms") or 0) + dt)
            self._release_shm(payload["msg"], resp)
            graph_ok = self._gs_submit(
                self.swapper.pool_kernel(payload["family"]),
                payload["family"]) if os.environ.get(
                    "PACT_GRAPH_SERVICE") == "1" else None
            if graph_ok is not None:
                plan["graph_retarget"] = bool(graph_ok)
            st["installs"] += 1
            st["install_ms"].append(dt)
            st["last_plan"] = plan
            return plan
        if kind == "kernel":
            resp = payload["resp"]
            t0 = time.monotonic()
            swapped = payload["gain_percent"] >= 0.0
            if swapped:
                self.swapper.swap(payload["kernel"])
            dt = (time.monotonic() - t0) * 1000.0
            plan = {
                "baseline_us": payload["base_us"],
                "candidate_us": payload["cand_us"],
                "post_us": None,
                "gain_percent": payload["gain_percent"],
                "swapped": swapped, "rolled_back": False,
                "cache_hit": payload["compile_ms"] < 50.0,
                "local_compile_ms": payload["compile_ms"],
                "measure_time_ms": payload["measure_ms"],
                "slot_swap_ms": dt, "async": True,
                "subproc": payload.get("subproc", False),
                "worker_ms": payload.get("worker_ms"),
                "service": {k: resp[k] for k in
                            ("family", "new_config", "profile_time_ms",
                             "compile_time_ms", "shm_name") if k in resp},
            }
            plan["overhead_ms"] = (
                float(resp.get("profile_time_ms") or 0) +
                float(resp.get("compile_time_ms") or 0) +
                float(plan.get("measure_time_ms") or 0))
            self._release_shm(payload["msg"], resp)
            if os.environ.get("PACT_GRAPH_SERVICE") == "1":
                plan["graph_retarget"] = self._gs_submit(
                    payload["kernel"], resp.get("family"))
            st["installs"] += 1
            st["install_ms"].append(dt)
            st["last_plan"] = plan
            if swapped:
                self._submit_bg(
                    lambda: self._bg_post_check(payload["base_us"], plan))
            return plan
        st["last_error"] = payload.get("error") or payload.get("resp_type")
        return None

    def _bg_post_check(self, base_us: float, plan: Dict[str, Any]):
        """Asynchronous G4 post-verification: re-measure the installed slot
        on the background worker and roll back if it regressed."""
        try:
            post_us = self.swapper.measure_kernel(
                self.swapper.current, max(_g4_iters() // 2, 3))
            plan["post_us"] = post_us
            if post_us >= base_us:
                self.swapper.swap(self.swapper.slots[0])
                self._gs_rollback()  # graph nodes back to the captured func
                plan["rolled_back"] = True
                plan["swapped"] = False
        except Exception as e:  # noqa: BLE001 - surfaced via async_state
            plan["post_check_error"] = repr(e)
        return plan

    def launch(self):
        self.launch_count += 1
        if os.environ.get("PACT_ASYNC_PGO") == "1" and \
                self._async_pending is not None:
            self.install_at_boundary()
        return self.swapper.launch()
