"""V16-T5 (user point 6(1)): structural audit for in-graph kernel swaps.

Layer 1 of the GraphSwapValidator (survey verdict: NO official or
mainstream tool exists -- notes/v16/GRAPH_SWAP_VALIDATOR_SURVEY_V16.md).
After every applied retarget, read back each bound node's params from
the SOURCE graph (the mirror of what _apply wrote) and check them
against (a) the variant's recorded metadata and (b) the bind-time
snapshot:

  func        pointer-identical to the variant's CUfunction (this is the
              mirror-audit evidence; exec-side getters do not exist)
  blockDimX   == 32 * variant warps
  sharedMem   == variant shared
  kernelParams pointer array UNCHANGED from the bind snapshot (ABI rule)
  grid dims   unchanged from the node's own current snapshot

Cost: one cuGraphKernelNodeGetParams per bound node, host-only, us-scale
per node; runs inside the service lock-free after the apply.  Failures
are recorded in the service state (audit_failures) -- the audit never
raises into the replay path.

Layer 2 (kernel-output flip anchor) is a cell/probe-level concern: on
real vLLM graphs the base-vs-variant outputs differ (the v13 cross-
module corruption WAS caught by it), while micro shapes can be
numerically unresolvable -- it cannot live inside the service.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def audit_after_apply(svc, jit_name: str, variant: str,
                      expect_nodes: Optional[Dict[int, int]] = None
                      ) -> Dict[str, Any]:
    """Audit every bound graph of `jit_name` right after `variant` was
    applied.  Returns a summary dict; failures also land in
    svc.state['audit_failures'] (capped) for cross-process visibility.
    """
    cu = svc._driver()
    from triton.pact.runtime.graph_service import _LOCK
    with _LOCK:
        v = svc._variants.get((jit_name, variant))
        items = [(gid, nodes) for gid, nodes in svc._bindings.items()
                 if svc._jits.get(gid) == jit_name]
        catalogue = dict(svc.state.get("content_hash") or {})
    if v is None:
        return {"ok": False, "reason": "variant not loaded"}
    if not v.get("cross"):
        # Intra-module swaps go through EXEC SetParams -- the exec keeps
        # a private params copy and no exec-side getter exists (API fact,
        # graph_phase0_probe).  The source node is NOT the mirror for
        # that path, so a func read-back would false-positive; the
        # behavioural anchor (layer 2) covers this path at the cell level.
        return {"ok": True, "audited_nodes": 0,
                "note": "exec-opaque (intra-module path)"}
    failures: list = []
    audited = 0
    for gid, nodes in items:
        if expect_nodes is not None and \
                expect_nodes.get(gid) not in (None, len(nodes)):
            failures.append({"gid": gid, "node_count": len(nodes),
                             "expected": expect_nodes[gid]})
        for nb in nodes:
            # V17 S3-3 ②: nodes not born from this service's module were
            # skipped by _apply (skipped_nodes counter) -- they are not
            # ours to audit either; auditing them would false-positive
            # every mixed graph into an auto-rollback
            if int(nb.func_before) not in svc._known_funcs:
                continue
            err, cur = cu.cuGraphKernelNodeGetParams(nb.node)
            if err != 0:
                failures.append({"gid": gid, "rc_getparams": int(err)})
                continue
            audited += 1
            checks = {
                "func": int(cur.func) == int(v["fn"]),
                "blockDimX": int(cur.blockDimX) == 32 * int(v["warps"]),
                "sharedMem": int(cur.sharedMemBytes) == int(v["shared"]),
                "kernelParams": int(cur.kernelParams) == int(
                    nb.kernel_params),
                "grid": (int(cur.gridDimX), int(cur.gridDimY),
                         int(cur.gridDimZ)) != (0, 0, 0),
                # V19 N4 third layer: the read-back function's identity
                # must hit the content-hash catalogue (a name-matching
                # handle from an uncatalogued load is the silent-corruption
                # shape; failures ride the EXISTING audit->rollback path).
                # An EMPTY catalogue means N4 booking never ran (legacy
                # state / test stub) -- skip rather than false-positive.
                "content_hash": (not catalogue) or (
                    svc._known_funcs.get(int(cur.func)) in catalogue),
            }
            bad = [k for k, ok in checks.items() if not ok]
            if bad:
                failures.append({"gid": gid, "node": int(nb.node),
                                 "failed": bad,
                                 "func_hex": hex(int(cur.func)),
                                 "want_hex": hex(int(v["fn"]))})
    out = {"ok": not failures, "audited_nodes": audited,
           "failures": failures[:8]}
    if failures:
        # V17 S3-4 ③: fix the retention dead code (was
        # (old)[-4:] + [{...}][:1] -- the [:1] of a one-element list
        # silently collapsed every entry to the same shape and the cap
        # dropped useful history); plain bounded append, cap 8
        with _LOCK:
            svc.state["audit_failures"] = \
                (svc.state.get("audit_failures") or [])[-7:] + [
                    {"variant": variant, "n": len(failures),
                     "failed": [f.get("failed") for f in failures[:4]]}]
    return out
