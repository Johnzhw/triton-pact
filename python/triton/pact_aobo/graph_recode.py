"""CUDA-Graph node re-code spike (V11-6b stage 1 feasibility).

Question: on this host (RTX 3080 SM86, CUDA 13 driver), can the kernel
FUNCTION of a node inside an ALREADY-INSTANTIATED graph exec be replaced via
cuGraphExecKernelNodeSetParams, so that subsequent launches of the SAME exec
run the new binary — no re-capture?

Method:
  1. one PTX module with two entries over a u32 buffer: ``addone(ptr, inc)``
     adds the parameter; ``addhundred(ptr, inc)`` adds a hard-coded 100 —
     a true FUNCTION swap, not a parameter tweak;
  2. build a 1-node graph via the driver API (torch ships cuda-python's
     ``cuda.bindings.driver`` — proper marshalling, no hand-rolled ctypes
     structs);
  3. instantiate + launch on the torch stream -> buffer == 1;
  4. cuGraphExecKernelNodeSetParams on the SAME node handle, swapping only
     ``func`` to addhundred -> relaunch the SAME exec -> buffer == 101.

Run: python -m triton.pact_aobo.graph_recode   (set PACT_RESULT_JSON to dump)
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

_PTX = b"""
.version 8.3
.target sm_86
.address_size 64
.visible .entry addone(.param .u64 p, .param .u32 inc) {
  .reg .b32 %r<8>; .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r5, [inc];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, %ctaid.x;
  mov.u32 %r2, %ntid.x;
  mov.u32 %r3, %tid.x;
  mad.lo.s32 %r4, %r2, %r1, %r3;
  mul.wide.s32 %rd3, %r4, 4;
  add.s64 %rd4, %rd2, %rd3;
  ld.global.u32 %r6, [%rd4];
  add.u32 %r7, %r6, %r5;
  st.global.u32 [%rd4], %r7;
  ret;
}
.visible .entry addhundred(.param .u64 p, .param .u32 inc) {
  .reg .b32 %r<8>; .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, %ctaid.x;
  mov.u32 %r2, %ntid.x;
  mov.u32 %r3, %tid.x;
  mad.lo.s32 %r4, %r2, %r1, %r3;
  mul.wide.s32 %rd3, %r4, 4;
  add.s64 %rd4, %rd2, %rd3;
  ld.global.u32 %r6, [%rd4];
  add.u32 %r7, %r6, 100;
  st.global.u32 [%rd4], %r7;
  ret;
}
"""


class _ArgBuf:
    """Classic ABI form: kernelParams = address of a void*[N] array whose
    elements are the addresses of the argument values.  Every buffer is kept
    alive here — the driver dereferences at Add AND at SetParams time."""

    def __init__(self, ptr_val: int, inc_val: int):
        import ctypes as _ct
        self._ptr = _ct.c_uint64(ptr_val)
        self._inc = _ct.c_uint32(inc_val)
        self._arr = (_ct.c_void_p * 2)(
            _ct.cast(_ct.byref(self._ptr), _ct.c_void_p),
            _ct.cast(_ct.byref(self._inc), _ct.c_void_p))
        self.abi_address = _ct.addressof(self._arr)


def spike() -> dict:
    import torch
    from cuda.bindings import driver as cu

    torch.zeros(1, device="cuda")  # ensure the runtime context exists

    (err,) = cu.cuInit(0)
    assert err == 0, err

    err, mod = cu.cuModuleLoadDataEx(_PTX, 0, [], [])
    assert err == 0, f"cuModuleLoadDataEx {err}"
    fns = {}
    for sym in (b"addone", b"addhundred"):
        err, fn = cu.cuModuleGetFunction(mod, sym)
        assert err == 0, f"cuModuleGetFunction({sym}) {err}"
        fns[sym] = fn

    buf = torch.zeros(256, dtype=torch.int32, device="cuda")
    args = _ArgBuf(int(buf.data_ptr()), 1)

    def node_params(fn):
        np_ = cu.CUDA_KERNEL_NODE_PARAMS()
        np_.func = fn
        np_.gridDimX, np_.gridDimY, np_.gridDimZ = 2, 1, 1
        np_.blockDimX, np_.blockDimY, np_.blockDimZ = 128, 1, 1
        np_.sharedMemBytes = 0
        np_.kernelParams = args.abi_address
        np_.extra = 0
        return np_

    err, graph = cu.cuGraphCreate(0)
    assert err == 0, err
    err, node = cu.cuGraphAddKernelNode(
        graph, (), 0, node_params(fns[b"addone"]))
    assert err == 0, f"cuGraphAddKernelNode {err}"

    err, gexec = cu.cuGraphInstantiate(graph, 0)
    assert err == 0, f"cuGraphInstantiate {err}"
    stream = torch.cuda.current_stream().cuda_stream

    def run_and_read():
        (e,) = cu.cuGraphLaunch(gexec, stream)
        assert e == 0, f"cuGraphLaunch {e}"
        torch.cuda.synchronize()
        return buf[0].item(), buf[255].item()

    a, b_ = run_and_read()

    (rc,) = cu.cuGraphExecKernelNodeSetParams(
        gexec, node, node_params(fns[b"addhundred"]))
    c, d = (run_and_read() if rc == 0 else (None, None))

    report = {
        "host": "RTX 3080 SM86, cuda.bindings.driver",
        "first_launch_addone": [a, b_],
        "recode_rc": int(rc),
        "recode_supported": int(rc) == 0,
        "relaunch_addhundred_same_exec": [c, d],
        "verdict": "PASS" if (a == 1 and int(rc) == 0 and c == 101) else "FAIL",
        "meaning": "an instantiated graph's kernel node can be re-pointed to "
                   "a different function; relaunching the same exec runs the "
                   "new binary — the 'graph 内调用点重编码' primitive",
    }
    try:
        cu.cuGraphExecDestroy(gexec)
        cu.cuGraphDestroy(graph)
    except Exception:
        pass
    return report


def _entry():
    rep = spike()
    txt = json.dumps(rep, indent=1)
    print(txt)
    out = os.environ.get("PACT_RESULT_JSON")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(txt + "\n")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_entry())
