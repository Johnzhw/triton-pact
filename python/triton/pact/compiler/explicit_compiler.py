"""Compile one variant through Triton's public JIT path and export a shm blob."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

from triton._C.libtriton import get_cache_invalidating_env_vars
from triton.runtime import driver
from triton.runtime.cache import get_cache_key


def compile_explicit(jit_fn, args, kwargs, extra_env: Dict[str, str],
                     options_override: Optional[Dict[str, Any]] = None):
    """Compile (or reuse from disk) one variant.  Returns CompiledKernel."""
    device = driver.active.get_current_device()
    kernel_cache, _k, target, backend, binder = jit_fn.device_caches[device]
    bound_args, specialization, options = binder(*args, **kwargs)
    options, signature, constexprs, attrs = jit_fn._pack_args(
        backend, kwargs, bound_args, specialization, options)
    src = jit_fn.ASTSource(jit_fn, signature, constexprs, attrs)
    if options_override:
        options = dataclasses.replace(options, **options_override)

    env_vars = dict(get_cache_invalidating_env_vars())
    env_vars.update(extra_env)
    old = {k: os.environ.get(k) for k in extra_env}
    os.environ.update(extra_env)
    try:
        kernel = jit_fn.compile(src, target=target, options=options.__dict__,
                                _env_vars=env_vars)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    cache_key = get_cache_key(src, backend, options, env_vars)
    kernel_cache[cache_key] = kernel
    return kernel, bound_args


def kernel_to_blob(kernel, extra_header: Optional[Dict[str, Any]] = None
                   ) -> Tuple[Dict[str, Any], bytes]:
    """Serialize cubin + metadata for shm.  CUDA module stays process-local."""
    md = kernel.metadata
    md_dict = md._asdict() if hasattr(md, "_asdict") else dict(md)
    if getattr(kernel, "n_regs", None) in (None, 0):
        # Force CUDA module load so n_regs is populated (same-process only).
        try:
            kernel._init_handles()
        except Exception:
            pass
    cubin = None
    if hasattr(kernel, "asm") and isinstance(kernel.asm, dict):
        cubin = kernel.asm.get("cubin")
    if cubin is None:
        cubin = getattr(kernel, "kernel", None)
    if cubin is None:
        raise RuntimeError("CompiledKernel has no cubin")
    header = {
        "name": getattr(kernel, "name", md_dict.get("name")),
        "shared": int(md_dict.get("shared") or 0),
        "num_warps": int(md_dict.get("num_warps") or 4),
        "num_stages": int(md_dict.get("num_stages") or 3),
        "n_regs": int(getattr(kernel, "n_regs", 0) or md_dict.get("n_regs") or 0),
        "hash": getattr(kernel, "hash", None),
    }
    if extra_header:
        header.update(extra_header)
    header["blob_sha256"] = hashlib.sha256(cubin).hexdigest()[:16]
    return header, cubin
