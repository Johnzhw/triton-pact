"""PACT PGO kernel variant compiler and atomic swapper.

Variants are compiled through Triton's public compile path with an explicit
`_env_vars` map, so PACT_PGO_HINTS_JSON participates in the on-disk cache key.
The active CompiledKernel can be swapped between decode steps under a lock;
launches bypass JITFunction.run and go directly through CompiledKernel[grid].
"""
import dataclasses
import os
import threading
from typing import Any, Dict, Optional, Tuple

from triton._C.libtriton import get_cache_invalidating_env_vars
from triton.runtime import driver
from triton.runtime.cache import get_cache_key
from triton.runtime.jit import compute_cache_key


class VariantCompileError(RuntimeError):
    pass


def _prepare(jit_fn, args, kwargs):
    device = driver.active.get_current_device()
    kernel_cache, kernel_key_cache, target, backend, binder = \
        jit_fn.device_caches[device]
    bound_args, specialization, options = binder(*args, **kwargs)
    options, signature, constexprs, attrs = jit_fn._pack_args(
        backend, kwargs, bound_args, specialization, options)
    src = jit_fn.ASTSource(jit_fn, signature, constexprs, attrs)
    return kernel_cache, target, backend, options, src, bound_args


def compile_variant(jit_fn, args, kwargs, extra_env: Dict[str, str],
                    options_override: Optional[Dict[str, Any]] = None):
    """Compile (or reuse from disk) one variant.  Returns (CompiledKernel, bound_args)."""
    kernel_cache, target, backend, options, src, bound_args = \
        _prepare(jit_fn, args, kwargs)
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


class PactKernelSwapper:
    def __init__(self, jit_fn, args, kwargs, grid, extra_env: Dict[str, str]):
        self.jit_fn = jit_fn
        self.args = args
        self.kwargs = kwargs
        self.grid = grid
        self.extra_env = dict(extra_env)
        self._lock = threading.Lock()
        self.active: Optional[Any] = None
        self.bound_args = None
        self.baseline, self.bound_args = compile_variant(
            jit_fn, args, kwargs, self.extra_env)

    def compile_candidate(self, hints: Dict[str, Any], hints_path: str):
        env = dict(self.extra_env)
        # The PGO candidate is the joint-decision build: P6 consumes
        # measured_iterations (always ON) and P11 consumes measured
        # regs_per_thread + active_warp_ratio_permille.  P11 is OFF by
        # default globally, so enable it explicitly for candidates; when the
        # collected facts are missing, P11/P6 fall back to their theory-only
        # paths and this still compiles a valid candidate.
        env.setdefault("PACT_ENABLE_AUTO_NUM_WARPS", "1")
        env["PACT_PGO_HINTS_JSON"] = hints_path
        with open(hints_path, "w") as f:
            import json
            json.dump(hints, f)
        return compile_variant(self.jit_fn, self.args, self.kwargs, env)[0]

    def compile_explicit(self, num_stages: int, num_warps: int):
        """Oracle-grid variant: explicit num_stages/num_warps with the PACT
        auto passes disabled so the knobs are not overridden."""
        env = dict(self.extra_env)
        env["PACT_ENABLE_AUTO_NUM_STAGES"] = "0"
        env["PACT_ENABLE_AUTO_NUM_WARPS"] = "0"
        return compile_variant(
            self.jit_fn, self.args, self.kwargs, env,
            options_override={"num_stages": num_stages,
                              "num_warps": num_warps})[0]

    def launch(self, kernel, grid=None) -> Any:
        g = self.grid
        if callable(g):
            g = g(self.bound_args)
        if len(g) == 2:
            g = (g[0], g[1], 1)
        return kernel[g](*self.bound_args.values())

    def swap(self, kernel) -> None:
        with self._lock:
            self.active = kernel

    def launch_active(self, grid=None) -> Any:
        with self._lock:
            kernel = self.active or self.baseline
        return self.launch(kernel, grid=grid)
