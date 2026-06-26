"""PACT JIT frontend annotation utilities.

Separated from compiler.py to keep the main compilation path clean.
"""
from triton._C.libtriton import ir


def attach_pact_func_attrs(module, attrs):
    """Attach pact.* attributes to module and tt.func ops.

    Called after ast_to_ttir. Sets module-level and function-level
    attributes so downstream C++ passes (PageTransform, PatternSpecialize,
    PrefetchInsert) can read them.
    """
    if not attrs or "pact.paged" not in attrs:
        return

    pact_attr_keys = [
        k for k in attrs if isinstance(k, str) and k.startswith("pact.")
    ]
    if not pact_attr_keys:
        return

    try:
        builder = ir.builder(module.context)

        # Set module-level attributes (accessible via module->getAttr)
        for key in pact_attr_keys:
            val = attrs[key]
            if isinstance(val, bool):
                if val:
                    module.set_attr(key, builder.get_unit_attr())
            elif isinstance(val, int):
                module.set_attr(key, builder.get_int32_attr(val))

        # Also set on tt.func ops via walk (for pass-level access)
        def _set_on_func(op):
            try:
                op_name = op.get_name() if hasattr(op, 'get_name') else ""
                if "func" in str(op_name).lower():
                    for key in pact_attr_keys:
                        val = attrs[key]
                        if isinstance(val, bool):
                            if val:
                                op.set_attr(key, builder.get_unit_attr())
                        elif isinstance(val, int):
                            op.set_attr(key, builder.get_int32_attr(val))
            except Exception:
                pass

        module.walk(_set_on_func)

    except Exception:
        pass  # Best-effort: never break compilation
