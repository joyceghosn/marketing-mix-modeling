"""Compatibility shim for lightweight_mmm==0.1.9 on modern JAX.

lightweight_mmm (last released 2023, effectively unmaintained) calls
`jnp.where(condition=..., x=..., y=...)` with keyword arguments in
`media_transforms.py`. JAX made `where`'s arguments positional-only at some
point after that, so this raises `TypeError: where() got some
positional-only arguments passed as keyword arguments` on any current JAX
(the version this project actually installs, since old jax/jaxlib/numpyro
pins from lightweight_mmm's own requirements have no Windows wheels and are
years behind numpy/scipy).

Rather than pinning that broken, unavailable-on-Windows stack, we patch
`jax.numpy.where` with a tiny wrapper that accepts the old keyword names and
forwards them positionally. This is applied once, before any
lightweight_mmm model is fit, and only changes behavior for callers that
pass condition=/x=/y= as keywords (lightweight_mmm's own code) -- normal
positional use of jnp.where is untouched.
"""
import jax.numpy as jnp

_PATCHED = False


def patch_jax_where_for_lightweight_mmm():
    global _PATCHED
    if _PATCHED:
        return
    original_where = jnp.where

    def where_compat(*args, **kwargs):
        if "condition" in kwargs or "x" in kwargs or "y" in kwargs:
            condition = kwargs.pop("condition", args[0] if args else None)
            x = kwargs.pop("x", None)
            y = kwargs.pop("y", None)
            pos = [condition]
            if x is not None:
                pos.append(x)
            if y is not None:
                pos.append(y)
            return original_where(*pos, **kwargs)
        return original_where(*args, **kwargs)

    jnp.where = where_compat
    _PATCHED = True
