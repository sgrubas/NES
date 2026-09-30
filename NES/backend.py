"""
Derivatives of a network w.r.t. its *inputs*, and float64-safe elementwise functions, for every Keras 3 backend.

Keras 3 has no backend-agnostic `ops.grad`, and some `keras.ops` lose float64 precision (see `sin` below),
so this is the only module that touches JAX / TensorFlow / PyTorch directly. Everything else in NES uses `keras.ops`.

All derivative functions assume `f` is sample-wise: row `i` of `f(x)` depends only on row `i` of `x`.
Then d(sum f)/dx is exactly the stack of per-sample gradients, obtained in one reverse pass.
The returned derivatives stay differentiable, so they can enter a training loss
(double backpropagation) or be differentiated again (Hessians).
"""
import keras
from keras import ops

BACKEND = keras.backend.backend()


def _value_and_grad_jax(f, x):
    import jax
    import jax.numpy as jnp
    y, vjp = jax.vjp(f, x)
    return y, vjp(jnp.ones_like(y))[0]


def _value_and_grad_tensorflow(f, x):
    import tensorflow as tf
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x)
        y = f(x)
    return y, tape.gradient(y, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)


def _value_and_grad_torch(f, x):
    import torch
    # Keep the graph if an outer differentiation is active (training step, or an outer
    # derivative in `hessian_rows`). `predict` runs under `no_grad`, where it is not needed.
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        y = f(x)
        g, = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=create_graph)
    return y, g


_IMPL = {'jax': _value_and_grad_jax,
         'tensorflow': _value_and_grad_tensorflow,
         'torch': _value_and_grad_torch}

if BACKEND not in _IMPL:
    raise ImportError(f"NES supports the 'jax', 'tensorflow' and 'torch' Keras backends, got '{BACKEND}'")

value_and_grad = _IMPL[BACKEND]
value_and_grad.__doc__ = """Returns `(f(x), d sum(f(x)) / dx)`; for sample-wise `f` the latter is the per-sample gradient."""


def grad(f, x):
    """Per-sample gradient of a sample-wise scalar function `f`, shape `x.shape`."""
    return value_and_grad(f, x)[1]


def hessian_rows(f, x, rows):
    """
        Rows of the per-sample Hessian: `H[:, k, :] = d/dx (df/dx_{rows[k]})`.

        Returns tensor of shape (N, len(rows), x.shape[-1]).
        One nested reverse pass per row; intended for inference, not for training losses.
    """
    return ops.stack([grad(lambda z, i=i: grad(f, z)[:, i:i + 1], x) for i in rows], axis=1)


# Keras 3.15 computes some ops in float32 even for float64 input, since `dtypes.result_type('float64', float)`
# gives 'float32' on the JAX and PyTorch backends: on JAX sin, cos, tan, the hyperbolic functions and all their
# inverses; on both var, norm, arctan2 and mean (which still returns float64). The elementwise functions NES needs
# are therefore taken from the backend for float64 tensors, which keeps the precision. Other dtypes, and
# symbolic `KerasTensor`s, go through `keras.ops` as before.

def _native_jax(name):
    import jax.numpy as jnp
    return getattr(jnp, name)


def _native_tensorflow(name):
    import tensorflow as tf
    return getattr(tf.math, {'arctan': 'atan'}.get(name, name))


def _native_torch(name):
    import torch
    return getattr(torch, name)


_NATIVE = {'jax': _native_jax, 'tensorflow': _native_tensorflow, 'torch': _native_torch}


def _float64_safe(name):
    keras_op, native = getattr(ops, name), _NATIVE[BACKEND](name)

    def fn(x):
        if not keras.backend.is_keras_tensor(x):
            x = ops.convert_to_tensor(x)
            if keras.backend.standardize_dtype(x.dtype) == 'float64':
                return native(x)
        return keras_op(x)

    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = f"`keras.ops.{name}` that keeps float64 precision"
    return fn


sin, tanh, arctan = (_float64_safe(name) for name in ('sin', 'tanh', 'arctan'))
