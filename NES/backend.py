"""
Derivatives of a network w.r.t. its *inputs*, for every Keras 3 backend.

Keras 3 has no backend-agnostic `ops.grad`, so this is the only module that touches
JAX / TensorFlow / PyTorch directly. Everything else in NES uses `keras.ops`.

All functions assume `f` is sample-wise: row `i` of `f(x)` depends only on row `i` of `x`.
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
