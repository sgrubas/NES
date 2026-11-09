"""Exact JAX reference of the NES training loss and its gradient, to test the C++ training against.

    loss, gradient = loss_and_gradient(nes, x, v)      # one batch
    theta = parameters(nes)                            # current weights, same order as `gradient`

Loss (default NES loss, receiver equation only, no sample weights):

    loss = mean_p | (|grad_r T(x_p)|^2 v_p^2 - 1) / 2 |

with `v_p` the velocity at the receiver of point p. Parameter order of `gradient` and `parameters`:

    for each hidden layer k = 1..L : kernel (n_in x n_out, row-major), bias
    head kernel (n_L values), head bias (1 value)
    `a` of each hidden layer (L values), only if the hidden activation is adaptive
    `a` of the output activation (1 value), only if it is adaptive

The gradient is computed by JAX in the active precision (tests use float64). It goes through NES's own residual
model, so the reference is the loss NES trains with, not a re-implementation of it.
"""
import numpy as np
import jax
import jax.numpy as jnp
import keras
from keras import ops


def _variables_in_order(net):
    """The network's trainable variables in the documented order (list of Keras variables)."""
    variables = []
    for layer in net.hidden + [net.head]:
        variables += [layer.kernel, layer.bias]
    variables += [act.a for act in net.acts if act.adapt]
    if net.out_act.adapt:
        variables.append(net.out_act.a)
    return variables


def _flatten(arrays):
    return np.concatenate([np.asarray(a, dtype='float64').reshape(-1) for a in arrays])


def parameters(nes):
    """Current weights as one flat float64 array, in the documented order."""
    return _flatten([ops.convert_to_numpy(v.value) for v in _variables_in_order(nes.net)])


def velocities_at_receivers(nes, x):
    """Velocity (N, 1) at the receiver of each point of `x`, as NES feeds it to the loss."""
    key = nes.model.keys[0]                                # 'E' (NES_OP) or 'Er' (NES_TP)
    return nes._prepare_inputs(x, key)['v' + key[1:]]


def loss_and_gradient(nes, x, v):
    """
        Loss on the batch `x` (N, n_in) with receiver velocities `v` (N, 1), and its gradient with respect to
        all trainable weights as one flat numpy array (see the module docstring for the order).
    """
    model = nes.model
    if model.keys not in (('E',), ('Er',)):
        raise ValueError(f"only the receiver equation 'E' / 'Er' is supported, the model trains {model.keys}")
    dtype = 'float64' if keras.config.floatx() == 'float64' else 'float32'
    inputs = {'x': np.asarray(x, dtype=dtype), 'v' + model.keys[0][1:]: np.asarray(v, dtype=dtype).reshape(-1, 1)}

    if not model.built:
        model(inputs)                                      # creates the model's variable lists; `stateless_call` needs it
    # `stateless_call` runs the model as a pure function of its variable values, which JAX can differentiate.
    # Values are passed for every variable; only the trainable ones are differentiated.
    trainable = [w.value for w in model.trainable_variables]
    non_trainable = [w.value for w in model.non_trainable_variables]

    def loss(values):
        residual, _ = model.stateless_call(values, non_trainable, inputs)
        # jnp, not keras.ops: `ops.mean` loses float64 precision on JAX in Keras 3.15
        return jnp.mean(jnp.abs(residual[0]))              # MAE of the residual against zero

    value, grads = jax.value_and_grad(loss)(trainable)
    by_variable = {id(w): g for w, g in zip(model.trainable_variables, grads)}
    # reorder from Keras's variable order to the documented one
    return float(value), _flatten([by_variable[id(v)] for v in _variables_in_order(nes.net)])
