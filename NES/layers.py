"""
Backend-agnostic building blocks (Keras 3, `keras.ops` only): activations and the traveltime network.
`sin`, `tanh` and `arctan` come from NES.backend, which keeps float64 precision where `keras.ops` does not.
"""
import numpy as np
import keras
from keras import ops

from .backend import sin, tanh, arctan


#######################################################################
                        ### ACTIVATIONS ###
#######################################################################


def _sinc(z):
    # `where` evaluates both branches: guard the division so the unused branch cannot produce NaN gradients
    nonzero = ops.not_equal(z, 0)
    safe = ops.where(nonzero, z, ops.ones_like(z))
    return ops.where(nonzero, sin(safe) / safe, ops.ones_like(z))


ACTS = {
        'tanh': tanh,
        'atan': arctan,
        'sigmoid': ops.sigmoid,
        'softplus': ops.softplus,
        'relu': ops.relu,
        'exp': ops.exp,
        'elu': ops.elu,
        'sin': sin,
        'sinc': _sinc,
        'linear': lambda z: z,
        'abs_linear': ops.abs,
        'gauss': lambda z: ops.exp(-z * z),
        'swish': lambda z: z * ops.sigmoid(z),
        'laplace': lambda z: ops.exp(-ops.abs(z)),
        'gauslace': lambda z: ops.exp(-z * z) + ops.exp(-ops.abs(z)),
        }


def parse_activation(act):
    """
        Parses activation spec into `(function, adaptive, n)`, meaning `act(x) = function(a * n * x)`.

        act : str or callable :
            1) name: 'tanh', 'gauss', ... (NES.layers.ACTS or any `keras.activations` name)
            2) '(ad)-name-n': 'ad' makes `a` trainable, `n` is a constant factor. E.g. 'ad-gauss-1', '-tanh-2'
            3) callable
    """
    if callable(act):
        return act, False, 1.0
    if not isinstance(act, str):
        raise ValueError("'act' must be either 'str' or 'callable'")
    parts = act.split('-')
    if len(parts) == 1:
        return ACTS.get(act) or keras.activations.get(act), False, 1.0
    if len(parts) != 3:
        raise ValueError(f"Activation '{act}' must have format '(ad)-name-n', e.g. 'ad-gauss-1'")
    fn = ACTS.get(parts[1]) or keras.activations.get(parts[1])
    return fn, 'ad' in parts[0], float(parts[2]) if parts[2] else 1.0


class AdaptiveActivation(keras.layers.Layer):
    """
        Activation layer `act(x) = f(a * n * x)` with trainable scalar `a` (if adaptive, initialized as 1).
        See `parse_activation` for the format of `act`.
    """
    def __init__(self, act, **kwargs):
        super().__init__(**kwargs)
        self.act = act
        self.fn, self.adapt, self.n = parse_activation(act)

    def build(self, input_shape=None):
        if self.adapt:
            self.a = self.add_weight(name='a', shape=(1,), initializer='ones', trainable=True)

    def call(self, x):
        return self.fn((self.n * self.a if self.adapt else self.n) * x)

    def get_config(self):
        config = super().get_config()
        config.update({'act': self.act})
        return config


def Activation(act):
    """ Kept for backward compatibility: returns `AdaptiveActivation(act)` """
    return AdaptiveActivation(act)


#######################################################################
                    ### TRAVELTIME NETWORK ###
#######################################################################


RECIPROCITY_MODES = (None, 'output', 'first_layer', 'invariant')
DEFAULT_RECIPROCITY = 'output'


def resolve_reciprocity(reciprocity):
    """ Maps `True` to the default mode and `False` to `None` """
    if reciprocity is True:
        return DEFAULT_RECIPROCITY
    if reciprocity is False:
        return None
    if reciprocity not in RECIPROCITY_MODES:
        raise ValueError(f"reciprocity must be one of {RECIPROCITY_MODES} or bool, got {reciprocity!r}")
    return reciprocity


def _distance(d):
    """
        |d| over the last axis (keepdims). Not `ops.norm`: Keras 3.15 computes it in float32 for float64 input
        on JAX and PyTorch.
        At d = 0 (the source) the value is 0 and the gradient is 0 on every backend: `where` guards the sqrt,
        whose derivative is infinite at 0.
    """
    s = ops.sum(d * d, axis=-1, keepdims=True)
    nonzero = ops.greater(s, 0)
    return ops.where(nonzero, ops.sqrt(ops.where(nonzero, s, ops.ones_like(s))), ops.zeros_like(s))


class TraveltimeNet(keras.Model):
    """
        Factored traveltime network.

            T = |xr - xs| * (1/vmax + (1/vmin - 1/vmax) * out_act(MLP(features)))

        Input `x` is `xr` (N, dim) for one-point NES (source `xs` fixed), or `[xs, xr]` (N, 2*dim) for two-point NES.

        Reciprocity T(xs, xr) = T(xr, xs) (two-point only):
            None          : features [xs, xr], no symmetry.
            'output'      : T ~ out_act((MLP(xs, xr) + MLP(xr, xs)) / 2). Full MLP evaluated twice (original NES).
            'first_layer' : the first hidden layer's activations are averaged over the swap,
                            h1 = (act(W [xs, xr] + b) + act(W [xr, xs] + b)) / 2, the rest evaluated once.
                            Averaging must come after the activation: before it, h1 would depend on xs + xr only.
            'invariant'   : single pass on swap-invariant features [(xs + xr)/2, triu(d d^T)], d = xr - xs.
                            Any smooth symmetric function is a smooth function of these (even in d).

        improved_mlp : U/V-gated MLP of Wang, Teng & Perdikaris (2021), https://doi.org/10.1137/20M1318043
    """
    def __init__(self, dim, xscale, vmin, vmax, xs=None, nl=4, nu=50, act='ad-gauss-1', out_act='ad-sigmoid-1',
                 input_scale=True, factored=True, out_vscale=True, reciprocity=None, improved_mlp=False,
                 name=None, **dense_kwargs):
        super().__init__(name=name)
        self.dim = int(dim)
        self.two_point = xs is None
        self.xs = None if self.two_point else np.asarray(xs, dtype='float64').reshape(1, self.dim)
        self.reciprocity = resolve_reciprocity(reciprocity) if self.two_point else None
        self.scale = 1.0 / float(xscale) if input_scale else 1.0
        self.slow_min, self.slow_max = 1.0 / float(vmax), 1.0 / float(vmin)
        self.factored, self.out_vscale, self.improved_mlp = factored, out_vscale, improved_mlp

        nu = [nu] * nl if isinstance(nu, (int, np.integer)) else list(nu)
        if len(nu) != nl:
            raise ValueError("Number of hidden layers 'nl' must be equal to 'len(nu)'")
        if improved_mlp and len(set(nu)) > 1:
            raise ValueError("'improved_mlp' requires equal widths of hidden layers")
        dense_kwargs.setdefault('kernel_initializer', 'he_normal')
        dense = lambda units: keras.layers.Dense(units, **dense_kwargs)

        self.hidden = [dense(n) for n in nu]
        self.acts = [AdaptiveActivation(act) for _ in nu]
        self.gated = improved_mlp and nl > 1
        if self.gated:
            self.u, self.u_act = dense(nu[0]), AdaptiveActivation(act)
            self.v, self.v_act = dense(nu[0]), AdaptiveActivation(act)
        self.head = dense(1)
        self.out_act = AdaptiveActivation(out_act)

    @property
    def n_features(self):
        if not self.two_point:
            return self.dim
        if self.reciprocity == 'invariant':
            return self.dim + self.dim * (self.dim + 1) // 2
        return 2 * self.dim

    def build(self, input_shape=None):
        first = [(self.hidden[0], self.acts[0])] + ([(self.u, self.u_act), (self.v, self.v_act)] if self.gated else [])
        for layer, act in first:
            layer.build((None, self.n_features))
            act.build()
        for layer, act, n_in in zip(self.hidden[1:], self.acts[1:], [h.units for h in self.hidden[:-1]]):
            layer.build((None, n_in))
            act.build()
        self.head.build((None, self.hidden[-1].units))
        self.out_act.build()

    # ----- MLP pieces -----

    def _first_layer(self, f):
        """ Activations of all layers fed by the features: [h1] or [h1, U, V] """
        out = [self.acts[0](self.hidden[0](f))]
        if self.gated:
            out += [self.u_act(self.u(f)), self.v_act(self.v(f))]
        return out

    def _rest(self, first):
        h = first[0]
        for layer, act in zip(self.hidden[1:], self.acts[1:]):
            h = act(layer(h))
            if self.gated:
                h = (1.0 - h) * first[1] + h * first[2]
        return self.head(h)

    def _swap_mean(self, t):
        a, b = ops.split(t, 2, axis=0)
        return 0.5 * (a + b)

    def _logits(self, xs, xr, d):
        if not self.two_point:
            return self._rest(self._first_layer(d * self.scale))
        if self.reciprocity is None:
            return self._rest(self._first_layer(ops.concatenate([xs, xr], axis=-1) * self.scale))
        if self.reciprocity == 'invariant':
            ds = d * self.scale
            quad = [ds[:, i:i + 1] * ds[:, j:j + 1] for i in range(self.dim) for j in range(i, self.dim)]
            f = ops.concatenate([(xs + xr) * (0.5 * self.scale)] + quad, axis=-1)
            return self._rest(self._first_layer(f))

        # both orderings stacked along the batch axis: one matmul per layer instead of two
        pairs = ops.concatenate([ops.concatenate([xs, xr], axis=-1),
                                 ops.concatenate([xr, xs], axis=-1)], axis=0) * self.scale
        first = self._first_layer(pairs)
        if self.reciprocity == 'output':
            return self._swap_mean(self._rest(first))
        return self._rest([self._swap_mean(t) for t in first])  # 'first_layer'

    def call(self, x):
        if self.two_point:
            xs, xr = x[:, :self.dim], x[:, self.dim:]
        else:
            xs, xr = ops.convert_to_tensor(self.xs, dtype=x.dtype), x
        d = xr - xs
        tau = self.out_act(self._logits(xs, xr, d))
        if self.out_vscale:
            tau = self.slow_min + (self.slow_max - self.slow_min) * tau
        if self.factored:
            tau = tau * _distance(d)
        return tau

    def flops(self):
        """
            Approximate FLOPs of one forward evaluation of T (one point for NES-OP, one pair for NES-TP):
            2*n_in*n_out (multiply-add) + n_out (bias) + n_out (activation) per dense layer, 4 per gating element
            of the improved MLP. Network passes duplicated by the reciprocity mode are counted twice.
            Deterministic and backend-independent, it serves as a proxy of runtime.
        """
        dense = lambda n_in, n_out: 2 * n_in * n_out + 2 * n_out
        n_groups = 3 if self.gated else 1                       # h1 (+ U, V)
        nu0 = self.hidden[0].units
        first = n_groups * dense(self.n_features, nu0)
        rest, n_prev = 0, nu0
        for layer in self.hidden[1:]:
            rest += dense(n_prev, layer.units) + (4 * layer.units if self.gated else 0)
            n_prev = layer.units
        head = dense(n_prev, 1)
        if self.reciprocity == 'output':
            body = 2 * (first + rest + head) + 2
        elif self.reciprocity == 'first_layer':
            body = 2 * first + 2 * n_groups * nu0 + rest + head
        else:
            body = first + rest + head
        features = 3 * self.dim * (self.dim + 1) // 2 + 2 * self.dim if self.reciprocity == 'invariant' else 0
        return int(body + features + 3 * self.dim + 4)          # + offset, norm, bounding and factorization

    def legacy_variables(self):
        """ Variables in the order of the original TF/Keras-2 NES weight lists (for loading old models) """
        order = [0] + ([1, 'u', 'v'] + list(range(2, len(self.hidden))) if self.gated
                       else list(range(1, len(self.hidden))))
        variables = []
        for k in order:
            layer, act = (getattr(self, k), getattr(self, f'{k}_act')) if isinstance(k, str) \
                else (self.hidden[k], self.acts[k])
            variables += [layer.kernel, layer.bias] + ([act.a] if act.adapt else [])
        variables += [self.head.kernel, self.head.bias] + ([self.out_act.a] if self.out_act.adapt else [])
        return [v for v in variables if v is not None]
