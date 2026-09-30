import keras
from keras import ops


class IsoEikonal(keras.layers.Layer):
    """
        Isotropic eikonal equation residual.

        Arguments:
            p : int or float : power of both sides of the equation, by default p=2
            hamiltonian : boolean : whether to use the hamiltonian form 'H = ((v * |grad T|)^p - 1) / p'.
                          Otherwise '(|grad T|^p - v^-p) / p'. By default is True

        Call:
            dT : tensor (N, dim) (or list of (N, 1) tensors) : traveltime gradient
            v : tensor (N, 1) : velocity
    """
    def __init__(self, p=2, hamiltonian=True, **kwargs):
        kwargs.setdefault('name', 'IsoEikonal')
        super().__init__(**kwargs)
        if not isinstance(p, (float, int)) or p == 0:
            raise ValueError("`p` must be a non-zero float or int")
        self.p = p
        self.hamiltonian = hamiltonian

    def call(self, dT, v):
        if isinstance(dT, (list, tuple)):
            dT = ops.concatenate(dT, axis=-1)
        s2 = ops.sum(dT * dT, axis=-1, keepdims=True)   # |grad T|^2, no sqrt: finite gradient at |grad T| = 0 for p=2
        if self.hamiltonian:
            lhs = s2 * v * v
            rhs = 1.0
        else:
            lhs = s2
            rhs = ops.power(v, -self.p)
        if self.p != 2:
            lhs = ops.power(lhs, self.p / 2)
        return (lhs - rhs) / self.p

    def get_config(self):
        config = super().get_config()
        config.update({"p": self.p, "hamiltonian": self.hamiltonian})
        return config
