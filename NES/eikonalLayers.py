from tensorflow import norm, pow, reduce_sum, ones_like, concat
from tensorflow.keras.layers import Layer
from tensorflow.math import log

class IsoEikonal(Layer):
    """
        Isotropic eikonal equation

        Arguments:
            P : callable or int or float : function for 'right hand side' and 'left hand side' of the equation. 
            If 'int' or 'float', the exponentiation is applied. By default P=2.
            hamiltonian : boolean : whether to use hamiltonian form of the equation 'H = P(v * |grad tau|) - P(1)'. By default is True
    """
    def __init__(self, p=3, hamiltonian=True, **kwargs):
        if kwargs.get('name') is None:
            kwargs['name'] = 'IsoEikonal'
        super(IsoEikonal, self).__init__(**kwargs)
        self.hamiltonian = hamiltonian

        assert isinstance(p, (float, int)), " `p` must be float or int"
        assert p != 0, "`p` must not be 0"
        self.p = p

    def call(self, dT, v):
        eik = norm(concat(dT, axis=-1), axis=-1, keepdims=True)

        if self.hamiltonian:
            lhs = eik * v
            rhs = ones_like(v)
        else:
            lhs = eik
            rhs = 1 / v

        eikp = pow(lhs, self.p) - pow(rhs, self.p)
        return eikp / self.p

    def get_config(self):
        config = super(IsoEikonal, self).get_config()
        config.update({"p": self.p, "hamiltonian": self.hamiltonian})
        return config
