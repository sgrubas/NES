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
    def __init__(self, P=3, hamiltonian=True, **kwargs):
        """
            Isotropic eikonal equation

            Arguments:
                P : callable or int or float : function for 'right hand side' and 'left hand side' of the equation. 
                If 'int' or 'float', the exponentiation is applied. By default P=2.
                hamiltonian : boolean : whether to use hamiltonian form of the equation 'H = P(v * |grad tau|) - P(1)'. By default is True
        """
        super(IsoEikonal, self).__init__(**kwargs)
        self.hamiltonian = hamiltonian
        if isinstance(P, (float, int)):
            if P != 0:
                self.P = lambda x: pow(x, P) / P
            else:
                self.P = log
        elif callable(P):
            self.P = P
        else:
            assert False, "Unknown type of 'P' argument"

    def call(self, dT, v):
        eik = norm(concat(dT, axis=-1), axis=-1, keepdims=True)

        if self.hamiltonian:
            lhs = eik * v
            rhs = ones_like(v)
        else:
            lhs = eik
            rhs = 1 / v

        eikp = self.P(lhs) - self.P(rhs)
        return eikp

    def get_config(self):
        config = super(IsoEikonal, self).get_config()
        config.update({"P": self.P, "hamiltonian": self.hamiltonian})
        return config
