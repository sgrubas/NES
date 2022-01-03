import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import initializers


#######################################################################
                        ### ACTIVATIONS ###
#######################################################################

ACTS = {
        'tanh' : tf.math.tanh,
        'atan' : tf.math.atan,
        'sigmoid' : tf.math.sigmoid, 
        'softplus' : tf.math.softplus, 
        'relu' : tf.nn.relu, 
        'elu' : tf.nn.elu,
        'exp' : tf.math.exp,
        'linear' : lambda z: tf.abs(z),
        'gauss' : lambda z: tf.math.exp(-z**2),
        'swish' : lambda z: z * tf.math.sigmoid(z),
        'laplace' : lambda z: tf.math.exp(-tf.abs(z))
        }

class Activation(L.Layer):
    """ Layer for activation function. Overriden to include adaptivity
    """
    def __init__(self, act, **kwargs):
        """
            act : formatted str : Activation in format '(ad) -activation_name- n'.
                                  Format of activation - 'act(x) = f(a * n * x)', where 'a' is adaptive term (trainable weight), 
                                  'n' constant term (degree of adaptivity). If 'ad' presents, 'a' is trainable, otherwise 'a=1'.
        """
        super(Activation, self).__init__(**kwargs)
        parts = act.split('-')
        assert len(parts) == 3
        self.adapt = 'ad' in parts[0]
        self.act = ACTS[parts[1]]
        self.n = float(parts[-1]) if parts[-1].isdigit() else 1.0
        if self.adapt:
            self.a = self.add_weight(name='a', 
                                    shape=(1,),
                                    initializer='ones',
                                    trainable=True)        

    def call(self, X):
        if self.adapt:
            return self.act(self.n * self.a * X)
        else: 
            return self.act(self.n * X)

    def get_config(self):
        config = super(Activation, self).get_config()
        config.update({"n": self.n, 
                        "adapt" : self.adapt, 
                        "act" : self.act, })
        return config


#######################################################################
                        ### API LAYERS ###
#######################################################################

def DenseBody(inputs, nu, nl, out_dim=1, act='lad-gauss-1', out_act='lad-sigmoid-1', **kwargs):
    """
        API function that construct the block of fully connected layers.

        Arguments:
            inputs : tf.Tensor : set of inputs of fully-connected block
            nl : int : Number of hidden layers, by default 'nl=4'
            nu : int or list of ints : Number of hidden units of each hidden layer, by default 'nu=50'
            out_dim : int : Output dimenstion, by default 'out_dim = 1'
            act : formatted str : Hidden activation in format '(ad) -activation_name- n'.
                                  Format of activation - 'act(x) = f(a * n * x)', where 'a' is adaptive term (trainable weight), 
                                  'n' constant term (degree of adaptivity). If 'ad' presents, 'a' is trainable, otherwise 'a=1'.
                                  By default "act = 'ad-gauss-1' "
            out_act : formatted str : Output activation in the same format as 'act'. By default "act = 'ad-sigmoid-1' "
            **kwargs : keyword arguments : Arguments for tf.keras.layers.Dense(**kwargs) such as 'kernel_initializer'.
                        If "kwargs.get('kernel_initializer')" is None then "kwargs['kernel_initializer'] = 'he_normal' "
        Returns:
            output : tf.Tensor : the output of fully-connected block
    """
    if kwargs.get('kernel_initializer') is None:
        kwargs['kernel_initializer'] = 'he_normal'

    if isinstance(nu, int):
        nu = [nu]*nl
    assert isinstance(nu, (list, np.ndarray)) and len(nu) == nl, "Number hidden layers 'nl' must be equal to 'len(nu)'"

    x = L.Dense(nu[0], activation=Activation(act), **kwargs)(inputs)
    for i in range(1, nl):
        x = L.Dense(nu[i], activation=Activation(act), **kwargs)(x)

    out = L.Dense(out_dim, activation=Activation(out_act), **kwargs)(x)
    return out

def Diff(**kwargs):
    """
        API function for differentiation using 'tf.gradients'

        Arguments:
            kwargs : keyword arguments : Arguments for 'tf.keras.layers.Lambda(**kwargs)' such as 'name'
    """
    return L.Lambda(lambda x: tf.gradients(x[0], x[1], 
        unconnected_gradients='zero'), **kwargs)


#######################################################################
                            ### OTHER ###
#######################################################################


class LossesHolder(tf.keras.callbacks.Callback):
    """
        Callback container that can save losses if training is launched multiple times (without overriding)
    """
    def __init__(self, ):
        self.logs = {'loss' : [], 'val_loss': []}
        
    def on_epoch_end(self, epoch, logs=None):
        self.logs['loss'].append(logs['loss'])
        if logs.get('val_loss') is not None:
            self.logs['val_loss'].append(logs['val_loss'])


class Initializer(initializers.Initializer):
    """
        Initializer that converts 'numpy array' to 'tf.Tensor'
    """

    def __init__(self, x):
        self.x = x

    def __call__(self, shape, dtype=None):
        return tf.convert_to_tensor(self.x, dtype=dtype)

    def get_config(self):
        return {'x': self.x}


class SourceLoc(L.Layer):
    """
    Class for Source location using tf.keras.layers.Layer.
    """
    def __init__(self, xs, **kwargs):
        """
        xs : source location [x, y, z]
        """
        super(SourceLoc, self).__init__(**kwargs)
        self.xs = self.add_weight(name='xs', shape=(len(xs),),
                                  trainable=False, 
                                  initializer=Initializer(xs))

    def call(self, x):
        return tf.ones_like(x) * self.xs

    def get_config(self):
        config = super(SourceLoc, self).get_config()
        config.update({"xs": self.xs.numpy()})
        return config