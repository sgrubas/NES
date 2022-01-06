import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping as EarlyStoppingCallback
from tensorflow.python.platform import tf_logging as logging


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

class AdaptiveActivation(L.Layer):
    """ Layer for activation function.
    """
    def __init__(self, act, **kwargs):
        """
            act : formatted str : Activation in format '(ad) -activation_name- n'.
                                  Format of activation - 'act(x) = f(a * n * x)', where 'a' is adaptive term (trainable weight), 
                                  'n' constant term (degree of adaptivity). If 'ad' presents, 'a' is trainable, otherwise 'a=1'.
        """
        super(AdaptiveActivation, self).__init__(**kwargs)
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
        config = super(AdaptiveActivation, self).get_config()
        config.update({"n": self.n, 
                        "adapt" : self.adapt, 
                        "act" : self.act, })
        return config


def Activation(act):
    """
        Checks adaptivity requirement for activation

        Arguments:
            act : str : It can be two types:
                        1) regular activation function - just name, 'tanh', 'relu', 'gauss', ...
                        2) adaptive activation - 'ad-name-n', where 'ad' means including trainable weight 'a',
                           'n' is constan integer value describing the adaptivity degree. Format - act(x) = name(a * n * x).
                           Example: 'ad-gauss-1' (adaptive) or '-tanh-2' (not adaptive)
    """
    parts = act.split('-')
    if (len(parts) == 1):
        if act in ACTS.keys():
            return ACTS[act]
        else:
            return act
    else:
        return AdaptiveActivation(act)

#######################################################################
                            ### API LAYERS ###
#######################################################################

def DenseBody(inputs, nu, nl, out_dim=1, act='ad-gauss-1', out_act='ad-sigmoid-1', **kwargs):
    """
        API function that construct the block of fully connected layers.

        Arguments:
            inputs : tf.Tensor : set of inputs of fully-connected block
            nl : int : Number of hidden layers, by default 'nl=4'
            nu : int or list of ints : Number of hidden units of each hidden layer, by default 'nu=50'
            out_dim : int : Output dimenstion, by default 'out_dim = 1'
            act : str : Hidden activation, see description 'NES.baseLayers.Activation'. By default 'ad-gauss-1'
            out_act : formatted str : Output activation, see description 'NES.baseLayers.Activation'. By default 'ad-sigmoid-1'
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
        for kw, v in logs.items():
            if self.logs.get(kw) is None:
                self.logs[kw] = []
            self.logs[kw].append(v)


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

        xs : source location [x, y, z]
    """
    def __init__(self, xs, **kwargs):
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


class NES_EarlyStopping(tf.keras.callbacks.Callback):
    """Stop training when a monitored metric has reached a tolerance value.
    Assuming the goal of a training is to minimize the loss. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss reached a tolerance value, considering the `patience`. 
    Once it's found lower than `tolerance`,
    `model.stop_training` is marked True and the training terminates.
    The quantity to be monitored needs to be available in `logs` dict and
    can 'loss' or 'val_loss' only.

    Args:
    monitor: Quantity to be monitored ('loss' or 'val_loss'). By default 'loss'
    tolerance: The baseline value of RMAE of the solution.
            Training will stop if the `conversion(monitor) < tolerance'. 
            By default 1e-2 (1 %).
    conversion: Callable function that maps `monitor` to units comparable with `tolerance`.
            It is empirical equation that defines the dependence between loss and RMAE.
            By default conversion(x) = 1.1 * exp(x)
    patience: Number of epochs `conversion(monitor)` must be lower than `tolerance` to be stopped.
    verbose: verbosity mode (0 or 1).
    """

    def __init__(self,
               monitor='loss',
               tolerance=0,
               patience=10,
               verbose=1,
               conversion=lambda x: x * 10**(-0.16),
               ):
        super(NES_EarlyStopping, self).__init__()
        assert monitor == 'loss' or monitor == 'val_loss', \
        "Only 'loss' and 'val_loss' are supported for monitor metric"

        self.monitor = monitor
        self.tolerance = tolerance
        self.patience = patience
        self.verbose = verbose
        self.conversion = conversion

        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if np.less(self.conversion(current), self.tolerance):
            self.wait += 1

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.monitor_last = current
            self.tolerance_last = self.conversion(current)
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
            print(f'{self.monitor}: {self.monitor_last:.5f}')
            print(f'Approximate RMAE of solution: {100*self.tolerance_last:.5f} %')

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
          logging.warning('Early stopping conditioned on metric `%s` '
                          'which is not available. Available metrics are: %s',
                          self.monitor, ','.join(list(logs.keys())))
        return monitor_value

