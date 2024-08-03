import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as L
from tensorflow.keras import initializers
from tensorflow.python.platform import tf_logging as logging
from . import experimental
from . import velocity


#######################################################################
                        ### ACTIVATIONS ###
#######################################################################


ACTS = {
        'tanh' : tf.math.tanh,
        'atan' : tf.math.atan,
        'sigmoid' : tf.math.sigmoid, 
        'softplus' : tf.math.softplus, 
        'relu' : tf.nn.relu, 
        'exp' : tf.math.exp,
        'elu' : tf.nn.elu,
        'sin' : tf.math.sin,
        'sinc' : lambda z: tf.where(tf.equal(z,0), tf.ones_like(z), tf.divide(tf.sin(z),z)),
        'linear' : lambda z: z,
        'abs_linear' : tf.abs,
        'gauss' : lambda z: tf.math.exp(-z**2),
        'swish' : lambda z: z * tf.math.sigmoid(z),
        'laplace' : lambda z: tf.math.exp(-tf.abs(z)),
        'gauslace' : lambda z: tf.math.exp(-z**2) + tf.math.exp(-tf.abs(z)),
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
                       "adapt": self.adapt,
                       "act": self.act})
        return config


def Activation(act):
    """
        Checks adaptivity requirement for activation

        Arguments:
            act : str : It can be two types:
                        1) regular activation function - just name, 'tanh', 'relu', 'gauss', ...
                        2) adaptive activation - 'ad-name-n', where 'ad' means including trainable weight 'a',
                           'n' is constant integer value describing the adaptivity degree. Format - act(x) = name(a * n * x).
                           Example: 'ad-gauss-1' (adaptive) or '-tanh-2' (not adaptive)
    """

    if callable(act): 
        return act

    elif isinstance(act, str): 
        parts = act.split('-')
        if (len(parts) == 1):
            if act in ACTS.keys():
                return ACTS[act]
            else:
                return act
        else:
            return AdaptiveActivation(act)
    else:
        raise ValueError("'act' must be either 'str' or 'callable'")

#######################################################################
                            ### API LAYERS ###
#######################################################################


def DenseBody(inputs, nu, nl, out_dim=1, act='ad-gauss-1', out_act='linear', improved_mlp=False, **kwargs):
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
    kwargs.setdefault('kernel_initializer', 'he_normal')
    
    if isinstance(nu, int):
        nu = [nu] * nl
    assert isinstance(nu, (list, tuple, np.ndarray)) and len(nu) == nl, \
                        "Number of hidden layers 'nl' must be equal to 'len(nu)'"

    UV_transform = L.Lambda(lambda x: (1.0 - x[0]) * x[1] + x[0] * x[2])

    if improved_mlp:
        U = L.Dense(nu[0], activation=Activation(act), **kwargs)(inputs)
        V = L.Dense(nu[0], activation=Activation(act), **kwargs)(inputs)

    x = L.Dense(nu[0], activation=Activation(act), **kwargs)(inputs)
    for i in range(1, nl):
        x = L.Dense(nu[i], activation=Activation(act), **kwargs)(x)
        if improved_mlp:
            x = UV_transform([x, U, V])

    out = L.Dense(out_dim, activation=Activation(out_act), **kwargs)(x)
    return out


def Diff(**kwargs):
    """
        API function for differentiation using 'tf.gradients'

        Arguments:
            kwargs : keyword arguments : Arguments for 'tf.keras.layers.Lambda(**kwargs)' such as 'name'
    """
    return L.Lambda(lambda x: tf.gradients(x[0], x[1], unconnected_gradients='zero'), **kwargs)


class SourceLoc(L.Layer):
    """
        Class for Source location using tf.keras.layers.Layer.

        xs : source location [x, y, z]
    """
    def __init__(self, xs, **kwargs):
        super(SourceLoc, self).__init__(**kwargs)
        self.xs = self.add_weight(name=kwargs.get('name', 'xs'),
                                  shape=(len(xs),),
                                  trainable=False, 
                                  initializer=Initializer(xs))

    def call(self, x):
        return tf.ones_like(x) * self.xs

    def get_config(self):
        config = super(SourceLoc, self).get_config()
        config.update({"xs": self.xs.numpy()})
        return config


#######################################################################
                            ### CALLBACKS ###
#######################################################################


class LossesHolder(tf.keras.callbacks.Callback):
    """
        Callback container that can save logs if training is launched multiple times (without overriding)

        Arguments:
            NES : NES instance : if None, callback collect all available logs. If given, you can pass
            validation : list of dicts : format : [{'out': 'E', 'x': x, 'y': y or None,
                                                    'freq': 10, 'batch' : 10000, 
                                                    'loss_func': lambda x: np.abs(x).mean()}]

    """
    def __init__(self, NES=None, validation=[]):
        self.NES = NES
        for v in validation:
            assert v.get('out') in NES.outs.keys(), \
            f"Validation set can be applied for available outputs {NES.outs.keys()}"
        self.validation = validation

    def on_train_begin(self, logs=None):
        self.logs = {'loss' : [], 'epoch': []}
        
        for v in self.validation:
            self.logs[v['out'] + '_loss'] = [] 
            self.logs[v['out'] + '_epoch'] = [] 
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        self.logs['epoch'].append(epoch)
        for kw, v in logs.items():
            if self.logs.get(kw) is None:
                self.logs[kw] = []
            self.logs[kw].append(v)

        for v in self.validation:
            if ((epoch % v['freq']) == 0):
                self.logs[v['out'] + '_loss'].append(self.evaluate_model(v))
                self.logs[v['out'] + '_epoch'].append(epoch) 

    def evaluate_model(self, v):
        y_eval = self.NES._predict(v['x'], v['out'], batch_size=v.get('batch_size'))
        if v.get('y') is not None:
            y_eval -= v['y']
        return v['loss_func'](y_eval)


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


class BestWeights(tf.keras.callbacks.Callback):
    """Keeps only the best weights of the model.

    Args:
        monitor: Quantity to be monitored ('loss' or 'val_loss'). By default, 'loss'
        start_monitor: below which 'monitor' value callback starts watching 'monitor' and saves best weights
        freq: how often save weights
        verbose: verbosity mode (0 or 1).
    """

    def __init__(self,
                 monitor='loss',
                 start_monitor=1e-2,
                 freq=1,
                 verbose=1,
                 ):
        super(BestWeights, self).__init__()
        assert monitor == 'loss' or monitor == 'val_loss', \
            "Only 'loss' and 'val_loss' are supported for monitor metric"

        self.monitor = monitor
        self.verbose = verbose
        self.freq = freq
        self.start_monitor = start_monitor
        self.best_monitor = np.inf
        self.best_epoch = None
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.best_monitor = np.inf
        self.best_epoch = None
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if (epoch > 0) and (epoch % self.freq == 0) and \
           (current < self.best_monitor) and (current < self.start_monitor):
            self.best_monitor = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()

            if self.verbose:
                print('Epoch %05d: saving best weights' % (epoch + 1))
                print(f'{self.monitor}: {self.best_monitor:.5f}')

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        if self.verbose:
            print("Set the best weights for the model\n")

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
          logging.warning('The callback conditioned on metric `%s` '
                          'which is not available. Available metrics are: %s',
                          self.monitor, ','.join(list(logs.keys())))
        return monitor_value


#######################################################################
                            ### OTHER ###
#######################################################################


def data_handler(x, y, **kwargs):
    callbacks = kwargs.get('callbacks', [])
    generator_required = False
    for c in callbacks:
        if isinstance(c, (experimental.ImportanceSampling, experimental.ImportanceWeighting, 
                          experimental.RARsampling, experimental.FromCoarseToFineResampling)):
            generator_required = True
            break

    if generator_required:
        data = experimental.Generator(x, len(y), 
                        batch_size=kwargs.pop('batch_size', None), 
                        sample_weights=kwargs.pop('sample_weights', None), 
                        shuffle=kwargs.pop('shuffle', True))
    else:
        data = (x, y)

    return data


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


class RegularGrid:
    """
        API for generating regular distribution in a given velocity model
    """
    limits = None 
    def __init__(self, velocity):
        """velocity: velocity class
        """
        self.xmins = velocity.xmin
        self.xmaxs = velocity.xmax

    def __call__(self, axes):
        """ axes : tuple of ints : (nx, ny, nz)
        """
        xi = [np.linspace(self.xmins[i], self.xmaxs[i], axes[i]) for i in range(len(axes))]
        X = np.meshgrid(*xi, indexing='ij')
        X = np.stack(X, axis=-1)
        return X

    @staticmethod
    def sou_rec_pairs(xs, xr):
        xs, xr = np.array(xs, ndmin=2), np.array(xr, ndmin=2)
        assert xr.shape[-1] == xs.shape[-1]
        dim = xs.shape[-1]

        Xs = np.expand_dims(xs, axis=tuple(i+len(xs.shape[:-1]) for i in range(len(xr.shape[:-1]))))
        Xr = np.expand_dims(xr, axis=tuple(i for i in range(len(xs.shape[:-1]))))
        Xs = np.tile(Xs, (1,)*len(xs.shape[:-1]) + tuple(j for j in xr.shape[:-1]) + (1,))
        Xr = np.tile(Xr, tuple(j for j in xs.shape[:-1]) + (1,)*len(xr.shape[:-1]) + (1,))
        return np.concatenate((Xs, Xr), axis=-1)


class Uniform_PDF:
    """
        API for generating uniform distribution in a domain limits
    """
    limits = None

    def __init__(self, *args):
        """
            args: 
                - instance of NES.velocity.BaseVelocity
                - (xmin, xmax) 
        """
        if len(args) == 1:
            assert isinstance(args[0], velocity.BaseVelocity)
            xmins, xmaxs = getattr(args[0], 'xmin'), getattr(args[0], 'xmax')
        elif len(args) == 2:
            assert len(args[0]) == len(args[1])
            xmins, xmaxs = args
        else:
            raise ValueError("Arguments must be either limits or instance `NES.velocity.BaseVelocity`")
        self.limits = np.array([xmins, xmaxs])

    def __call__(self, num_points, rep=1):
        """ Return random points from uniform distribution in a given domain
        """
        dim = self.limits.shape[1] * rep
        lim0 = list(self.limits[0]) * rep
        lim1 = list(self.limits[1]) * rep
        return np.random.uniform(lim0, lim1, size=(num_points, dim))