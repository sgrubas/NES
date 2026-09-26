import warnings
import numpy as np
import keras
from . import velocity
from .layers import ACTS, AdaptiveActivation, Activation  # noqa: F401 (backward-compatible imports)


#######################################################################
                            ### CALLBACKS ###
#######################################################################


class LossesHolder(keras.callbacks.Callback):
    """
        Callback container that can save logs if training is launched multiple times (without overriding)

        Arguments:
            NES : NES instance : if None, callback collect all available logs. If given, you can pass
            validation : list of dicts : format : [{'out': 'E', 'x': x, 'y': y or None,
                                                    'freq': 10, 'batch_size' : 10000,
                                                    'loss_func': lambda x: np.abs(x).mean()}]

    """
    def __init__(self, NES=None, validation=()):
        super().__init__()
        self.NES = NES
        for v in validation:
            assert v.get('out') in NES.output_keys, \
                f"Validation set can be applied for available outputs {NES.output_keys}"
        self.validation = list(validation)

    def on_train_begin(self, logs=None):
        self.logs = {'loss': [], 'epoch': []}
        for v in self.validation:
            self.logs[v['out'] + '_loss'] = []
            self.logs[v['out'] + '_epoch'] = []

    def on_epoch_end(self, epoch, logs=None):
        self.logs['epoch'].append(epoch)
        for kw, v in (logs or {}).items():
            self.logs.setdefault(kw, []).append(v)

        for v in self.validation:
            if (epoch % v['freq']) == 0:
                self.logs[v['out'] + '_loss'].append(self.evaluate_model(v))
                self.logs[v['out'] + '_epoch'].append(epoch)

    def evaluate_model(self, v):
        y_eval = self.NES._predict(v['x'], v['out'], batch_size=v.get('batch_size', v.get('batch')))
        if v.get('y') is not None:
            y_eval = y_eval - v['y']
        return v['loss_func'](y_eval)


class _MonitorMixin:
    def get_monitor_value(self, logs):
        monitor_value = (logs or {}).get(self.monitor)
        if monitor_value is None:
            warnings.warn(f"Callback is conditioned on metric `{self.monitor}` which is not available. "
                          f"Available metrics are: {','.join(list((logs or {}).keys()))}")
        return monitor_value


class NES_EarlyStopping(_MonitorMixin, keras.callbacks.Callback):
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
        conversion: Callable function that maps `monitor` to units comparable with `tolerance`.
                It is empirical equation that defines the dependence between loss and RMAE.
                By default conversion(x) = x * 10**(-0.16)
        patience: Number of epochs `conversion(monitor)` must be lower than `tolerance` to be stopped.
        verbose: verbosity mode (0 or 1).
    """

    def __init__(self, monitor='loss', tolerance=0, patience=10, verbose=1, conversion=lambda x: x * 10**(-0.16)):
        super().__init__()
        assert monitor in ('loss', 'val_loss'), "Only 'loss' and 'val_loss' are supported for monitor metric"
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


class BestWeights(_MonitorMixin, keras.callbacks.Callback):
    """Keeps only the best weights of the model.

    Args:
        monitor: Quantity to be monitored ('loss' or 'val_loss'). By default, 'loss'
        start_monitor: below which 'monitor' value callback starts watching 'monitor' and saves best weights
        freq: how often save weights
        verbose: verbosity mode (0 or 1).
        conversion: maps `monitor` to approximate RMAE for printing (see `NES_EarlyStopping`)
    """

    def __init__(self, monitor='loss', start_monitor=1e-2, freq=1, verbose=1, conversion=lambda x: x * 10**(-0.16)):
        super().__init__()
        assert monitor in ('loss', 'val_loss'), "Only 'loss' and 'val_loss' are supported for monitor metric"
        self.monitor = monitor
        self.verbose = verbose
        self.freq = freq
        self.start_monitor = start_monitor
        self.conversion = conversion
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
                print(f'RMAE ~ {100 * self.conversion(self.best_monitor):.5f} %\n')

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            if self.verbose:
                print("Set the best weights for the model\n")


NES_BestWeights = BestWeights  # alias used in some notebooks


#######################################################################
                            ### SAMPLING ###
#######################################################################


class RegularGrid:
    """
        API for generating regular distribution in a given velocity model
    """
    def __init__(self, velocity):
        """velocity: velocity class
        """
        self.xmins = velocity.xmin
        self.xmaxs = velocity.xmax

    def __call__(self, axes):
        """ axes : tuple of ints : (nx, ny, nz)
        """
        xi = [np.linspace(self.xmins[i], self.xmaxs[i], n) for i, n in enumerate(axes)]
        return np.stack(np.meshgrid(*xi, indexing='ij'), axis=-1)

    @staticmethod
    def sou_rec_pairs(xs, xr):
        """ All source-receiver pairs: xs (*S, dim), xr (*R, dim) -> (*S, *R, 2*dim) """
        xs, xr = np.array(xs, ndmin=2), np.array(xr, ndmin=2)
        assert xr.shape[-1] == xs.shape[-1]
        S, R, dim = xs.shape[:-1], xr.shape[:-1], xs.shape[-1]
        Xs = np.broadcast_to(xs.reshape(S + (1,) * len(R) + (dim,)), S + R + (dim,))
        Xr = np.broadcast_to(xr.reshape((1,) * len(S) + R + (dim,)), S + R + (dim,))
        return np.concatenate((Xs, Xr), axis=-1)


class Uniform_PDF:
    """
        API for generating uniform distribution in a domain limits
    """
    def __init__(self, *args):
        """
            args:
                - instance of NES.velocity.BaseVelocity
                - (xmin, xmax)
        """
        if len(args) == 1:
            assert isinstance(args[0], velocity.BaseVelocity)
            xmins, xmaxs = args[0].xmin, args[0].xmax
        elif len(args) == 2:
            assert len(args[0]) == len(args[1])
            xmins, xmaxs = args
        else:
            raise ValueError("Arguments must be either limits or instance `NES.velocity.BaseVelocity`")
        self.limits = np.array([xmins, xmaxs])

    def __call__(self, num_points, rep=1):
        """ Return random points from uniform distribution in a given domain
        """
        lim0 = np.tile(self.limits[0], rep)
        lim1 = np.tile(self.limits[1], rep)
        return np.random.uniform(lim0, lim1, size=(num_points, len(lim0)))
