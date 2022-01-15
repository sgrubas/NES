import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.python.platform import tf_logging as logging
from .misc import Uniform_PDF
from sklearn.cluster import KMeans


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

#######################################################################
                        ### DATA GENERATOR ###
#######################################################################


class Generator(tf.keras.utils.Sequence):
    def __init__(self, x, y_num, batch_size=None, sample_weights=None, shuffle=True):
        self._set_state(x, y_num, batch_size, sample_weights, shuffle)

    def _set_state(self, x, y_num, batch_size, sample_weights, shuffle):
        if isinstance(x, dict):
            self.x = np.array(list(x.values())).T
        else:
            self.x = x
        self.y = np.zeros((self.x.shape[0], y_num))
        self.size = self.x.shape[0]
        if sample_weights is None:
            sample_weights = np.ones(self.size)
        self.sample_weights = sample_weights
        if batch_size is None:
            batch_size = np.ceil(self.size / 4).astype(int)
        self.batch_size = batch_size
        self.num_batches = np.ceil(self.size / batch_size).astype(int)
        self.shuffle = shuffle
        self.ids = np.arange(0, self.size)
        self.p = None
        self.shuffle_batch_size = self.size
        self.reshuffle()
        self.print_status()

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self.p is None:
            return self._get_normal_batch(idx)
        else:
            return self._get_important_batch()
    
    def on_epoch_end(self,):
        if self.p is None:
            self.reshuffle()

    def get_data(self):
        return self.x, self.y, self.sample_weights

    def reshuffle(self):
        if self.num_batches > 1 and self.shuffle:
            self.ids = np.random.choice(self.size, self.shuffle_batch_size, replace=False, p=self.p)

    def _get_normal_batch(self, idx):
        i1 = idx * self.batch_size
        i2 = min(i1 + self.batch_size, self.size)
        ids = self.ids[i1 : i2]
        inputs = [*self.x[ids].T]
        outputs = [*self.y[ids].T]
        sample_weights = self.sample_weights[ids]
        return inputs, outputs, sample_weights

    def _get_important_batch(self,):
        self.reshuffle()
        inputs = [*self.x[self.ids].T]
        outputs = [*self.y[self.ids].T]
        sample_weights = self.sample_weights[self.ids]
        return inputs, outputs, sample_weights

    def add_data(self, x, sample_weights=None):
        if isinstance(x, dict):
            x = np.array(list(x.values())).T
        x_new = np.concatenate((self.x, x.reshape(x.shape[0], self.x.shape[-1])), axis=0)
        if sample_weights is None:
            sample_weights = np.ones(x.shape[0])
        sample_weights = np.concatenate((self.sample_weights, sample_weights), axis=0)
        batch_size = np.ceil(x_new.shape[0] / self.num_batches).astype(int)
        self._set_state(x_new, self.y.shape[-1], batch_size, sample_weights, self.shuffle)

    def set_probabilities(self, p):
        self.p = p.ravel()
        self.shuffle_batch_size = self.batch_size
        self.set_weights(1.0 / (self.p * self.size))

    def reset_probabilities(self,):
        self.p = None
        self.shuffle_batch_size = self.size
        self.sample_weights = np.ones(self.size)

    def set_weights(self, sample_weights):
        self.sample_weights = sample_weights * self.size / np.sum(sample_weights)

    def print_status(self,):
        print("\nTotal samples: {} ".format(self.size))
        print("Batch size: {} ".format(min(self.batch_size, self.size)))
        print("Total batches: {} \n".format(self.num_batches))


#######################################################################
                            ### CALLBACKS ###
#######################################################################


class LossesHolder(tf.keras.callbacks.Callback):
    """
        Callback container that can save losses if training is launched multiple times (without overriding)
    """
    def __init__(self, NES=None, mae_test=None, freq=10, eval_batch_size=1e5):
        self.mae_test = mae_test
        self.NES = NES
        self.freq = freq
        self.eval_batch_size = int(eval_batch_size)
    
    def on_train_begin(self, logs=None):
        self.logs = {'loss' : [], 'epoch': []}
        if self.mae_test is not None:
            self.logs['mae'] = []
            self.logs['mae_epoch'] = []

        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        self.logs['epoch'].append(epoch)
        for kw, v in logs.items():
            if self.logs.get(kw) is None:
                self.logs[kw] = []
            self.logs[kw].append(v)
        
        self.wait += 1

        if ((self.wait % self.freq) == 0) and \
            (self.mae_test is not None):
            self.evaluate_model()
            self.logs['mae_epoch'].append(epoch)

    def evaluate_model(self,):
        t_pred = self.NES.Traveltime(self.mae_test[0], batch_size=self.eval_batch_size)
        mae = np.abs(t_pred - self.mae_test[1]).mean()
        self.logs['mae'].append(mae)


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


class RARsampling(tf.keras.callbacks.Callback):
    def __init__(self,
                 NES,
                 m=100,
                 res_pts=1000,
                 freq=10,
                 eps=1e-2,
                 verbose=1,
                 eval_batch_size=1e5,
                 loss_func=lambda x: x+1,
                ):
        super(RARsampling, self).__init__()

        self.NES = NES
        self.pdf = Uniform_PDF(self.NES.velocity)
        self.m = m
        self.res_pts = res_pts
        self.freq = freq
        self.verbose = verbose
        self.eps = eps
        self.eval_batch_size = int(eval_batch_size)
        self.loss_func = loss_func

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.freq) == 0 and epoch > 0:
            log = self.evaluate_model()
            if isinstance(log, tuple):
                if self.verbose > 0:
                    print('Epoch %05d: RAR' % (epoch + 1))
                    print(f'Evaluated loss on test set: {log[1]:.5f}')
                    print(f'{self.m} additional points added to the training set.')
                self.NES.data_generator.add_data(log[0])
            else:
                if self.verbose > 1:
                    print('Epoch %05d: RAR' % (epoch + 1))
                    print(f'Evaluated loss on test set {log:.5f} is lower than eps={self.eps:.5f}')          

    def evaluate_model(self,):
        x_eval = self.eval_data_generator(self.res_pts)
        y_eval = self.loss_func(self.model.predict(x_eval, batch_size=self.eval_batch_size))
        res = y_eval.mean()

        if res > self.eps:
            inds = np.argsort(y_eval.ravel())[-self.m:].ravel()
            x_add = {kw : v[inds] for kw, v in x_eval.items()}
            return x_add, res
        else:
            return res

    def eval_data_generator(self, num_pts):
        x = self.pdf(num_pts)        
        if np.any((x == self.NES.xs[None, ...]).prod(axis=-1)):
            self.eval_data_generator(num_pts)
            print("Accidentally source point in training set, resetting...")
        x_eval = self.NES._prepare_inputs(self.model, x, self.NES.velocity)
        return x_eval


class ImportanceSampling(tf.keras.callbacks.Callback):
    def __init__(self,
                 NES,
                 num_seeds=100,
                 p_min=0.0,
                 freq=1,
                 duration=10,
                 verbose=1,
                 seeds_batch_size=None,
                 loss_func=lambda x: x+1,
                ):
        super(ImportanceSampling, self).__init__()

        self.NES = NES
        self.num_seeds = num_seeds
        self.x_seeds = None
        if seeds_batch_size is None:
            seeds_batch_size = num_seeds
        self.seeds_batch_size = int(seeds_batch_size)
        self.loss_func = loss_func 

        self.freq = freq
        self.p_min = p_min
        if self.freq == 1:
            self.duration = np.inf
        else:
            self.duration = duration
        self.verbose = verbose
        self.cntr = 0
        self.cntr2 = 0
        self.epoch = 0

    def on_train_begin(self, logs=None):
        self.define_seeds()

    def on_batch_begin(self, batch, logs=None):
        if self.cntr == 0 and self.cntr2 == 0 and self.epoch > 0:
            self.cntr = self.duration
            if self.verbose:
                print('Epoch %05d: Importance Based Sampling : started' % (self.epoch + 1))
        elif self.epoch == 0:
            self.cntr2 = self.freq
        else: pass

        if self.cntr > 0:
            p = self.evaluate_seeds()
            self.NES.data_generator.set_probabilities(p)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if self.cntr > 0:
            self.cntr -= 1
            if self.cntr == 0:
                self.NES.data_generator.reset_probabilities()
                self.cntr2 = self.freq
                if self.verbose:
                    print('Epoch %05d: Importance Based Sampling : finished' % (self.epoch + 1))
        else:
            self.cntr2 -= 1

    def evaluate_seeds(self,):
        y_eval = self.loss_func(self.model.predict(self.x_seeds, batch_size=self.seeds_batch_size))
        inds = self.kmeans.predict(self.NES.data_generator.x[:, :self.NES.dim]).squeeze()
        loss = y_eval[inds].squeeze()
        p = loss / np.sum(loss)
        if self.p_min > 0:
            p[p < self.p_min] = self.p_min
            p /= np.sum(p)
        return p

    def define_seeds(self,):
        self.kmeans = KMeans(n_clusters=self.num_seeds, n_init=1).fit(self.NES.data_generator.x[:, :self.NES.dim])
        self.seeds = self.kmeans.cluster_centers_
        if np.any((self.seeds == self.NES.xs[None, ...]).prod(axis=-1)):
            self.define_seeds()
            print("Accidentally source point in training set, resetting...")
        self.x_seeds = self.NES._prepare_inputs(self.model, self.seeds, self.NES.velocity)


class ImportanceWeighting(tf.keras.callbacks.Callback):
    def __init__(self,
                 NES,
                 num_seeds=100,
                 w_lims=(0.1, 0.9),
                 freq=1,
                 verbose=1,
                 seeds_batch_size=None,
                 loss_func=lambda x: x + 1,
                ):
        super(ImportanceWeighting, self).__init__()

        self.NES = NES
        self.num_seeds = num_seeds
        self.w_lims = w_lims
        self.x_seeds = None
        self.seeds_batch_size = seeds_batch_size
        self.loss_func = loss_func

        self.freq = freq
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.define_seeds()

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.freq == 0 and epoch > 0:
            self.NES.data_generator.set_weights(self.evaluate_weights())
            if self.verbose:
                print('Epoch %05d: Importance Based Weighting' % (epoch + 1))

    def evaluate_weights(self,):
        y_eval = self.loss_func(self.model.predict(self.x_seeds, batch_size=self.seeds_batch_size))
        x_train = self.NES.data_generator.x[:, :self.NES.dim]
        if len(y_eval) < len(x_train):
            inds = self.kmeans.predict(x_train).squeeze()
            loss = y_eval[inds].squeeze()
        else:
            loss = y_eval.squeeze()
        w = loss / np.max(loss)
        if self.w_lims is not None:
            w[w < self.w_lims[0]] = self.w_lims[0]
            w[w > self.w_lims[1]] = self.w_lims[1]
        return w

    def define_seeds(self,):
        if self.num_seeds is not None:
            self.kmeans = KMeans(n_clusters=self.num_seeds, n_init=1).fit(self.NES.data_generator.x[:, :self.NES.dim])
            self.seeds = self.kmeans.cluster_centers_
            if np.any((self.seeds == self.NES.xs[None, ...]).prod(axis=-1)):
                self.define_seeds()
                print("Accidentally source point in training set, resetting...")
            self.x_seeds = self.NES._prepare_inputs(self.model, self.seeds, self.NES.velocity)
        else:
            self.x_seeds = self.NES.x_train

        if self.seeds_batch_size is None:
            if self.num_seeds is None:
                self.seeds_batch_size = self.NES.data_generator.batch_size
            else:
                self.seeds_batch_size = self.num_seeds


#######################################################################
                            ### OTHER ###
#######################################################################

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