"""
Experimental adaptive sampling of collocation points (Keras 3, backend-agnostic).

Callbacks with `requires_generator = True` make `NES.train` feed data through `Generator`,
which they can grow during training via `NES.data_generator.add_data(...)`.
"""
import numpy as np
import keras
from . import utils


#######################################################################
                        ### DATA GENERATOR ###
#######################################################################


class Generator(keras.utils.PyDataset):
    """
        Shuffled mini-batches of a (growing) training set.

        x : dict of arrays, as `NES._prepare_inputs` returns ({'x': (N, n_in), 'vr': (N, 1), ...})
        Batches are `(x_batch, dummy_zero_targets, sample_weights)`; the residual loss ignores the targets.
    """
    def __init__(self, x, batch_size=None, sample_weights=None, shuffle=True, verbose=0, **kwargs):
        super().__init__(**kwargs)
        self.shuffle = shuffle
        self._set_state(x, batch_size, sample_weights, verbose)

    def _set_state(self, x, batch_size, sample_weights, verbose):
        self.x = {k: np.asarray(v) for k, v in x.items()}
        self.size = len(self.x['x'])
        self.sample_weights = np.ones(self.size, dtype='float32') if sample_weights is None \
            else np.asarray(sample_weights, dtype='float32')
        self.batch_size = int(np.ceil(self.size / 4)) if batch_size is None else int(batch_size)
        self.n_batches = int(np.ceil(self.size / self.batch_size))
        self.ids = np.arange(self.size)
        self._shuffle()
        if verbose:
            self.print_status()

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        ids = self.ids[idx * self.batch_size: (idx + 1) * self.batch_size]
        return ({k: v[ids] for k, v in self.x.items()},
                np.zeros((len(ids), 1), dtype='float32'),
                self.sample_weights[ids])

    def on_epoch_end(self):
        self._shuffle()

    def _shuffle(self):
        # permuting indices keeps inputs and sample weights aligned
        if self.shuffle and self.n_batches > 1:
            np.random.shuffle(self.ids)

    def get_data(self):
        return self.x, self.sample_weights

    def add_data(self, x, sample_weights=None, verbose=0):
        """ Appends points (dict as `self.x`), keeping the number of batches per epoch """
        x_new = {k: np.concatenate((v, np.asarray(x[k], dtype=v.dtype)), axis=0) for k, v in self.x.items()}
        n_add = len(x_new['x']) - self.size
        w_add = np.ones(n_add) if sample_weights is None else np.asarray(sample_weights)
        batch_size = int(np.ceil(len(x_new['x']) / self.n_batches))
        self._set_state(x_new, batch_size, np.concatenate((self.sample_weights, w_add)), verbose)

    def set_weights(self, sample_weights):
        self.sample_weights = np.asarray(sample_weights * self.size / np.sum(sample_weights), dtype='float32')

    def print_status(self):
        print("\nTotal samples: {} ".format(self.size))
        print("Batch size: {} ".format(min(self.batch_size, self.size)))
        print("Total batches: {} \n".format(self.n_batches))


#######################################################################
                            ### CALLBACKS ###
#######################################################################


class LRScheduler(keras.callbacks.Callback):
    """
        Steps the learning rate down (`lrs`) each time the loss stays below the next of `bounds` for `patience` epochs.
        Requires a constant learning rate: compile with `decay=0`.
    """
    def __init__(self, monitor='loss', bounds=(3e-2, 5e-3), lrs=(5e-3, 1e-3, 7.5e-4),
                 patience=10, verbose=0, cooldown=5):
        super().__init__()
        self.monitor = monitor
        self.bounds = list(bounds)
        self.lrs = list(lrs)
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        self.cooldown_counter = 0
        self.wait = 0
        self.ctr = 0

    def _get_lr(self):
        return float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))

    def _set_lr(self, lr):
        optimizer = self.model.optimizer
        if isinstance(optimizer._learning_rate, keras.optimizers.schedules.LearningRateSchedule):
            raise ValueError("LRScheduler needs a constant learning rate: compile with `decay=0`")
        optimizer.learning_rate = lr

    def on_train_begin(self, logs=None):
        self._reset()
        self._set_lr(self.lrs[0])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        lr = self._get_lr()
        logs['lr'] = lr
        current = logs.get(self.monitor)

        if current is not None and lr > self.lrs[-1]:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.wait = 0
            elif self.ctr < len(self.bounds):
                self.wait += np.less(current, self.bounds[self.ctr])
                if self.wait >= self.patience:
                    new_lr = self.lrs[self.ctr + 1]
                    self._set_lr(new_lr)
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch + 1}: LR scheduler: changed learning rate to {new_lr}.')
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
                    self.ctr += 1


def _residual_magnitude(outputs, loss_func):
    """ Sums `loss_func` of all residual outputs of the training model -> (N,) """
    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    return sum(np.asarray(loss_func(np.asarray(r))).reshape(-1) for r in outputs)


def _sample(NES, pdf, num_pts):
    """ Uniform points without singular (source = receiver) points, as model inputs """
    x = pdf(num_pts, rep=NES._n_in // NES.dim)
    singular = NES._singular(x)
    while np.any(singular):
        x[singular] = pdf(int(singular.sum()), rep=NES._n_in // NES.dim)
        singular = NES._singular(x)
    return NES._prepare_inputs(x)


class RARsampling(keras.callbacks.Callback):
    """
        Residual-based adaptive refinement: every `freq` epochs evaluates residuals on `res_pts` random points
        and adds the `m` worst to the training set, unless the mean residual is below `eps`.
    """
    requires_generator = True

    def __init__(self, NES, m=100, res_pts=1000, freq=10,
                 eps=1e-2, verbose=1, eval_batch_size=1e5,
                 loss_func=np.abs):
        super().__init__()
        self.NES = NES
        self.pdf = utils.Uniform_PDF(self.NES.velocity)
        self.m = m
        self.res_pts = max(res_pts, m)
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
                self.NES.data_generator.add_data(log[0], sample_weights=None, verbose=self.verbose)
            elif self.verbose > 1:
                print('Epoch %05d: RAR' % (epoch + 1))
                print(f'Evaluated loss on test set {log:.5f} is lower than eps={self.eps:.5f}')

    def evaluate_model(self):
        x_eval = self.eval_data_generator(self.res_pts)
        y_eval = _residual_magnitude(self.model.predict(x_eval, batch_size=self.eval_batch_size, verbose=0),
                                     self.loss_func)
        res = y_eval.mean()
        if res > self.eps:
            inds = np.argsort(y_eval)[-self.m:]
            return {kw: v[inds] for kw, v in x_eval.items()}, res
        return res

    def eval_data_generator(self, num_pts):
        return _sample(self.NES, self.pdf, num_pts)


class FromCoarseToFineResampling(keras.callbacks.Callback):
    """
        Grows the training set through `set_pts` sizes each time the loss stays below `tolerance` for `patience` epochs.
    """
    requires_generator = True

    def __init__(self, NES, set_pts=(10, 100, 1000, 10000, 100000), tolerance=1e-3, patience=10, verbose=1):
        super().__init__()
        self.NES = NES
        self.pdf = utils.Uniform_PDF(self.NES.velocity)
        self.set_pts = list(set_pts)
        self.tolerance = tolerance
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.N = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.N = 0

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get('loss')
        if current is None:
            return
        if np.less(current, self.tolerance):
            self.wait += 1
        # Only check after the first epoch.
        if (self.wait >= self.patience) and (epoch > 0) and (self.N + 1 < len(self.set_pts)):
            self.wait = 0
            self.N += 1
            if self.verbose > 0:
                print(f'Epoch {epoch + 1:5d}: Resampling to {self.set_pts[self.N]:7d}')
                print(f'Reached loss: {current:.5f}')
            self.NES.data_generator.add_data(self.data_generator(self.set_pts[self.N] - self.set_pts[self.N - 1]),
                                             sample_weights=None, verbose=self.verbose)

    def data_generator(self, num_pts):
        return _sample(self.NES, self.pdf, num_pts)


def TP_solver(NES):
    """ Whether NES is two-point """
    return NES._n_in == 2 * NES.dim


def check_singularities(x, NES):
    """ Whether `x` contains singular (source = receiver) points """
    return bool(np.any(NES._singular(np.asarray(x))))


######################################################
        ### GENERATION OF COLLOCATION POINTS ###
######################################################


class GradientBased_PDF:
    """
        API for generating gradient based distribution in a given 2D velocity model
    """
    def __init__(self, velocity):
        """velocity: velocity class
        """
        self.grad_func = velocity.gradient
        self.limits = np.array([velocity.xmin, velocity.xmax]).T

    def __call__(self, num_pts, regular=0.7, random_base=20000):
        """ Returns 'regular' fraction of points on a regular grid, the rest sampled proportionally to |grad v| + mean
        """
        reg_num = int(np.sqrt(num_pts * regular))
        x_reg = [np.linspace(*limi, reg_num) for limi in self.limits]
        x_reg = np.stack(np.meshgrid(*x_reg, indexing='ij'), axis=-1)
        x_reg = x_reg.reshape(-1, x_reg.shape[-1])

        rand_num = num_pts - reg_num**2
        x_rand_base = np.random.uniform(*self.limits.T, size=(random_base, len(self.limits)))
        dv = self.grad_func(x_rand_base)
        dv = np.linalg.norm(dv, axis=-1)
        dv += dv.mean()
        dv /= dv.sum()
        rand_ids = np.random.choice(random_base, rand_num, p=dv, replace=False)
        x_rand = x_rand_base[rand_ids]

        return np.concatenate((x_reg, x_rand), axis=0)
