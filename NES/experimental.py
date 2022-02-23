import tensorflow as tf
import numpy as np
from tensorflow.python.platform import tf_logging as logging
from .misc import Uniform_PDF
from sklearn.cluster import KMeans

#######################################################################
                        ### DATA GENERATOR ###
#######################################################################


class Generator(tf.keras.utils.Sequence):
    def __init__(self, x, len_y, batch_size=None, sample_weights=None, shuffle=True, verbose=0):
        self._set_state(x, len_y, batch_size, sample_weights, shuffle, verbose)

    @staticmethod
    def _format_input(x):
        if isinstance(x, dict):
            X = np.array(list(x.values())).squeeze().T
        elif isinstance(x, list):
            X = np.array(x).squeeze().T
        elif isinstance(x, np.ndarray):
            X = x.reshape(-1, x.shape[-1])
        return X

    def _set_state(self, x, len_y, batch_size, sample_weights, shuffle, verbose):
        
        self.x = self._format_input(x)
        self.size = self.x.shape[0]
        self.y = np.zeros((self.size, len_y))
        if sample_weights is None:
            sample_weights = np.ones(self.size)
        self.sample_weights = sample_weights
        if batch_size is None:
            batch_size = np.ceil(self.size / 4).astype(int)

        self.batch_size = batch_size
        self.num_batches = np.ceil(self.size / batch_size).astype(int)
        self.shuffle = shuffle
        self.ids = np.arange(self.size)

        self._normal_shuffle()
        if verbose:
            self.print_status()

        self.p = None
        self.importance_batch_size = None

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self.p is None:
            return self._get_normal_batch(idx)
        else:
            return self._get_importance_batch()
    
    def on_epoch_end(self,):
        if self.p is None:
            self._normal_shuffle()

    def get_data(self):
        return self.x, self.y, self.sample_weights

    def _normal_shuffle(self):
        if self.num_batches > 1 and self.shuffle:
            np.random.shuffle(self.x)

    def _importance_shuffle(self):
        np.random.choice(self.ids, self.importance_batch_size, replace=True, p=self.p)

    def _get_normal_batch(self, idx):
        i1 = idx * self.batch_size
        i2 = min(i1 + self.batch_size, self.size)
        inputs = [*self.x[i1:i2].T]
        outputs = [*self.y[i1:i2].T]
        return inputs, outputs, self.sample_weights[i1:i2]

    def _get_importance_batch(self,):
        self._importance_shuffle()
        inputs = [*self.x[self.ids].T]
        outputs = [*self.y[:self.importance_batch_size].T]
        return inputs, outputs, self.sample_weights[self.ids]

    def add_data(self, x, sample_weights=None, verbose=0):
        x = self._format_input(x)
        x_new = np.concatenate((self.x, x), axis=0)

        if sample_weights is not None:
            sample_weights = np.concatenate((self.sample_weights, sample_weights), axis=0)
        else:
            sample_weights = np.concatenate((self.sample_weights, np.ones(x_new.shape[0])), axis=0)

        batch_size = np.ceil(x_new.shape[0] / self.num_batches).astype(int)
        self._set_state(x_new, self.y.shape[-1], batch_size, sample_weights, self.shuffle, verbose)

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


class LRScheduler(tf.keras.callbacks.Callback):
  def __init__(self,
               monitor='loss',
               bounds=[3e-2, 5e-3],
               lrs=[5e-3, 1e-3, 7.5e-4],
               patience=10,
               verbose=0,
               cooldown=5):
    super(LRScheduler, self).__init__()

    self.monitor = monitor
    self.bounds = bounds
    self.lrs = lrs
    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.cooldown_counter = 0  # Cooldown counter.
    self.wait = 0
    self.ctr = 0
    self._reset()

  def _reset(self):
    """Resets wait counter and cooldown counter.
    """
    self.cooldown_counter = 0
    self.wait = 0
    self.ctr = 0

  def on_train_begin(self, logs=None):
    self._reset()
    tf.keras.backend.set_value(self.model.optimizer.lr, self.lrs[0])

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
    current = logs.get(self.monitor)
    
    if current is not None and logs['lr'] > self.lrs[-1]:
      if self.in_cooldown():
        self.cooldown_counter -= 1
        self.wait = 0
      elif self.ctr < len(self.bounds):
        self.wait += np.less(current, self.bounds[self.ctr])
        if self.wait >= self.patience:
          old_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
          new_lr = self.lrs[self.ctr+1]
          tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
          if self.verbose > 0:
            print(f'\nEpoch {epoch +1}: '
                  f'LR scheduler: changed learning rate to {new_lr}.')
          self.cooldown_counter = self.cooldown
          self.wait = 0
          self.ctr += 1
      else:
        pass

  def in_cooldown(self):
    return self.cooldown_counter > 0


class RARsampling(tf.keras.callbacks.Callback):
    def __init__(self, NES, m=100, res_pts=1000, freq=10,
                 eps=1e-2, verbose=1, eval_batch_size=1e5,
                 loss_func=np.abs):
        super(RARsampling, self).__init__()

        self.NES = NES
        self.tp = TP_solver(NES)
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
                self.NES.data_generator.add_data(log[0], sample_weights=None, verbose=self.verbose)
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
        if self.tp:
            x = np.concatenate((x, self.pdf(num_pts)), axis=-1)
        if check_singularities(x, self.NES):
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
                 loss_func=np.abs,
                ):
        super(ImportanceSampling, self).__init__()

        self.NES = NES
        self.tp = TP_solver(NES)
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
        inds = self.kmeans.predict(self.NES.data_generator.x[:, :self.NES.dim*(1+self.tp)]).squeeze()
        loss = y_eval[inds].squeeze()
        p = loss / np.sum(loss)
        if self.p_min > 0:
            p[p < self.p_min] = self.p_min
            p /= np.sum(p)
        return p

    def define_seeds(self,):
        c = TP_solver(self.NES)
        self.kmeans = KMeans(n_clusters=self.num_seeds, n_init=1).fit(self.NES.data_generator.x[:, :self.NES.dim*(1+self.tp)])
        self.seeds = self.kmeans.cluster_centers_
        if check_singularities(self.seeds, self.NES):
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
        self.tp = TP_solver(NES)
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
        x_train = self.NES.data_generator.x[:, :self.NES.dim*(1+self.tp)]
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
            self.kmeans = KMeans(n_clusters=self.num_seeds, n_init=1).fit(self.NES.data_generator.x[:, :self.NES.dim*(1+self.tp)])
            self.seeds = self.kmeans.cluster_centers_
            if check_singularities(self.seeds, self.NES):
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

def TP_solver(NES):
    if getattr(NES, 'xs', False):
        return False
    else:
        return True

def check_singularities(x, NES):
    if not TP_solver(NES):
        if np.any((x == NES.xs[None, ...]).prod(axis=-1)):
            return True
    else:
        if np.any((x[..., :NES.dim] == x[..., NES.dim:]).prod(axis=-1)):
            return True 