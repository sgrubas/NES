import pathlib
import pickle
import numpy as np
import keras
from keras import ops

from .backend import value_and_grad, hessian_rows
from .layers import TraveltimeNet, resolve_reciprocity
from .eikonalLayers import IsoEikonal
from .velocity import BaseVelocity
from .utils import Uniform_PDF, RegularGrid, NES_EarlyStopping
from . import legacy

FORMAT_VERSION = 2


###############################################################################
                    ### KERAS MODELS (backend-agnostic) ###
###############################################################################


def _evaluate(net, equation, dim, layout, inputs, keys):
    """
        Computes solution quantities for `keys`, sharing the forward/backward passes between them.

        T : traveltime (N, 1)
        G<s> : gradient w.r.t. block <s> (N, dim)
        E<s> : eikonal residual w.r.t. block <s> (N, 1), needs velocity input 'v<s>'
        L<s> : laplacian (N, 1),   H<s> : upper triangle of hessian (N, dim*(dim+1)/2),   Hsr : mixed (N, dim, dim)

        <s> is '' for NES-OP, 'r' (receiver) or 's' (source) for NES-TP; `layout` maps it to the column offset.
    """
    x = inputs['x']
    if any(k[0] in 'GE' for k in keys):
        T, G = value_and_grad(net, x)
    elif 'T' in keys:
        T = net(x)

    out = []
    for k in keys:
        if k == 'T':
            out.append(T)
        elif k[0] in 'GE':
            o = layout[k[1:]]
            g = G[:, o:o + dim]
            out.append(g if k[0] == 'G' else equation(g, inputs['v' + k[1:]]))
        elif k == 'Hsr':
            o_s, o_r = layout['s'], layout['r']
            out.append(hessian_rows(net, x, range(o_s, o_s + dim))[:, :, o_r:o_r + dim])
        else:  # 'L' or 'H'
            o = layout[k[1:]]
            H = hessian_rows(net, x, range(o, o + dim))[:, :, o:o + dim]
            if k[0] == 'L':
                out.append(ops.sum(ops.diagonal(H, axis1=1, axis2=2), axis=-1, keepdims=True))
            else:
                out.append(ops.stack([H[:, i, j] for i in range(dim) for j in range(i, dim)], axis=-1))
    return out


class _SolverModel(keras.Model):
    """ Keras model evaluating `keys` of the solution for inputs {'x': coordinates, 'v<s>': velocities} """
    def __init__(self, net, equation, dim, layout, keys, **kwargs):
        super().__init__(**kwargs)
        self.net, self.equation = net, equation  # attributes, so that the variables are tracked (JAX)
        self.dim, self.layout, self.keys = dim, dict(layout), tuple(keys)

    def build(self, input_shape=None):
        pass  # `net` is built by the solver

    def call(self, inputs):
        out = _evaluate(self.net, self.equation, self.dim, self.layout, inputs, self.keys)
        return out[0] if len(out) == 1 else out


class _ResidualModel(_SolverModel):
    """
        Training model: outputs eikonal residuals, the loss drives them to zero (no targets are needed).
        Works with `fit` on every backend, since the input gradient is taken inside `call`.
    """
    def __init__(self, *args, loss='mae', **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_loss = loss

    def call(self, inputs):
        return _evaluate(self.net, self.equation, self.dim, self.layout, inputs, self.keys)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        loss_fn = keras.losses.get(self.residual_loss)
        total = 0.0
        for r in y_pred:
            if isinstance(loss_fn, keras.losses.Loss):
                total = total + loss_fn(ops.zeros_like(r), r, sample_weight=sample_weight)
                continue
            per_sample = loss_fn(ops.zeros_like(r), r)
            if sample_weight is not None:
                per_sample = per_sample * ops.reshape(ops.cast(sample_weight, per_sample.dtype), (-1,))
            total = total + ops.mean(per_sample)
        if self.losses:
            total = total + ops.sum(self.losses)
        return total


class _Outputs(dict):
    """ Lazily built inference models: `outs['T']`, `outs['Gr']`, `outs[('T', 'Gs')]`, ... """
    def __init__(self, solver):
        super().__init__()
        self._solver = solver

    def __missing__(self, key):
        keys = (key,) if isinstance(key, str) else tuple(key)
        unknown = set(keys) - set(self._solver.output_keys)
        if unknown:
            raise KeyError(f"Unknown outputs {unknown}, available: {self._solver.output_keys}")
        s = self._solver
        model = _SolverModel(s.net, s.equation, s.dim, s._layout, keys, name='_'.join(keys))
        self[key] = model
        return model


###############################################################################
                        ### SHARED SOLVER LOGIC ###
###############################################################################


class _NESBase:
    """ Shared logic of NES_OP and NES_TP: training, prediction, saving/loading """
    output_keys = ()
    loss_keys = ()
    _default_lr = 3e-3

    def __init__(self, velocity, eikonal=None, name=None):
        assert isinstance(velocity, BaseVelocity), "Must be instance of BaseVelocity"
        self.velocity = velocity
        self.name = name
        # Input scale factor
        self.xscale = np.max(np.abs([self.velocity.xmin, self.velocity.xmax]))
        # Eikonal equation layer
        if eikonal is None:
            eikonal = IsoEikonal()
        assert isinstance(eikonal, keras.layers.Layer), "Eikonal should be an instance of keras Layer"
        self.equation = eikonal

        self.sing_eps = 1e-5    # tolerance for source singularity (to remove from training)
        self.x_train = None     # input training data
        self.compiled = False   # compilation status
        self.config = {}        # config data of NN model to be reproducible
        self.net = None         # traveltime network, `keras.Model` mapping coordinates to T
        self.model = None       # training model (eikonal residuals)
        self.outs = _Outputs(self)

    # ----- to be defined by subclasses -----

    @property
    def _layout(self):
        raise NotImplementedError

    @property
    def _n_in(self):
        raise NotImplementedError

    def _net_kwargs(self):
        raise NotImplementedError

    def _init_kwargs(self):
        raise NotImplementedError

    def _singular(self, x):
        raise NotImplementedError

    # ----- model -----

    def _build(self, losses, **config):
        losses = list(self.loss_keys[:1] if losses is None else losses)
        wrong = set(losses) - set(self.loss_keys)
        if wrong:
            raise ValueError(f"Losses {wrong} are not supported, available: {self.loss_keys}")
        self.config = dict(config, losses=losses)
        self.net = TraveltimeNet(**self._net_kwargs(), **config, name=f'{self.name}_traveltime')
        self.net.build()
        self.model = _ResidualModel(self.net, self.equation, self.dim, self._layout, losses,
                                    name=f"{self.name}_{'_'.join(losses)}")
        self.outs = _Outputs(self)
        self.compiled = False

    def __call__(self, x, **kwargs):
        """
            Calls the model for training (see keras.Model.__call__)
        """
        return self.model(x, **kwargs)

    # ----- inputs and predictions -----

    def _prepare_inputs(self, x, keys=None):
        """ Dict of model inputs: coordinates 'x' (N, n_in) and velocities 'v<s>' (N, 1) needed for `keys` """
        keys = self.config['losses'] if keys is None else ((keys,) if isinstance(keys, str) else keys)
        x = np.asarray(x)
        assert x.shape[-1] == self._n_in, "Dimensions do not coincide"
        X = x.reshape(-1, self._n_in)
        dtype = 'float64' if keras.config.floatx() == 'float64' else 'float32'   # no rounding in float64 mode
        inputs = {'x': X.astype(dtype)}
        for k in keys:
            if k[0] == 'E':
                o = self._layout[k[1:]]
                v = self.velocity(X[:, o:o + self.dim])
                inputs['v' + k[1:]] = np.asarray(v, dtype=dtype).reshape(-1, 1)
        return inputs

    def _predict(self, x, out, **kwargs):
        """ Predicts output `out` (key or tuple of keys, see `output_keys`), reshaped to `x.shape[:-1] + ...` """
        if kwargs.get('batch_size') is None:
            kwargs['batch_size'] = 100000
        kwargs.setdefault('verbose', 0)
        x = np.asarray(x)
        keys = (out,) if isinstance(out, str) else tuple(out)
        P = self.outs[out].predict(self._prepare_inputs(x, keys), **kwargs)
        reshape = lambda p: np.asarray(p).reshape(x.shape[:-1] + p.shape[1:]).squeeze()
        return reshape(P) if isinstance(out, str) else tuple(reshape(p) for p in P)

    def predict(self, x, outputs=('T',), **kwargs):
        """
            Predicts several outputs at once, sharing the computations (e.g. `('T', 'Gs')` needs one pass).

            Arguments:
                x : numpy array (..., n_in) : coordinates
                outputs : str or tuple of str : keys from `output_keys`
                **kwargs : arguments for keras.Model.predict such as 'batch_size'
        """
        return self._predict(x, outputs, **kwargs)

    # ----- training -----

    def train_inputs(self, x):
        """
            Creates dictionary of inputs for training. Removes singular points (xr=xs)
        """
        x = np.asarray(x)
        self.x_train = self._prepare_inputs(x[~self._singular(x)])

    def compile(self, optimizer=None, loss='mae', lr=None, decay=5e-4, **kwargs):
        """
            Compiles the neural-network model for training.

            Arguments:
                optimizer : instance of 'keras.optimizers.Optimizer' : If 'None', Adam is used.
                loss : str or callable (see 'keras.losses') : Loss applied to the residuals. By default "loss = 'mae'"
                lr : float : Learning rate, by default 3e-3 (NES_OP) and 5e-3 (NES_TP).
                decay : float : Inverse-time decay of learning rate, lr_t = lr / (1 + decay * t),
                                where t is the iteration. By default 'decay = 5e-4'.
                **kwargs : keyword arguments : Arguments for 'keras.Model.compile(**kwargs)' such as 'jit_compile'
        """
        if optimizer is None:
            lr = self._default_lr if lr is None else lr
            schedule = keras.optimizers.schedules.InverseTimeDecay(lr, decay_steps=1, decay_rate=decay) \
                if decay else lr
            optimizer = keras.optimizers.Adam(learning_rate=schedule)
        self.model.residual_loss = loss
        self.model.compile(optimizer=optimizer, **kwargs)
        self.compiled = True

    def train(self, x_train, tolerance=None, **train_kw):
        """
            Trains the neural-network model.

            Arguments:
                x_train : numpy array (N, n_in) of floats or int :
                                Array of collocation points. If 'int' is given, then random uniform distribution is used.
                tolerance : It can be:
                    1) float - Tolerance value for early stopping in RMAE units for traveltimes.
                               NES_EarlyStopping callback will be created with default options
                               (see `NES.utils.NES_EarlyStopping`)
                    2) instance of NES_EarlyStopping callback.
                **train_kw : keyword arguments : Arguments for 'keras.Model.fit(**train_kw)' such as 'batch_size', 'epochs'
        """
        if isinstance(x_train, (int, np.integer)):
            x_train = Uniform_PDF(self.velocity)(x_train, rep=self._n_in // self.dim)
        self.train_inputs(x_train)

        callbacks = list(train_kw.pop('callbacks', None) or [])
        if isinstance(tolerance, float):
            callbacks.append(NES_EarlyStopping(tolerance=tolerance))
        elif isinstance(tolerance, keras.callbacks.Callback):
            callbacks.append(tolerance)

        if any(getattr(c, 'requires_generator', False) for c in callbacks):
            from .experimental import Generator
            self.data_generator = Generator(self.x_train,
                                            batch_size=train_kw.pop('batch_size', None),
                                            sample_weights=train_kw.pop('sample_weight',
                                                                        train_kw.pop('sample_weights', None)),
                                            shuffle=train_kw.pop('shuffle', True))
            data = self.data_generator
        else:
            data = self.x_train

        if not self.compiled:
            self.compile()
        return self.model.fit(data, callbacks=callbacks, **train_kw)

    # ----- saving / loading -----

    def _optimizer_state(self):
        optimizer = getattr(self.model, 'optimizer', None)
        if optimizer is None or not optimizer.built:
            return None
        return {'optimizer': keras.optimizers.serialize(optimizer),
                'variables': [keras.ops.convert_to_numpy(v) for v in optimizer.variables],
                'loss': self.model.residual_loss}

    def _restore_optimizer(self, state):
        if state is None:
            return
        optimizer = keras.optimizers.deserialize(state['optimizer'])
        optimizer.build(self.model.trainable_variables)
        for var, value in zip(optimizer.variables, state['variables']):
            var.assign(value)
        self.compile(optimizer=optimizer, loss=state['loss'])

    def save(self, filepath, save_optimizer=False, training_data=False):
        """
            Saves the model to `filepath` directory: configuration + velocity model ('{folder}_config.pkl'),
            network weights ('{folder}_weights.weights.h5').
            `save_optimizer` saves the optimizer state to continue the training from the last point,
            `training_data` saves the last training set used for training.
        """
        path = pathlib.Path(filepath)
        path.mkdir(parents=True, exist_ok=True)
        filename = lambda kw: path / f'{path.name}_{kw}'
        for kw in ('optimizer.pkl', 'train_data.pkl', 'weights.pkl', 'weights.h5'):  # stale files of previous saves
            filename(kw).unlink(missing_ok=True)

        config = dict(self._init_kwargs(),
                      format_version=FORMAT_VERSION,
                      equation=(type(self.equation), self.equation.get_config()),
                      build=dict(self.config))
        with open(filename('config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        self.net.save_weights(filename('weights.weights.h5'))
        if save_optimizer:
            with open(filename('optimizer.pkl'), 'wb') as f:
                pickle.dump(self._optimizer_state(), f)
        if training_data:
            with open(filename('train_data.pkl'), 'wb') as f:
                pickle.dump(self.x_train, f)

    @classmethod
    def load(cls, filepath):
        """
            Creates an instance according to the configuration and pretrained weights in `filepath`.
            Also loads models saved by the original TensorFlow/Keras-2 NES (<= 0.2.x).
        """
        path = pathlib.Path(filepath)
        filename = lambda kw: path / f'{path.name}_{kw}'
        with open(filename('config.pkl'), 'rb') as f:
            config = pickle.load(f)
        if config.get('format_version', 1) < 2:
            return legacy.load(cls, path, config)

        eq_class, eq_config = config.pop('equation')
        build = config.pop('build')
        config.pop('format_version')
        instance = cls(eikonal=eq_class.from_config(eq_config), **config)
        instance.build_model(**build)
        instance.net.load_weights(filename('weights.weights.h5'))
        print(f'Loaded model from "{filepath}"')

        if filename('optimizer.pkl').is_file():
            with open(filename('optimizer.pkl'), 'rb') as f:
                instance._restore_optimizer(pickle.load(f))
            print('Compiled the model with saved optimizer')
        if filename('train_data.pkl').is_file():
            with open(filename('train_data.pkl'), 'rb') as f:
                instance.x_train = pickle.load(f)
            print(f'Loaded last training data: see {cls.__name__}.x_train')
        return instance

    def _transfer(self, **init_overrides):
        kwargs = dict(self._init_kwargs(), eikonal=self.equation, name=f"{self.name}_copy")
        kwargs.update({k: v for k, v in init_overrides.items() if v is not None})
        instance = type(self)(**kwargs)
        instance.build_model(**self.config)
        instance.net.set_weights(self.net.get_weights())
        instance._restore_optimizer(self._optimizer_state())
        return instance


###############################################################################
                    ### ONE POINT NEURAL EIKONAL SOLVER ###
###############################################################################


class NES_OP(_NESBase):
    """
    Neural Eikonal Solver for solving the equation in One-Point formulation tau(xr)

    Arguments:
        xs : list or array (dim,) of floats : Source location. 'dim' - dimension
        velocity : object : Velocity model class. Must be callable in a format 'v(xr) = velocity(xr)'.
                            See example in 'NES.velocity.Interpolator'
        eikonal : instance of keras.layers.Layer : Layer that mimics the eikonal equation.
                  There must be two inputs: gradient (N, dim), velocity (N, 1).
                  If 'None', 'eikonalLayers.IsoEikonal(p=2, hamiltonian=True)' is used.
    """
    instance_num = 0
    output_keys = ('T', 'G', 'E', 'L', 'H')
    loss_keys = ('E',)
    _default_lr = 3e-3

    def __init__(self, xs, velocity, eikonal=None, name=None):
        if not isinstance(xs, (list, tuple, np.ndarray)):
            raise ValueError("Unrecognized 'xs' type")
        name = name if name is not None else f"NESOP{NES_OP.instance_num}"
        NES_OP.instance_num += 1
        super().__init__(velocity, eikonal, name)
        self.xs = np.array(xs, dtype=float).reshape(-1)
        self.dim = len(self.xs)

    _layout = {'': 0}

    @property
    def _n_in(self):
        return self.dim

    def _net_kwargs(self):
        return dict(dim=self.dim, xscale=self.xscale, vmin=self.velocity.min, vmax=self.velocity.max, xs=self.xs)

    def _init_kwargs(self):
        return dict(xs=self.xs, velocity=self.velocity, name=self.name)

    def _singular(self, xr):
        return np.abs(xr - self.xs).sum(axis=-1) <= self.sing_eps

    def build_model(self, nl=4, nu=50, act='ad-gauss-1', out_act='ad-sigmoid-1',
                    input_scale=True, factored=True, out_vscale=True, improved_mlp=False, **kwargs):
        """
            Builds the neural-network model.

            Arguments:
                nl : int : Number of hidden layers, by default 'nl=4'
                nu : int or list of ints : Number of hidden units of each hidden layer, by default 'nu=50'
                act : formatted str : Hidden activation in format '(ad) -activation_name- n'.
                                      Format of activation - 'act(x) = f(a * n * x)', where 'a' is adaptive term (trainable weight),
                                      'n' constant term (degree of adaptivity). If 'ad' presents, 'a' is trainable, otherwise 'a=1'.
                                      By default "act = 'ad-gauss-1' "
                out_act : formatted str : Output activation in the same format as 'act'. By default "act = 'ad-sigmoid-1' "
                input_scale : boolean : Scale of inputs. By default 'input_scale=True'
                factored : boolean : Conventional factorization 'tau = R * out_act'. By default 'factored=True'
                out_vscale : boolean : Improved factorization 'tau = R * (1/vmin - 1/vmax) * out_act + 1/vmax'.
                                       If 'True', the 'out_act' must be bounded in [0, 1]. By default 'out_vscale=True'
                improved_mlp : boolean : whether to apply an Improved MLP structure (https://doi.org/10.1137/20M1318043)
                **kwargs : keyword arguments : 'losses' (default ['E']) and arguments for keras.layers.Dense(**kwargs)
                            such as 'kernel_initializer' (by default 'he_normal')
        """
        losses = kwargs.pop('losses', None)
        self._build(losses, nl=nl, nu=nu, act=act, out_act=out_act, input_scale=input_scale, factored=factored,
                    out_vscale=out_vscale, improved_mlp=improved_mlp, **kwargs)

    def Traveltime(self, xr, **kwargs):
        """
            Computes traveltimes.

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                **kwargs : keyword arguments : Arguments for keras.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                T : numpy array (N,) of floats : Traveltimes from the source 'NES_OP.xs' at 'xr'
        """
        return self._predict(xr, 'T', **kwargs)

    def Gradient(self, xr, **kwargs):
        """
            Computes gradients - vector (tau_dx, tau_dy, tau_dz). Returns numpy array (N, dim)
        """
        return self._predict(xr, 'G', **kwargs)

    def Velocity(self, xr, **kwargs):
        """
            Computes predicted velocity - 1 / ||( tau_dx, tau_dy, tau_dz) ||. Returns numpy array (N,)
        """
        return 1 / np.linalg.norm(self.Gradient(xr, **kwargs), axis=-1)

    def Laplacian(self, xr, **kwargs):
        """
            Computes laplacian - tau_dxdx + tau_dydy + tau_dzdz. Returns numpy array (N,)
        """
        return self._predict(xr, 'L', **kwargs)

    def Hessian(self, xr, **kwargs):
        """
            Computes full Hessian in a form of:
            1D: [tau_dxdx]
            2D: [tau_dxdx, tau_dxdy, tau_dydy]
            3D: [tau_dxdx, tau_dxdy, tau_dxdz, tau_dydy, tau_dydz, tau_dzdz]
            Returns numpy array (N, dim*(dim+1)/2)
        """
        return self._predict(xr, 'H', **kwargs)

    def transfer(self, xs=None, velocity=None, eikonal=None):
        """
            Transfers trained NES-OP to a new copy with new source 'xs', 'velocity', 'eikonal' (if not None).
            Returns NES_OP instance with the trained weights.
        """
        return self._transfer(xs=xs, velocity=velocity, eikonal=eikonal)


##############################################################################
                    ### TWO POINT NEURAL EIKONAL SOLVER ###
##############################################################################


class NES_TP(_NESBase):
    """
    Neural Eikonal Solver for solving the equation in Two-Point formulation T(xs, xr)

    Arguments:
        velocity : object : Velocity model class. Must be callable in a format 'v(xr) = velocity(xr)'.
                            See example in 'NES.velocity.Interpolator'
        eikonal : instance of keras.layers.Layer : Layer that mimics the eikonal equation.
                  There must be two inputs: gradient (N, dim), velocity (N, 1).
                  If 'None', 'eikonalLayers.IsoEikonal(p=2, hamiltonian=True)' is used.
    """
    instance_num = 0
    output_keys = ('T', 'Gr', 'Gs', 'Er', 'Es', 'Lr', 'Ls', 'Hr', 'Hs', 'Hsr')
    loss_keys = ('Er', 'Es')
    _default_lr = 5e-3

    def __init__(self, velocity, eikonal=None, name=None):
        name = name if name is not None else f"NESTP{NES_TP.instance_num}"
        NES_TP.instance_num += 1
        super().__init__(velocity, eikonal, name)
        self.dim = self.velocity.dim

    @property
    def _layout(self):
        return {'s': 0, 'r': self.dim}

    @property
    def _n_in(self):
        return 2 * self.dim

    def _net_kwargs(self):
        return dict(dim=self.dim, xscale=self.xscale, vmin=self.velocity.min, vmax=self.velocity.max)

    def _init_kwargs(self):
        return dict(velocity=self.velocity, name=self.name)

    def _singular(self, x):
        return np.abs(x[..., self.dim:] - x[..., :self.dim]).sum(axis=-1) <= self.sing_eps

    def build_model(self, nl=4, nu=50, act='ad-gauss-1', out_act='ad-sigmoid-1', factored=True, out_vscale=True,
                    input_scale=True, reciprocity=True, improved_mlp=False, **kwargs):
        """
            Builds the neural-network model.

            Arguments:
                nl, nu, act, out_act, input_scale, factored, out_vscale, improved_mlp : see `NES_OP.build_model`
                reciprocity : str or bool : how the reciprocity principle T(xs, xr) = T(xr, xs) is built in:
                              'output' (average of network outputs over the swap, original NES),
                              'first_layer' (average of first hidden layer activations, one pass of the rest),
                              'invariant' (swap-invariant input features, single pass), None or False (not imposed).
                              True means the default mode `NES.layers.DEFAULT_RECIPROCITY`.
                **kwargs : keyword arguments : 'losses' (default ['Er']; 'Er' is equation w.r.t. 'xr',
                            'Es' w.r.t. 'xs') and arguments for keras.layers.Dense(**kwargs)
                            such as 'kernel_initializer' (by default 'he_normal')
        """
        losses = kwargs.pop('losses', None)
        self._build(losses, nl=nl, nu=nu, act=act, out_act=out_act, input_scale=input_scale, factored=factored,
                    out_vscale=out_vscale, reciprocity=resolve_reciprocity(reciprocity),
                    improved_mlp=improved_mlp, **kwargs)

    def Traveltime(self, x, **kwargs):
        """
            Computes traveltimes.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **kwargs : keyword arguments : Arguments for keras.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                T : numpy array (N,) of floats : Traveltimes
        """
        return self._predict(x, 'T', **kwargs)

    def GradientR(self, x, **kwargs):
        """ Gradient of traveltimes w.r.t. 'xr', numpy array (N, dim) """
        return self._predict(x, 'Gr', **kwargs)

    def GradientS(self, x, **kwargs):
        """ Gradient of traveltimes w.r.t. 'xs', numpy array (N, dim) """
        return self._predict(x, 'Gs', **kwargs)

    def VelocityR(self, x, **kwargs):
        """ Predicted velocity at 'xr', numpy array (N,) """
        return 1 / np.linalg.norm(self.GradientR(x, **kwargs), axis=-1)

    def VelocityS(self, x, **kwargs):
        """ Predicted velocity at 'xs', numpy array (N,) """
        return 1 / np.linalg.norm(self.GradientS(x, **kwargs), axis=-1)

    def LaplacianR(self, x, **kwargs):
        """ Laplacian w.r.t. 'xr' - tau_dxdx + tau_dydy + tau_dzdz, numpy array (N,) """
        return self._predict(x, 'Lr', **kwargs)

    def LaplacianS(self, x, **kwargs):
        """ Laplacian w.r.t. 'xs' - tau_dxdx + tau_dydy + tau_dzdz, numpy array (N,) """
        return self._predict(x, 'Ls', **kwargs)

    def HessianR(self, x, **kwargs):
        """
            Full Hessian w.r.t. 'xr' in a form of:
            1D: [tau_dxdx]
            2D: [tau_dxdx, tau_dxdy, tau_dydy]
            3D: [tau_dxdx, tau_dxdy, tau_dxdz, tau_dydy, tau_dydz, tau_dzdz]
        """
        return self._predict(x, 'Hr', **kwargs)

    def HessianS(self, x, **kwargs):
        """ Full Hessian w.r.t. 'xs', same format as `HessianR` """
        return self._predict(x, 'Hs', **kwargs)

    def HessianSR(self, x, **kwargs):
        """
            Mixed Hessian w.r.t. 'xs' and 'xr', numpy array (N, dim, dim):
            2D: [[tau_dxs_dxr, tau_dxs_dyr],
                 [tau_dys_dxr, tau_dys_dyr]]
        """
        return self._predict(x, 'Hsr', **kwargs)

    def Multisource(self, Xs, Xr, **kwargs):
        """
            Computes first-arrival traveltimes from complex source 'Xs' (e.g. line).

            Arguments:
                Xs: float array (Ns, dim) : source coordinates (potentially it can be multiple sources, e.g. line)
                Xr: float array (Nr, dim) : receiver points

            Returns:
                T: float array (Nr,) : traveltimes from multisource 'Xs' to receivers 'Xr'
        """
        X = RegularGrid.sou_rec_pairs(Xs, Xr)
        T = self.Traveltime(X, **kwargs)
        if Xs[..., 0].size > 1:
            ndim = len(Xs.shape[:-1])
            T = T.min(axis=tuple(i for i in range(ndim)))
        return T

    def Reflection(self, Xs, Xd, Xr, **kwargs):
        """
            Simulates traveltimes of the wave originated in source 'xs' and reflected at 'xd'.

            Arguments:
                Xs: float array (Ns, dim) : source coordinates (potentially it can be multiple sources, e.g. line)
                Xd: float array (Nd, dim) : diffractions points for simulation of reflection
                Xr: float array (Nr, dim) : receiver points

            Returns:
                Ts: float array (Nr,) : traveltimes from (multi)source 'Xs' to receivers 'Xr'
                Td: float array (Nr,) : reflection traveltimes from 'Xd'
        """
        Xs = np.array(Xs, ndmin=2)
        Ts = self.Multisource(Xs, Xr, **kwargs)
        Tsd = self.Multisource(Xs, Xd, **kwargs)
        Td = self.Traveltime(RegularGrid.sou_rec_pairs(Xd, Xr), **kwargs)
        ndim = len(Tsd.shape)
        Tsd = np.expand_dims(Tsd, axis=tuple(i + len(Tsd.shape) for i in range(len(Xr.shape[:-1]))))
        Trefl = (Td + Tsd).min(axis=tuple(i for i in range(ndim)))
        return Ts, Trefl

    def Raylets(self, xs1, xs2, Xc, traveltimes=False, **kwargs):
        """
            Computes the norm of gradient of combined traveltime field between 'xs1' and 'xs2'.
            'xs1' and 'xs2' define a source-receiver pair. The norm of gradient is used to calculate raylets
            (Rawlinson, N., Sambridge, M., & Hauser, J. (2010).
            Multipathing, reciprocal traveltime fields and raylets.
            Geophysical Journal International, 181(2), 1077-1092.)

            Computes '|grad_c T_12| = |grad_c T_1c + grad_c T_c2|', where 'grad' stands for gradient operation,
            'c' denotes a point in a medium. Points where '|grad_c T_12| = 0' are stationary points,
            which can be used to build raylets (raypaths between 'xs1' and 'xs2').

            Arguments:
                xs1 : float array (dim,) : first source (or receiver) coordinates
                xs2 : float array (dim,) : second source (or receiver) coordinates.
                Xc : float array (N, dim) : Points 'c' where '|grad_c T_12|' is computed
                traveltimes : boolean : whether to compute combined traveltime field
                **kwargs : keyword arguments : Arguments for keras.Model.predict(**kwargs) such as 'batch_size'
            Returns:
                grad_c T_12 : gradient of combined traveltime field between 'xs1' and 'xs2'
                T_12 : combined traveltime field (if `traveltimes` is True)
        """
        Xc1 = RegularGrid.sou_rec_pairs(xs1, Xc)
        Xc2 = RegularGrid.sou_rec_pairs(Xc, xs2)
        if not traveltimes:
            dTc = self.GradientR(Xc1, **kwargs) + self.GradientS(Xc2, **kwargs)
            return np.linalg.norm(dTc, axis=-1).reshape(Xc.shape[:-1])
        T1, G1 = self.predict(Xc1, ('T', 'Gr'), **kwargs)
        T2, G2 = self.predict(Xc2, ('T', 'Gs'), **kwargs)
        return np.linalg.norm(G1 + G2, axis=-1).reshape(Xc.shape[:-1]), (T1 + T2).reshape(Xc.shape[:-1])

    def transfer(self, velocity=None, eikonal=None):
        """
            Transfers trained NES-TP to a new copy with new 'velocity', 'eikonal' (if not None).
            Returns NES_TP instance with the trained weights.
        """
        return self._transfer(velocity=velocity, eikonal=eikonal)
