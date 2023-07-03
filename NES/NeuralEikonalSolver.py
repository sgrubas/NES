import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as L
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model
from .utils import Uniform_PDF, RegularGrid, DenseBody, Diff, SourceLoc, NES_EarlyStopping, data_handler, Activation
from .eikonalLayers import IsoEikonal
import pickle, pathlib, shutil, os

###############################################################################
                    ### ONE POINT NEURAL EIKONAL SOLVER ###
###############################################################################

class NES_OP:
    """
    Neural Eikonal Solver for solving the equation in One-Point formulation tau(xr)

    Arguments:
        xs : list or array (dim,) of floats : Source location. 'dim' - dimension
        velocity : object : Velocity model class. Must be callable in a format 'v(xr) = velocity(xr)'. 
                            See example in 'misc.Interpolator'
        eikonal : instance of tf.keras.layers.Layer : Layer that mimics the eikonal equation. 
                  There must two inputs: list of spatial derivatives, velocity value. 
                  Format example - 'eikonal([tau_x, tau_y, tau_z], v(x, y, z))'. 
                  If 'None', 'eikonalLayers.IsoEikonal(P=2, hamiltonian=True)' is used.

    """
    def __init__(self, xs, velocity, eikonal=None):
        
        # Source
        if isinstance(xs, (list, np.ndarray)):
            self.xs = np.array(xs).squeeze()
            self.dim = len(xs)
        else: 
            raise ValueError("Unrecognized 'xs' type")
        
        # Velocity
        assert callable(velocity), "Must be callable in a format 'v(xr) = velocity(xr)'"
        self.velocity = velocity

        # Input scale factor
        self.xscale = np.max(np.abs([self.velocity.xmin, 
                                     self.velocity.xmax]))

        # Eikonal equation layer 
        if eikonal is None:
            self.equation = IsoEikonal(name='IsoEikonal')
        else:
            assert isinstance(eikonal, L.Layer), "Eikonal should be an instance of keras Layer"   
            self.equation = eikonal

        self.sing_eps = 1e-5    # tolerance for source singularity (to remove from training)
        self.x_train = None     # input training data
        self.y_train = None     # output training data
        self.compiled = False   # compilation status
        # config data of NN model to be reproducible
        self.config = {}
    
    def build_model(self, nl=4, nu=50, act='ad-gauss-1', out_act='ad-sigmoid-1', 
                    input_scale=True, factored=True, out_vscale=True, **kwargs):       
        """
            Build a neural-network model using Tensorflow.

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
                **kwargs : keyword arguments : Arguments for tf.keras.layers.Dense(**kwargs) such as 'kernel_initializer'.
                            If "kwargs.get('kernel_initializer')" is None then "kwargs['kernel_initializer'] = 'he_normal' "

        """
        #### The best initializer
        kwargs.setdefault('kernel_initializer', 'he_normal')

        losses = kwargs.pop('losses', None)
        
        # Saving configuration for reproducibility, to save and load model
        self.config['nl'] = nl
        self.config['nu'] = nu
        self.config['act'] = act
        self.config['out_act'] = out_act
        self.config['input_scale'] = input_scale
        self.config['factored'] = factored
        self.config['out_vscale'] = out_vscale
        self.config['losses'] = losses
        for kw, v in kwargs.items():
            self.config[kw] = v

        #### Receiver coordinate input
        xr_list = [L.Input(shape=(1,), name=f'xr{i}') for i in range(self.dim)]
        self.xr_list = xr_list

        #### Source coordinate reduction
        xr = L.Concatenate(name='xr', axis=-1)(xr_list)
        xs = SourceLoc(self.xs, name='SourceLoc')(xr)
        x = L.Subtract(name='Centering')([xr, xs])

        #### Input scaling
        if input_scale:
            x_sc = Rescaling(1 / self.xscale, name='input_scaling')(x)
        else:
            x_sc = x
        
        #### Trainable body with Traveltime Output
        T = DenseBody(x_sc, nu, nl, out_dim=1, act=act, out_act=out_act, **kwargs)

        #### Factorized solution
        if out_vscale:
            vmin, vmax = self.velocity.min, self.velocity.max
            T = L.Lambda(lambda z: (1 / vmin - 1 / vmax) * z + 1 / vmax, name='V_factor')(T)
        if factored:
            D = L.Lambda(lambda z: tf.norm(z, axis=-1, keepdims=True), name='R_factor')(x)
            T = L.Multiply(name='Traveltime')([T, D])

        #### Final Traveltime Model
        Tm = Model(inputs=xr_list, outputs=T, name=f"Model_{T.name.split('/')[0]}")

        #### Gradient
        dT_list = Diff(name='gradients')([T, xr_list])
        self._dT_list = dT_list
        dT = L.Concatenate(name='Gradient', axis=-1)(dT_list)
        Gm = Model(inputs=xr_list, outputs=dT, name=f"Model_{dT.name.split('/')[0]}")

        #### Eikonal equation
        v = L.Input(shape=(1,), name='v') # Velocity input
        Eq = self.equation(dT_list, v)
        Em = Model(inputs=xr_list + [v], outputs=Eq, name=f"Model_{Eq.name.split('/')[0]}")

        # #### Full Hessian
        # H_list = []
        # for i, dTri in enumerate(dT_list):
        #     H_list += Diff(name=f'd2T{i}')([dTri, xr_list[i:]])
        # H = L.Concatenate(axis=-1, name='Hessians')(H_list)
        # Hm = Model(inputs=xr_list, outputs=H, name=f"Model_{H.name.split('/')[0]}")

        # All callable models
        self.outs = dict(T=Tm, E=Em, G=Gm)
        # self.outs = dict(H=Hm)

        # Trainable model
        kw_models = dict(T=T, E=Eq)
        if losses is None: losses = ['E']
        inputs = xr_list + [v]
        model_outputs = [kw_models[kw] for kw in losses]
        model_name = ''
        for si in [out.name.split('/')[0] for out in model_outputs]: model_name += '_' + si 
        self.model = Model(inputs=inputs, outputs=model_outputs, name=f"Model{model_name}")

    def __call__(self, x, **kwargs):
        """
            Calls the model for training (see tf.keras.models.Model.__call__)
        """
        return self.model(x, **kwargs)

    def Traveltime(self, xr, **kwargs):
        """
            Computes traveltimes.

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                T : numpy array (N,) of floats : Traveltimes from the source 'NES_OP.xs' at 'xr'
        """
        return self._predict(xr, 'T', **kwargs)

    def Gradient(self, xr, **kwargs):
        """
            Computes gradients - vector (tau_dx, tau_dy, tau_dz).

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                dT : numpy array (N, dim) of floats : Gradient of traveltimes from the source 'NES_OP.xs' at 'xr'
        """
        return self._predict(xr, 'G', **kwargs)
        
    def Velocity(self, xr, **kwargs):
        """
            Computes predicted velocity - 1 / ||( tau_dx, tau_dy, tau_dz) ||.

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                V : numpy array (N,) of floats : Predicted velocity from the source 'NES_OP.xs' at 'xr'
        """
        G = self.Gradient(x=x, **kwargs)
        return 1 / np.linalg.norm(G, axis=-1)

    def Laplacian(self, xr, **kwargs):
        """
            Computes laplacian - tau_dxdx + tau_dydy + tau_dzdz.

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                Laplacian : numpy array (N,) of floats : Laplacian from the source 'NES_OP.xs' at 'xr'
        """
        kw = 'L'
        if self.outs.get(kw) is None: 
            L_list = []
            for i, dTi in enumerate(self._dT_list):
                L_list += Diff(name=f'L{i}')([dTi, self.xr_list[i]])
            LT = L.Add(name='Laplacian')(L_list)
            self.outs[kw] = Model(inputs=self.xr_list, outputs=LT, 
                                   name=f"Model_{LT.name.split('/')[0]}")

        return self._predict(xr, kw, **kwargs)


    def Hessian(self, xr, **kwargs):
        """
            Computes full Hessian in a form of:
            1D: [tau_dxdx]
            2D: [tau_dxdx, tau_dxdy, tau_dydy]
            3D: [tau_dxdx, tau_dxdy, tau_dxdz, tau_dydy, tau_dydz, tau_dzdz]

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                Hessian : numpy array (N, dim*((dim - 1)/2 + 1) ) of floats : Laplacian from the source 'NES_OP.xs' at 'xr'
        """
        kw = 'H'
        if self.outs.get(kw) is None: 
            H_list = []
            for i, dTri in enumerate(self._dT_list):
                H_list += Diff(name=f'd2T{i}')([dTri, self.xr_list[i:]])
            H = L.Concatenate(axis=-1, name='Hessians')(H_list)
            self.outs[kw] = Model(inputs=xr_list, outputs=H, name=f"Model_{H.name.split('/')[0]}")

        return self._predict(xr, kw, **kwargs)

    @staticmethod
    def _prepare_inputs(model, x, velocity):
        """
            Creates dictionary according to the input names of model
        """
        assert x.shape[-1] == velocity.dim, "Dimensions do not coincide"
        X = {}
        for kwi, inp in zip(model.input_names, model.inputs):
            shape = (-1,) + tuple(inp.shape.as_list()[1:])
            if kwi == 'v':
                X[kwi] = velocity(x).reshape(*shape)
            else:
                X[kwi] = x[..., int(kwi[-1])].reshape(*shape)
        return X

    def _predict(self, xr, out, **kwargs):
        kwargs.setdefault('batch_size', 100000)
        X = self._prepare_inputs(self.outs[out], xr, self.velocity)
        P = self.outs[out].predict(X, **kwargs).reshape(*xr.shape[:-1], -1).squeeze()
        return P

    def train_inputs(self, xr):
        """
            Creates dictionary of inputs for training. Removes singular points (xr=xs)

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
        """

        ids = abs(xr - self.xs[None, ...]).sum(axis=-1) > self.sing_eps # removing singular point
        self.x_train = self._prepare_inputs(self.model, xr[ids], self.velocity)

    def train_outputs(self,):
        """
            Creates dictionary of output for training (target values). All target values will be zero.
        """
        self.y_train = {}
        xi = list(self.x_train.values())[0]
        for nm, otp in zip(self.model.output_names, self.model.outputs):
            self.y_train[nm] = np.zeros((len(xi),) + tuple(otp.shape.as_list()[1:]))

    def compile(self, optimizer=None, loss='mae', lr=3e-3, decay=5e-4, **kwargs):
        """
            Compiles the neural-network model for training.

            Arguments:
                optimizer : Instance of 'tf.optimizers.Optimizer' : Optimizer of weights. 
                            If 'None', 'tf.optimizers.Adam' is used.
                loss : str (shortcuts of 'tf.keras.losses') : Loss type. By default "loss = 'mae'"
                lr : float : Learning rate, by default 'lr = 3e-3'.
                decay : float : Decay of learning rate, by default 'decay = 5e-4'.
                **kwargs : keyword arguments : Arguments for 'tf.keras.models.Model.compile(**kwargs)'
        """
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=lr, weight_decay=decay)

        self.model.compile(optimizer=optimizer, loss=loss, **kwargs)
        self.compiled = True

    def train(self, x_train, tolerance=None, **train_kw):
        """
            Traines the neural-network model.

            Arguments:
                x_train : numpy array (N, dim) of floats or int : 
                                Array of receivers. 'N' - number of receivers, 'dim' - dimension.
                                If 'int' is given, then random uniform distribution is used with 'x_train' points.
                tolerance : It can be:
                    1) float - Tolerance value for early stopping in RMAE units for traveltimes.
                               NES_EarlyStopping callback will be created with default options 
                               (see `baseLayers.NES_EarlyStopping`)
                    2) instance of NES_EarlyStopping callback. 
                              
                **train_kw : keyword arguments : Arguments for 'tf.keras.models.Model.fit(**train_kw)' such as 'batch_size', 'epochs'
        """
        if isinstance(x_train, int):
            x_train = Uniform_PDF(self.velocity)(x_train)

        self.train_inputs(x_train)
        self.train_outputs()
        data = data_handler(self.x_train, self.y_train, **train_kw)
        if isinstance(data, tf.keras.utils.Sequence):
            self.data_generator = data
            data = (data,)

        if not self.compiled:
            self.compile()

        train_kw.setdefault('callbacks', [])
        if isinstance(tolerance, float):
            train_kw['callbacks'].append(NES_EarlyStopping(tolerance=tolerance))
        elif isinstance(tolerance, tf.keras.callbacks.Callback):
            train_kw['callbacks'].append(tolerance)
        else:
            pass

        h = self.model.fit(*data, **train_kw)
        return h

    @staticmethod
    def _pack_opt_config(model):
        if model.optimizer is None: return None
        opt_config = {}
        opt_config['optimizer.config'] = model.optimizer.get_config()
        opt_config['optimizer.weights'] = model.optimizer.get_weights()
        opt_config['loss'] = model.loss
        return opt_config

    def _unpack_opt_config(self, opt_config):
        if opt_config is None: return None
        optimizer = tf.keras.optimizers.get(opt_config['optimizer.config']['name'])
        optimizer = optimizer.from_config(opt_config['optimizer.config'])
        optimizer._create_all_weights(self.model.trainable_variables)
        optimizer.set_weights(opt_config['optimizer.weights'])
        self.compile(optimizer=optimizer, loss=opt_config['loss'])

    def save(self, filepath, save_optimizer=False, training_data=False):
        """
            Saves the current NES_OP model to `filepath` directory.
            `save_optimizer` saves the optimizer state to continue the training from the last point,
            `training_data` saves the last training set used for training.

            NES_OP includes: the neural-network weights ('filepath_weights'), 
                             the velocity model and configures ('filepath_config')
        """

        config = self.config
        config['velocity'] = self.velocity
        config['xs'] = self.xs
        config['equation.config'] = self.equation.get_config()

        # makedir
        if pathlib.Path(filepath).is_dir():
            shutil.rmtree(filepath)
        pathlib.Path(filepath).mkdir(parents=True, exist_ok=False)

        folder = os.path.split(filepath)[-1]
        filename = lambda kw: os.path.join(filepath, f'{folder}_{kw}')

        # Model configuration
        config_filename = filename('config.pkl')
        with open(config_filename, 'wb') as f: 
            pickle.dump(config, f)

        # Model weights
        weights_filename = filename('weights.h5')
        self.outs['T'].save_weights(weights_filename)

        # Optimizer state
        if save_optimizer:
            opt_filename = filename('optimizer.pkl')
            with open(opt_filename, 'wb') as f: 
                pickle.dump(NES_OP._pack_opt_config(self.model), f)

        # Training set of collocation points
        if training_data:
            data_filename = filename('train_data.pkl')
            with open(data_filename, 'wb') as f: 
                pickle.dump(self.x_train, f)

    @staticmethod
    def load(filepath):
        """
            Creates an NES_OP instance according to the configuration 
            and pretrained_weights in `filepath`
            
            Arguments:
                filepath : directory with the saved model

            Returns:
                NES_OP instance 
        """
        folder = os.path.split(filepath)[-1]
        filename = lambda kw: os.path.join(filepath, f'{folder}_{kw}')

        # Importing configuration data
        config_filename = filename('config.pkl')
        with open(config_filename, 'rb') as f: 
            config = pickle.load(f)

        # Loading configuration
        # TODO anisotropy
        eikonal = IsoEikonal.from_config(config.pop('equation.config'))
        NES_OP_instance = NES_OP(xs=config.pop('xs'), 
                                 velocity=config.pop('velocity'), 
                                 eikonal=eikonal)
        NES_OP_instance.build_model(**config)

        # Loading weights
        weights_filename = filename('weights.h5')
        NES_OP_instance.outs['T'].load_weights(weights_filename, by_name=False)
        print(f'Loaded model from "{filepath}"')

        # Loading optimizer state if available
        opt_filename = filename('optimizer.pkl')
        if pathlib.Path(opt_filename).is_file():
            with open(opt_filename, 'rb') as f: 
                opt_config = pickle.load(f)
            NES_OP_instance._unpack_opt_config(opt_config)
            print('Compiled the model with saved optimizer')

        # Loading training data if available
        data_filename = filename('train_data.pkl')
        if pathlib.Path(data_filename).is_file():
            with open(data_filename, 'rb') as f: 
                NES_OP_instance.x_train = pickle.load(f)
            print('Loaded last training data: see NES_OP.x_train')

        return NES_OP_instance

    def transfer(self, xs=None, velocity=None, eikonal=None):
        """
            Transfers trained NES-OP to a new copy

            Arguments:
                xs: new source location for transfer. If None, previous is used
                velocity: new velocity for transfer. If None, previous is used
                eikonal: new eikonal equation for transfer. If None, previous is used

            Returns:
                NES_OP instance: new NES_OP with transfered achitecture (trained weights)
        """
        if xs is None:
            xs = self.xs
        if velocity is None:
            velocity = self.velocity
        if eikonal is None:
            eikonal = self.equation

        NES_OP_instance = NES_OP(xs=xs, velocity=velocity, eikonal=eikonal)
        NES_OP_instance.build_model(**self.config)
        NES_OP_instance.outs['T'].set_weights(self.outs['T'].get_weights())
        NES_OP_instance._unpack_opt_config(NES_OP._pack_opt_config(self.model))
        return NES_OP_instance


##############################################################################
                    ### TWO POINT NEURAL EIKONAL SOLVER ###
##############################################################################

class NES_TP:
    """
    Neural Eikonal Solver for solving the equation in Two-Point formulation T(xs, xr)

    Arguments:
        velocity : object : Velocity model class. Must be callable in a format 'v(xr) = velocity(xr)'. 
                            See example in 'misc.Interpolator'
        eikonal : instance of tf.keras.layers.Layer : Layer that mimics the eikonal equation. 
                  There must be two inputs: list of spatial derivatives, velocity value. 
                  Format example - 'eikonal([tau_x, tau_y, tau_z], v(x, y, z))'. 
                  If 'None', 'eikonalLayers.IsoEikonal(P=2, hamiltonian=True)' is used.
    """

    def __init__(self, velocity, eikonal=None):
        
        # Velocity
        assert callable(velocity), "Must be callable in a format 'v(xr) = velocity(xr)'"
        self.velocity = velocity
        self.dim = self.velocity.dim

        # Input scale factor
        self.xscale = np.max(np.abs([self.velocity.xmin, 
                                     self.velocity.xmax]))

        # Eikonal equation layer 
        if eikonal is None:
            self.equation = IsoEikonal(name='IsoEikonal')
        else:
            assert isinstance(eikonal, L.Layer), "Eikonal should be an instance of keras Layer"   
            self.equation = eikonal

        self.sing_eps = 1e-5    # tolerance for source singularity (to remove from training)
        self.x_train = None     # input training data
        self.y_train = None     # output training data
        self.compiled = False   # compilation status
        self.config = {}        # config data of NN model to be reproducible
        
    def build_model(self, nl=4, nu=50, act='ad-gauss-1', out_act='ad-sigmoid-1', 
                    factored=True, out_vscale=True, input_scale=True, reciprocity=True, **kwargs):
        """
            Build a neural-network model using Tensorflow.

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
                reciprocity : boolean : Enhanced factorization for NES_TP incorporating the reciprocity principle tau(xs, xr) = tau(xr, xs).
                                        By default 'True'
                **kwargs : keyword arguments : Arguments for tf.keras.layers.Dense(**kwargs) such as 'kernel_initializer'.
                            If "kwargs.get('kernel_initializer')" is None then "kwargs['kernel_initializer'] = 'he_normal' "

        """
        #### The best initializer
        kwargs.setdefault('kernel_initializer', 'he_normal')

        # outputs for training losses. Since this is Two-Point formulation, 
        # we can solve two eikonal equations according to the reciprocity principle.
        # 'Er' is equation w.r.t. 'xr', 'Es' is equation w.r.t. 'xr'.
        losses = kwargs.pop('losses', None)

        # Saving configuration for reproducibility, to save and load model
        self.config['nl'] = nl
        self.config['nu'] = nu
        self.config['act'] = act
        self.config['out_act'] = out_act
        self.config['input_scale'] = input_scale
        self.config['factored'] = factored
        self.config['out_vscale'] = out_vscale
        self.config['reciprocity'] = reciprocity
        self.config['losses'] = losses
        for kw, v in kwargs.items():
            self.config[kw] = v

        #### Input for Source part
        xs_list = [L.Input(shape=(1,), name='xs' + str(i)) for i in range(self.dim)]
        self.xs_list = xs_list
        xs = L.Concatenate(name='xs', axis=-1)(xs_list)

        #### Input for Receiver part
        xr_list = [L.Input(shape=(1,), name='xr' + str(i)) for i in range(self.dim)]
        self.xr_list = xr_list
        xr = L.Concatenate(name='xr', axis=-1)(xr_list)
        
        #### Input list
        inputs = xs_list + xr_list

        #### Input scaling
        X = L.Concatenate(name='x', axis=-1)([xs, xr])
        if input_scale:
            X_sc = Rescaling(1 / self.xscale, name='X_scaling')(X)
        else:
            X_sc = X

        #### Trainable body
        T = DenseBody(X_sc, nu, nl, out_dim=1, act=act, out_act='linear', **kwargs)

        #### Reciprocity 
        if reciprocity: # T(xs,xr)=T(xr,xs)
            t = Model(inputs=inputs, outputs=T, name='Model_No_Reciprocity')
            xsr = xs_list + xr_list; xrs = xr_list + xs_list;
            tsr = t(xsr); trs = t(xrs)
            T = L.Lambda(lambda x: 0.5*(x[0] + x[1]), name='Reciprocity')([tsr, trs])

        #### Output activation
        T = L.Activation(Activation(out_act))(T)
        # Scaling to the range of [1/vmax , 1/vmin]. T is assumed to be in [0, 1]
        if out_vscale:
            vmin, vmax = self.velocity.min, self.velocity.max
            T = L.Lambda(lambda z: (1 / vmin - 1 / vmax) * z + 1 / vmax, name='V_factor')(T)

        #### Factorization
        if factored:
            xr_xs = L.Subtract(name='xr_xs_difference')([xr, xs])
            D = L.Lambda(lambda z: tf.norm(z, axis=-1, keepdims=True), name='R_factor')(xr_xs)    
            T = L.Multiply(name='Traveltime')([T, D])

        #### Final Traveltime Model
        Tm = Model(inputs=inputs, outputs=T, name=f"Model_{T.name.split('/')[0]}")

        #### Gradient 'xr'
        dTr_list = Diff(name='gradients_xr')([T, xr_list])
        self._dTr_list = dTr_list
        dTr = L.Concatenate(axis=-1, name='Gradient_xr')(dTr_list)
        Gr = Model(inputs=inputs, outputs=dTr, name=f"Model_{dTr.name.split('/')[0]}")

        #### Eikonal 'xr'
        vr = L.Input(shape=(1,), name='vr') # velocity input for 'xr'
        Er = self.equation(dTr_list, vr)
        Emr = Model(inputs=inputs + [vr], outputs=Er, name=f"Model_{Er.name.split('/')[0]}")

        #### Gradient 'xs'
        dTs_list = Diff(name='gradients_xs')([T, xs_list])
        self._dTs_list = dTs_list
        dTs = L.Concatenate(axis=-1, name='Gradient_xs')(dTs_list)
        Gs = Model(inputs=inputs, outputs=dTs, name=f"Model_{dTs.name.split('/')[0]}")

        #### Eikonal 'xs'
        vs = L.Input(shape=(1,), name='vs') # velocity input for 'xs'
        Es = self.equation(dTs_list, vs)
        Ems = Model(inputs=inputs + [vs], outputs=Es, name=f"Model_{Es.name.split('/')[0]}")

        #### Trainable model ####
        kw_models = dict(T=T, Er=Er, Es=Es)
        if losses is None: losses = ['Er']
        model_outputs = [kw_models[kw] for kw in losses]
        model_inputs = inputs + [vr] * ('Er' in losses) + [vs] * ('Es' in losses)
        model_name = ''
        for si in [out.name.split('/')[0] for out in model_outputs]: model_name += '_' + si 
        self.model = Model(inputs=model_inputs, outputs=model_outputs, name=f"Model{model_name}")

        #### All callable models ####
        self.outs = dict(T=Tm, Er=Emr, Es=Ems, Gr=Gr, Gs=Gs)

    def __call__(self, x, **kwargs):
        """
            Calls the model for training (see tf.keras.models.Model.__call__)
        """
        return self.model(x, **kwargs)

    def Traveltime(self, x, **kwargs):
        """
            Computes traveltimes.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                T : numpy array (N,) of floats : Traveltimes
        """
        return self._predict(x, 'T', **kwargs)

    def GradientR(self, x, **kwargs):
        """
            Computes gradient of traveltimes w.r.t. 'xr'.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                dTr : numpy array (N, dim) of floats : gradient of traveltimes w.r.t. 'xr'
        """
        return self._predict(x, 'Gr', **kwargs)
    
    def GradientS(self, x, **kwargs):
        """
            Computes gradient of traveltimes w.r.t. 'xs'.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                dTs : numpy array (N, dim) of floats : gradient of traveltimes w.r.t. 'xs'
        """
        return self._predict(x, 'Gs', **kwargs)

    def VelocityR(self, x, **kwargs):
        """
            Predicted velocity at 'xr'.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                Vr : numpy array (N,) of floats : Predicted velocity at 'xr'.
        """
        G = self.GradientR(x=x, **kwargs)
        return 1 / np.linalg.norm(G, axis=-1)
        
    def VelocityS(self, x, **kwargs):
        """
            Predicted velocity at 'xs'.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                Vs : numpy array (N,) of floats : Predicted velocity at 'xs'.
        """
        G = self.GradientS(x=x, **kwargs)
        return 1 / np.linalg.norm(G, axis=-1)

    def LaplacianR(self, x, **kwargs):
        """
            Computes laplacian w.r.t. 'xr' - tau_dxdx + tau_dydy + tau_dzdz.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                Laplacian : numpy array (N,) of floats
        """
        kw = 'Lr'
        if self.outs.get(kw) is None: # create model for the first call
            Lr_list = []
            for i, dTi in enumerate(self._dTr_list):
                Lr_list += Diff(name=f'Lr{i}')([dTi, self.xr_list[i]])
            Lr = L.Add(name='Laplacian_xr')(Lr_list)
            inputs = self.xs_list + self.xr_list
            self.outs[kw] = Model(inputs=inputs, outputs=Lr, name=f"Model_{Lr.name.split('/')[0]}")

        return self._predict(x, kw, **kwargs)

    def LaplacianS(self, x, **kwargs):
        """
            Computes laplacian w.r.t. 'xs' - tau_dxdx + tau_dydy + tau_dzdz.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                Laplacian : numpy array (N,) of floats
        """
        kw = 'Ls'
        if self.outs.get(kw) is None: # create model for the first call
            Ls_list = []
            for i, dTi in enumerate(self._dTs_list):
                Ls_list += Diff(name=f'Ls{i}')([dTi, self.xs_list[i]])
            Ls = L.Add(name='Laplacian_xs')(Ls_list)
            inputs = self.xs_list + self.xr_list
            self.outs[kw] = Model(inputs=inputs, outputs=Ls, name=f"Model_{Ls.name.split('/')[0]}")

        return self._predict(x, kw, **kwargs)

    def HessianR(self, x, **kwargs):
        """
            Computes full Hessian w.r.t. 'xr' in a form of:
            1D: [tau_dxdx]
            2D: [tau_dxdx, tau_dxdy, tau_dydy]
            3D: [tau_dxdx, tau_dxdy, tau_dxdz, tau_dydy, tau_dydz, tau_dzdz]

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                Hessian : numpy array (N, dim*((dim - 1)/2 + 1) ) of floats
        """
        kw = 'Hr'
        if self.outs.get(kw) is None: # create model for the first call
            Hr_list = []
            for i, dTri in enumerate(self._dTr_list):
                Hr_list += Diff(name=f'Hr{i}')([dTri, self.xr_list[i:]])
            Hr = L.Concatenate(axis=-1, name='Hessian_xr')(Hr_list)
            inputs = self.xs_list + self.xr_list
            self.outs[kw] = Model(inputs=inputs, outputs=Hr, name=f"Model_{Hr.name.split('/')[0]}")

        return self._predict(x, kw, **kwargs)

    def HessianS(self, x, **kwargs):
        """
            Computes full Hessian w.r.t. 'xs' in a form of:
            1D: [tau_dxdx]
            2D: [tau_dxdx, tau_dxdy, tau_dydy]
            3D: [tau_dxdx, tau_dxdy, tau_dxdz, tau_dydy, tau_dydz, tau_dzdz]

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                Hessian : numpy array (N, dim*((dim - 1)/2 + 1) ) of floats
        """
        kw = 'Hs'
        if self.outs.get(kw) is None: # create model for the first call
            Hs_list = []
            for i, dTsi in enumerate(self._dTs_list):
                Hs_list += Diff(name=f'Hs{i}')([dTsi, self.xs_list[i:]])
            Hs = L.Concatenate(axis=-1, name='Hessian_xs')(Hs_list)
            inputs = self.xs_list + self.xr_list
            self.outs[kw] = Model(inputs=inputs, outputs=Hs, name=f"Model_{Hs.name.split('/')[0]}")

        return self._predict(x, kw, **kwargs)

    def HessianSR(self, x, **kwargs):
        """
            Computes full mixed Hessian w.r.t. 'xs' and 'xr' in a form of:
            1D: [[tau_dxs_dxr]]
            2D: [[tau_dxs_dxr, tau_dxs_dyr], 
                 [tau_dys_dxr, tau_dys_dyr]]
            3D: [[tau_dxs_dxr, tau_dxs_dyr, tau_dxs_dzr], 
                 [tau_dys_dxr, tau_dys_dyr, tau_dys_dzr],
                 [tau_dzs_dxr, tau_dzs_dyr, tau_dzs_dzr]]

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                HessianSR : numpy array (N, dim, dim) of floats
        """
        kw = 'Hsr'
        if self.outs.get(kw) is None: # create model for the first call
            Hsr_list = []
            for i, dTsi in enumerate(self._dTs_list):
                Hsr_list += Diff(name=f'Hsr{i}')([dTsi, self.xr_list])
            Hsr = L.Concatenate(axis=-1, name='hessian_xsxr')(Hsr_list)
            Hsr = L.Reshape((self.dim, self.dim), name='Hessian_xsxr')(Hsr)
            inputs = self.xs_list + self.xr_list
            self.outs[kw] = Model(inputs=inputs, outputs=Hsr, name=f"Model_{Hsr.name.split('/')[0]}")

        return self._predict(x, kw, **kwargs)

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
        X = RegularGrid.sou_rec_pairs(Xd, Xr)
        Td = self.Traveltime(X, **kwargs)
        ndim = len(Tsd.shape)
        Tsd = np.expand_dims(Tsd, axis=tuple(i+len(Tsd.shape) for i in range(len(Xr.shape[:-1]))))
        Td += Tsd
        Trefl = Td.min(axis=tuple(i for i in range(ndim)))
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
                        Actually, 'xs1' and 'xs2' define a source-receiver pair.
                Xc : float array (N, dim) : Points 'c' where '|grad_c T_12|' is computed
                traveltimes : boolean : whether to compute combined traveltime field
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'
            Returns:
                grad_c T_12 : gradient of combined traveltime field between 'xs1' and 'xs2'
                T_12 : combined traveltime field (if `traveltimes` is True)
        """
        Xc1 = RegularGrid.sou_rec_pairs(xs1, Xc)
        dTc = self.GradientR(Xc1, **kwargs)
        Xc2 = RegularGrid.sou_rec_pairs(Xc, xs2)
        dTc += self.GradientS(Xc2, **kwargs)
        abs_dTc = np.linalg.norm(dTc, axis=-1).reshape(Xc.shape[:-1])

        if traveltimes:
            Tc = self.Traveltime(Xc1, **kwargs)
            Tc += self.Traveltime(Xc2, **kwargs)
            return abs_dTc, Tc.reshape(Xc.shape[:-1])
        else:
            return abs_dTc

    @staticmethod
    def _prepare_inputs(model, x, velocity):
        dim = velocity.dim
        assert x.shape[-1] == 2*dim, "Dimensions do not coincide"
        xs = x[..., :dim]
        xr = x[..., dim:]
        vx = {'vs': xs, 'vr': xr}
        X = {}
        for kwi, inp in zip(model.input_names, model.inputs):
            shape = (-1,) + tuple(inp.shape.as_list()[1:])
            if 'v' in kwi:
                X[kwi] = velocity(vx[kwi]).reshape(shape)
            elif 'xs' in kwi:
                i = int(kwi[-1])
                X[kwi] = xs[..., i].reshape(shape)
            elif 'xr' in kwi:
                i = int(kwi[-1])
                X[kwi] = xr[..., i].reshape(shape)
        return X

    def _predict(self, x, out, **kwargs):
        kwargs.setdefault('batch_size', 100000)
        X = self._prepare_inputs(self.outs[out], x, self.velocity)
        P = self.outs[out].predict(X, **kwargs)
        shape = x.shape[:-1] + P.shape[1:]
        P = P.reshape(shape).squeeze()
        return P

    def train_inputs(self, x):
        """
            Creates dictionary of inputs for training.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
        """
        xs = x[..., :self.dim]
        xr = x[..., self.dim:]

        ids = abs(xr - xs).sum(axis=-1) > self.sing_eps # removing singular points
        self.x_train = self._prepare_inputs(self.model, x[ids], self.velocity)

    def train_outputs(self,):
        """
            Creates dictionary of output for training (target values). All target values will be zero.
        """
        self.y_train = {}
        xi = list(self.x_train.values())[0]
        for nm, otp in zip(self.model.output_names, self.model.outputs):
            self.y_train[nm] = np.zeros((len(xi),) + tuple(otp.shape.as_list()[1:]))

    def compile(self, optimizer=None, loss='mae', lr=5e-3, decay=5e-4, **kwargs):
        """
            Compiles the neural-network model for training.

            Arguments:
                optimizer : Instance of 'tf.optimizers.Optimizer' : Optimizer of weights. 
                            If 'None', 'tf.optimizers.Adam' is used.
                loss : str (shortcuts for 'tf.keras.losses') : Loss type. By default "loss = 'mae'"
                lr : float : Learning rate, by default 'lr = 3e-3'.
                decay : float : Decay of learning rate, by default 'decay = 5e-4'.
                **kwargs : keyword arguments : Arguments for 'tf.keras.models.Model.compile(**kwargs)'
        """
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=lr, weight_decay=decay)

        self.model.compile(optimizer=optimizer, loss=loss, **kwargs)
        self.compiled = True

    def train(self, x_train, tolerance=None, **train_kw):
        """
            Traines the neural-network model.

            Arguments:
                x_train : numpy array (N, dim*2) of floats or int : 
                                Array of receivers. 'N' - number of source-receivers pairs, 'dim' - dimension.
                                If 'int' is given, then random uniform distribution is used with 'x_train' points.
                tolerance : It can be:
                    1) float - Tolerance value for early stopping in RMAE units for traveltimes.
                               NES_EarlyStopping callback will be created with default options 
                               (see `baseLayers.NES_EarlyStopping`)
                    2) instance of NES_EarlyStopping callback. 
                              
                **train_kw : keyword arguments : Arguments for 'tf.keras.models.Model.fit(**train_kw)' such as 'batch_size', 'epochs'
        """
        if isinstance(x_train, int):
            pdf = Uniform_PDF(self.velocity)
            x_train = pdf(x_train, rep=2)

        self.train_inputs(x_train)
        self.train_outputs()
        data = data_handler(self.x_train, self.y_train, **train_kw)
        if isinstance(data, tf.keras.utils.Sequence):
            self.data_generator = data
            data = (data,)

        if not self.compiled:
            self.compile()

        train_kw.setdefault('callbacks', [])
        if isinstance(tolerance, float):
            train_kw['callbacks'].append(NES_EarlyStopping(tolerance=tolerance))
        elif isinstance(tolerance, tf.keras.callbacks.Callback):
            train_kw['callbacks'].append(tolerance)
        else:
            pass

        h = self.model.fit(*data, **train_kw)
        return h

    @staticmethod
    def _pack_opt_config(model):
        if model.optimizer is None: return None
        opt_config = {}
        opt_config['optimizer.config'] = model.optimizer.get_config()
        opt_config['optimizer.weights'] = model.optimizer.get_weights()
        opt_config['loss'] = model.loss
        return opt_config

    def _unpack_opt_config(self, opt_config):
        if opt_config is None: return None
        optimizer = tf.keras.optimizers.get(opt_config['optimizer.config']['name'])
        optimizer = optimizer.from_config(opt_config['optimizer.config'])
        optimizer._create_all_weights(self.model.trainable_variables)
        optimizer.set_weights(opt_config['optimizer.weights'])
        self.compile(optimizer=optimizer, loss=opt_config['loss'])

    def save(self, filepath, save_optimizer=False, training_data=False):
        """
            Saves the current NES_OP model to `filepath` directory.
            `save_optimizer` saves the optimizer state to continue the training from the last point,
            `training_data` saves the last training set used for training.

            NES_TP includes: the neural-network weights ('filepath_weights'), 
                             the velocity model and configures ('filepath_config')
        """
        config = self.config
        config['velocity'] = self.velocity
        config['equation.config'] = self.equation.get_config()

        # makedir
        if pathlib.Path(filepath).is_dir():
            shutil.rmtree(filepath)
        pathlib.Path(filepath).mkdir(parents=True, exist_ok=False)

        folder = os.path.split(filepath)[-1]
        filename = lambda kw: os.path.join(filepath, f'{folder}_{kw}')

        # Model configuration
        config_filename = filename('config.pkl')
        with open(config_filename, 'wb') as f:
            pickle.dump(config, f)

        # Model weights
        weights_filename = filename('weights.pkl')
        weights = self.outs['T'].get_weights()
        with open(weights_filename, 'wb') as f:
            pickle.dump(weights, f)

        # Optimizer state
        if save_optimizer:
            opt_filename = filename('optimizer.pkl')
            with open(opt_filename, 'wb') as f: 
                pickle.dump(NES_TP._pack_opt_config(self.model), f)

        # Training set of collocation points
        if training_data:
            data_filename = filename('train_data.pkl')
            with open(data_filename, 'wb') as f: 
                pickle.dump(self.x_train, f)

    @staticmethod
    def load(filepath):
        """
            Creates an NES_TP instance according to the configuration 
            and pretrained_weights in `filepath`
            
            Arguments:
                filepath : directory with the saved model

            Returns:
                NES_TP instance 
        """
        folder = os.path.split(filepath)[-1]
        filename = lambda kw: os.path.join(filepath, f'{folder}_{kw}')

        # Importing configuration data
        config_filename = filename('config.pkl')
        with open(config_filename, 'rb') as f: 
            config = pickle.load(f)

        # Loading configuration
        # TODO anisotropy
        eikonal = IsoEikonal.from_config(config.pop('equation.config'))
        NES_TP_instance = NES_TP(velocity=config.pop('velocity'), 
                                 eikonal=eikonal)
        NES_TP_instance.losses = config.pop('losses')
        NES_TP_instance.build_model(**config)
        
        # Loading weights
        weights_filename = filename('weights.pkl')
        with open(weights_filename, 'rb') as f: 
            weights = pickle.load(f)
        NES_TP_instance.outs['T'].set_weights(weights)
        print(f'Loaded model from "{filepath}"')

        # Loading optimizer state if available
        opt_filename = filename('optimizer.pkl')
        if pathlib.Path(opt_filename).is_file():
            with open(opt_filename, 'rb') as f: 
                opt_config = pickle.load(f)
            NES_TP_instance._unpack_opt_config(opt_config)
            print('Compiled the model with saved optimizer')

        # Loading training data if available
        data_filename = filename('train_data.pkl')
        if pathlib.Path(data_filename).is_file():
            with open(data_filename, 'rb') as f: 
                NES_TP_instance.x_train = pickle.load(f)
            print('Loaded last training data: see NES_TP.x_train')

        return NES_TP_instance

    def transfer(self, velocity=None, eikonal=None):
        """
            Transfers trained NES-TP to a new copy

            Arguments:
                velocity: new velocity for transfer. If None, previous is used
                eikonal: new eikonal equation for transfer. If None, previous is used

            Returns:
                NES_TP instance: new NES_TP with transfered achitecture (trained weights)
        """
        if velocity is None:
            velocity = self.velocity
        if eikonal is None:
            eikonal = self.equation

        NES_TP_instance = NES_TP(velocity=velocity, eikonal=eikonal)
        NES_TP_instance.build_model(**self.config)
        NES_TP_instance.outs['T'].set_weights(self.outs['T'].get_weights())
        NES_TP_instance._unpack_opt_config(NES_TP._pack_opt_config(self.model))
        return NES_TP_instance
