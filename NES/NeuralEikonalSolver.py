import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as L
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model
from .utils import DenseBody, Diff, SourceLoc, NES_EarlyStopping, data_handler
from .eikonalLayers import IsoEikonal
from .misc import Interpolator
import pickle, pathlib, shutil

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
                  If 'None', 'eikonalLayers.IsoEikonal(P=3, hamiltonian=True)' is used.

    """
    def __init__(self, xs, velocity, eikonal=None):
        
        # Source
        if isinstance(xs, (list, np.ndarray)):
            self.xs = np.array(xs).squeeze()
            self.dim = len(xs)
        else: 
            assert False, "Unrecognized 'xs' type"
        
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

        self.x_train = None     # input training data
        self.y_train = None     # output training data
        self.compiled = False   # compilation status
        self.config = {}        # config data of NN model to be reproducible
    
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
        if kwargs.get('kernel_initializer') is None:
            kwargs['kernel_initializer'] = 'he_normal'

        # Saving configuration for reproducibility, to save and load model
        self.config['nl'] = nl
        self.config['nu'] = nu
        self.config['act'] = act
        self.config['out_act'] = out_act
        self.config['input_scale'] = input_scale
        self.config['factored'] = factored
        self.config['out_vscale'] = out_vscale
        for kw, v in kwargs.items():
            self.config[kw] = v

        #### Receiver coordinate input
        xr_list = [L.Input(shape=(1,), name=f'xr{i}') for i in range(self.dim)]

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
            D = L.Lambda(lambda z: tf.norm(z, axis=-1, keepdims=True), name='D_factor')(x)
            T = L.Multiply(name='Traveltime')([T, D])

        #### Final Traveltime Model
        Tm = Model(inputs=xr_list, outputs=T)

        #### Gradient
        dT_list = Diff(name='gradients')([T, xr_list])
        dT = L.Concatenate(name='Gradient', axis=-1)(dT_list)
        Gm = Model(inputs=xr_list, outputs=dT)

        #### Eikonal equation
        v = L.Input(shape=(1,), name='v') # Velocity input
        Eq = self.equation(dT_list, v)
        Em = Model(inputs=xr_list + [v], outputs=Eq)

        #### Velocity
        V = L.Lambda(lambda z: 1 / tf.norm(z, axis=-1, keepdims=True), name='Velocity')(dT)
        Vm = Model(inputs=xr_list, outputs=V)

        #### Laplacian
        L_list = []
        for i, dTi in enumerate(dT_list):
            L_list += Diff(name=f'L{i}')([dTi, xr_list[i]])
        LT = L.Add(name='Laplacian')(L_list)
        Lm = Model(inputs=xr_list, outputs=LT)

        #### Full Hessian
        H_list = []
        for i, dTri in enumerate(dT_list):
            H_list += Diff(name=f'd2T{i}')([dTri, xr_list[i:]])
        H = L.Concatenate(axis=-1, name='Hessians')(H_list)
        Hm = Model(inputs=xr_list, outputs=H)

        # All callable models
        self.outs = dict(T=Tm, E=Em, G=Gm, V=Vm, L=Lm, H=Hm)

        # Trainable model
        inputs = xr_list + [v]
        outputs = Eq
        self.model = Model(inputs=inputs, outputs=outputs)

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
        return self._predict(xr, 'V', **kwargs)

    def Laplacian(self, xr, **kwargs):
        """
            Computes laplacian - tau_dxdx + tau_dydy + tau_dzdz.

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                Laplacian : numpy array (N,) of floats : Laplacian from the source 'NES_OP.xs' at 'xr'
        """        
        return self._predict(xr, 'L', **kwargs)


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
        return self._predict(xr, 'H', **kwargs)

    @staticmethod
    def _prepare_inputs(model, x, velocity):
        """
            Creates dictionary according to the input names of model
        """
        X = {}
        dim = velocity.dim
        for kwi in model.input_names:
            if kwi == 'v':
                X[kwi] = velocity(x).ravel()
            else:
                X[kwi] = x[..., int(kwi[-1])].ravel()
        return X

    def _predict(self, xr, out, **kwargs):
        if kwargs.get('batch_size') is None:
            kwargs['batch_size'] = 100000
        X = self._prepare_inputs(self.outs[out], xr, self.velocity)
        P = self.outs[out].predict(X, **kwargs).reshape(*xr.shape[:-1], -1).squeeze()
        return P

    def train_inputs(self, xr):
        """
            Creates dictionary of inputs for training. Removes singular points (xr=xs)

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
        """

        ids = abs(xr - self.xs[None, ...]).sum(axis=-1) != 0 # removing singular point
        self.x_train = self._prepare_inputs(self.model, xr[ids], self.velocity)

    def train_outputs(self,):
        """
            Creates dictionary of output for training (target values). All target values will be zero.
        """
        self.y_train = {}
        xi = list(self.x_train.values())[0]
        for l in self.model.output_names:
            self.y_train[l] = np.zeros(len(xi))

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
            optimizer = tf.optimizers.Adam(learning_rate=lr, decay=decay)

        self.model.compile(optimizer=optimizer, loss=loss, **kwargs)
        self.compiled = True

    def train(self, x_train, tolerance=None, **train_kw):
        """
            Traines the neural-network model.

            Arguments:
                x_train : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension.
                                If 'None', previous `NES_OP.x_train` may be used if available
                tolerance : It can be:
                    1) float - Tolerance value for early stopping in RMAE units for traveltimes.
                               NES_EarlyStopping callback will be created with default options 
                               (see `baseLayers.NES_EarlyStopping`)
                    2) instance of NES_EarlyStopping callback. 
                              
                **train_kw : keyword arguments : Arguments for 'tf.keras.models.Model.fit(**train_kw)' such as 'batch_size', 'epochs'
        """
        self.train_inputs(x_train)
        self.train_outputs()
        data = data_handler(self.x_train, self.y_train, **train_kw)
        if isinstance(data, tf.keras.utils.Sequence):
            self.data_generator = data
            data = (data,)

        if not self.compiled:
            self.compile()

        callbacks = []
        if isinstance(tolerance, (float, tf.keras.callbacks.Callback)):
            if isinstance(tolerance, float):
                EarlyStopping = NES_EarlyStopping(tolerance=tolerance)
            else:
                EarlyStopping = tolerance
            callbacks.append(EarlyStopping)

        if train_kw.get('callbacks') is None:
            train_kw['callbacks'] = callbacks
        else:
            train_kw['callbacks'] += callbacks

        h = self.model.fit(*data, **train_kw)
        return h

    def save(self, filepath, save_optimizer=False, training_data=False):
        """
            Saves the current NES_OP model to `filepath` directory.
            `save_optimizer` saves the optimizer state to continue the training from the last point,
            `training_data` saves the last training set used for training.
        """
        config = self.config
        config['velocity'] = self.velocity
        config['xs'] = self.xs
        config['equation.config'] = self.equation.get_config()

        # makedir
        if pathlib.Path(filepath).is_dir():
            shutil.rmtree(filepath)
        pathlib.Path(filepath).mkdir(parents=True, exist_ok=False)
        filename = filepath.split('/')[-1].split('.')[0]

        # Model configuration
        config_filename = filepath + f'/{filename}_config'
        with open(config_filename, 'wb') as f: 
            pickle.dump(config, f)

        # Model weights
        weights_filename = filepath + f'/{filename}_weights.h5'
        self.outs['T'].save_weights(weights_filename)

        # Optimizer state
        if save_optimizer:
            opt_config = {}
            opt_config['optimizer.config'] = self.model.optimizer.get_config()
            opt_config['optimizer.weights'] = self.model.optimizer.get_weights()
            opt_config['loss'] = self.model.loss
            opt_filename = filepath + f'/{filename}_optimizer'
            with open(opt_filename, 'wb') as f: 
                pickle.dump(opt_config, f)

        # Training set of collocation points
        if training_data:
            data_filename = filepath + f'/{filename}_train_data'
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
        filename = filepath.split('/')[-1].split('.')[0]

        # Importing configuration data
        config_filename = filepath + f'/{filename}_config'
        with open(config_filename, 'rb') as f: 
            config = pickle.load(f)

        # Loading configuration
        eikonal = IsoEikonal.from_config(config.pop('equation.config'))
        NES_OP_instance = NES_OP(xs=config.pop('xs'), 
                                 velocity=config.pop('velocity'), 
                                 eikonal=eikonal)
        NES_OP_instance.build_model(**config)

        # Loading weights
        weights_filename = filepath + f'/{filename}_weights.h5'
        NES_OP_instance.outs['T'].load_weights(weights_filename, by_name=False)
        print(f'Loaded model from "{filepath}"')

        # Loading optimizer state if available
        opt_filename = filepath + f'/{filename}_optimizer'
        if pathlib.Path(opt_filename).is_file():
            with open(opt_filename, 'rb') as f: 
                opt_config = pickle.load(f)
            optimizer = tf.keras.optimizers.get(opt_config['optimizer.config']['name'])
            optimizer = optimizer.from_config(opt_config['optimizer.config'])
            optimizer._create_all_weights(NES_OP_instance.model.trainable_variables)
            optimizer.set_weights(opt_config['optimizer.weights'])
            NES_OP_instance.compile(optimizer=optimizer, loss=opt_config['loss'])
            print('Compiled the model with saved optimizer')

        # Loading training data if available
        data_filename = filepath + f'/{filename}_train_data'
        if pathlib.Path(data_filename).is_file():
            with open(data_filename, 'rb') as f: 
                NES_OP_instance.x_train = pickle.load(f)
            print('Loaded last training data: see NES_OP.x_train')

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
                  If 'None', 'eikonalLayers.IsoEikonal(P=3, hamiltonian=True)' is used.
    """

    def __init__(self, velocity, eikonal=None):
        
        # Velocity
        assert callable(velocity), "Must be callable in a format 'v(xr) = velocity(xr)'"
        self.velocity = velocity
        self.dim = len(self.velocity.xmin)

        # Input scale factor
        self.xscale = np.max(np.abs([self.velocity.xmin, 
                                     self.velocity.xmax]))

        # Eikonal equation layer 
        if eikonal is None:
            self.equation = IsoEikonal(name='IsoEikonal')
        else:
            assert isinstance(eikonal, L.Layer), "Eikonal should be an instance of keras Layer"   
            self.equation = eikonal

        self.x_train = None     # input training data
        self.y_train = None     # output training data
        self.compiled = False   # compilation status
        self.config = {}        # config data of NN model to be reproducible
        self.losses = ['Er']    # outputs for training losses. Since this is Two-Point formulation, 
                                # we can solve two eikonal equations according to the reciprocity principle.
                                # 'Er' is equation w.r.t. 'xr', 'Es' is equation w.r.t. 'xr'.
        
    def build_model(self, nl=4, nu=50, act='lad-gauss-1', out_act='lad-sigmoid-1', 
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
        if kwargs.get('kernel_initializer') is None:
            kwargs['kernel_initializer'] = 'he_normal'

        # Saving configuration for reproducibility, to save and load model
        self.config['nl'] = nl
        self.config['nu'] = nu
        self.config['act'] = act
        self.config['out_act'] = out_act
        self.config['input_scale'] = input_scale
        self.config['factored'] = factored
        self.config['out_vscale'] = out_vscale
        self.config['reciprocity'] = reciprocity
        self.config['losses'] = self.losses
        for kw, v in kwargs.items():
            self.config[kw] = v

        #### Input for Source part
        xs_list = [L.Input(shape=(1,), name='xs' + str(i)) for i in range(self.dim)]
        xs = L.Concatenate(name='xs', axis=-1)(xs_list)

        #### Input for Receiver part
        xr_list = [L.Input(shape=(1,), name='xr' + str(i)) for i in range(self.dim)]
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
        T = DenseBody(X_sc, nu, nl, out_dim=1, act=act, out_act=out_act, **kwargs)

        #### Factorization
        # Scaling to the range of [1/vmax , 1/vmin]. T is assumed to be in [0, 1]
        if out_vscale:
            vmin, vmax = self.velocity.min, self.velocity.max
            T = L.Lambda(lambda z: (1 / vmin - 1 / vmax) * z + 1 / vmax, name='V_factor')(T)
        if factored:
            xr_xs = L.Subtract(name='xr_xs_difference')([xr, xs])
            D = L.Lambda(lambda z: tf.norm(z, axis=-1, keepdims=True), name='D_factor')(xr_xs)    
            T = L.Multiply(name='Traveltime')([T, D])

        #### Reciprocity 
        if reciprocity: # T(xs,xr)=T(xr,xs)
            t = Model(inputs=inputs, outputs=T)
            xsr = xs_list + xr_list; xrs = xr_list + xs_list;
            tsr = t(xsr); trs = t(xrs)
            T = L.Lambda(lambda x: 0.5*(x[0] + x[1]), name='Reciprocity')([tsr, trs])

        #### Final Traveltime Model
        Tm = Model(inputs=inputs, outputs=T)

        #### Gradient 'xr'
        dTr_list = Diff(name='gradients_xr')([T, xr_list])
        dTr = L.Concatenate(axis=-1, name='Gradient_xr')(dTr_list)
        Gr = Model(inputs=inputs, outputs=dTr)

        #### Eikonal 'xr'
        vr = L.Input(shape=(1,), name='vr') # velocity input for 'xr'
        Er = self.equation(dTr_list, vr)
        Emr = Model(inputs=inputs + [vr], outputs=Er)

        #### Gradient 'xs'
        dTs_list = Diff(name='gradients_xs')([T, xs_list])
        dTs = L.Concatenate(axis=-1, name='Gradient_xs')(dTs_list)
        Gs = Model(inputs=inputs, outputs=dTs)

        #### Eikonal 'xs'
        vs = L.Input(shape=(1,), name='vs') # velocity input for 'xs'
        Es = self.equation(dTs_list, vs)
        Ems = Model(inputs=inputs + [vs], outputs=Es)

        #### Laplacian 'xr'
        Lr_list = []
        for i, dTi in enumerate(dTr_list):
            Lr_list += Diff(name=f'Lr{i}')([dTi, xr_list[i]])
        Lr = L.Add(name='Laplacian_xr')(Lr_list)
        Lrm = Model(inputs=inputs, outputs=Lr)

        #### Laplacian 'xs'
        Ls_list = []
        for i, dTi in enumerate(dTs_list):
            Ls_list += Diff(name=f'Ls{i}')([dTi, xs_list[i]])
        Ls = L.Add(name='Laplacian_xs')(Lr_list)
        Lsm = Model(inputs=inputs, outputs=Ls)

        #### Full Hessian 'xr'
        Hr_list = []
        for i, dTri in enumerate(dTr_list):
            Hr_list += Diff(name=f'Hr{i}')([dTri, xr_list[i:]])
        Hr = L.Concatenate(axis=-1, name='Hessian_xr')(Hr_list)
        Hrm = Model(inputs=inputs, outputs=Hr)

        #### Hessians ss
        Hs_list = []
        for i, dTsi in enumerate(dTs_list):
            Hs_list += Diff(name=f'Hs{i}')([dTsi, xs_list[i:]])
        Hs = L.Concatenate(axis=-1, name='Hessian_xs')(Hs_list)
        Hsm = Model(inputs=inputs, outputs=Hs)

        #### Trainable model ####
        kw_models = dict(Er=Er, Es=Es)
        model_outputs = [kw_models[kw] for kw in self.losses]
        model_inputs = inputs + [vr] * ('Er' in self.losses) + [vs] * ('Es' in self.losses)
        self.model = Model(inputs=model_inputs, outputs=model_outputs)

        #### All callable models ####
        self.outs = dict(T=Tm, Er=Emr, Es=Ems, Gr=Gr, Gs=Gs, 
                         Lr=Lrm, Ls=Lsm, Hr=Hrm, Hs=Hsm)

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
        return self._predict(x, 'Lr', **kwargs)

    def LaplacianS(self, x, **kwargs):
        """
            Computes laplacian w.r.t. 'xs' - tau_dxdx + tau_dydy + tau_dzdz.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **kwargs : keyword arguments : Arguments for tf.keras.models.Model.predict(**kwargs) such as 'batch_size'

            Returns:
                Laplacian : numpy array (N,) of floats
        """        
        return self._predict(x, 'Ls', **kwargs)

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
        return self._predict(x, 'Hr', **kwargs)

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
        return self._predict(x, 'Hs', **kwargs)

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
        dims = Xc.shape[:-1]
        xs1 = np.array(xs1).squeeze().reshape([1]*len(dims) + [len(xs1)])
        xs2 = np.array(xs2).squeeze().reshape([1]*len(dims) + [len(xs2)])

        Xc1 = np.concatenate((np.tile(xs1, Xc.shape[:-1] + (1,)), Xc), axis=-1)
        dTc = self.GradientR(Xc1, **kwargs)
        Xc2 = np.concatenate((Xc, np.tile(xs2, Xc.shape[:-1] + (1,))), axis=-1)
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
        xs = x[..., :dim]
        xr = x[..., dim:]
        X = {}
        for kwi in model.input_names:
            if 'vr' in kwi:
                X['vr'] = velocity(xr).ravel()
            elif 'vs' in kwi:
                X['vs'] = velocity(xs).ravel()
            else:
                i = int(kwi[-1])
                r = dim*('r' in kwi)
                X[kwi] = x[..., r + i].ravel()
        return X

    def _predict(self, x, out, **kwargs):
        if kwargs.get('batch_size') is None:
            kwargs['batch_size'] = 100000
        X = self._prepare_inputs(self.outs[out], x, self.velocity)
        P = self.outs[out].predict(X, **kwargs).reshape(*x.shape[:-1], -1).squeeze()
        return P

    def train_inputs(self, x):
        """
            Creates dictionary of inputs for training.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
        """
        xs = x[..., :self.dim]
        xr = x[..., self.dim:]

        ids = abs(xr - xs).sum(axis=-1) != 0 # removing singular points

        self.x_train = self._prepare_inputs(self.model, x[ids], self.velocity)

    def train_outputs(self,):
        """
            Creates dictionary of output for training (target values). All target values will be zero.
        """
        self.y_train = {}
        xi = list(self.x_train.values())[0]
        for l in self.model.output_names:
            self.y_train[l] = np.zeros(len(xi))

    def compile(self, optimizer=None, loss='mae', lr=3e-3, decay=5e-4, **kwargs):
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
            optimizer = tf.optimizers.Adam(learning_rate=lr, decay=decay)

        self.model.compile(optimizer=optimizer, loss=loss, **kwargs)
        self.compiled = True

    def train(self, x_train=None, tolerance=None, **train_kw):
        """
            Traines the neural-network model.

            Arguments:
                x_train : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension. 
                                If 'None', "NES_OP.x" are used.
                tol : float : Tolerance value for early stopping in RMAE units for traveltimes. 
                              Empiric dependence 'RMAE = C exp(-Loss)' is used. If 'None', 'tol' is not used
                **train_kw : keyword arguments : Arguments for 'tf.keras.models.Model.fit(**train_kw)' such as 'batch_size', 'epochs'
        """
        self.train_inputs(x_train)
        self.train_outputs()
        data = data_handler(self.x_train, self.y_train, **train_kw)
        if isinstance(data, tf.keras.utils.Sequence):
            self.data_generator = data
            data = (data,)

        if not self.compiled:
            self.compile()

        callbacks = []
        if isinstance(tolerance, (float, tf.keras.callbacks.Callback)):
            if isinstance(tolerance, float):
                EarlyStopping = NES_EarlyStopping(tolerance=tolerance)
            else:
                EarlyStopping = tolerance
            callbacks.append(EarlyStopping)

        if train_kw.get('callbacks') is None:
            train_kw['callbacks'] = callbacks
        else:
            train_kw['callbacks'] += callbacks

        h = self.model.fit(*data, **train_kw)
        return h

    def save(self, filepath, save_optimizer=False, training_data=False):
        """
            Saves the current NES_OP model to `filepath` directory.
            `save_optimizer` saves the optimizer state to continue the training from the last point,
            `training_data` saves the last training set used for training.
        """
        config = self.config
        config['velocity'] = self.velocity
        config['equation.config'] = self.equation.get_config()

        # makedir
        if pathlib.Path(filepath).is_dir():
            shutil.rmtree(filepath)
        pathlib.Path(filepath).mkdir(parents=True, exist_ok=False)
        filename = filepath.split('/')[-1].split('.')[0]

        # Model configuration
        config_filename = filepath + f'/{filename}_config'
        with open(config_filename, 'wb') as f: 
            pickle.dump(config, f)

        # Model weights
        weights_filename = filepath + f'/{filename}_weights'
        weights = self.outs['T'].get_weights()
        with open(weights_filename, 'wb') as f: 
            pickle.dump(weights, f)

        # Optimizer state
        if save_optimizer:
            opt_config = {}
            opt_config['optimizer.config'] = self.model.optimizer.get_config()
            opt_config['optimizer.weights'] = self.model.optimizer.get_weights()
            opt_config['loss'] = self.model.loss
            opt_filename = filepath + f'/{filename}_optimizer'
            with open(opt_filename, 'wb') as f: 
                pickle.dump(opt_config, f)

        # Training set of collocation points
        if training_data:
            data_filename = filepath + f'/{filename}_train_data'
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
        filename = filepath.split('/')[-1].split('.')[0]

        # Importing configuration data
        config_filename = filepath + f'/{filename}_config'
        with open(config_filename, 'rb') as f: 
            config = pickle.load(f)

        # Loading configuration
        eikonal = IsoEikonal.from_config(config.pop('equation.config'))
        NES_TP_instance = NES_TP(velocity=config.pop('velocity'), 
                                 eikonal=eikonal)
        NES_TP_instance.losses = config.pop('losses')
        NES_TP_instance.build_model(**config)
        
        # Loading weights
        weights_filename = filepath + f'/{filename}_weights'
        with open(weights_filename, 'rb') as f: 
            weights = pickle.load(f)
        NES_TP_instance.outs['T'].set_weights(weights)
        print(f'Loaded model from "{filepath}"')

        # Loading optimizer state if available
        opt_filename = filepath + f'/{filename}_optimizer'
        if pathlib.Path(opt_filename).is_file():
            with open(opt_filename, 'rb') as f: 
                opt_config = pickle.load(f)
            optimizer = tf.keras.optimizers.get(opt_config['optimizer.config']['name'])
            optimizer = optimizer.from_config(opt_config['optimizer.config'])
            optimizer._create_all_weights(NES_TP_instance.model.trainable_variables)
            optimizer.set_weights(opt_config['optimizer.weights'])
            NES_TP_instance.compile(optimizer=optimizer, loss=opt_config['loss'])
            print('Compiled the model with saved optimizer')

        # Loading training data if available
        data_filename = filepath + f'/{filename}_train_data'
        if pathlib.Path(data_filename).is_file():
            with open(data_filename, 'rb') as f: 
                NES_TP_instance.x_train = pickle.load(f)
            print('Loaded last training data: see NES_TP.x_train')

        return NES_TP_instance
