import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as L
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model
from .baseLayers import DenseBody, Diff, SourceLoc, NES_EarlyStopping
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
        # Saving configuration for reproducibility, to save and load model
        self.config['nl'] = nl
        self.config['nu'] = nu
        self.config['act'] = act
        self.config['out_act'] = out_act
        self.config['input_scale'] = input_scale
        self.config['factored'] = factored
        self.config['out_vscale'] = out_vscale

        # The best initializer
        if kwargs.get('kernel_initializer') is None:
            kwargs['kernel_initializer'] = 'he_normal'

        for kw, v in kwargs.items():
            self.config[kw] = v

        # Receiver coordinate input
        xr_list = [L.Input(shape=(1,), name=f'xr{i}') for i in range(self.dim)]

        # Source coordinate reduction
        xr = L.Concatenate(name='xr', axis=-1)(xr_list)
        xs = SourceLoc(self.xs, name='SourceLoc')(xr)
        x = L.Subtract(name='Centering')([xr, xs])

        # Trainable body with Traveltime Output
        if input_scale:
            x_sc = Rescaling(1 / self.xscale, name='input_scaling')(x)
        else:
            x_sc = x

        T = DenseBody(x_sc, nu, nl, out_dim=1, act=act, out_act=out_act, **kwargs)

        # Factorized solution
        if out_vscale:
            vmin, vmax = self.velocity.min, self.velocity.max
            T = L.Lambda(lambda z: (1 / vmin - 1 / vmax) * z + 1 / vmax, name='V_factor')(T)
        if factored:
            D = L.Lambda(lambda z: tf.norm(z, axis=-1, keepdims=True), name='D_factor')(x)
            T = L.Multiply(name='Traveltime')([T, D])

        # Final Traveltime Model
        Tm = Model(inputs=xr_list, outputs=T)

        # Gradient
        dT_list = Diff(name='gradients')([T, xr_list])
        dT = L.Concatenate(name='Gradient', axis=-1)(dT_list)
        Gm = Model(inputs=xr_list, outputs=dT)

        # Eikonal equation
        v = L.Input(shape=(1,), name='v') # Velocity input
        Eq = self.equation(dT_list, v)
        Em = Model(inputs=xr_list + [v], outputs=Eq)

        # Velocity
        V = L.Lambda(lambda z: 1 / tf.norm(z, axis=-1, keepdims=True), name='Velocity')(dT)
        Vm = Model(inputs=xr_list, outputs=V)

        # Laplacian calculation
        d2T_list = []
        for i, dTi in enumerate(dT_list):
            d2Ti = Diff(name=f'd2T{i}')([dTi, xr_list[i]])
            d2T_list += d2Ti
        d2T = L.Concatenate(name='d2T', axis=-1)(d2T_list)
        LT = L.Lambda(lambda z: tf.reduce_sum(z, axis=-1, keepdims=True), name='Laplacian')(d2T)
        Lm = Model(inputs=xr_list, outputs=LT)

        # All callable models
        self.outs = dict(T=Tm, E=Em, G=Gm, V=Vm, L=Lm, gE=gEm)

        # Trainable model
        inputs = xr_list + [v]
        outputs = Eq
        self.model = Model(inputs=inputs, outputs=outputs)

    def Traveltime(self, xr, **pred_kw):
        """
            Computes traveltimes.

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                **pred_kw : keyword arguments : Arguments for tf.keras.models.Model.predict(**pred_kw) such as 'batch_size'

            Returns:
                T : numpy array (N,) of floats : Traveltimes from the source 'NES_OP.xs' at 'xr'
        """
        X = self.predict_inputs(xr, 'T')
        T = self.outs['T'].predict(X, **pred_kw)
        return T

    def Gradient(self, xr, **pred_kw):
        """
            Computes gradients - vector (tau_dx, tau_dy, tau_dz).

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                **pred_kw : keyword arguments : Arguments for tf.keras.models.Model.predict(**pred_kw) such as 'batch_size'

            Returns:
                dT : numpy array (N, dim) of floats : Gradient of traveltimes from the source 'NES_OP.xs' at 'xr'
        """
        X = self.predict_inputs(xr, 'G')
        G = self.outs['G'].predict(X, **pred_kw)
        return G

    def Laplacian(self, xr, **pred_kw):
        """
            Computes laplacian - tau_dxdx + tau_dydy + tau_dzdz.

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                **pred_kw : keyword arguments : Arguments for tf.keras.models.Model.predict(**pred_kw) such as 'batch_size'

            Returns:
                Laplacian : numpy array (N,) of floats : Laplacian from the source 'NES_OP.xs' at 'xr'
        """        
        X = self.predict_inputs(xr, 'L')
        L = self.outs['L'].predict(X, **pred_kw)
        return L
        
    def Velocity(self, xr, **pred_kw):
        """
            Computes predicted velocity - 1 / ||( tau_dx, tau_dy, tau_dz) ||.

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                **pred_kw : keyword arguments : Arguments for tf.keras.models.Model.predict(**pred_kw) such as 'batch_size'

            Returns:
                V : numpy array (N,) of floats : Predicted velocity from the source 'NES_OP.xs' at 'xr'
        """
        X = self.predict_inputs(xr, 'V')
        V = self.outs['V'].predict(X, **pred_kw)
        return V

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

    def train_inputs(self, xr):
        """
            Creates dictionary of inputs for training. Removes singular points (xr=xs)

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
        """

        ids = abs(xr - self.xs[None, ...]).sum(axis=-1) != 0 # removing singular point
        self.x_train = self._prepare_inputs(self.model, xr[ids], self.velocity)
    

    def predict_inputs(self, xr, out='T'):
        """
            Creates dictionary of inputs for prediction for different output models `out`.

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                out : str : For which model inputs should be prepared. See available in 'NES_OP.outs.keys()'

            Returns:
                X : dict : dictionary of inputs to be feed in a model 'out'
        """
        return self._prepare_inputs(self.outs[out], xr, self.velocity)

    def train_outputs(self,):
        """
            Creates dictionary of output for training (target values). All target values will be zero.
        """
        self.y_train = {}
        xi = list(self.x_train.values())[0]
        for l in self.model.output_names:
            self.y_train[l] = np.zeros(len(xi))

    def compile(self, optimizer=None, loss='mae', lr=2.5e-3, decay=1e-4, **kwargs):
        """
            Compiles the neural-network model for training.

            Arguments:
                optimizer : Instance of 'tf.optimizers.Optimizer' : Optimizer of weights. 
                            If 'None', 'tf.optimizers.Adam' is used.
                loss : str (shortcuts for 'tf.keras.losses') : Loss type. By default "loss = 'mae'"
                lr : float : Learning rate, by default 'lr = 2.5e-3'.
                decay : float : Decay of learning rate, by default 'decay = 1e-4'.
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
                                If 'None', previous `NES_OP.x_train` may be used if available
                tolerance : It can be:
                    1) float - Tolerance value for early stopping in RMAE units for traveltimes.
                               NES_EarlyStopping callback will be created with default options 
                               (see `baseLayers.NES_EarlyStopping`)
                    2) instance of NES_EarlyStopping callback. 
                              
                **train_kw : keyword arguments : Arguments for 'tf.keras.models.Model.fit(**train_kw)' such as 'batch_size', 'epochs'
        """
        if x_train is None:
            assert self.x_train is not None, " if `x_train` is not given, `NES_OP.x_train` must be defined"
        else:
            self.train_inputs(x_train)

        if self.y_train is None:
            self.train_outputs()

        if not self.compiled:
            self.compile()

        if isinstance(tolerance, (float, tf.keras.callbacks.Callback)):
            if isinstance(tolerance, float):
                EarlyStopping = NES_EarlyStopping(tolerance=tolerance)
            else:
                EarlyStopping = tolerance

            if train_kw.get('callbacks') is None:
                train_kw['callbacks'] = [EarlyStopping]
            else:
                train_kw['callbacks'].append(EarlyStopping)

        h = self.model.fit(x=self.x_train, y=self.y_train, **train_kw)
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

    def train_evolution(self, max_epochs=1000, step_epochs=10, x_train=None, tqdm=None, T_test_set=None,
                 t_evol=False, compile_kw=dict(lr=1e-3, decay=1e-4, loss='mae'), 
                 pred_kw=dict(batch_size=100000), **train_kw):
        """
            Traines the neural-network model and tracks the evolution of traveltime accuracy comparing to the reference.

            Arguments:
                max_epochs : int : Total number of epochs for training
                step_epochs : int : Step for each we track the traveltime accuracy
                x_train : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension. 
                                If 'None', "NES_OP.x" are used.
                tqdm : instance of tqdm : Progress bar
                T_test_set : tuple (x_test, y_test) : 'x_test' is numpy array (N, dim) of floats (receiver coordinates),
                                                      'y_test' is numpy array (N,) of floats (traveltimes).
                t_evol : boolean : Saves the solution on 'x_test' for each 'step_epochs'
                compile_kw : dict : Dict of keyword arguments for 'tf.keras.models.Model.compile(**compile_kw)'
                pred_kw : dict : Dict of keyword arguments for 'tf.keras.models.Model.predict(**pred_kw)'
                **train_kw : keyword arguments : Arguments for 'tf.keras.models.Model.fit(**train_kw)' such as 'batch_size'

            Returns:
                Logs : dict : Evolution of solution and MAE with the reference solution in a form {'mae':mae, 't_evolve': t_evolve}
        """
        def mae_assess():
            if mae_bool:
                t_pred = self.Traveltime(T_test_set[0], **pred_kw).ravel()
                mae_i = np.mean(np.abs(t_pred - T_test_set[1].ravel()))
                mae.append(mae_i)
                if t_evol:
                    t_evolve.append(t_pred)

        # Train set preparation
        self.train_inputs(x_train)
        self.train_outputs()

        # Outer Callbacks
        mae_bool = isinstance(T_test_set, (list, tuple))
        mae = [] # if 'T_test_set' is given
        t_evolve = [] # if 't_evol'

        # Compilation
        self.compile(**compile_kw)

        if step_epochs is not None:
            train_kw['epochs'] = step_epochs

        # Progress bar
        steps = np.ceil(max_epochs / train_kw['epochs']).astype(int)
        if tqdm is not None:
            p_bar = tqdm(total=steps)

        for i in range(steps):
            # Metric with reference solution
            mae_assess()
            # Train
            self.model.fit(x=self.x_train, y=self.y_train, **train_kw)
            if tqdm is not None: p_bar.update()

        # Metric with reference solution
        mae_assess()

        return {'mae':mae, 't_evolve': t_evolve}


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

        # The best initializer
        if kwargs.get('kernel_initializer') is None:
            kwargs['kernel_initializer'] = 'he_normal'

        for kw, v in kwargs.items():
            self.config[kw] = v

        ## Input for Source part
        xs_list = [L.Input(shape=(1,), name='xs' + str(i)) for i in range(self.dim)]
        xs = L.Concatenate(name='xs', axis=-1)(xs_list)

        ## Input for Receiver part
        xr_list = [L.Input(shape=(1,), name='xr' + str(i)) for i in range(self.dim)]
        xr = L.Concatenate(name='xr', axis=-1)(xr_list)
        
        # Input list
        inputs = xs_list + xr_list

        # Trainable body
        X = L.Concatenate(name='x', axis=-1)([xs, xr])
        if input_scale:
            X_sc = Rescaling(1 / self.xscale, name='X_scaling')(X)
        else:
            X_sc = X

        T = DenseBody(X_sc, nu, nl, out_dim=1, act=act, out_act=out_act, **kwargs)

        ### Factorization ###
        # Scaling to the range of [1/vmax , 1/vmin]. T is assumed to be in [0, 1]
        if out_vscale:
            vmin, vmax = self.velocity.min, self.velocity.max
            T = L.Lambda(lambda z: (1 / vmin - 1 / vmax) * z + 1 / vmax, name='V_factor')(T)
        if factored:
            xr_xs = L.Subtract(name='xr_xs_difference')([xr, xs])
            D = L.Lambda(lambda z: tf.norm(z, axis=-1, keepdims=True), name='D_factor')(xr_xs)    
            T = L.Multiply(name='Traveltime')([T, D])

        # Reciprocity T(xs,xr)=T(xr,xs)
        if reciprocity:
            t = Model(inputs=inputs, outputs=T)
            xsr = xs_list + xr_list; xrs = xr_list + xs_list;
            tsr = t(xsr); trs = t(xrs)
            T = L.Lambda(lambda x: 0.5*(x[0] + x[1]), name='Reciprocity')([tsr, trs])

        Tm = Model(inputs=inputs, outputs=T)

        # Gradient over 'xr'
        dTr_list = Diff(name='gradients_xr')([T, xr_list])
        dTr = L.Concatenate(axis=-1, name='Gradient_xr')(dTr_list)
        Gr = Model(inputs=inputs, outputs=dTr)

        # Eikonal over 'xr'
        vr = L.Input(shape=(1,), name='vr') # velocity input for 'xr'
        Er = self.equation(dTr_list, vr)
        Emr = Model(inputs=inputs + [vr], outputs=Er)

        # Gradient over 'xs'
        dTs_list = Diff(name='gradients_xs')([T, xs_list])
        dTs = L.Concatenate(axis=-1, name='Gradient_xs')(dTs_list)
        Gs = Model(inputs=inputs, outputs=dTs)

        # Eikonal over 'xs'
        vs = L.Input(shape=(1,), name='vs') # velocity input for 'xs'
        Es = self.equation(dTs_list, vs)
        Ems = Model(inputs=inputs + [vs], outputs=Es)

        #### Hessians rr ####
        d2Tr_list = []
        for i, dTri in enumerate(dTr_list):
            d2Tr_list += Diff(name='d2Trr' + str(i))([dTri, xr_list[i:]])
        HTr = L.Concatenate(axis=-1, name='Hessians_xr')(d2Tr_list)
        Hmr = Model(inputs=inputs, outputs=HTr)

        #### Hessians ss ####
        d2Ts_list = []
        for i, dTsi in enumerate(dTs_list):
            d2Ts_list += Diff(name='d2Tss' + str(i))([dTsi, xs_list[i:]])
        HTs = L.Concatenate(axis=-1, name='Hessians_xs')(d2Ts_list)
        Hms = Model(inputs=inputs, outputs=HTs)
        
        # Trainable model
        kw_models = dict(Er=Er, Es=Es)
        model_outputs = [kw_models[kw] for kw in self.losses]
        model_inputs = inputs + [vr] * ('Er' in self.losses) + [vs] * ('Es' in self.losses)
        self.model = Model(inputs=model_inputs, outputs=model_outputs)

        # All callable models
        self.outs = dict(T=Tm, Er=Emr, Es=Ems, 
                         Gr=Gr, Gs=Gs, HTr=Hmr, HTs=Hms)

    def Traveltime(self, x, **pred_kw):
        """
            Computes traveltimes.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **pred_kw : keyword arguments : Arguments for tf.keras.models.Model.predict(**pred_kw) such as 'batch_size'

            Returns:
                T : numpy array (N,) of floats : Traveltimes
        """
        X = self.predict_inputs(x, 'T')
        T = self.outs['T'].predict(X, **pred_kw)
        return T

    def GradientR(self, x, **pred_kw):
        """
            Computes gradient of traveltimes w.r.t. 'xr'.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **pred_kw : keyword arguments : Arguments for tf.keras.models.Model.predict(**pred_kw) such as 'batch_size'

            Returns:
                dTr : numpy array (N, dim) of floats : gradient of traveltimes w.r.t. 'xr'
        """
        X = self.predict_inputs(x, 'Gr')
        G = self.outs['Gr'].predict(X, **pred_kw)
        return G
    
    def GradientS(self, x, **pred_kw):
        """
            Computes gradient of traveltimes w.r.t. 'xs'.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **pred_kw : keyword arguments : Arguments for tf.keras.models.Model.predict(**pred_kw) such as 'batch_size'

            Returns:
                dTs : numpy array (N, dim) of floats : gradient of traveltimes w.r.t. 'xs'
        """
        X = self.predict_inputs(x, 'Gs')
        G = self.outs['Gs'].predict(X, **pred_kw)
        return G
        
    def VelocityR(self, x, **pred_kw):
        """
            Predicted velocity at 'xr'.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **pred_kw : keyword arguments : Arguments for tf.keras.models.Model.predict(**pred_kw) such as 'batch_size'

            Returns:
                Vr : numpy array (N,) of floats : Predicted velocity at 'xr'.
        """
        G = self.GradientR(x=x, **pred_kw)
        return 1 / np.linalg.norm(G, axis=-1)
        
    def VelocityS(self, x, **pred_kw):
        """
            Predicted velocity at 'xs'.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **pred_kw : keyword arguments : Arguments for tf.keras.models.Model.predict(**pred_kw) such as 'batch_size'

            Returns:
                Vs : numpy array (N,) of floats : Predicted velocity at 'xs'.
        """
        G = self.GradientS(x=x, **pred_kw)
        return 1 / np.linalg.norm(G, axis=-1)

    def HessianR(self, x, **pred_kw):
        """
            Computes hessians at w.r.t. 'xr'.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **pred_kw : keyword arguments : Arguments for tf.keras.models.Model.predict(**pred_kw) such as 'batch_size'

            Returns:
                H : numpy array (N, dim*2) of floats : Hessians w.r.t. 'xr' in a form (tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz).
        """
        X = self.predict_inputs(x, 'HTr')
        H = self.outs['HTr'].predict(X, **pred_kw)
        return H

    def HessianS(self, x, **pred_kw):
        """
            Computes hessians at w.r.t. 'xs'.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                **pred_kw : keyword arguments : Arguments for tf.keras.models.Model.predict(**pred_kw) such as 'batch_size'

            Returns:
                H : numpy array (N, dim*2) of floats : Hessians w.r.t. 'xs' in a form (tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz).
        """
        X = self.predict_inputs(x, 'HTs')
        H = self.outs['HTs'].predict(X, **pred_kw)
        return H

    @staticmethod
    def _prepare_inputs(model, x, velocity):
        dim = x.shape[-1] // 2
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
        

    def predict_inputs(self, x, out='T'):
        """
            Creates dictionary of inputs for prediction.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                out : str : For which model inputs should be prepared. See available in 'NES_TP.outs.keys()'

            Returns:
                X : dict : dictionary of inputs to be feed in a model 'out'
        """
        return self._prepare_inputs(self.outs[out], x, self.velocity)

    def train_outputs(self,):
        """
            Creates dictionary of output for training (target values). All target values will be zero.
        """
        self.y_train = {}
        xi = list(self.x_train.values())[0]
        for l in self.model.output_names:
            self.y_train[l] = np.zeros(len(xi))

    def compile(self, optimizer=None, loss='mae', lr=2.5e-3, decay=1e-4, **kwargs):
        """
            Compiles the neural-network model for training.

            Arguments:
                optimizer : Instance of 'tf.optimizers.Optimizer' : Optimizer of weights. 
                            If 'None', 'tf.optimizers.Adam' is used.
                loss : str (shortcuts for 'tf.keras.losses') : Loss type. By default "loss = 'mae'"
                lr : float : Learning rate, by default 'lr = 2.5e-3'.
                decay : float : Decay of learning rate, by default 'decay = 1e-4'.
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
        if x_train is None:
            assert self.x_train is not None, " if `x_train` is not given, `NES_OP.x_train` must be defined"
        else:
            self.train_inputs(x_train)

        if self.y_train is None:
            self.train_outputs()
        if not self.compiled:
            self.compile()

        if isinstance(tolerance, (float, tf.keras.callbacks.Callback)):
            if isinstance(tolerance, float):
                EarlyStopping = NES_EarlyStopping(tolerance=tolerance)
            else:
                EarlyStopping = tolerance

            if train_kw.get('callbacks') is None:
                train_kw['callbacks'] = [EarlyStopping]
            else:
                train_kw['callbacks'].append(EarlyStopping)

        h = self.model.fit(x=self.x_train, y=self.y_train, **train_kw)
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
        try:
            # topology may have multiple weights 
            # with the same names if `reciprocity=True`
            self.outs['T'].save_weights(weights_filename + '.h5')
        except:
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
        try:
            NES_TP_instance.outs['T'].load_weights(weights_filename + '.h5', by_name=False)
        except:
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
            print('Loaded last training data: see NES_OP.x_train')

        return NES_TP_instance

    def train_evolution(self, max_epochs=1000, step_epochs=10, x_train=None, 
                        tqdm=None, T_test_set=None, t_evol=False, 
                        compile_kw=dict(lr=1e-3, decay=1e-4, loss='mae'), 
                        pred_kw=dict(batch_size=100000), 
                        **train_kw):
        """
            Traines the neural-network model and tracks the evolution of traveltime accuracy comparing to the reference.

            Arguments:
                max_epochs : int : Total number of epochs for training
                step_epochs : int : Step for each we track the traveltime accuracy
                x_train : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension. 
                                If 'None', "NES_OP.x" are used.
                tqdm : instance of tqdm : Progress bar
                T_test_set : tuple (x_test, y_test) : 'x_test' is numpy array (N, dim) of floats (receiver coordinates),
                                                      'y_test' is numpy array (N,) of floats (traveltimes).
                t_evol : boolean : Saves the solution on 'x_test' for each 'step_epochs'
                compile_kw : dict : Dict of keyword arguments for 'tf.keras.models.Model.compile(**compile_kw)'
                pred_kw : dict : Dict of keyword arguments for 'tf.keras.models.Model.predict(**pred_kw)'
                **train_kw : keyword arguments : Arguments for 'tf.keras.models.Model.fit(**train_kw)' such as 'batch_size'

            Returns:
                Logs : dict : Evolution of solution and MAE with the reference solution in a form {'mae':mae, 't_evolve': t_evolve}
        """

        def mae_assess():
            if mae_bool:
                t_pred = self.Traveltime(T_test_set[0], **pred_kw).ravel()
                mae_i = np.mean(np.abs(t_pred - T_test_set[1].ravel()))
                mae.append(mae_i)
                if t_evol:
                    t_evolve.append(t_pred)

        # Train set preparation
        self.train_inputs(x_train)
        self.train_outputs()

        # Outer Callbacks
        mae_bool = isinstance(T_test_set, (list, tuple))
        mae = [] # if 'T_test_set' is given
        t_evolve = [] # if 't_evol'

        # Compilation
        self.compile(**compile_kw)

        if step_epochs is not None:
            train_kw['epochs'] = step_epochs

        # Progress bar
        steps = np.ceil(max_epochs / train_kw['epochs']).astype(int)
        if tqdm is not None:
            p_bar = tqdm(total=steps)

        for i in range(steps):
            # Metric with reference solution
            mae_assess()
            # Train
            self.model.fit(x=self.x_train, y=self.y_train, **train_kw)
            if tqdm is not None: p_bar.update()

        # Metric with reference solution
        mae_assess()

        return {'mae':mae, 't_evolve': t_evolve}