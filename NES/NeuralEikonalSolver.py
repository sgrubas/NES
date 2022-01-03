import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as L
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model
from .baseLayers import DenseBody, Diff, SourceLoc
from .eikonalLayers import IsoEikonal
from .misc import Interpolator
from scipy.spatial.distance import cdist

###############################################################################
                    ### ONE POINT NEURAL EIKONAL SOLVER ###
###############################################################################

class NES_OP():
    """
    Neural Eikonal Solver for solving the equation in One-Point formulation tau(xr)
    """
    def __init__(self, xr, xs, xscale=None, 
                 vmin=None, vmax=None,
                 velocity=None, eikonal=None):
        """
            Initializer of NES_OP.

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                xs : list or array (dim,) of floats : Source location. 'dim' - dimension
                xscale : float : Constant for scaling inputs. If 'None', 'max|xr|' is used
                vmin : flaot : Minimal velocity value for improved factorization. If 'None', 'min v(xr)' is used
                vmax : flaot : Maximal velocity value for improved factorization. If 'None', 'max v(xr)' is used
                velocity : object : Velocity model class. Must be callable in a format 'v(xr) = velocity(xr)'. 
                                    See example in 'misc.Interpolator'
                eikonal : instance of tf.keras.layers.Layer : Layer that mimics the eikonal equation. 
                          There must two inputs: list of spatial derivatives, velocity value. 
                          Format example - 'eikonal([tau_x, tau_y, tau_z], v(x, y, z))'. 
                          If 'None', 'eikonalLayers.IsoEikonal(P=3, hamiltonian=True)' is used.

        """

        # Receivers
        self.xr = xr
        assert len(xr.shape) == 2
        self.dim = xr.shape[-1]

        # Source
        if isinstance(xs, (list, np.ndarray)):
            self.xs = np.array(xs).squeeze()
            assert self.dim == len(self.xs), "Dimension for 'xs' and 'xr' must coincide"
        else: 
            assert False, "Unrecognized 'xs' type "

        # Relative coordinates
        self.x = self.xr - self.xs[None, ...]

        # Input scale factor
        if xscale is None: 
            self.xscale = np.abs(self.x).max()
        else: 
            self.xscale = xscale
        
        # Velocity
        assert callable(velocity), "Must be callable in a format 'v(xr) = velocity(xr)'"
        self.velocity = velocity

        if vmin is None: 
            self.vmin = self.velocity(self.xr).min()
        else: 
            self.vmin = vmin

        if vmax is None: 
            self.vmax = self.velocity(self.xr).max()
        else: 
            self.vmax = vmax

        # Eikonal equation layer 
        if eikonal is None:
            self.equation = IsoEikonal(name='IsoEikonal')
        else:
            assert isinstance(eikonal, L.Layer), "Eikonal should be an instance of keras Layer"   
            self.equation = eikonal

        self.x_train = None
        self.y_train = None
        
    
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

        if kwargs.get('kernel_initializer') is None:
            kwargs['kernel_initializer'] = 'he_normal'

        # Receiver coordinate input
        xr_list = [L.Input(shape=(1,), name=f'xr{i}') for i in range(self.dim)]

        # Source coordinate reduction
        xr = L.Concatenate(axis=-1)(xr_list)
        xs = SourceLoc(self.xs)(xr)
        x = L.Subtract(name='xr_xs')([xr, xs])

        # Velocity input
        v = L.Input(shape=(1,), name='v')

        # Trainable body with Traveltime Output
        if input_scale:
            x_sc = Rescaling(1 / self.xscale, name='x_scaling')(x)
        else:
            x_sc = x

        T = DenseBody(x_sc, nu, nl, out_dim=1, act=act, out_act=out_act, **kwargs)

        # Factorized solution
        D = L.Lambda(lambda z: tf.norm(z, axis=-1, keepdims=True), name='D_factor')(x)
        if out_vscale:
            T = L.Lambda(lambda z: (1 / self.vmin - 1 / self.vmax) * z + 1 / self.vmax, name='V_factor')(T)
        T = L.Multiply(name='Traveltime')([T, D])

        # Final Traveltime Model
        Tm = Model(inputs=xr_list, outputs=T)

        # Eikonal equation
        dT_list = Diff(name='gradient')([T, xr_list])
        dT = L.Concatenate(axis=-1)(dT_list)
        Gm = Model(inputs=xr_list, outputs=dT)
        
        Eq = self.equation(dT_list, v)
        Em = Model(inputs=xr_list + [v], outputs=Eq)

        # Trainable model
        self.model = Model(inputs=xr_list + [v], outputs=Eq)

        ### Other callable outputs 
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

        # Callable outputs
        self.outs = dict(T=Tm, E=Em, G=Gm, V=Vm, L=Lm)

    def Traveltime(self, xr=None, **pred_kw):
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

    def Gradient(self, xr=None, **pred_kw):
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

    def Laplacian(self, xr=None, **pred_kw):
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
        
    def Velocity(self, xr=None, **pred_kw):
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

    def train_inputs(self, xr=None):
        """
            Creates dictionary of inputs for training.

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
        """
        if xr is not None:
            self.xr = xr.reshape(-1, self.dim)
            self.x = self.xr - self.xs[None, ...]

        self.x_train = {}
        ids = (abs(self.x)).sum(axis=-1) != 0 # removing singular point
        self.xr = self.xr[ids]
        self.x = self.x[ids]

        for kwi in self.model.input_names:
            if 'v' in kwi:
                self.x_train['v'] = self.velocity(self.xr).ravel()
            else:
                self.x_train[kwi] = self.xr[..., int(kwi[-1])]

    def predict_inputs(self, xr=None, out='E'):
        """
            Creates dictionary of inputs for prediction.

            Arguments:
                xr : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension
                out : str : For which model inputs should be prepared. See available in 'NES_OP.outs.keys()'

            Returns:
                X : dict : dictionary of inputs to be feed in a model 'out'
        """
        if xr is None:
            xr = self.xr
        else:
            xr = xr.reshape(-1, self.dim)

        X = {}
        for kwi in self.outs[out].input_names:
            if 'v' in kwi:
                X['v'] = self.velocity(xr).ravel()
            else:
                X[kwi] = xr[..., int(kwi[-1])]

        return X

    def train_outputs(self,):
        """
            Creates dictionary of output for training (target values). All target values will be zero.
        """
        if self.x_train is None:
            self.train_inputs()

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

    def train(self, x_train=None, tol=None, **train_kw):
        """
            Traines the neural-network model.

            Arguments:
                x_train : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension. 
                                If 'None', "NES_OP.x" are used.
                tol : float : Tolerance value for early stopping in RMAE units for traveltimes. 
                              Empiric dependence 'RMAE = C exp(-Loss)' is used. If 'None', 'tol' is not used
                **train_kw : keyword arguments : Arguments for 'tf.keras.models.Model.fit(**train_kw)' such as 'batch_size', 'epochs'
        """
        if self.x_train is None:
            self.train_inputs(x_train)
        if self.y_train is None:
            self.train_outputs()
        h = self.model.fit(x=self.x_train, y=self.y_train, **train_kw)
        return h

    def train_evolution(self, max_epochs=1000, x_train=None, tqdm=None, T_test_set=None,
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

class NES_TP():
    """
    Neural Eikonal Solver for solving the equation in Two-Point formulation tau(xs, xr)
    """
    def __init__(self, x, xscale=None, 
                 vmin=None, vmax=None,
                 velocity=None, eikonal=None):
        """
            Initializer of NES_TP.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs, where sources (N, :dim) and receivers (N, dim:). 
                                'N' - number of source-receiver pairs, 'dim' - dimension
                xscale : float : Constant for scaling inputs. If 'None', 'max|x|' is used
                vmin : flaot : Minimal velocity value for improved factorization. If 'None', 'min v(xr)' is used
                vmax : flaot : Maximal velocity value for improved factorization. If 'None', 'max v(xr)' is used
                velocity : object : Velocity model class. Must be callable in a format 'v(xr) = velocity(xr)'. 
                                    See example in 'misc.Interpolator'
                eikonal : instance of tf.keras.layers.Layer : Layer that mimics the eikonal equation. 
                          There must two inputs: list of spatial derivatives, velocity value. 
                          Format example - 'eikonal([tau_x, tau_y, tau_z], v(x, y, z))'. 
                          If 'None', 'eikonalLayers.IsoEikonal(P=3, hamiltonian=True)' is used.

        """


        # Grid 
        self.x = x
        self.dim = x.shape[-1] // 2
        self.xs = self.x[..., :self.dim]
        self.xr = self.x[..., self.dim:]

        # Input scale factor
        if xscale is None: 
            self.xscale = np.abs(self.x).max()
        else: 
            self.xscale = xscale
        
        # Velocity
        assert callable(velocity)
        self.velocity = velocity

        if vmin is None: 
            self.vmin = self.velocity(self.xr).min()
        else: 
            self.vmin = vmin

        if vmax is None: 
            self.vmax = self.velocity(self.xr).max()
        else: 
            self.vmax = vmax

        # Eikonal equation layer 
        if eikonal is None:
            self.equation = IsoEikonal(name='IsoEikonal')
        else:
            assert isinstance(eikonal, L.Layer), "Eikonal should be an instance of keras Layer"   
            self.equation = eikonal

        self.x_train = None
        self.y_train = None
        
    def build_model(self, nl=4, nu=50, act='lad-gauss-1', out_act='lad-sigmoid-1', 
                    factored=True, out_vscale=True, input_scale=True, reciprocity=True, 
                    losses=['Er'], **kwargs):

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
                losses : list of str : Since this is Two-Point formulation, we can solve two eikonal equations according to the reciprocity principle.
                                       'Er' is equation w.r.t. 'xr', 'Es' is equation w.r.t. 'xr'. By default 'losses = ['Er']'
                **kwargs : keyword arguments : Arguments for tf.keras.layers.Dense(**kwargs) such as 'kernel_initializer'.
                            If "kwargs.get('kernel_initializer')" is None then "kwargs['kernel_initializer'] = 'he_normal' "

        """

        if kwargs.get('kernel_initializer') is None:
            kwargs['kernel_initializer'] = 'he_normal'

        ## Input for Source part
        xs_list = [L.Input(shape=(1,), name='xs' + str(i)) for i in range(self.dim)]
        xs = L.Concatenate(axis=-1)(xs_list)

        vs = L.Input(shape=(1,), name='vs')

        ## Input for Receiver part
        xr_list = [L.Input(shape=(1,), name='xr' + str(i)) for i in range(self.dim)]
        xr = L.Concatenate(axis=-1)(xr_list)
        
        vr = L.Input(shape=(1,), name='vr')

        # Input list
        inputs = xs_list + xr_list

        # Trainable body
        X = L.Concatenate(axis=-1)([xs, xr])
        if input_scale:
            X_sc = Rescaling(1 / self.xscale, name='X_scaling')(X)
        else:
            X_sc = X
        T = DenseBody(X, nu, nl, out_dim=1, act=act, out_act=out_act, **kwargs)

        ### Factorization ###
        # Scaling to the range of [1/vmax , 1/vmin]. T is assumed to be in [0, 1]
        if out_vscale:
            T = L.Lambda(lambda z: (1 / self.vmin - 1 / self.vmax) * z + 1 / self.vmax, name='V_factor')(T)

        if factored:
            xr_xs = L.Subtract(name='xr_xs_difference')([xr, xs])
            D = L.Lambda(lambda z: tf.norm(z, axis=-1, keepdims=True), name='Distance')(xr_xs)    
            T = L.Multiply(name='D_factor')([T, D])

        # Symmetric function T(xs,xr)=T(xr,xs)
        if symmetric:
            t = Model(inputs=inputs, outputs=T)
            xsr = xs_list + xr_list; xrs = xr_list + xs_list;
            tsr = t(xsr); trs = t(xrs)
            T = L.Lambda(lambda x: 0.5*(x[0] + x[1]), name='Symmetry')([tsr, trs])

        Tm = Model(inputs=inputs, outputs=T)
        

        # Eikonal over 'xr'
        dTr_list = Diff(name='gradient_xr')([T, xr_list])
        dTr = L.Concatenate(axis=-1)(dTr_list)
        Gr = Model(inputs=inputs, outputs=dTr)

        Er = self.equation(dTr_list, vr)
        Emr = Model(inputs=inputs + [vr], outputs=Er)

        # Eikonal over 'xs'
        dTs_list = Diff(name='gradient_xs')([T, xs_list])
        dTs = L.Concatenate(axis=-1)(dTs_list)
        Gs = Model(inputs=inputs, outputs=dTs)

        Es = self.equation(dTs_list, vs)
        Ems = Model(inputs=inputs + [vs], outputs=Es)

        #### Hessians rr ####
        d2Tr_list = []
        for i, dTri in enumerate(dTr_list):
            d2Tr_list += Diff(name='d2Trr' + str(i))([dTri, xr_list[i:]])
        HTr = L.Concatenate(axis=-1)(d2Tr_list)
        Hmr = Model(inputs=inputs, outputs=HTr)

        # Models
        self.outs = dict(T=Tm, Er=Emr, Es=Ems, Gr=Gr, Gs=Gs, HTr=Hmr)
        
        # Trainable model
        kw_models = dict(Er=Er, Es=Es)
        model_outputs = [kw_models[kw] for kw in losses]
        model_inputs = inputs + [vr] * ('Er' in losses) + [vs] * ('Es' in losses)
        self.model = Model(inputs=model_inputs, outputs=model_outputs)
        

    def Traveltime(self, x=None, **pred_kw):
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

    def GradientR(self, x=None, **pred_kw):
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
    
    def GradientS(self, x=None, **pred_kw):
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
        
    def VelocityR(self, x=None, **pred_kw):
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
        
    def VelocityS(self, x=None, **pred_kw):
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

    def HessianR(self, x=None, **pred_kw):
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

    def train_inputs(self, x=None):
        """
            Creates dictionary of inputs for training.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
        """
        if x is not None:
            self.x = x.reshape(-1, self.dim*2)

        self.x_train = {}
        r = self.xs - self.xr
        ids = abs(r).sum(axis=-1) != 0 # removing singular point 
        self.x = self.x[ids]
        self.xs = self.x[..., :self.dim]
        self.xr = self.x[..., self.dim:]

        for kwi in self.model.input_names:
            if 'vr' in kwi:
                self.x_train['vr'] = self.velocity(self.xr).ravel()
            elif 'vs' in kwi:
                self.x_train['vs'] = self.velocity(self.xs).ravel()
            else:
                i = int(kwi[-1])
                r = self.dim * ('r' in kwi)
                self.x_train[kwi] = self.x[..., r + i]

    def predict_inputs(self, x=None, out='T'):
        """
            Creates dictionary of inputs for prediction.

            Arguments:
                x : numpy array (N, dim*2) of floats : Array of source-receiver pairs.
                out : str : For which model inputs should be prepared. See available in 'NES_TP.outs.keys()'

            Returns:
                X : dict : dictionary of inputs to be feed in a model 'out'
        """
        if x is None:
            x = self.x
        else:
            x = x.reshape(-1, self.dim*2)

        xs = x[..., :self.dim]
        xr = x[..., self.dim:]
        X = {}
        for kwi in self.outs[out].input_names:
            if 'vr' in kwi:
                X['vr'] = self.velocity(xr).ravel()
            elif 'vs' in kwi:
                X['vs'] = self.velocity(xs).ravel()
            else:
                i = int(kwi[-1])
                r = self.dim*('r' in kwi)
                X[kwi] = x[..., r + i]

        return X

    def train_outputs(self,):
        """
            Creates dictionary of output for training (target values). All target values will be zero.
        """
        if self.x_train is None:
            self.train_inputs()

        self.y_train = {}
        xi = list(self.x_train.values())[0]
        for l in self.model.output_names:
            self.y_train[l] = np.zeros(len(xi))

    def compile(self, optimizer=None, loss='mae', lr=1e-3, decay=1e-4, **kwargs):
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

    def train(self, x_train=None, tol=None, **train_kw):
        """
            Traines the neural-network model.

            Arguments:
                x_train : numpy array (N, dim) of floats : Array of receivers. 'N' - number of receivers, 'dim' - dimension. 
                                If 'None', "NES_OP.x" are used.
                tol : float : Tolerance value for early stopping in RMAE units for traveltimes. 
                              Empiric dependence 'RMAE = C exp(-Loss)' is used. If 'None', 'tol' is not used
                **train_kw : keyword arguments : Arguments for 'tf.keras.models.Model.fit(**train_kw)' such as 'batch_size', 'epochs'
        """
        if self.x_train is None:
            self.train_inputs(x_train)
        if self.y_train is None:
            self.train_outputs()
        h = self.model.fit(x=self.x_train, y=self.y_train, **train_kw)
        return h

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
        self.train_outputs(None)

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