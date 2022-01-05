import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


######################################################
                ### MISCELLANEOUS ###
######################################################


class Uniform_PDF:
    """
        API for generating uniform distribution in a given domain
    """
    limits = None 
    def __init__(self, limits):
        """limits : [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        """
        self.limits = np.array(limits, ndmin=2)

    def __call__(self, num_points):
        """ Return random points from uniform distribution in a given domain
        """
        return np.random.uniform(*self.limits.T, 
            size=(num_points, len(self.limits)))


class Interpolator:
    """
        Interpolator using 'scipy.interpolate.RegularGridInterpolator'

    """    
    dim = None
    F = None
    dF = None
    LF = None
    axes = None
    Func = None
    dFunc = None
    LFunc = None
    xmin = None
    xmax = None

    def __init__(self, F, *axes, **interp_kw):
        """
        The interpolator uses 'scipy.interpolate.RegularGridInterpolator'
        
        Arguments:
            F: numpy array (nx,) or (nx,ny) or (nx,ny,nz)
                Values 
            axes: tuple of numpy arrays (nx,), (ny), (nz)
                Grid
            interp_kw: dictionary of keyword arguments for 'scipy.interpolate.RegularGridInterpolator'
        """
        self.dim = len(F.shape)
        self.axes = axes
        self.F = F
        self.Func = RegularGridInterpolator(axes, F, **interp_kw)

        self.xmin = [xi.min() for xi in axes]
        self.xmax = [xi.max() for xi in axes]

    def __call__(self, X):
        """
        Computes values of function using interpolation at points X
        """
        return self.Func(X)

    def gradient(self, X, **interp_kw):
        """
        Computes partial derivatives (using default np.gradient) of function using interpolation at points X
        """
        if self.dFunc is None:
            self.dF = np.stack(np.gradient(self.F, *self.axes), axis=-1)
            self.dFunc = RegularGridInterpolator(self.axes, self.dF, **interp_kw)
        return self.dFunc(X)

    def laplacian(self, X, **interp_kw):
        """
        Computes laplacian (using default np.gradient) of function using interpolation at points X
        """
        if self.dFunc is None:
            self.dF = np.stack(np.gradient(self.F, *self.axes), axis=-1)
            self.dFunc = RegularGridInterpolator(self.axes, self.dF, **interp_kw)

        if self.LFunc is None:
            d2F = [np.gradient(self.dF[...,i], xi, axis=i) for i, xi in enumerate(self.axes)]
            L = np.sum(np.stack(d2F, axis=-1), axis=-1)
            self.LFunc = RegularGridInterpolator(self.axes, L, **interp_kw)

        return self.LFunc(X)

class VerticalGradient:
    """
        Velocity class for vertical gradient model
    """
    v0 = None
    a = None
    def __init__(self, v0, a):
        """ v0 : initial velocity ar z=0
            a : gradient of velocity
        """
        self.v0 = v0
        self.a = a

    def __call__(self, X):
        """ Computes the velocity value at 'X', where X is (...., dim), and z=X[..., -1]
        """
        return self.v0 + self.a * X[..., -1]

    def gradient(self, X):
        """ Computes the gradient of velocity value at 'X'
        """
        return np.concatenate([np.zeros_like(X[..., :-1]), 
            np.full_like(X[..., -2:-1], self.a)], axis=-1)

    def time(self, X, xs):
        """ Computes the analytical traveltimes at 'X'
        """
        Xdiff = X - xs[None,:]
        Vxszs = self(xs)
        up = self.a**2 * (Xdiff**2).sum(axis=-1)
        down = 2 * Vxszs * (self.a * Xdiff[...,-1] + Vxszs)
        tau = np.arccosh(up / down + 1) / self.a
        return tau

    def dtime(self, X, xs):
        """ Computes the analytical gradient of traveltimes at 'X'
        """
        Xdiff = X - xs[None,:]
        Vxszs = self(xs)
        up = self.a*2 * (Xdiff**2).sum(axis=-1)
        down = 2 * Vxszs * (self.a * Xdiff[...,-1] + Vxszs)
        A = 1 / self.a / np.sqrt((up / down + 1)**2 - 1)
        dt_dx = 2 * self.a**2 * Xdiff[...,0] / down * A
        dt_dz = (2 * self.a**2 * Xdiff[...,-1] / down - 2 * self.a * Vxszs * up / down**2) * A
        return np.stack([dt_dx, dt_dz], axis=-1)

class LocAnomaly:
    """
        Velocity class for model with gaussian anomaly
    """
    mus = None
    sigmas = None
    vmin = None
    vmax = None

    def __init__(self, vmin, vmax, mus, sigmas):
        """ vmin : minimal velocity
            vmax : maximal velocity
            mus : center of gaussian anomaly
            sigmas : width of gaussian anomaly
        """
        self.mus = mus.reshape(1, -1)
        self.sigmas = sigmas.reshape(1, -1)
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, X):
        """ Computes the velocity value at 'X'
        """
        V = (self.vmax - self.vmin) 
        V *= np.exp(- ((X - self.mus)**2 / 2 / self.sigmas**2).sum(axis=-1))
        return V + self.vmin

    def gradient(self, X):
        """ Computes the analytical gradient of velocity at 'X'
        """
        return (self.__call__(X) - self.vmin)[..., None] * (self.mus - X) / self.sigmas

def Marmousi(smooth=None, section=None):
    """
        Creates Interpolator of Marmousi model

        Arguments:
            smooth : float : smoothes model using "scipy.ndimage.gaussian_filter(sigma=smooth)"
                             If 'None', smoothing is not applied
            section : list of ints : indices to cut out a rectangle part of Marmousi model. 
                                     Example - 'section = [[100, 200], [0, 150]]' 
                                     where the first pair is for axis 0, the second - axis 1
        Return:
            Vel : instance of 'NES.Interpolator' for Marmousi model in 'km/s' units
    """

    V = np.load('../data/Marmousi_Pwave_smooth_12_5m.npy') / 1000.0
    if section is not None:
        i = [0, V.shape[0]+1] if section[0] is None else section[0]
        j = [0, V.shape[1]+1] if section[1] is None else section[1]
        V = V[i[0] : i[1], j[0] : j[1]]
    if smooth is not None:
        V = gaussian_filter(V, sigma=smooth)

    nx, nz = V.shape
    xmin, xmax = 0.0, .0125 * nx
    zmin, zmax = 0.0, .0125 * nz
    x = np.linspace(xmin, xmax, nx)
    z = np.linspace(zmin, zmax, nz)
    Vel = Interpolator(V, x, z)
    return Vel

def MarmousiSmoothedPart():
    """
        Return smoothed central part of Marmousi model 'NES.Marmousi(smooth=3, section=[[600, 900], None])' 
    """
    return Marmousi(smooth=3, section=[[600, 900], [0, 10000]])