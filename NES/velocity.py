from importlib import resources
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


class BaseVelocity:
    r"""
        Base class for velocity models used to train NES. 
        There are a few mandatory properties that must be defined in `__init__` 
        or `__call__` methods BEFORE passing it to NES.

        Properties:
            dim : int : dimensions
            xmin : list of floats : minimum coordinates of the domain
            xmax : list of floats : maximum coordinates of the domain
            min : float : minimum velocity value
            max : float : maximum velocity value
    """

    dim = None # dimensions
    xmin = None # minimum coordinates of the domain
    xmax = None # maximum coordinates of the domain
    min = None # minimum velocity value
    max = None # maximum velocity value

    def __init__(self, **kwargs):
        """
            Arguments:
                kwargs : dict : arguments defining a specific velocity model 
        """
        pass

    def __call__(self, X):
        """
            Arguments:
                X : numpy NDarray : has shape `(... , dim)` where `dim` refers to `BaseVelocity.dim`

            Return:
                V : numpy NDarray : velocity at `X` coordinates with shape `(... , 1)` or `(... , )`
        """
        if self.xmin is None:
            self.xmin = X.reshape(-1, X.shape[-1]).min(axis=0)
        if self.xmax is None:
            self.xmax = X.reshape(-1, X.shape[-1]).max(axis=0)
        if self.dim is None:
            self.dim = X.shape[-1]
        pass

    def gradient(self, X):
        """
            Optional methods returning gradient of velocity

            Arguments:
                X : numpy NDarray : has shape `(... , dim)` where `dim` refers to `BaseVelocity.dim`

            Return:
                dV : numpy NDarray : gradient of velocity at `X` coordinates with shape `(... , dim)` 
        """
        pass

    def laplacian(self, X):
        """
            Optional methods returning laplacian of velocity

            Arguments:
                X : numpy NDarray : has shape `(... , dim)` where `dim` refers to `BaseVelocity.dim`

            Return:
                LV : numpy NDarray : laplacian of velocity at `X` coordinates with shape `(... , 1)` or `(... , )`
        """
        pass


class Interpolator(BaseVelocity):
    """
        Interpolator using 'scipy.interpolate.RegularGridInterpolator'

    """    
    
    F = None 
    dF = None
    LF = None
    axes = None
    Func = None
    dFunc = None
    LFunc = None

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
        self.min = F.min()
        self.max = F.max()

    def __call__(self, X):
        """
        Computes values of function using interpolation at points X
        """
        return self.Func(X)

    _INTERP_KEYS = ('method', 'bounds_error', 'fill_value')

    def __getstate__(self):
        # scipy interpolators are not stable across scipy versions: pickle the grid and rebuild on load
        state = self.__dict__.copy()
        state['interp_kw'] = {k: getattr(self.Func, k) for k in self._INTERP_KEYS if hasattr(self.Func, k)}
        for k in ('Func', 'dFunc', 'LFunc', 'dF'):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        # also handles models pickled by older NES versions (with a stale `RegularGridInterpolator` inside)
        old = state.pop('Func', None)
        interp_kw = state.pop('interp_kw', None)
        if interp_kw is None:
            old_dict = getattr(old, '__dict__', {})
            interp_kw = {k: old_dict[k] for k in self._INTERP_KEYS if k in old_dict}
        for k in ('dFunc', 'LFunc', 'dF'):
            state.pop(k, None)
        self.__dict__.update(state)
        self.Func = RegularGridInterpolator(self.axes, self.F, **interp_kw)

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


class VerticalGradient(BaseVelocity):
    """
        Velocity class for vertical gradient model
    """
    v0 = None
    a = None

    def __init__(self, v0, a, xmin=None, xmax=None):
        """ v0 : initial velocity ar z=0
            a : gradient of velocity
        """

        self.v0 = v0
        self.a = a
        self.xmin = xmin
        self.xmax = xmax
        if (xmin is not None) and (xmax is not None):
            self.dim = len(xmin)
            v_bounds = [v0 + a * xmin[-1], v0 + a * xmax[-1]]
            self.min, self.max = min(v_bounds), max(v_bounds)

    def __call__(self, X):
        """ Computes the velocity value at 'X', where X is (...., dim), and z=X[..., -1]
        """
        super(VerticalGradient, self).__call__(X)

        V = self.v0 + self.a * X[..., -1]

        if self.min is None:
            self.min = V.min()
        if self.max is None:
            self.max = V.max()

        return V

    def gradient(self, X):
        """ Computes the gradient of velocity value at 'X'
        """
        return np.concatenate([np.zeros_like(X[..., :-1]), 
            np.full_like(X[..., -2:-1], self.a)], axis=-1)

    def time(self, X, xs):
        """ Computes the analytical traveltimes at 'X'
        """
        xs = np.asarray(xs)
        Xdiff = X - xs[None, :]
        Vxszs = self(xs)
        up = self.a**2 * (Xdiff**2).sum(axis=-1)
        down = 2 * Vxszs * (self.a * Xdiff[...,-1] + Vxszs)
        tau = np.arccosh(up / down + 1) / self.a
        return tau

    def dtime(self, X, xs):
        """ Computes the analytical gradient of traveltimes at 'X'
        """
        xs = np.asarray(xs)
        Xdiff = X - xs[None, :]
        Vxszs = self(xs)
        up = self.a**2 * (Xdiff**2).sum(axis=-1)
        down = 2 * Vxszs * (self.a * Xdiff[...,-1] + Vxszs)
        A = 1 / self.a / np.sqrt((up / down + 1)**2 - 1)
        dt_dx = 2 * self.a**2 * Xdiff[...,0] / down * A
        dt_dz = (2 * self.a**2 * Xdiff[...,-1] / down - 2 * self.a * Vxszs * up / down**2) * A
        return np.stack([dt_dx, dt_dz], axis=-1)


class LocAnomaly(BaseVelocity):
    """
        Velocity class for model with gaussian anomaly
    """
    mus = None
    sigmas = None

    def __init__(self, vmin, vmax, mus, sigmas, xmin=None, xmax=None):
        """ vmin : minimal velocity
            vmax : maximal velocity
            mus : center of gaussian anomaly
            sigmas : width of gaussian anomaly
            xmin : [x_min, y_min, z_min] lower bound of the domain
            xmax : [x_max, y_max, z_max] upper bound of the domain
        """
        self.mus = np.array(mus).reshape(1, -1)
        self.sigmas = np.array(sigmas).reshape(1, -1)      
        self.vmin = vmin
        self.vmax = vmax

        self.min = min(vmin, vmax)
        self.max = max(vmin, vmax)
        self.xmin = xmin
        self.xmax = xmax
        self.dim = len(mus)

    def __call__(self, X):
        """ Computes the velocity value at 'X'
        """
        super(LocAnomaly, self).__call__(X)

        V = (self.vmax - self.vmin) 
        V *= np.exp(- ((X - self.mus)**2 / 2 / self.sigmas**2).sum(axis=-1))
        return V + self.vmin

    def gradient(self, X):
        """ Computes the analytical gradient of velocity at 'X'
        """
        return (self.__call__(X) - self.vmin)[..., None] * (self.mus - X) / self.sigmas**2


class MaxwellFishEye(BaseVelocity):
    r"""
        Lens models with closed-form two-point traveltimes, for testing solvers against exact solutions.

            v(x) = v0 * (1 + k |x - c|^2 / R^2)

        k = +1 (default) : Maxwell's fish-eye, a low-velocity lens (v0 is the minimum, at the centre `c`).
                           Every ray from a source x1 refocuses at the image point c - R^2 (x1 - c) / |x1 - c|^2,
                           where the first-arrival traveltime has a kink (a point caustic).
        k = -1 (`high_velocity=True`) : hyperbolic lens, a high-velocity lens (v0 is the maximum), needs |x - c| < R.
                           No caustics.

        With u = (x - c) / R, the optical metric |dx| / v is the stereographic image of a sphere (k=+1) or the
        Poincare ball (k=-1) of radius R / (2 v0), so the traveltime is that radius times the geodesic distance:

            T(x1, x2) = (R / v0) * asin_k( |u1 - u2| / sqrt((1 + k |u1|^2) (1 + k |u2|^2)) ),

        asin_k = arcsin for k=+1 and arcsinh for k=-1. Valid in any dimension; for R -> inf, T -> |x1 - x2| / v0.
        This is the first arrival of the unbounded medium. In a bounded domain it stays the first arrival while the
        fastest rays stay inside: for the fish-eye (rays bend away from the centre) keep sources within ~0.6 R of the
        centre in a [-R, R] box; the hyperbolic lens bends rays towards the centre and has no such restriction.
    """
    def __init__(self, v0=1.0, R=1.0, center=(0.0, 0.0), high_velocity=False, xmin=None, xmax=None):
        """ v0 : velocity at the centre
            R : lens radius (the velocity changes by factor 2 at |x - c| = R for the fish-eye)
            center : centre of the lens, defines the dimension
            high_velocity : False for Maxwell's fish-eye (low-velocity lens), True for the hyperbolic lens
            xmin, xmax : bounds of the domain (the hyperbolic lens needs the domain inside |x - c| < R)
        """
        self.v0, self.R = float(v0), float(R)
        self.center = np.asarray(center, dtype=float)
        self.k = -1.0 if high_velocity else 1.0
        self.dim = len(self.center)
        self.xmin, self.xmax = xmin, xmax
        if (xmin is not None) and (xmax is not None):
            lo, hi = np.asarray(xmin, dtype=float), np.asarray(xmax, dtype=float)
            r_near = np.linalg.norm(np.clip(self.center, lo, hi) - self.center)
            r_far = np.linalg.norm(np.maximum(np.abs(lo - self.center), np.abs(hi - self.center)))
            if self.k < 0 and r_far >= self.R:
                raise ValueError("The hyperbolic lens needs the domain inside |x - center| < R")
            v_near, v_far = self._v(r_near**2), self._v(r_far**2)
            self.min, self.max = min(v_near, v_far), max(v_near, v_far)

    def _v(self, r2):
        return self.v0 * (1 + self.k * r2 / self.R**2)

    def __call__(self, X):
        """ Velocity at X (..., dim) """
        super().__call__(X)
        return self._v(((np.asarray(X) - self.center)**2).sum(axis=-1))

    def gradient(self, X):
        """ Analytical gradient of velocity at X (..., dim) """
        return 2 * self.v0 * self.k * (np.asarray(X) - self.center) / self.R**2

    def _q(self, X, xs):
        u2 = (np.asarray(X) - self.center) / self.R
        u1 = (np.asarray(xs) - self.center) / self.R
        d = u2 - u1
        a1 = 1 + self.k * (u1**2).sum(axis=-1)
        a2 = 1 + self.k * (u2**2).sum(axis=-1)
        dist = np.linalg.norm(d, axis=-1)
        return dist / np.sqrt(a1 * a2), d, dist, a1, a2, u2

    def time(self, X, xs):
        """ Analytical traveltimes between xs (dim,) or (..., dim) and X (..., dim), broadcast """
        q = self._q(X, xs)[0]
        asin = np.arcsin if self.k > 0 else np.arcsinh
        return self.R / self.v0 * asin(np.clip(q, None, 1.0) if self.k > 0 else q)

    def dtime(self, X, xs):
        """ Analytical gradient of traveltimes w.r.t. X (receiver), shape (..., dim) """
        q, d, dist, a1, a2, u2 = self._q(X, xs)
        dq = d / (dist * np.sqrt(a1 * a2))[..., None] - (q * self.k / a2)[..., None] * u2
        return dq / (self.v0 * np.sqrt(1 - self.k * q**2))[..., None]

    def focus(self, xs):
        """ Point where all rays from `xs` refocus (fish-eye only) """
        u = (np.asarray(xs, dtype=float) - self.center)
        return self.center - self.R**2 * u / (u**2).sum()



class LuneburgLens(BaseVelocity):
    r"""
        Luneburg-type lens: a local lens of radius R in a homogeneous background, with closed-form traveltimes
        between any two points inside the lens.

            n(x)^2 = n0^2 + (1 - n0^2) |x - c|^2 / R^2   inside (|x - c| <= R),   n = 1 outside,   v = v_out / n

        n0 = sqrt(2) (default) : the classic Luneburg lens, low velocity v_out / sqrt(2) at the centre; rays from any
                                 point on the lens surface leave it as a plane wave.
        1 < n0 < sqrt(2)       : weaker low-velocity (focusing) lens.
        0 < n0 < 1             : high-velocity (defocusing) lens, velocity v_out / n0 at the centre.

        Inside, rays are the orbits of a (inverted, for n0 < 1) harmonic oscillator. With w = sqrt|1 - n0^2| / R,
        y = w (x - c) / n0 and k = sign(n0^2 - 1), the traveltime between y1 and y2 is

            T = n0^2 / (w v_out) * [ s/2 + S(s) C(s) / 2 - k S(s) (y1 . y2) ],
            C(s) = k (y1 . y2) + sqrt( (y1 . y2)^2 - k (|y1|^2 + |y2|^2) + 1 ),

        with S, C = sin, cos for k = +1 and sinh, cosh for k = -1. For n0 <= sqrt(2) this is the first arrival of the
        whole medium whenever both points are inside the lens: leaving the lens is never faster. Outside the lens the
        traveltime needs a ray-entry search and `time` returns NaN there.
    """
    def __init__(self, v_out=1.0, R=1.0, center=(0.0, 0.0), n0=np.sqrt(2), xmin=None, xmax=None):
        """ v_out : velocity outside the lens
            R : lens radius
            center : centre of the lens, defines the dimension
            n0 : refractive index at the centre, 0 < n0 <= sqrt(2) (n0 > 1 low velocity, n0 < 1 high velocity)
            xmin, xmax : bounds of the domain
        """
        if not 0 < n0 <= np.sqrt(2) + 1e-12 or np.isclose(n0, 1):
            raise ValueError("n0 must be in (0, 1) or (1, sqrt(2)]")
        self.v_out, self.R, self.n0 = float(v_out), float(R), float(n0)
        self.center = np.asarray(center, dtype=float)
        self.dim = len(self.center)
        self.k = 1.0 if n0 > 1 else -1.0
        self.w = np.sqrt(abs(1 - n0**2)) / self.R
        self.xmin, self.xmax = xmin, xmax
        v_c = self.v_out / self.n0
        self.min, self.max = min(v_c, self.v_out), max(v_c, self.v_out)

    def _n2(self, X):
        r2 = ((np.asarray(X) - self.center)**2).sum(axis=-1)
        return np.where(r2 <= self.R**2, self.n0**2 + (1 - self.n0**2) * r2 / self.R**2, 1.0)

    def __call__(self, X):
        """ Velocity at X (..., dim) """
        super().__call__(X)
        return self.v_out / np.sqrt(self._n2(X))

    def gradient(self, X):
        """ Analytical gradient of velocity at X (..., dim), zero outside the lens """
        d = np.asarray(X) - self.center
        n2 = self._n2(X)
        inside = ((d**2).sum(axis=-1) <= self.R**2)[..., None]
        return np.where(inside, -0.5 * self.v_out * n2[..., None]**-1.5 * 2 * (1 - self.n0**2) * d / self.R**2, 0.0)

    def inside(self, X):
        return ((np.asarray(X) - self.center)**2).sum(axis=-1) <= self.R**2 * (1 + 1e-12)

    def _parts(self, X, xs):
        y2 = self.w * (np.asarray(X) - self.center) / self.n0
        y1 = self.w * (np.asarray(xs) - self.center) / self.n0
        y1, y2 = np.broadcast_arrays(y1, y2)
        dot = (y1 * y2).sum(axis=-1)
        root = np.sqrt(np.maximum(dot**2 - self.k * ((y1**2).sum(-1) + (y2**2).sum(-1)) + 1, 0.0))
        C = self.k * dot + root
        if self.k > 0:
            s = np.arccos(np.clip(C, -1, 1))
            S = np.sin(s)
        else:
            s = np.arccosh(np.maximum(C, 1))
            S = np.sinh(s)
        return s, S, C, dot, y1, y2, root

    def time(self, X, xs):
        """ Analytical traveltimes between xs (dim,) or (..., dim) and X (..., dim), both inside the lens (else NaN) """
        s, S, C, dot, *_ = self._parts(X, xs)
        T = self.n0**2 / (self.w * self.v_out) * (0.5 * s + 0.5 * S * C - self.k * S * dot)
        return np.where(self.inside(X) & self.inside(xs), T, np.nan)

    def dtime(self, X, xs):
        """ Analytical gradient of traveltimes w.r.t. X (receiver), shape (..., dim); NaN outside the lens """
        s, S, C, dot, y1, y2, root = self._parts(X, xs)
        # T(y2) = A [s/2 + S C / 2 - k S dot], C = k dot + root, s = acos_k(C); implicit derivative w.r.t. y2
        safe_root = np.where(root > 0, root, 1.0)[..., None]
        dC = self.k * y1 + (dot[..., None] * y1 - self.k * y2) / safe_root
        # dS/dC = C / S * (-k)... for trig: S' = -C/S * ds? use ds/dC = -1/S (trig) or 1/S (hyperbolic)
        safe_S = np.where(S > 0, S, 1.0)[..., None]
        ds = -self.k * dC / safe_S
        dS = C[..., None] * ds
        dCs = -self.k * S[..., None] * ds          # dC/ds * ds, consistent with C = cos / cosh
        dT = 0.5 * ds + 0.5 * (dS * C[..., None] + S[..., None] * dCs) - self.k * (dS * dot[..., None] + S[..., None] * y1)
        g = self.n0**2 / (self.w * self.v_out) * dT * self.w / self.n0
        return np.where((self.inside(X) & self.inside(xs))[..., None], g, np.nan)

class HorLayeredModel(BaseVelocity):
    """
        Velocity class for horizontally layered model
    """
    v = None
    z = None

    def __init__(self, depths, v, xmin=None, xmax=None):
        """ depths : depths of interfaces (lower bounds of layers)
            v : velocities in layers
            xmin : [x_min, y_min, z_min] lower bound of the domain
            xmax : [x_max, y_max, z_max] upper bound of the domain
        """
        self.z = np.array(depths).squeeze()[:-1]
        self.v = np.array(v).squeeze()

        self.min = min(v)
        self.max = max(v)
        self.xmin = xmin
        self.xmax = xmax
        if (xmin is not None) or (xmax is not None):
            self.dim = len(xmin) if (xmin is not None) else len(xmax)

    def __call__(self, X):
        """ Computes the velocity value at 'X'
        """
        super(HorLayeredModel, self).__call__(X)

        Z = X[..., -1]
        layer_id = np.stack([Z > zi for zi in self.z], axis=-1)
        layer_id = np.cumprod(layer_id, axis=-1).sum(axis=-1)
        V = self.v[layer_id.ravel()].reshape(Z.shape)

        return V


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
    with resources.files('NES').joinpath('data', 'Marmousi_Pwave_smooth_12_5m.npy').open('rb') as fp:
        V = np.load(fp) / 1000.0

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
    return Marmousi(smooth=3, section=[[600, 900], None])
