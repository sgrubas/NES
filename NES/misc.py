import numpy as np
from scipy.ndimage import gaussian_filter
import importlib_resources
from .velocity import Interpolator


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
    # f = pkg_resources.resource_stream(__name__, "data//Marmousi_Pwave_smooth_12_5m.npy")
    ref = importlib_resources.files(__name__).joinpath('data//Marmousi_Pwave_smooth_12_5m.npy')
    with ref.open('rb') as fp:
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
