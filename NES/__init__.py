"""
Neural Eikonal Solver (NES) on Keras 3: runs on the JAX, TensorFlow or PyTorch backend.
Select the backend before importing NES (or keras): `os.environ["KERAS_BACKEND"] = "jax"`.
"""
__version__ = '0.3.0'

from .NeuralEikonalSolver import NES_OP, NES_TP
from .utils import NES_EarlyStopping, LossesHolder, Uniform_PDF, RegularGrid
from .eikonalLayers import IsoEikonal
from .velocity import Interpolator
from . import backend, layers, velocity, misc, ray_tracing, experimental  # noqa: F401
