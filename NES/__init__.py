from .NeuralEikonalSolver import NES_OP, NES_TP
from .utils import NES_EarlyStopping, LossesHolder, Uniform_PDF, RegularGrid
from .eikonalLayers import IsoEikonal
from .velocity import Interpolator
import NES.velocity
import NES.misc
import NES.ray_tracing

__version__ = '0.2.0'