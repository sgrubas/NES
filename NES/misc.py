"""
Kept for backward compatibility: `NES.misc` used to hold a copy of `NES.velocity`.
"""
from .velocity import *  # noqa: F401,F403
from .velocity import Marmousi, MarmousiSmoothedPart  # noqa: F401
from .utils import RegularGrid, Uniform_PDF  # noqa: F401
