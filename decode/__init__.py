__version__ = "0.7.0"
__author__ = "DESHIMA software team"


# submodules
from . import array
from . import cube
from . import io
from . import models
from . import logging
from . import plot
from . import utils


# aliases
from .array import *
from .cube import *
from .logging import *


# for sphinx docs
__all__ = dir()
