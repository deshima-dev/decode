__version__ = '0.5.4'

from . import utils
from .core import *
from . import io
from . import plot
from .logging import *
from . import models
del core
del logging

# for sphinx docs
__all__ = dir()
