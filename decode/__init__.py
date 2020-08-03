__version__ = '0.5.6'
__author__ = 'DESHIMA software team'


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
