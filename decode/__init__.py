# flake8: noqa
__version__ = "0.6.1"
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
