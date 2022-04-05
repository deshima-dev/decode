# flake8: noqa
__version__ = "0.6.1"
__author__ = "DESHIMA software team"


# subpackages
from . import core


# submodules
from . import cube
from . import io
from . import models
from . import logging
from . import plot
from . import utils


# aliases
from .cube import *
from .core import *
from .logging import *


# for sphinx docs
__all__ = dir()
