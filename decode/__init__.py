# flake8: noqa
__version__ = "0.6.0"
__author__ = "DESHIMA software team"


# subpackages
from . import core
from . import utils


# submodules
from . import io
from . import models
from . import logging
from . import plot


# aliases
from .core import *
from .logging import *


# for sphinx docs
__all__ = dir()
