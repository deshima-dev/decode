# coding: utf-8

# information
__version__ = '0.1'
__author__  = 'snoopython'
__email__   = 'taniguchi@ioa.s.u-tokyo.ac.jp'

from .array import *
from .logging import *
from . import utils

# default logger
import logging
logger = logging.getLogger('decode')
setlogfile(logger=logger)
setloglevel(logger=logger)
del logging
