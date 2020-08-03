# coding: utf-8
# flake8: noqa


# classes
class BaseAccessor(object):
    def __init__(self, dataarray):
        """Initialize the base accessor."""
        self._dataarray = dataarray

    def __getattr__(self, name):
        """self._dataarray.name <=> self.name."""
        return getattr(self._dataarray, name)

    def __setstate__(self, state):
        """A method used for pickling."""
        self.__dict__ = state

    def __getstate__(self):
        """A method used for unpickling."""
        return self.__dict__

    @property
    def scalarcoords(self):
        """A dictionary of values that don't label any axes (point-like)."""
        return {k: v.values for k, v in self.coords.items() if v.dims == ()}


# dependent packages
from .array.classes import *
from .array.decorators import *
from .array.functions import *
from .cube.classes import *
from .cube.functions import *
