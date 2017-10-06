# coding: utf-8

# public items
__all__ = []

# standard library
from collections import OrderedDict
from datetime import datetime

# dependent packages
import decode as dc
import numpy as np
import xarray as xr

# local constants
TCOORDS = lambda array: OrderedDict([
    ('vrad', ('t', np.zeros(array.shape[0], dtype=float))),
    ('x',    ('t', np.zeros(array.shape[0], dtype=float))),
    ('y',    ('t', np.zeros(array.shape[0], dtype=float))),
    ('time', ('t', np.full(array.shape[0], datetime(2000,1,1)))),
])

CHCOORDS = lambda array: OrderedDict([
    ('kidid', ('ch', np.zeros(array.shape[1], dtype=float))),
    ('kidfq', ('ch', np.zeros(array.shape[1], dtype=float))),
])

PTCOORDS = OrderedDict([
    ('coordsys', 'RADEC'),
    ('xref', 0.0),
    ('yref', 0.0),
])


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
    def tcoords(self):
        """A dictionary of arrays that label time axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('t',)}

    @property
    def chcoords(self):
        """A dictionary of arrays that label channel axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('ch',)}

    @property
    def ptcoords(self):
        """A dictionary of values that don't label any axes (point-like)."""
        return {k: v.values for k, v in self.coords.items() if v.dims==()}


@xr.register_dataarray_accessor('dca')
class DecodeArrayAccessor(BaseAccessor):
    def __init__(self, array):
        """Initialize the Decode accessor of an array.

        Note:
            This method is only for the internal use.
            Users can create an array with Decode accessor using dc.array.

        Args:
            array (xarray.DataArray): An array to which Decode accessor is added.

        """
        super().__init__(array)

    def _initcoords(self):
        """Initialize coords with default values.

        Warning:
            Do not use this method after an array is created.
            This forcibly replaces all vaules of coords with default ones.

        """
        self.coords.update(TCOORDS(self))
        self.coords.update(CHCOORDS(self))
        self.coords.update(PTCOORDS)
