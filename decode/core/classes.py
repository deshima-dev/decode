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
TCOORDS = lambda size=0: OrderedDict([
    ('vrad', ('t', np.zeros(size, dtype=float))),
    ('xrel', ('t', np.zeros(size, dtype=float))),
    ('yrel', ('t', np.zeros(size, dtype=float))),
    ('time', ('t', np.tile(datetime(2000,1,1), size))),
])

CHCOORDS = lambda size=0: OrderedDict([
    ('kidid', ('ch', np.zeros(size, dtype=int))),
    ('kidfq', ('ch', np.zeros(size, dtype=float))),
])

PTCOORDS = OrderedDict([
    ('xref', 0.0),
    ('yref', 0.0),
])


# classes
@xr.register_dataarray_accessor('dc')
class DecodeAccessor(object):
    def __init__(self, array):
        """Initialize the Decode accessor of an array.

        Note:
            This method is only for the internal use.
            Users can create an array with Decode accessor using dc.array.

        Args:
            array (xarray.DataArray): An array to which Decode accessor is added.

        """
        self._array = array

    @property
    def tcoords(self):
        """A dictionary of arrays that label time axis."""
        return {key: getattr(self, key).values for key in TCOORDS()}

    @property
    def chcoords(self):
        """A dictionary of arrays that label channel axis."""
        return {key: getattr(self, key).values for key in CHCOORDS()}

    @property
    def ptcoords(self):
        """A dictionary of values that don't label any axes (point-like)."""
        return {key: getattr(self, key).item() for key in PTCOORDS}

    def _initcoords(self):
        """Initialize coords with default values.

        Warning:
            Do not use this method after an array is created.
            This forcibly replaces all vaules of coords with default ones.

        """
        self.coords.update(TCOORDS(self.shape[0]))
        self.coords.update(CHCOORDS(self.shape[1]))
        self.coords.update(PTCOORDS)

    def __getattr__(self, name):
        """array.dc.name <=> array.name"""
        return getattr(self._array, name)
