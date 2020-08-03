# coding: utf-8


# public items
__all__ = []


# standard library
from collections import OrderedDict
from logging import getLogger


# dependent packages
import numpy as np
import xarray as xr
from .. import BaseAccessor


# module logger
logger = getLogger(__name__)


# local constants
def XCOORDS(array):
    return OrderedDict([("x", ("x", np.zeros(array.shape[0], dtype=float)))])


def YCOORDS(array):
    return OrderedDict([("y", ("y", np.zeros(array.shape[1], dtype=float)))])


def CHCOORDS(array):
    return OrderedDict(
        [
            ("masterid", ("ch", np.zeros(array.shape[2], dtype=int))),
            ("kidid", ("ch", np.zeros(array.shape[2], dtype=int))),
            ("kidfq", ("ch", np.zeros(array.shape[2], dtype=float))),
            ("kidtp", ("ch", np.zeros(array.shape[2], dtype=int))),
        ]
    )


def DATACOORDS(array):
    return OrderedDict(
        [("noise", (("x", "y", "ch"), np.ones(array.shape, dtype=float)))]
    )


SCALARCOORDS = OrderedDict(
    [
        ("coordsys", "RADEC"),
        ("datatype", "temperature"),
        ("xref", 0.0),
        ("yref", 0.0),
        ("type", "dcc"),
    ]
)


# classes
@xr.register_dataarray_accessor("dcc")
class DecodeCubeAccessor(BaseAccessor):
    def __init__(self, array):
        """Initialize the Decode accessor of an array.

        Note:
            This method is only for the internal use.
            Users can create an array with Decode accessor using dc.array.

        Args:
            array (xarray.DataArray): Array to which Decode accessor is added.
        """
        super().__init__(array)

    def _initcoords(self):
        """Initialize coords with default values.

        Warning:
            Do not use this method after an array is created.
            This forcibly replaces all vaules of coords with default ones.
        """
        self.coords.update(XCOORDS(self))
        self.coords.update(YCOORDS(self))
        self.coords.update(CHCOORDS(self))
        self.coords.update(DATACOORDS(self))
        self.coords.update(SCALARCOORDS)

    @property
    def xcoords(self):
        """Dictionary of arrays that label x axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims == ("x",)}

    @property
    def ycoords(self):
        """Dictionary of arrays that label y axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims == ("y",)}

    @property
    def chcoords(self):
        """Dictionary of arrays that label channel axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims == ("ch",)}

    @property
    def datacoords(self):
        """Dictionary of arrays that label x, y, and channel axis."""
        return {
            k: v.values for k, v in self.coords.items() if v.dims == ("x", "y", "ch")
        }
