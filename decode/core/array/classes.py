# coding: utf-8


# public items
__all__ = []


# standard library
from collections import OrderedDict


# dependent packages
import numpy as np
import xarray as xr
from .. import BaseAccessor


# local constants
def TCOORDS(array):
    return OrderedDict(
        [
            ("vrad", ("t", np.zeros(array.shape[0], dtype=float))),
            ("x", ("t", np.zeros(array.shape[0], dtype=float))),
            ("y", ("t", np.zeros(array.shape[0], dtype=float))),
            ("time", ("t", np.zeros(array.shape[0], dtype=float))),
            ("temp", ("t", np.zeros(array.shape[0], dtype=float))),
            ("pressure", ("t", np.zeros(array.shape[0], dtype=float))),
            ("vapor-pressure", ("t", np.zeros(array.shape[0], dtype=float))),
            ("windspd", ("t", np.zeros(array.shape[0], dtype=float))),
            ("winddir", ("t", np.zeros(array.shape[0], dtype=float))),
            ("scantype", ("t", np.full(array.shape[0], "GRAD", dtype="U4"))),
            ("scanid", ("t", np.zeros(array.shape[0], dtype=int))),
        ]
    )


def CHCOORDS(array):
    return OrderedDict(
        [
            ("masterid", ("ch", np.zeros(array.shape[1], dtype=int))),
            ("kidid", ("ch", np.zeros(array.shape[1], dtype=int))),
            ("kidfq", ("ch", np.zeros(array.shape[1], dtype=float))),
            ("kidtp", ("ch", np.zeros(array.shape[1], dtype=int))),
        ]
    )


def DATACOORDS(array):
    return OrderedDict([("weight", (("t", "ch"), np.ones(array.shape, dtype=float)))])


SCALARCOORDS = OrderedDict(
    [
        ("coordsys", "RADEC"),
        ("datatype", "Temperature"),
        ("xref", 0.0),
        ("yref", 0.0),
        ("type", "dca"),
    ]
)


@xr.register_dataarray_accessor("dca")
class DecodeArrayAccessor(BaseAccessor):
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
        self.coords.update(TCOORDS(self))
        self.coords.update(CHCOORDS(self))
        self.coords.update(DATACOORDS(self))
        self.coords.update(SCALARCOORDS)

    @property
    def tcoords(self):
        """Dictionary of arrays that label time axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims == ("t",)}

    @property
    def chcoords(self):
        """Dictionary of arrays that label channel axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims == ("ch",)}

    @property
    def datacoords(self):
        """Dictionary of arrays that label time and channel axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims == ("t", "ch")}
