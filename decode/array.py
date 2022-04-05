__all__ = [
    "array",
    "ones",
    "zeros",
    "full",
    "empty",
    "ones_like",
    "zeros_like",
    "full_like",
    "empty_like",
    "concat",
]


# standard library
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Tuple


# dependencies
import numpy as np
import xarray as xr
from typing_extensions import Literal
from xarray_dataclasses import AsDataArray, Coord, Data


# type hints
_ = Tuple[()]
Ti = Literal["t"]
Ch = Literal["ch"]


# module logger
logger = getLogger(__name__)


# runtime classes
@dataclass(frozen=True)
class Array(AsDataArray):
    """Specification for de:code arrays."""

    data: Data[Tuple[Ti, Ch], Any]
    vrad: Coord[Ti, float] = 0.0
    x: Coord[Ti, float] = 0.0
    y: Coord[Ti, float] = 0.0
    time: Coord[Ti, float] = 0.0
    temp: Coord[Ti, float] = 0.0
    pressure: Coord[Ti, float] = 0.0
    vapor_pressure: Coord[Ti, float] = 0.0
    windspd: Coord[Ti, float] = 0.0
    winddir: Coord[Ti, float] = 0.0
    scantype: Coord[Ti, str] = "GRAD"
    scanid: Coord[Ti, int] = 0
    masterid: Coord[Ch, int] = 0
    kidid: Coord[Ch, int] = 0
    kidfq: Coord[Ch, float] = 0.0
    kidtp: Coord[Ch, int] = 0
    weight: Coord[Tuple[Ti, Ch], float] = 1.0
    coordsys: Coord[_, str] = "RADEC"
    datatype: Coord[_, str] = "Temperature"
    xref: Coord[_, float] = 0.0
    yref: Coord[_, float] = 0.0
    type: Coord[_, str] = "dca"


@xr.register_dataarray_accessor("dca")
@dataclass(frozen=True)
class ArrayAccessor:
    """Accessor for de:code arrays."""

    array: xr.DataArray

    @property
    def tcoords(self):
        """Dictionary of arrays that label time axis."""
        return {k: v.values for k, v in self.array.coords.items() if v.dims == ("t",)}

    @property
    def chcoords(self):
        """Dictionary of arrays that label channel axis."""
        return {k: v.values for k, v in self.array.coords.items() if v.dims == ("ch",)}

    @property
    def datacoords(self):
        """Dictionary of arrays that label time and channel axis."""
        return {
            k: v.values for k, v in self.array.coords.items() if v.dims == ("t", "ch")
        }

    @property
    def scalarcoords(self):
        """Dictionary of values that don't label any axes (point-like)."""
        return {k: v.values for k, v in self.array.coords.items() if v.dims == ()}

    def __setstate__(self, state):
        """A method used for pickling."""
        self.__dict__ = state

    def __getstate__(self):
        """A method used for unpickling."""
        return self.__dict__


# runtime functions
def array(
    data,
    tcoords=None,
    chcoords=None,
    scalarcoords=None,
    datacoords=None,
    attrs=None,
    name=None,
):
    """Create an array as an instance of xarray.DataArray with Decode accessor.

    Args:
        data (numpy.ndarray): 2D (time x channel) array.
        tcoords (dict, optional): Dictionary of arrays that label time axis.
        chcoords (dict, optional): Dictionary of arrays that label channel axis.
        scalarcoords (dict, optional): Dictionary of values
            that don't label any axes (point-like).
        datacoords (dict, optional): Dictionary of arrays
            that label time and channel axes.
        attrs (dict, optional): Dictionary of attributes to add to the instance.
        name (str, optional): String that names the instance.

    Returns:
        array (decode.array): Decode array.
    """
    # initialize coords with default values
    array = Array.new(data)

    # update coords with input values (if any)
    if tcoords is not None:
        array.coords.update({k: ("t", v) for k, v in tcoords.items()})

    if chcoords is not None:
        array.coords.update({k: ("ch", v) for k, v in chcoords.items()})

    if datacoords is not None:
        array.coords.update({k: (("t", "ch"), v) for k, v in datacoords.items()})

    if scalarcoords is not None:
        array.coords.update({k: ((), v) for k, v in scalarcoords.items()})

    if attrs is not None:
        array.attrs.update(attrs)

    if name is not None:
        array.name = name

    return array


def zeros(shape, dtype=None, **kwargs):
    """Create an array of given shape and type, filled with zeros.

    Args:
        shape (sequence of ints): 2D shape of the array.
        dtype (data-type, optional): Desired data-type for the array.
        kwargs (optional): Other arguments of the array (*coords, attrs, and name).

    Returns:
        array (decode.array): Decode array filled with zeros.
    """
    data = np.zeros(shape, dtype)
    return array(data, **kwargs)


def ones(shape, dtype=None, **kwargs):
    """Create an array of given shape and type, filled with ones.

    Args:
        shape (sequence of ints): 2D shape of the array.
        dtype (data-type, optional): Desired data-type for the array.
        kwargs (optional): Other arguments of the array (*coords, attrs, and name).

    Returns:
        array (decode.array): Decode array filled with ones.
    """
    data = np.ones(shape, dtype)
    return array(data, **kwargs)


def full(shape, fill_value, dtype=None, **kwargs):
    """Create an array of given shape and type, filled with `fill_value`.

    Args:
        shape (sequence of ints): 2D shape of the array.
        fill_value (scalar or numpy.ndarray): Fill value or array.
        dtype (data-type, optional): Desired data-type for the array.
        kwargs (optional): Other arguments of the array (*coords, attrs, and name).

    Returns:
        array (decode.array): Decode array filled with `fill_value`.
    """
    return (zeros(shape, **kwargs) + fill_value).astype(dtype)


def empty(shape, dtype=None, **kwargs):
    """Create an array of given shape and type, without initializing entries.

    Args:
        shape (sequence of ints): 2D shape of the array.
        dtype (data-type, optional): Desired data-type for the array.
        kwargs (optional): Other arguments of the array (*coords, attrs, and name).

    Returns:
        array (decode.array): Decode array without initializing entries.
    """
    data = np.empty(shape, dtype)
    return array(data, **kwargs)


def zeros_like(array, dtype=None, keepmeta=True):
    """Create an array of zeros with the same shape and type as the input array.

    Args:
        array (xarray.DataArray): The shape and data-type of it define
            these same attributes of the output array.
        dtype (data-type, optional): If specified, this function overrides
            the data-type of the output array.
        keepmeta (bool, optional): Whether *coords, attrs, and name of the input
            array are kept in the output one. Default is True.

    Returns:
        array (decode.array): Decode array filled with zeros.
    """
    if keepmeta:
        return xr.zeros_like(array, dtype)
    else:
        return zeros(array.shape, dtype)


def ones_like(array, dtype=None, keepmeta=True):
    """Create an array of ones with the same shape and type as the input array.

    Args:
        array (xarray.DataArray): The shape and data-type of it define
            these same attributes of the output array.
        dtype (data-type, optional): If spacified, this function overrides
            the data-type of the output array.
        keepmeta (bool, optional): Whether *coords, attrs, and name of the input
            array are kept in the output one. Default is True.

    Returns:
        array (decode.array): Decode array filled with ones.
    """
    if keepmeta:
        return xr.ones_like(array, dtype)
    else:
        return ones(array.shape, dtype)


def full_like(array, fill_value, reverse=False, dtype=None, keepmeta=True):
    """Create an array of `fill_value` with the same shape and type as the input array.

    Args:
        array (xarray.DataArray): The shape and data-type of it define
            these same attributes of the output array.
        fill_value (scalar or numpy.ndarray): Fill value or array.
        dtype (data-type, optional): If spacified, this function overrides
            the data-type of the output array.
        keepmeta (bool, optional): Whether *coords, attrs, and name of the input
            array are kept in the output one. Default is True.

    Returns:
        array (decode.array): Decode array filled with `fill_value`.
    """
    if keepmeta:
        return (zeros_like(array) + fill_value).astype(dtype)
    else:
        return full(array.shape, fill_value, dtype)


def empty_like(array, dtype=None, keepmeta=True):
    """Create an array of empty with the same shape and type as the input array.

    Args:
        array (xarray.DataArray): The shape and data-type of it define
            these same attributes of the output array.
        dtype (data-type, optional): If spacified, this function overrides
            the data-type of the output array.
        keepmeta (bool, optional): Whether *coords, attrs, and name of the input
            array are kept in the output one. Default is True.

    Returns:
        array (decode.array): Decode array without initializing entries.
    """
    if keepmeta:
        return empty(
            array.shape,
            dtype,
            tcoords=array.dca.tcoords,
            chcoords=array.dca.chcoords,
            scalarcoords=array.dca.scalarcoords,
            attrs=array.attrs,
            name=array.name,
        )
    else:
        return empty(array.shape, dtype)


def concat(objs, dim=None, **kwargs):
    xref = objs[0].xref.values
    yref = objs[0].yref.values
    for obj in objs:
        obj.coords.update({"xref": xref, "yref": yref})
    return xr.concat(objs, dim=dim, **kwargs)
