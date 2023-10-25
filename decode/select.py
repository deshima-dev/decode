__all__ = ["by"]


# standard library
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Optional, TypeVar, Union


# dependencies
import xarray as xr


# type hints
T = TypeVar("T", bool, int, float, str, datetime, timedelta)
Multiple = Union[Sequence[T], T]


def by(
    dems: xr.DataArray,
    coord_name: str,
    /,
    *,
    min: Optional[T] = None,
    max: Optional[T] = None,
    include: Optional[Multiple[T]] = None,
    exclude: Optional[Multiple[T]] = None,
    sort: bool = False,
    as_dim: bool = False,
) -> xr.DataArray:
    """Select DEMS by values of a coordinate.

    Args:
        dems: DEMS DataArray to be selected.
        coord_name: Name of the coordinate for the selection.
        min: Minimum selection bound (inclusive).
            If not specified, no bound is set.
        max: Maximum selection bound (exclusive).
            If not specified, no bound is set.
        include: Coordinate values to be included.
            If not specified, all values are included.
        exclude: Coordinate values to be excluded.
            If not specified, any values are not excluded.
        sort: Whether to sort by the coordinate after selection.
        as_dim: Whether to use the coordinate as a dimension.

    Returns:
        Selected DEMS.

    """
    coord = dems[coord_name]

    if not isinstance(coord, xr.DataArray):
        raise TypeError("Coordinate must be DataArray.")

    if not coord.ndim == 1:
        raise ValueError("Coordinate must be one-dimensional.")

    coord_dim = coord.dims[0]

    if min is not None:
        dems = dems.sel({coord_dim: dems[coord_name] >= min})

    if max is not None:
        dems = dems.sel({coord_dim: dems[coord_name] < max})

    if include is not None:
        dems = dems.sel({coord_dim: dems[coord_name].isin(include)})

    if exclude is not None:
        dems = dems.sel({coord_dim: ~dems[coord_name].isin(exclude)})

    if sort:
        dems = dems.sortby(coord_name)

    if as_dim:
        dems = dems.swap_dims({coord_dim: coord_name})

    return dems
