__all__ = ["apply"]


# standard library
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any, Literal, Optional, Union


# dependencies
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray.core.types import Dims


# constants
FIRST_INDEX = 0
LAST_INDEX = -1
NUMERIC_KINDS = "biufcmM"


# type hints
Boundary = Literal["exact", "trim", "pad"]
Side = Literal["left", "right"]
Stat = Union[Callable[..., Any], str]


def apply(
    da: xr.DataArray,
    func: Stat,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    skipna: Optional[bool] = None,
    keep_attrs: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) statistical operation to a DataArray.

    Args:
        da: Input DataArray.
        func: Function or name of the statistical operation (e.g. ``'mean'``).
        dim: Name(s) of the dimension(s) along which the statistical operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the statistical operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        skipna: Whether to ignore missing values in the statistical operation.
        keep_attrs: Whether to keep attributes in the statistical operation.
        **options: Other options to be passed to the statistical operation.

    Returns:
        DataArray that the (chunked) statistical operation is applied.

    """
    if isinstance(dim, dict):
        pass
    elif dim is ... or dim is None:
        dim = da.sizes
    elif isinstance(dim, str):
        dim = {dim: da.sizes[dim]}
    elif isinstance(dim, Iterable):
        dim = {d: da.sizes[d] for d in dim}

    coord_func: dict[Hashable, Stat] = {}

    for name, coord in da.coords.items():
        if coord.dtype.kind in NUMERIC_KINDS:
            coord_func[name] = numeric_coord_func
        else:
            if nonnumeric_coord_func == "first":
                coord_func[name] = _first
            elif nonnumeric_coord_func == "last":
                coord_func[name] = _last
            else:
                coord_func[name] = nonnumeric_coord_func

    coarsened = da.coarsen(
        dim,
        boundary=boundary,
        coord_func=coord_func,
        side=side,
    )

    if isinstance(func, str):
        return getattr(coarsened, func)(
            skipna=skipna,
            keep_attrs=keep_attrs,
            **options,
        )

    if callable(func):
        return coarsened.reduce(
            func=func,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **options,
        )

    raise TypeError("Func must be either callable or string.")


def _first(
    array: NDArray[Any],
    axis: Union[Sequence[int], int, None],
    **kwargs: Any,
) -> NDArray[Any]:
    """Similar to numpy.take(array, 0, axis) but supports multiple axes."""
    if not isinstance(axis, Sequence):
        return np.take(array, FIRST_INDEX, axis=axis)

    slices: list[Union[slice, int]] = [slice(None)] * array.ndim

    for ax in axis:
        slices[ax] = FIRST_INDEX

    return array[tuple(slices)]


def _last(
    array: NDArray[Any],
    axis: Union[Sequence[int], int, None],
    **kwargs: Any,
) -> NDArray[Any]:
    """Similar to numpy.take(array, -1, axis) but supports multiple axes."""
    if not isinstance(axis, Sequence):
        return np.take(array, LAST_INDEX, axis=axis)

    slices: list[Union[slice, int]] = [slice(None)] * array.ndim

    for ax in axis:
        slices[ax] = LAST_INDEX

    return array[tuple(slices)]
