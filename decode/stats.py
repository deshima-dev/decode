__all__ = [
    "all",
    "any",
    "apply",
    "count",
    "first",
    "last",
    "max",
    "mean",
    "median",
    "min",
    "prod",
    "std",
    "sum",
    "var",
]


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

    if numeric_coord_func == "first":
        numeric_coord_func = _first
    elif numeric_coord_func == "last":
        numeric_coord_func = _last

    if nonnumeric_coord_func == "first":
        nonnumeric_coord_func = _first
    elif nonnumeric_coord_func == "last":
        nonnumeric_coord_func = _last

    for name, coord in da.coords.items():
        if coord.dtype.kind in NUMERIC_KINDS:
            coord_func[name] = numeric_coord_func
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
            keep_attrs=keep_attrs,
            **options,
        ).squeeze()

    if callable(func):
        return coarsened.reduce(
            func=func,
            keep_attrs=keep_attrs,
            **options,
        ).squeeze()

    raise TypeError("Func must be either callable or string.")


def all(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``all`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``all``  operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``all``  operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``all``  operation.
        **options: Other options to be passed to the ``all``  operation.

    Returns:
        DataArray that the (chunked) ``all``  operation is applied.

    """
    return apply(
        da,
        "all",
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        keep_attrs=keep_attrs,
        **options,
    )


def any(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``any`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``any`` operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``any`` operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``any`` operation.
        **options: Other options to be passed to the ``any`` operation.

    Returns:
        DataArray that the (chunked) ``any`` operation is applied.

    """
    return apply(
        da,
        "any",
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        keep_attrs=keep_attrs,
        **options,
    )


def count(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``count`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``count`` operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``count`` operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``count`` operation.
        **options: Other options to be passed to the ``count`` operation.

    Returns:
        DataArray that the (chunked) ``count`` operation is applied.

    """
    return apply(
        da,
        "count",
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        keep_attrs=keep_attrs,
        **options,
    )


def first(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``first`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``first`` operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``first`` operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``first`` operation.
        **options: Other options to be passed to the ``first`` operation.

    Returns:
        DataArray that the (chunked) ``first`` operation is applied.

    """
    return apply(
        da,
        _first,
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        keep_attrs=keep_attrs,
        **options,
    )


def last(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``last`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``last`` operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``last`` operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``last`` operation.
        **options: Other options to be passed to the ``last`` operation.

    Returns:
        DataArray that the (chunked) ``last`` operation is applied.

    """
    return apply(
        da,
        _last,
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        keep_attrs=keep_attrs,
        **options,
    )


def max(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    skipna: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``max`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``max`` operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``max`` operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``max`` operation.
        skipna: Whether to ignore missing values in the ``max`` operation.
        **options: Other options to be passed to the ``max`` operation.

    Returns:
        DataArray that the (chunked) ``max`` operation is applied.

    """
    return apply(
        da,
        "max",
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        keep_attrs=keep_attrs,
        skipna=skipna,
        **options,
    )


def mean(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    skipna: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``mean`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``mean`` operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``mean`` operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``mean`` operation.
        skipna: Whether to ignore missing values in the ``mean`` operation.
        **options: Other options to be passed to the ``mean`` operation.

    Returns:
        DataArray that the (chunked) ``mean`` operation is applied.

    """
    return apply(
        da,
        "mean",
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        keep_attrs=keep_attrs,
        skipna=skipna,
        **options,
    )


def median(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    skipna: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``median`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``median`` operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``median`` operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``median`` operation.
        skipna: Whether to ignore missing values in the ``median`` operation.
        **options: Other options to be passed to the ``median`` operation.

    Returns:
        DataArray that the (chunked) ``median`` operation is applied.

    """
    return apply(
        da,
        "median",
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        keep_attrs=keep_attrs,
        skipna=skipna,
        **options,
    )


def min(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    skipna: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``min`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``min`` operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``min`` operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``min`` operation.
        skipna: Whether to ignore missing values in the ``min`` operation.
        **options: Other options to be passed to the ``min`` operation.

    Returns:
        DataArray that the (chunked) ``min`` operation is applied.

    """
    return apply(
        da,
        "min",
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        keep_attrs=keep_attrs,
        skipna=skipna,
        **options,
    )


def prod(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    skipna: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``prod`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``prod`` operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``prod`` operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``prod`` operation.
        skipna: Whether to ignore missing values in the ``prod`` operation.
        **options: Other options to be passed to the ``prod`` operation.

    Returns:
        DataArray that the (chunked) ``prod`` operation is applied.

    """
    return apply(
        da,
        "prod",
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        keep_attrs=keep_attrs,
        skipna=skipna,
        **options,
    )


def std(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    skipna: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``std`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``std`` operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``std`` operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``std`` operation.
        skipna: Whether to ignore missing values in the ``std`` operation.
        **options: Other options to be passed to the ``std`` operation.

    Returns:
        DataArray that the (chunked) ``std`` operation is applied.

    """
    return apply(
        da,
        "std",
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        keep_attrs=keep_attrs,
        skipna=skipna,
        **options,
    )


def sum(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    skipna: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``sum`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``sum`` operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``sum`` operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``sum`` operation.
        skipna: Whether to ignore missing values in the ``sum`` operation.
        **options: Other options to be passed to the ``sum`` operation.

    Returns:
        DataArray that the (chunked) ``sum`` operation is applied.

    """
    return apply(
        da,
        "sum",
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        keep_attrs=keep_attrs,
        skipna=skipna,
        **options,
    )


def var(
    da: xr.DataArray,
    /,
    *,
    dim: Union[Dims, dict[Hashable, int]] = None,
    boundary: Boundary = "trim",
    side: Union[Side, dict[Hashable, Side]] = "left",
    numeric_coord_func: Stat = "mean",
    nonnumeric_coord_func: Stat = "first",
    keep_attrs: Optional[bool] = None,
    skipna: Optional[bool] = None,
    **options: Any,
) -> xr.DataArray:
    """Apply a (chunked) ``var`` operation to a DataArray.

    Args:
        da: Input DataArray.
        dim: Name(s) of the dimension(s) along which the ``var`` operation
            will be applied. If a dictionary such as ``{dim: size, ...}``
            is specified, then the ``var`` operation will be applied
            to every data chunk of given size.
        boundary: Same option as ``xarray.DataArray.coarsen`` but defaults to ``'trim'``.
        side: Same option as ``xarray.DataArray.coarsen`` and defualts to ``'left'``.
        numeric_coord_func: Function or name of the statistical operation
            for the numeric coordinates (bool, numbers, datetime, timedelta).
        nonnumeric_coord_func: Function or name of the statistical operation
            for the non-numeric coordinates (str, bytes, and general object).
        keep_attrs: Whether to keep attributes in the ``var`` operation.
        skipna: Whether to ignore missing values in the ``var`` operation.
        **options: Other options to be passed to the ``var`` operation.

    Returns:
        DataArray that the (chunked) ``var`` operation is applied.

    """
    return apply(
        da,
        "var",
        dim=dim,
        boundary=boundary,
        side=side,
        numeric_coord_func=numeric_coord_func,
        nonnumeric_coord_func=nonnumeric_coord_func,
        skipna=skipna,
        keep_attrs=keep_attrs,
        **options,
    )


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
