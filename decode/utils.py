__all__ = ["mad", "phaseof"]


# dependencies
from typing import Any, Optional, cast
import numpy as np
import xarray as xr
from xarray.core.types import Dims


def mad(
    da: xr.DataArray,
    dim: Dims = None,
    skipna: Optional[bool] = None,
    keep_attrs: Optional[bool] = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Calculate the median absolute deviation (MAD) of a DataArray.

    Args:
        da: Input DataArray.
        dim: Name of dimension(s) along which the MAD is calculated.
        skipna: Same-name option to be passed to ``DataArray.median``.
        keep_attrs: Same-name option to be passed to ``DataArray.median``.
        kwargs: Same-name option(s) to be passed to ``DataArray.median``.

    Returns:
        The MAD of the input DataArray.

    """

    def median(da: xr.DataArray) -> xr.DataArray:
        return da.median(
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    return median(cast(xr.DataArray, np.abs(da - median(da))))


def phaseof(
    da: xr.DataArray,
    /,
    *,
    keep_attrs: bool = False,
    keep_coords: bool = False,
) -> xr.DataArray:
    """Assign a phase to each value in a 1D DataArray.

    The function assigns a unique phase (int64) to consecutive
    identical values in the DataArray. The phase increases
    sequentially whenever the value in the DataArray changes.

    Args:
        da: Input 1D DataArray.
        keep_attrs: Whether to keep attributes of the input.
        keep_coords: Whether to keep coordinates of the input.

    Returns:
        1D int64 DataArray of phases.

    """
    if da.ndim != 1:
        raise ValueError("Input DataArray must be 1D.")

    is_transision = xr.zeros_like(da, bool)
    is_transision.data[1:] = da.data[1:] != da.data[:-1]

    phase = is_transision.cumsum(keep_attrs=keep_attrs)
    return phase if keep_coords else phase.reset_coords(drop=True)
