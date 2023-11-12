__all__ = ["mad"]


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
    """Calculate median absolute deviation (MAD) of a DataArray.

    Args:
        da: Input DataArray.
        dim: Name of dimension(s) along which MAD is calculated.
        skipna: Same-name option to be passed to ``DataArray.median``.
        keep_attrs: Same-name option to be passed to ``DataArray.median``.
        kwargs: Same-name option(s) to be passed to ``DataArray.median``.

    Returns:
        MAD of the input DataArray.

    """

    def median(da: xr.DataArray) -> xr.DataArray:
        return da.median(
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    return median(cast(xr.DataArray, np.abs(da - median(da))))
