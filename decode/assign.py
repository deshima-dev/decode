__all__ = ["scan"]


# standard library
from typing import Literal, Optional


# dependencies
import numpy as np
import xarray as xr


def scan(
    dems: xr.DataArray,
    /,
    *,
    by: Literal["beam", "scan", "state"] = "state",
    dt: Optional[np.timedelta64] = None,
    inplace: bool = False,
) -> xr.DataArray:
    """Assign scan labels (scan coordinate) to DEMS.

    Args:
        dems: Input DEMS DataArray to be assigned.
        by: By what coordinate the scan labels are inferred.
        dt: Minimum time difference to assign different scan labels
            even if adjacent coordinate values are the same.
        inplace: Whether the scan labels are assigned in-place.
            If False, they are assigned to the copy of the input.

    Returns:
        DEMS DataArray to which the scan label are assigned.

    """
    if not inplace:
        # deepcopy except for data
        dems = dems.copy(data=dems.data)

    is_div = xr.zeros_like(dems.scan, dtype=bool)

    ref = dems.coords[by].data
    is_div[1:] |= ref[1:] != ref[:-1]

    if dt is not None:
        is_div[1:] |= np.diff(dems.time) >= dt

    new_scan = is_div.cumsum().astype(dems.scan.dtype)
    return dems.assign_coords(scan=new_scan)
