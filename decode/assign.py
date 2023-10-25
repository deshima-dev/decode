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
    by: Literal["state"] = "state",
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
        dems = dems.copy()

    is_cut = np.zeros_like(dems.scan, dtype=bool)

    if by == "state":
        state = dems.state.values
        is_cut |= np.hstack([False, state[1:] != state[:-1]])

    if dt is not None:
        is_cut |= np.hstack([False, np.diff(dems.time) >= dt])

    dems.scan.values[:] = np.cumsum(is_cut)
    return dems
