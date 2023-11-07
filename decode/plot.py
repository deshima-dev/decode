__all__ = ["data", "state"]


# standard library
from typing import Any, Literal


# dependencies
import xarray as xr
from dems.d2 import MS_DIMS as DEMS_DIMS
from matplotlib.artist import Artist


def data(
    data: xr.DataArray,
    /,
    *,
    squeeze: bool = True,
    **options: Any,
) -> Artist:
    """Plot 1D or 2D data of DEMS or its coordinate.

    Args:
        data: DEMS DataArray or coordinate DataArray.
        squeeze: Whether to squeeze the data before plotting.

    Keyword Args:
        options: Plotting options to be passed to Matplotlib.

    Returns:
        Matplotlib artist object of the plotted data.

    """
    if squeeze:
        data = data.squeeze()

    if data.dims == (DEMS_DIMS[0],):
        options = {
            "edgecolors": "none",
            "hue": "state",
            "s": 3,
            "x": DEMS_DIMS[0],
            **options,
        }
        return data.plot.scatter(**options)  # type: ignore

    if data.dims == (DEMS_DIMS[1],):
        options = {
            "edgecolors": "none",
            "hue": "state",
            "s": 3,
            "x": DEMS_DIMS[1],
            **options,
        }
        return data.plot.scatter(**options)  # type: ignore

    if data.dims == DEMS_DIMS:
        return data.plot.pcolormesh(**options)  # type: ignore

    raise ValueError(f"Dimensions must be subset of {DEMS_DIMS}.")


def state(
    dems: xr.DataArray,
    /,
    *,
    on: Literal["time", "sky"] = "time",
    **options: Any,
) -> Artist:
    """Plot the state coordinate of DEMS.

    Args:
        dems: DEMS DataArray to be plotted.
        on: On which plane the state coordinate is plotted.

    Keyword Args:
        options: Plotting options to be passed to Matplotlib.

    Returns:
        Matplotlib artist object of the plotted data.

    """
    if on == "time":
        options = {
            "edgecolors": "none",
            "hue": "state",
            "s": 3,
            "x": "time",
            **options,
        }
        return dems.state.sortby("state").plot.scatter(**options)

    if on == "sky":
        options = {
            "edgecolors": "none",
            "hue": "state",
            "s": 3,
            "x": "lon",
            **options,
        }
        return dems.lat.plot.scatter(**options)

    raise ValueError("On must be either time or sky.")
