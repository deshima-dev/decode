__all__ = ["state"]


# standard library
from typing import Any, Literal, Optional


# dependencies
import xarray as xr
from matplotlib.artist import Artist


def state(
    dems: xr.DataArray,
    *,
    on: Literal["time", "sky"] = "time",
    **options: Any,
) -> Artist:
    """Plot the state coordinate of DEMS.

    Args:
        dems: DEMS DataArray to be plotted.

    Keyword Args:
        on: On which plane the state coordinate is plotted.
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
