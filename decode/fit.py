__all__ = []


# standard library
from collections.abc import Sequence


# dependencies
import xarray as xr
from . import load


def dtau_dpwv(freq: Sequence[float]) -> xr.DataArray:
    """Calculate dtau/dpwv as a function of frequency.

    Args:
        freq: Frequency in units of Hz.

    Returns:
        DataArray that stores dtau/dpwv.

    """
    tau = load.atm(type="tau").interp(freq=freq, method="linear")
    fit = tau.curvefit("pwv", lambda x, a, b: a * x + b)
    return fit["curvefit_coefficients"].sel(param="a", drop=True)
