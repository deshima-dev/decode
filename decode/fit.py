__all__ = ["baseline"]


# standard library
from typing import Any, Optional, Union


# dependencies
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from sklearn import linear_model
from . import load


def baseline(
    dems: xr.DataArray,
    /,
    *,
    order: int = 0,
    model: str = "LinearRegression",
    weight: Optional[Union[NDArray[np.float_], float]] = None,
    **options: Any,
) -> xr.DataArray:
    """Fit baseline by polynomial and atmospheric models.

    Args:
        dems: DEMS DataArray to be fit.
        order: Maximum order of the polynomial model.
        weight: One-dimensional weight along channel axis.
            If it is a scalar, then ``(dtau/dpwv)^weight`` will be used.
            It is only for ``'LinearRegression'`` or ``'Ridge'`` models.
        model: Name of the model class in ``sklearn.linear_model``.
        options: Optional arguments used for the model initialization.

    Returns:
        baseline: DataArray of the fit baseline.

    """
    freq = dems.d2_mkid_frequency.values
    slope = dtau_dpwv(freq).values
    n_freq, n_poly = len(freq), order + 1

    # create data to be fit
    X = np.zeros([n_freq, n_poly + 1])
    X[:, 0] = slope

    for exp in range(n_poly):
        X[:, exp + 1] = (freq - freq.mean()) ** exp

    X /= np.linalg.norm(X, axis=0)
    y = dems.values.T

    if weight is None:
        weight = np.ones_like(freq)
    elif isinstance(weight, float):
        weight = slope**weight
    else:
        weight = np.array(weight)

    # fit model to data
    options = {"fit_intercept": False, **options}
    model = getattr(linear_model, model)(**options)

    if model in ("LinearRegression", "Ridge"):
        model.fit(X, y, sample_weight=weight)  # type: ignore
    else:
        model.fit(X, y)  # type: ignore

    coeff: NDArray[np.float_] = model.coef_  # type: ignore

    # create baseline
    baseline = xr.zeros_like(dems)
    baseline += np.outer(coeff[:, 0], X[:, 0])

    for exp in range(n_poly + 1):
        baseline.coords[f"basis_{exp}"] = "chan", X[:, exp]
        baseline.coords[f"coeff_{exp}"] = "time", coeff[:, exp]

    return baseline


def dtau_dpwv(freq: NDArray[np.float_]) -> xr.DataArray:
    """Calculate dtau/dpwv as a function of frequency.

    Args:
        freq: Frequency in units of Hz.

    Returns:
        DataArray that stores dtau/dpwv.

    """
    tau = load.atm(type="tau").interp(freq=freq, method="linear")
    fit = tau.curvefit("pwv", lambda x, a, b: a * x + b)
    return fit["curvefit_coefficients"].sel(param="a", drop=True)
