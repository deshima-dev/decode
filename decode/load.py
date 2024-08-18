__all__ = ["atm", "dems"]


# standard library
from pathlib import Path
from typing import Any, Literal, Union
from warnings import catch_warnings, simplefilter


# dependencies
import numpy as np
import pandas as pd
import xarray as xr
from . import convert


# constants
ALMA_ATM = "alma_atm.txt"
DATA_DIR = Path(__file__).parent / "data"
NETCDF_ENGINE = "scipy"
NETCDF_SUFFIX = ".nc"
ZARR_ENGINE = "zarr"
ZARR_SUFFIX = ".zarr"


def atm(*, type: Literal["eta", "tau"] = "tau") -> xr.DataArray:
    """Load an ALMA ATM model as a DataArray.

    Args:
        type: Type of model to be stored in the DataArray.
            Either ``'eta'`` (transmission) or ``'tau'`` (opacity).

    Returns:
        DataArray that stores the ALMA ATM model.

    """
    atm = pd.read_csv(
        DATA_DIR / ALMA_ATM,
        comment="#",
        index_col=0,
        sep=r"\s+",
    )
    freq = xr.DataArray(
        atm.index * 1e9,
        dims="freq",
        attrs={
            "long_name": "Frequency",
            "units": "Hz",
        },
    )
    pwv = xr.DataArray(
        atm.columns.astype(float),
        dims="pwv",
        attrs={
            "long_name": "Precipitable water vapor",
            "units": "mm",
        },
    )

    if type == "eta":
        return xr.DataArray(atm, coords=(freq, pwv))
    elif type == "tau":
        with catch_warnings():
            simplefilter("ignore")
            return xr.DataArray(-np.log(atm), coords=(freq, pwv))
    else:
        raise ValueError("Type must be either eta or tau.")


def dems(
    dems: Union[Path, str],
    /,
    measure: Literal["brightness", "df/f"] = "brightness",
    **options: Any,
) -> xr.DataArray:
    """Load a DEMS file as a DataArray.

    Args:
        dems: Path of the DEMS file.
        measure: Measure of the DataArray (either brightness or df/f).
        **options: Options to be passed to ``xarray.open_dataarray``.

    Returns:
        Loaded DEMS DataArray.

    Raises:
        ValueError: Raised if file type or measure is not supported.

    """
    suffixes = Path(dems).suffixes

    if NETCDF_SUFFIX in suffixes:
        options = {
            "engine": NETCDF_ENGINE,
            **options,
        }
    elif ZARR_SUFFIX in suffixes:
        options = {
            "chunks": "auto",
            "engine": ZARR_ENGINE,
            **options,
        }
    else:
        raise ValueError(
            f"File type of {dems} is not supported."
            "Use netCDF (.nc) or Zarr (.zarr, .zarr.zip)."
        )

    da = xr.open_dataarray(dems, **options)

    if da.long_name == "Brightness" and measure == "brightness":
        return da

    if da.long_name == "df/f" and measure == "df/f":
        return da

    if da.long_name == "df/f" and measure == "brightness":
        return convert.dfof_to_brightness(da)

    if da.long_name == "Brightness" and measure == "df/f":
        raise ValueError("Brightness-to-df/f conversion is not supported.")

    raise ValueError("Measure must be either brightness or df/f.")
