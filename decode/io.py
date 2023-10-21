__all__ = ["open_dems"]


# standard library
from pathlib import Path
from typing import Any, Union


# dependencies
import xarray as xr


# constants
NETCDF_ENGINE = "scipy"
NETCDF_SUFFIX = ".nc"
ZARR_ENGINE = "zarr"
ZARR_SUFFIX = ".zarr"


def open_dems(dems: Union[Path, str], **kwargs: Any) -> xr.DataArray:
    """Open a DEMS file as a DataArray.

    Args:
        dems: Path of the DEMS file.
        kwargs: Arguments to be passed to ``xarray.open_dataarray``.

    Return:
        A DataArray of the opened DEMS file.

    Raises:
        ValueError: Raised if the file type is not supported.

    """
    engine: str
    suffixes = Path(dems).suffixes

    if NETCDF_SUFFIX in suffixes:
        engine = kwargs.pop("engine", NETCDF_ENGINE)
    elif ZARR_SUFFIX in suffixes:
        engine = kwargs.pop("engine", ZARR_ENGINE)
    else:
        raise ValueError(
            f"File type of {dems} is not supported."
            "Use netCDF (.nc) or Zarr (.zarr, .zarr.zip)."
        )

    return xr.open_dataarray(dems, engine=engine, **kwargs)
