__all__ = ["dems"]


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


def dems(dems: Union[Path, str], /, **options: Any) -> xr.DataArray:
    """Load a DEMS file as a DataArray.

    Args:
        dems: Path of the DEMS file.

    Keyword Args:
        options: Arguments to be passed to ``xarray.open_dataarray``.

    Return:
        Loaded DEMS DataArray.

    Raises:
        ValueError: Raised if the file type is not supported.

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

    return xr.open_dataarray(dems, **options)
