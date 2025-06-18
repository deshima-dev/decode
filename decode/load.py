__all__ = ["atm", "dems"]


# standard library
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Optional, Union
from warnings import catch_warnings, simplefilter


# dependencies
import numpy as np
import pandas as pd
import xarray as xr
from astropy.units import Quantity
from ndtools import Range
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
    dems: Union[PathLike[str], str],
    /,
    *,
    # options for data selection
    include_mkid_types: Optional[Sequence[str]] = ("filter",),
    exclude_mkid_types: Optional[Sequence[str]] = None,
    include_mkid_ids: Optional[Sequence[int]] = None,
    exclude_mkid_ids: Optional[Sequence[int]] = None,
    min_frequency: Optional[str] = None,
    max_frequency: Optional[str] = None,
    # options for coordinate conversion
    frequency_units: Optional[str] = "GHz",
    skycoord_units: Optional[str] = "arcsec",
    skycoord_frame: Optional[str] = None,
    # options for data conversion
    data_scaling: Optional[Literal["brightness", "df/f"]] = None,
    T_amb: float = 273.0,  # K
    T_room: float = 293.0,  # K
    # other options for loading
    **options: Any,
) -> xr.DataArray:
    """Load a DEMS file as a DataArray.

    Args:
        dems: Path of the DEMS file.
        include_mkid_types: MKID types to be included.
            Defaults to filter-only.
        exclude_mkid_types: MKID types to be excluded.
            Defaults to no MKID types.
        include_mkid_ids: MKID IDs to be included.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded.
            Defaults to no MKID IDs.
        min_frequency: Minimum frequency to be included.
            Defaults to no minimum frequency bound.
        max_frequency: Maximum frequency to be included.
            Defaults to no maximum frequency bound.
        frequency_units: Units of the frequency-related coordinates.
            Defaults to GHz.
        skycoord_units: Units of the skycoord-related coordinates.
            Defaults to arcsec.
        skycoord_frame: Frame of the skycoord.
            Defaults to the skycoord of the input DEMS file.
        data_scaling: Data scaling (either brightness or df/f).
            Defaults to the data scaling of the input DEMS file.
        T_amb: Default ambient temperature value for the data scaling
            to be used when the ``temperature`` coordinate is all-NaN.
        T_room: Default room temperature value for the data scaling
            to be used when the ``aste_cabin_temperature`` coordinate is all-NaN.
        **options: Other options for loading (e.g. ``chunks=None``, etc).

    Return:
        DataArray of the loaded DEMS file.

    """
    # load DEMS as DataArray
    if NETCDF_SUFFIX in (suffixes := Path(dems).suffixes):
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

    # load DataArray coordinates on memory
    for name in da.coords:
        da.coords[name].load()

    # select DataArray by MKID types
    if include_mkid_types is not None:
        da = da.sel(chan=da.d2_mkid_type.isin(include_mkid_types))

    if exclude_mkid_types is not None:
        da = da.sel(chan=~da.d2_mkid_type.isin(exclude_mkid_types))

    # select DataArray by MKID master IDs
    if include_mkid_ids is not None:
        da = da.sel(chan=da.d2_mkid_id.isin(include_mkid_ids))

    if exclude_mkid_ids is not None:
        da = da.sel(chan=~da.d2_mkid_id.isin(exclude_mkid_ids))

    # select DataArray by frequency range
    if min_frequency is not None:
        min_frequency = Quantity(min_frequency).to(da.frequency.units).value

    if max_frequency is not None:
        max_frequency = Quantity(max_frequency).to(da.frequency.units).value

    if min_frequency is not None or max_frequency is not None:
        da = da.sel(chan=da.frequency == Range(min_frequency, max_frequency))

    # convert frequency units
    if frequency_units is not None:
        da = convert.coord_units(
            da,
            ["bandwidth", "frequency", "d2_mkid_frequency"],
            frequency_units,
        )

    # convert skycoord units and frame
    if skycoord_units is not None:
        da = convert.coord_units(
            da,
            ["lat", "lat_origin", "lon", "lon_origin"],
            skycoord_units,
        )

    if skycoord_frame is not None:
        da = convert.frame(da, skycoord_frame)

    # convert data scaling
    if data_scaling == "brightness":
        return convert.to_brightness(da, T_amb=T_amb, T_room=T_room)
    elif data_scaling == "df/f":
        return convert.to_dfof(da, T_amb=T_amb, T_room=T_room)
    else:
        return da
