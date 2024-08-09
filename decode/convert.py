__all__ = ["coord_units", "dfof_to_brightness", "frame", "units"]


# standard library
from typing import Optional, Sequence, TypeVar, Union


# dependencies
import numpy as np
import xarray as xr
from astropy.units import Equivalency, Quantity, Unit


# type hints
T = TypeVar("T")
Multiple = Union[Sequence[T], T]
UnitLike = Union[xr.DataArray, Unit, str]


# constants
DEFAULT_T_ATM = 273.0  # K
DEFAULT_T_ROOM = 293.0  # K


def coord_units(
    da: xr.DataArray,
    coord_names: Multiple[str],
    new_units: UnitLike,
    /,
    equivalencies: Optional[Equivalency] = None,
) -> xr.DataArray:
    """Convert the units of coordinate(s) of a DataArray.

    Args:
        da: Input DataArray.
        coord_names: Name(s) of the coordinate(s) to be converted.
        new_units: Units to be converted from the current ones.
            A DataArray that has units attribute is also accepted.
        equivalencies: Optional Astropy equivalencies.

    Returns:
        DataArray with the units of the coordinate converted.

    """
    # deepcopy except for data
    da = da.copy(data=da.data)

    if isinstance(coord_names, str):
        coord_names = [coord_names]

    for coord_name in coord_names:
        coord = da.coords[coord_name]
        new_coord = units(coord, new_units, equivalencies)
        da = da.assign_coords({coord_name: new_coord})

    return da


def dfof_to_brightness(da: xr.DataArray, /) -> xr.DataArray:
    """Convert a DataArray of df/f to that of brightness."""
    if np.isnan(T_atm := da.temperature.mean().data):
        T_atm = DEFAULT_T_ATM

    if np.isnan(T_room := da.aste_cabin_temperature.mean().data):
        T_room = DEFAULT_T_ROOM

    fwd = da.d2_resp_fwd.data
    p0 = da.d2_resp_p0.data
    T0 = da.d2_resp_t0.data

    return (
        da.copy(
            deep=True,
            data=(da.data + p0 * np.sqrt(T_room + T0)) ** 2 / (p0**2 * fwd)
            - T0 / fwd
            - (1 - fwd) / fwd * T_atm,
        )
        .astype(da.dtype)
        .assign_attrs(long_name="Brightness", units="K")
    )


def frame(da: xr.DataArray, new_frame: str, /) -> xr.DataArray:
    """Convert the skycoord frame of a DataArray.

    Args:
        da: Input DataArray.
        new_frame: Frame to be converted from the current one.

    Returns:
        DataArray with the skycoord frame converted.

    """
    # deepcopy except for data
    da = da.copy(data=da.data)

    if not new_frame == "relative":
        raise ValueError("Relative is only available.")

    lon = da.coords["lon"]
    lat = da.coords["lat"]
    lon_origin = da.coords["lon_origin"]
    lat_origin = da.coords["lat_origin"]

    # do not change the order below!
    lon -= units(lon_origin, lon)
    lon *= np.cos(units(lat, "rad"))
    lat -= units(lat_origin, lat)
    lon_origin *= 0.0
    lat_origin *= 0.0

    new_frame = da.frame.copy(data=new_frame)
    return da.assign_coords(frame=new_frame)


def units(
    da: xr.DataArray,
    new_units: UnitLike,
    /,
    equivalencies: Optional[Equivalency] = None,
) -> xr.DataArray:
    """Convert the units of a DataArray.

    Args:
        da: Input DataArray.
        new_units: Units to be converted from the current ones.
            A DataArray that has units attribute is also accepted.
        equivalencies: Optional Astropy equivalencies.

    Returns:
        DataArray with the units converted.

    """
    if (units := da.attrs.get("units")) is None:
        raise ValueError("Units must exist in DataArray attrs.")

    if isinstance(new_units, xr.DataArray):
        new_units = new_units.attrs["units"]

    new_data = Quantity(da, units).to(new_units, equivalencies)
    return da.copy(data=new_data).assign_attrs(units=new_units)
