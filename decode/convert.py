__all__ = ["coord_units", "frame", "to_brightness", "to_dfof", "units"]


# standard library
from logging import getLogger
from typing import Optional, Sequence, TypeVar, Union


# dependencies
import numpy as np
import xarray as xr
from astropy.units import Equivalency, Quantity, Unit


# constants
LOGGER = getLogger(__name__)


# type hints
T = TypeVar("T")
Multiple = Union[Sequence[T], T]
UnitLike = Union[xr.DataArray, Unit, str]


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


def to_brightness(
    dems: xr.DataArray,
    /,
    *,
    T_amb: float = 273.0,
    T_room: float = 293.0,
) -> xr.DataArray:
    """Convert DEMS on the df/f scale to the brightness temperature scale.

    Args:
        dems: Input DEMS DataArray on the df/f scale.
        T_amb: Default ambient temperature value to be used
            when all ``dems.temperature`` values are NaN.
        T_room: Default room temperature value to be used
            when all ``dems.aste_cabin_temperature`` values are NaN.

    Returns:
        DEMS DataArray on the brightness temperature scale.

    """
    if dems.long_name == "Brightness" or dems.units == "K":
        LOGGER.warning("DEMS may already be on the brightness temperature scale.")

    if np.isnan(T_room_ := dems.aste_cabin_temperature.mean().data):
        T_room_ = T_room

    if np.isnan(T_amb_ := dems.temperature.mean().data):
        T_amb_ = T_amb

    fwd = dems.d2_resp_fwd.data
    p0 = dems.d2_resp_p0.data
    T0 = dems.d2_resp_t0.data

    return (
        dems.copy(
            deep=True,
            data=(
                (dems.data + p0 * (T_room_ + T0) ** 0.5) ** 2 / (p0**2 * fwd)
                - (T0 + (1 - fwd) * T_amb_) / fwd
            ),
        )
        .astype(dems.dtype)
        .assign_attrs(long_name="Brightness", units="K")
    )


def to_dfof(
    dems: xr.DataArray,
    /,
    *,
    T_amb: float = 273.0,
    T_room: float = 293.0,
) -> xr.DataArray:
    """Convert DEMS on the brightness temperature scale to the df/f scale.

    Args:
        dems: Input DEMS DataArray on the brightness temperature scale.
        T_amb: Default ambient temperature value to be used
            when all ``dems.temperature`` values are NaN.
        T_room: Default room temperature value to be used
            when all ``dems.aste_cabin_temperature`` values are NaN.

    Returns:
        DEMS DataArray on the df/f scale.

    """
    if dems.long_name == "df/f" or dems.units == "dimensionless":
        LOGGER.warning("DEMS may already be on the df/f scale.")

    if np.isnan(T_room_ := dems.aste_cabin_temperature.mean().data):
        T_room_ = T_room

    if np.isnan(T_amb_ := dems.temperature.mean().data):
        T_amb_ = T_amb

    fwd = dems.d2_resp_fwd.data
    p0 = dems.d2_resp_p0.data
    T0 = dems.d2_resp_t0.data

    return (
        dems.copy(
            deep=True,
            data=(
                p0
                * (
                    (fwd * dems.data + (1 - fwd) * T_amb_ + T0) ** 0.5
                    - (T_room_ + T0) ** 0.5
                )
            ),
        )
        .astype(dems.dtype)
        .assign_attrs(long_name="df/f", units="dimensionless")
    )


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

    new_data = Quantity(da, units).to(new_units, equivalencies).value
    return da.copy(data=new_data).assign_attrs(units=new_units)
