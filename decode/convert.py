__all__ = ["coord_units", "frame", "units"]


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
