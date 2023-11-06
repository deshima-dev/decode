__all__ = ["frame", "units"]


# standard library
from typing import Optional, Union


# dependencies
import numpy as np
import xarray as xr
from astropy.units import Equivalency, Quantity, Unit


# type hints
UnitLike = Union[Unit, str]


def frame(
    dems: xr.DataArray,
    new_frame: str,
    /,
    *,
    inplace: bool = False,
) -> xr.DataArray:
    """Convert skycoord frame of DEMS.

    Args:
        dems: Target DEMS DataArray.
        new_frame: Skycoord frame to be converted from the current ones.
        inplace: Whether the skycoord frame are converted in-place.

    Returns:
        DEMS DataArray with the skycoord frame converted.

    """
    if not inplace:
        # deepcopy except for data
        dems = dems.copy(data=dems.data)

    if not new_frame == "relative":
        raise ValueError("Relative is only available.")

    lon, lon_origin = dems["lon"], dems["lon_origin"]
    lat, lat_origin = dems["lat"], dems["lat_origin"]
    cos = np.cos(Quantity(lat, lat.attrs["units"]).to("rad"))

    if lon.attrs["units"] != lon_origin.attrs["units"]:
        raise ValueError("Units of lon and lon_origin must be same.")

    if lat.attrs["units"] != lat_origin.attrs["units"]:
        raise ValueError("Units of lat and lat_origin must be same.")

    # do not change the order below!
    lon -= lon_origin
    lon *= cos
    lat -= lat_origin
    lon_origin[:] = 0.0
    lat_origin[:] = 0.0

    return dems


def units(
    dems: xr.DataArray,
    coord_name: str,
    new_units: UnitLike,
    /,
    *,
    equivalencies: Optional[Equivalency] = None,
    inplace: bool = False,
) -> xr.DataArray:
    """Convert units of a coordinate of DEMS.

    Args:
        dems: Target DEMS DataArray.
        coord_name: Name of the coordinate for the conversion.
        new_units: Units to be converted from the current ones.
        equivalencies: Optional Astropy equivalencies.
        inplace: Whether the units are converted in-place.

    Returns:
        DEMS DataArray with the coordinate units converted.

    """
    if not inplace:
        # deepcopy except for data
        dems = dems.copy(data=dems.data)

    coord = dems[coord_name]

    if (units := coord.attrs.get("units")) is None:
        return dems

    coord.values = Quantity(coord, units).to(new_units, equivalencies)
    coord.attrs["units"] = new_units
    return dems
