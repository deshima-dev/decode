__all__ = ["coord_units", "frame", "units"]


# standard library
from typing import Optional, Union


# dependencies
import numpy as np
import xarray as xr
from astropy.units import Equivalency, Quantity, Unit


# type hints
UnitLike = Union[Unit, str]


def coord_units(
    da: xr.DataArray,
    coord_name: str,
    new_units: UnitLike,
    /,
    *,
    equivalencies: Optional[Equivalency] = None,
) -> xr.DataArray:
    """Convert units of a coordinate of a DataArray.

    Args:
        da: Input DataArray.
        coord_name: Name of the coordinate to be converted.
        new_units: Units to be converted from the current ones.
        equivalencies: Optional Astropy equivalencies.

    Returns:
        DataArray with the units of the coordinate converted.

    """
    new_coord = units(da[coord_name], new_units, equivalencies=equivalencies)
    return da.assign_coords({coord_name: new_coord})


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

    return dems.assign_coords(frame=dems.frame.copy(False, new_frame))


def units(
    da: xr.DataArray,
    new_units: UnitLike,
    /,
    *,
    equivalencies: Optional[Equivalency] = None,
) -> xr.DataArray:
    """Convert units of a DataArray.

    Args:
        da: Input DataArray.
        new_units: Units to be converted from the current ones.
        equivalencies: Optional Astropy equivalencies.

    Returns:
        DataArray with the units converted.

    """
    if (units := da.attrs.get("units")) is None:
        raise ValueError("Units must exist in DataArray attrs.")

    new_data = Quantity(da, units).to(new_units, equivalencies)
    return da.copy(False, new_data).assign_attrs(units=str(new_units))
