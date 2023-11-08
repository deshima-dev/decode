__all__ = []


# standard library
from pathlib import Path
from typing import Any, Literal, Optional, Sequence


# dependencies
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from fire import Fire
from . import load, plot, select


# constants
# fmt: off
BAD_MKID_IDS = (
    18, 77, 117, 118, 140, 141, 161, 182, 183, 184,
    201, 209, 211, 232, 233, 239, 258, 259, 278, 282,
    283, 296, 297, 299, 301, 313,
)
# fmt: on


def skydip(
    dems: Path,
    /,
    *,
    include_mkid_ids: Optional[Sequence[int]] = None,
    exclude_mkid_ids: Optional[Sequence[int]] = BAD_MKID_IDS,
    data_type: Literal["df/f", "brightness"] = "brightness",
    outdir: Path = Path(),
    format: str = "png",
) -> None:
    """Quick-look at a skydip observation.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to bad MKID IDs found on 2023-11-07.
        data_type: Data type of the input DEMS file.
        outdir: Output directory for analysis results.
        format: Output image format of analysis results.

    """
    dems = Path(dems)
    result = Path(outdir) / dems.with_suffix(f".skydip.{format}").name

    # load DEMS
    da = load.dems(dems, chunks=None)

    if data_type == "df/f":
        da: xr.DataArray = np.abs(da)
        da.attrs.update(long_name="|df/f|", units="dimensionless")

    # add sec(Z) coordinate
    secz = 1 / np.cos(np.deg2rad(90.0 - da.lat))
    secz.attrs.update(long_name="sec(Z)", units="dimensionless")
    da = da.assign_coords(secz=secz)

    # select DEMS
    da = select.by(da, "d2_mkid_type", include="filter")
    da = select.by(
        da,
        "d2_mkid_id",
        include=include_mkid_ids,
        exclude=exclude_mkid_ids,
    )
    da_on = select.by(da, "state", include="SCAN")
    da_off = select.by(da, "state", exclude="GRAD")

    # plotting
    weight = da_off.std("time") ** -2
    series = (da_on * weight).sum("chan") / weight.sum("chan")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    plot.data(series, hue="secz", ax=ax)
    ax.set_title(Path(dems).name)
    ax.grid(True)

    ax = axes[1]
    plot.data(series, x="secz", ax=ax)
    ax.set_title(Path(dems).name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(result)
    print(str(result))


def zscan(
    dems: Path,
    /,
    *,
    include_mkid_ids: Optional[Sequence[int]] = None,
    exclude_mkid_ids: Optional[Sequence[int]] = BAD_MKID_IDS,
    data_type: Literal["df/f", "brightness"] = "brightness",
    outdir: Path = Path(),
    format: str = "png",
) -> None:
    """Quick-look at an observation of subref axial focus scan.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to bad MKID IDs found on 2023-11-07.
        data_type: Data type of the input DEMS file.
        outdir: Output directory for analysis results.
        format: Output image format of analysis results.

    """
    dems = Path(dems)
    result = Path(outdir) / dems.with_suffix(f".zscan.{format}").name

    # load DEMS
    da = load.dems(dems, chunks=None)

    if data_type == "df/f":
        da.attrs.update(long_name="df/f", units="dimensionless")

    # select DEMS
    da = select.by(da, "d2_mkid_type", include="filter")
    da = select.by(
        da,
        "d2_mkid_id",
        include=include_mkid_ids,
        exclude=exclude_mkid_ids,
    )
    da_on = select.by(da, "state", include="ON")
    da_off = select.by(da, "state", exclude="GRAD")

    # plotting
    weight = da_off.std("time") ** -2
    series = (da_on * weight).sum("chan") / weight.sum("chan")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    plot.data(series, hue="aste_subref_z", ax=ax)
    ax.set_title(Path(dems).name)
    ax.grid(True)

    ax = axes[1]
    plot.data(series, x="aste_subref_z", ax=ax)
    ax.set_title(Path(dems).name)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(result)
    print(str(result))


def main() -> None:
    """Entry point of the decode-qlook command."""
    with xr.set_options(keep_attrs=True):
        Fire(
            {
                "skydip": skydip,
                "zscan": zscan,
            }
        )
