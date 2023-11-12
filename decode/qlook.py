__all__ = ["still", "pswsc", "raster", "skydip", "zscan"]


# standard library
from pathlib import Path
from typing import Literal, Optional, Sequence, cast


# dependencies
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from fire import Fire
from . import assign, convert, load, make, plot, select, utils


# constants
# fmt: off
BAD_MKID_IDS = (
    18, 77, 117, 118, 140, 141, 161, 182, 183, 184,
    201, 209, 211, 232, 233, 239, 258, 259, 278, 282,
    283, 296, 297, 299, 301, 313,
)
# fmt: on
DFOF_TO_TSKY = (300 - 77) / 3e-5
TSKY_TO_DFOF = 3e-5 / (300 - 77)


def still(
    dems: Path,
    /,
    *,
    include_mkid_ids: Optional[Sequence[int]] = None,
    exclude_mkid_ids: Optional[Sequence[int]] = BAD_MKID_IDS,
    data_type: Literal["df/f", "brightness"] = "brightness",
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    cabin_temperature: float = 273.0,
    outdir: Path = Path(),
    format: str = "png",
) -> None:
    """Quick-look at a still observation.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to bad MKID IDs found on 2023-11-07.
        data_type: Data type of the input DEMS file.
        chan_weight: Weighting method along the channel axis.
            uniform: Uniform weight (i.e. no channel dependence).
            std: Inverse square of temporal standard deviation of sky.
            std/tx: Same as std but std is divided by the atmospheric
            transmission calculated by the ATM model.
        pwv: PWV in units of mm. Only used for the calculation of
            the atmospheric transmission when chan_weight is std/tx.
        cabin_temperature: Temperature at the ASTE cabin.
            Only used for the df/f-to-Tsky conversion.
        outdir: Output directory for the analysis result.
        format: Output data format of the analysis result.

    """
    dems = Path(dems)
    out = Path(outdir) / dems.with_suffix(f".still.{format}").name

    # load DEMS
    da = load.dems(dems, chunks=None)
    da = assign.scan(da)
    da = convert.frame(da, "relative")

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
    da_off = select.by(da, "state", exclude=["ON", "SCAN"])

    # make continuum series
    weight = get_weight(da_off, method=chan_weight, pwv=pwv)
    series = (da * weight).sum("chan") / weight.sum("chan")

    # export output
    if format == "csv":
        series.to_dataset(name=data_type).to_pandas().to_csv(out)
    elif format == "nc":
        series.to_netcdf(out)
    elif format.startswith("zarr"):
        series.to_zarr(out)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        plot.state(da, add_colorbar=False, add_legend=False, ax=ax)
        ax.set_title(Path(dems).name)
        ax.grid(True)

        ax = axes[1]
        plot.data(series, add_colorbar=False, ax=ax)
        ax.set_title(Path(dems).name)
        ax.grid(True)

        if data_type == "df/f":
            ax = ax.secondary_yaxis(
                "right",
                functions=(
                    lambda x: DFOF_TO_TSKY * x + cabin_temperature,
                    lambda x: TSKY_TO_DFOF * (x - cabin_temperature),
                ),
            )
            ax.set_ylabel("Approx. brightness [K]")

        fig.tight_layout()
        fig.savefig(out)

    print(str(out))


def pswsc(
    dems: Path,
    /,
    *,
    include_mkid_ids: Optional[Sequence[int]] = None,
    exclude_mkid_ids: Optional[Sequence[int]] = BAD_MKID_IDS,
    data_type: Literal["df/f", "brightness"] = "brightness",
    frequency_units: str = "GHz",
    outdir: Path = Path(),
    format: str = "png",
) -> None:
    """Quick-look at a PSW observation with sky chopper.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to bad MKID IDs found on 2023-11-07.
        data_type: Data type of the input DEMS file.
        frequency_units: Units of the frequency axis.
        outdir: Output directory for the analysis result.
        format: Output data format of the analysis result.

    """
    dems = Path(dems)
    out = Path(outdir) / dems.with_suffix(f".pswsc.{format}").name

    # load DEMS
    da = load.dems(dems, chunks=None)
    da = assign.scan(da)
    da = convert.frame(da, "relative")
    da = convert.coord_units(da, "frequency", frequency_units)
    da = convert.coord_units(da, "d2_mkid_frequency", frequency_units)

    if data_type == "df/f":
        da = cast(xr.DataArray, np.abs(da))
        da.attrs.update(long_name="|df/f|", units="dimensionless")

    # select DEMS
    da = select.by(da, "d2_mkid_type", include="filter")
    da = select.by(
        da,
        "d2_mkid_id",
        include=include_mkid_ids,
        exclude=exclude_mkid_ids,
    )
    da = select.by(da, "state", include=["ON", "OFF"])
    da_sub = da.groupby("scan").map(subtract_per_scan)

    # export output
    spec = da_sub.mean("scan")
    mad = utils.mad(spec)

    if format == "csv":
        spec.to_dataset(name=data_type).to_pandas().to_csv(out)
    elif format == "nc":
        spec.to_netcdf(out)
    elif format.startswith("zarr"):
        spec.to_zarr(out)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        plot.data(da.scan, ax=ax)
        ax.set_title(Path(dems).name)
        ax.grid(True)

        ax = axes[1]
        plot.data(spec, x="frequency", s=5, hue=None, ax=ax)
        ax.set_ylim(-mad, spec.max() + mad)
        ax.set_title(Path(dems).name)
        ax.grid(True)

        if data_type == "df/f":
            ax = ax.secondary_yaxis(
                "right",
                functions=(
                    lambda x: DFOF_TO_TSKY * x,
                    lambda x: TSKY_TO_DFOF * x,
                ),
            )
            ax.set_ylabel("Approx. brightness [K]")

        fig.tight_layout()
        fig.savefig(out)

    print(str(out))


def raster(
    dems: Path,
    /,
    *,
    include_mkid_ids: Optional[Sequence[int]] = None,
    exclude_mkid_ids: Optional[Sequence[int]] = BAD_MKID_IDS,
    data_type: Literal["df/f", "brightness"] = "brightness",
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    skycoord_grid: str = "6 arcsec",
    skycoord_units: str = "arcsec",
    outdir: Path = Path(),
    format: str = "png",
) -> None:
    """Quick-look at a raster scan observation.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to bad MKID IDs found on 2023-11-07.
        data_type: Data type of the input DEMS file.
        chan_weight: Weighting method along the channel axis.
            uniform: Uniform weight (i.e. no channel dependence).
            std: Inverse square of temporal standard deviation of sky.
            std/tx: Same as std but std is divided by the atmospheric
            transmission calculated by the ATM model.
        pwv: PWV in units of mm. Only used for the calculation of
            the atmospheric transmission when chan_weight is std/tx.
        skycoord_grid: Grid size of the sky coordinate axes.
        skycoord_units: Units of the sky coordinate axes.
        outdir: Output directory for analysis results.
        format: Output image format of analysis results.

    """
    dems = Path(dems)
    result = Path(outdir) / dems.with_suffix(f".raster.{format}").name

    # load DEMS
    da = load.dems(dems, chunks=None)
    da = assign.scan(da)
    da = convert.frame(da, "relative")

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
    da_on = select.by(da, "state", include="SCAN")
    da_off = select.by(da, "state", exclude="SCAN")

    # subtract temporal baseline
    da_base = (
        da_off.groupby("scan")
        .map(mean_in_time)
        .interp_like(
            da_on,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )
    )
    da_sub = da_on - da_base.values

    # make continuum series
    weight = get_weight(da_off, method=chan_weight, pwv=pwv)
    series = (da_sub * weight).sum("chan") / weight.sum("chan")

    # make continuum map
    cube = make.cube(
        da_sub,
        skycoord_grid=skycoord_grid,
        skycoord_units=skycoord_units,
    )
    cont = (cube * weight).sum("chan") / weight.sum("chan")

    if data_type == "df/f":
        cont.attrs.update(long_name="df/f", units="dimensionless")

    # plotting
    map_lim = max(abs(cube.lon).max(), abs(cube.lat).max())
    max_pix = cont.where(cont == cont.max(), drop=True)
    max_lon = float(max_pix.lon)
    max_lat = float(max_pix.lat)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    plot.data(series, ax=ax)
    ax.set_title(Path(dems).name)
    ax.grid(True)

    ax = axes[1]
    cont.plot(ax=ax)  # type: ignore
    ax.set_title(
        "Maxima: "
        f"dAz = {max_lon:+.1f} {cont.lon.attrs['units']}, "
        f"dEl = {max_lat:+.1f} {cont.lat.attrs['units']}"
    )
    ax.set_xlim(-map_lim, map_lim)
    ax.set_ylim(-map_lim, map_lim)
    ax.grid()

    fig.tight_layout()
    fig.savefig(result)
    print(str(result))


def skydip(
    dems: Path,
    /,
    *,
    include_mkid_ids: Optional[Sequence[int]] = None,
    exclude_mkid_ids: Optional[Sequence[int]] = BAD_MKID_IDS,
    data_type: Literal["df/f", "brightness"] = "brightness",
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
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
        chan_weight: Weighting method along the channel axis.
            uniform: Uniform weight (i.e. no channel dependence).
            std: Inverse square of temporal standard deviation of sky.
            std/tx: Same as std but std is divided by the atmospheric
            transmission calculated by the ATM model.
        pwv: PWV in units of mm. Only used for the calculation of
            the atmospheric transmission when chan_weight is std/tx.
        outdir: Output directory for analysis results.
        format: Output image format of analysis results.

    """
    dems = Path(dems)
    result = Path(outdir) / dems.with_suffix(f".skydip.{format}").name

    # load DEMS
    da = load.dems(dems, chunks=None)

    if data_type == "df/f":
        da = cast(xr.DataArray, np.abs(da))
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
    da_off = select.by(da, "state", exclude="SCAN")

    # plotting
    weight = get_weight(da_off, method=chan_weight, pwv=pwv)
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
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
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
        chan_weight: Weighting method along the channel axis.
            uniform: Uniform weight (i.e. no channel dependence).
            std: Inverse square of temporal standard deviation of sky.
            std/tx: Same as std but std is divided by the atmospheric
            transmission calculated by the ATM model.
        pwv: PWV in units of mm. Only used for the calculation of
            the atmospheric transmission when chan_weight is std/tx.
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
    da_off = select.by(da, "state", exclude="ON")

    # plotting
    weight = get_weight(da_off, method=chan_weight, pwv=pwv)
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


def get_weight(
    dems: xr.DataArray,
    /,
    *,
    method: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "3.0",
) -> xr.DataArray:
    """Calculate weight for the channel axis.

    Args:
        dems: Input DEMS DataArray to be considered.
        method: Method for calculating the weight.
            uniform: Uniform weight (i.e. no channel dependence).
            std: Inverse square of temporal standard deviation of sky.
            std/tx: Same as std but std is divided by the atmospheric
            transmission calculated by the ATM model.
        pwv: PWV in units of mm. Only used for the calculation of
            the atmospheric transmission when chan_weight is std/tx.

    Returns:
        The weight DataArray for the channel axis.

    """
    if method == "uniform":
        return xr.ones_like(dems.mean("time"))

    if method == "std":
        return dems.std("time") ** -2

    if method == "std/tx":
        tx = (
            load.atm(type="eta")
            .sel(pwv=float(pwv))
            .interp(
                freq=dems.d2_mkid_frequency,
                method="linear",
            )
        )
        return (dems.std("time") / tx) ** -2

    raise ValueError("Method must be either uniform, std, or std/tx.")


def mean_in_time(dems: xr.DataArray) -> xr.DataArray:
    """Similar to DataArray.mean but keeps middle time."""
    middle = dems[len(dems) // 2 : len(dems) // 2 + 1]
    return xr.zeros_like(middle) + dems.mean("time")


def subtract_per_scan(dems: xr.DataArray) -> xr.DataArray:
    """Apply source-sky subtraction to a single-scan DEMS."""
    if len(states := np.unique(dems.state)) != 1:
        raise ValueError("State must be unique.")

    if (state := states[0]) == "ON":
        src = select.by(dems, "beam", include="B")
        sky = select.by(dems, "beam", include="A")
        return src.mean("time") - sky.mean("time").data

    if state == "OFF":
        src = select.by(dems, "beam", include="A")
        sky = select.by(dems, "beam", include="B")
        return src.mean("time") - sky.mean("time").data

    raise ValueError("State must be either ON or OFF.")


def main() -> None:
    """Entry point of the decode-qlook command."""
    with xr.set_options(keep_attrs=True):
        Fire(
            {
                "default": still,
                "still": still,
                "pswsc": pswsc,
                "raster": raster,
                "skydip": skydip,
                "zscan": zscan,
            }
        )
