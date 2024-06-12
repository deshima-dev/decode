__all__ = ["auto", "pswsc", "raster", "skydip", "still", "zscan"]


# standard library
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union, cast
from warnings import catch_warnings, simplefilter


# dependencies
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from fire import Fire
from matplotlib.figure import Figure
from . import assign, convert, load, make, plot, select, utils


# constants
DATA_FORMATS = "csv", "nc", "zarr", "zarr.zip"
DEFAULT_DATA_TYPE = "auto"
# fmt: off
DEFAULT_EXCL_MKID_IDS = (
    0, 18, 26, 73, 130, 184, 118, 119, 201, 202,
    208, 214, 261, 266, 280, 283, 299, 304, 308, 321,
)
# fmt: on
DEFAULT_FIGSIZE = 12, 4
DEFAULT_FORMAT = "png"
DEFAULT_FREQUENCY_UNITS = "GHz"
DEFAULT_INCL_MKID_IDS = None
DEFAULT_OUTDIR = Path()
DEFAULT_OVERWRITE = False
DEFAULT_SKYCOORD_GRID = "6 arcsec"
DEFAULT_SKYCOORD_UNITS = "arcsec"
SIGMA_OVER_MAD = 1.4826


def auto(dems: Path, /, **options: Any) -> Path:
    """Quick-look at an observation with auto-selected command.

    The used command will be selected based on the observation name
    stored as the ``observation`` attribute in an input DEMS file.

    Args:
        dems: Input DEMS file (netCDF or Zarr).

    Keyword Args:
        options: Options for the selected command.
            See the command help for all available options.

    Returns:
        Absolute path of the saved file.

    """
    with xr.set_options(keep_attrs=True):
        da = load.dems(dems, chunks=None)
        obs: str = da.attrs["observation"]

        if "pswsc" in obs:
            return pswsc(dems, **options)

        if "raster" in obs:
            return raster(dems, **options)

        if "skydip" in obs:
            return skydip(dems, **options)

        if "still" in obs:
            return still(dems, **options)

        if "zscan" in obs:
            return zscan(dems, **options)

        raise ValueError(
            f"Could not infer the command to be used from {obs!r}: "
            "Observation name must include one of the command names. "
        )


def pswsc(
    dems: Path,
    /,
    *,
    # options for loading
    include_mkid_ids: Optional[Sequence[int]] = DEFAULT_INCL_MKID_IDS,
    exclude_mkid_ids: Optional[Sequence[int]] = DEFAULT_EXCL_MKID_IDS,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    frequency_units: str = DEFAULT_FREQUENCY_UNITS,
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "pswsc",
    **options: Any,
) -> Path:
    """Quick-look at a PSW observation with sky chopper.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to bad MKID IDs found on 2023-11-19.
        data_type: Data type of the input DEMS file.
            Defaults to the ``long_name`` attribute in it.
        frequency_units: Units of the frequency axis.
        format: Output data format of the quick-look result.
        outdir: Output directory for the quick-look result.
        overwrite: Whether to overwrite the output if it exists.
        suffix: Suffix that precedes the file extension.

    Keyword Args:
        options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with xr.set_options(keep_attrs=True):
        da = load_dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            data_type=data_type,
            frequency_units=frequency_units,
        )

        # make spectrum
        da_scan = select.by(da, "state", ["ON", "OFF"])
        da_sub = da_scan.groupby("scan").map(subtract_per_scan)
        spec = da_sub.mean("scan")

        # save result
        suffixes = f".{suffix}.{format}"
        file = Path(outdir) / Path(dems).with_suffix(suffixes).name

        if format in DATA_FORMATS:
            return save_qlook(spec, file, overwrite=overwrite, **options)

        fig, axes = plt.subplots(1, 2, figsize=DEFAULT_FIGSIZE)

        ax = axes[0]
        plot.data(da.scan, ax=ax)

        ax = axes[1]
        plot.data(spec, x="frequency", s=5, hue=None, ax=ax)
        ax.set_ylim(get_robust_lim(spec))

        for ax in axes:
            ax.set_title(Path(dems).name)
            ax.grid(True)

        fig.tight_layout()
        return save_qlook(fig, file, overwrite=overwrite, **options)


def raster(
    dems: Path,
    /,
    *,
    # options for loading
    include_mkid_ids: Optional[Sequence[int]] = DEFAULT_INCL_MKID_IDS,
    exclude_mkid_ids: Optional[Sequence[int]] = DEFAULT_EXCL_MKID_IDS,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    # options for analysis
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    skycoord_grid: str = DEFAULT_SKYCOORD_GRID,
    skycoord_units: str = DEFAULT_SKYCOORD_UNITS,
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "raster",
    **options: Any,
) -> Path:
    """Quick-look at a raster scan observation.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to bad MKID IDs found on 2023-11-19.
        data_type: Data type of the input DEMS file.
            Defaults to the ``long_name`` attribute in it.
        chan_weight: Weighting method along the channel axis.
            uniform: Uniform weight (i.e. no channel dependence).
            std: Inverse square of temporal standard deviation of sky.
            std/tx: Same as std but std is divided by the atmospheric
            transmission calculated by the ATM model.
        pwv: PWV in units of mm. Only used for the calculation of
            the atmospheric transmission when chan_weight is std/tx.
        skycoord_grid: Grid size of the sky coordinate axes.
        skycoord_units: Units of the sky coordinate axes.
        format: Output image format of quick-look result.
        outdir: Output directory for the quick-look result.
        suffix: Suffix that precedes the file extension.
        overwrite: Whether to overwrite the output if it exists.

    Keyword Args:
        options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with xr.set_options(keep_attrs=True):
        da = load_dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            data_type=data_type,
            skycoord_units=skycoord_units,
        )

        # subtract temporal baseline
        da_on = select.by(da, "state", include="SCAN")
        da_off = select.by(da, "state", exclude="SCAN")
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
        weight = get_chan_weight(da_off, method=chan_weight, pwv=pwv)
        series = da_sub.weighted(weight.fillna(0)).mean("chan")

        # make continuum map
        cube = make.cube(
            da_sub,
            skycoord_grid=skycoord_grid,
            skycoord_units=skycoord_units,
        )
        cont = cube.weighted(weight.fillna(0)).mean("chan")

        # save result
        suffixes = f".{suffix}.{format}"
        file = Path(outdir) / Path(dems).with_suffix(suffixes).name

        if format in DATA_FORMATS:
            return save_qlook(cont, file, overwrite=overwrite, **options)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        ax = axes[0]
        plot.data(series, ax=ax)
        ax.set_title(Path(dems).name)

        ax = axes[1]
        map_lim = max(abs(cube.lon).max(), abs(cube.lat).max())
        max_pix = cont.where(cont == cont.max(), drop=True)

        cont.plot(ax=ax)  # type: ignore
        ax.set_xlim(-map_lim, map_lim)
        ax.set_ylim(-map_lim, map_lim)
        ax.set_title(
            "Maximum: "
            f"dAz = {float(max_pix.lon):+.1f} {cont.lon.attrs['units']}, "
            f"dEl = {float(max_pix.lat):+.1f} {cont.lat.attrs['units']}"
        )

        for ax in axes:
            ax.grid(True)

        fig.tight_layout()
        return save_qlook(fig, file, overwrite=overwrite, **options)


def skydip(
    dems: Path,
    /,
    *,
    # options for loading
    include_mkid_ids: Optional[Sequence[int]] = DEFAULT_INCL_MKID_IDS,
    exclude_mkid_ids: Optional[Sequence[int]] = DEFAULT_EXCL_MKID_IDS,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    # options for analysis
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "skydip",
    **options: Any,
) -> Path:
    """Quick-look at a skydip observation.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to bad MKID IDs found on 2023-11-19.
        data_type: Data type of the input DEMS file.
            Defaults to the ``long_name`` attribute in it.
        chan_weight: Weighting method along the channel axis.
            uniform: Uniform weight (i.e. no channel dependence).
            std: Inverse square of temporal standard deviation of sky.
            std/tx: Same as std but std is divided by the atmospheric
            transmission calculated by the ATM model.
        pwv: PWV in units of mm. Only used for the calculation of
            the atmospheric transmission when chan_weight is std/tx.
        format: Output image format of quick-look result.
        outdir: Output directory for the quick-look result.
        suffix: Suffix that precedes the file extension.
        overwrite: Whether to overwrite the output if it exists.

    Keyword Args:
        options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with xr.set_options(keep_attrs=True):
        da = load_dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            data_type=data_type,
        )

        # make continuum series
        da_on = select.by(da, "state", include="SCAN")
        da_off = select.by(da, "state", exclude="SCAN")
        weight = get_chan_weight(da_off, method=chan_weight, pwv=pwv)
        series = da_on.weighted(weight.fillna(0)).mean("chan")

        # save result
        suffixes = f".{suffix}.{format}"
        file = Path(outdir) / Path(dems).with_suffix(suffixes).name

        if format in DATA_FORMATS:
            return save_qlook(series, file, overwrite=overwrite, **options)

        fig, axes = plt.subplots(1, 2, figsize=DEFAULT_FIGSIZE)

        ax = axes[0]
        plot.data(series, hue="secz", ax=ax)

        ax = axes[1]
        plot.data(series, x="secz", ax=ax)

        for ax in axes:
            ax.set_title(Path(dems).name)
            ax.grid(True)

        fig.tight_layout()
        return save_qlook(fig, file, overwrite=overwrite, **options)


def still(
    dems: Path,
    /,
    *,
    # options for loading
    include_mkid_ids: Optional[Sequence[int]] = DEFAULT_INCL_MKID_IDS,
    exclude_mkid_ids: Optional[Sequence[int]] = DEFAULT_EXCL_MKID_IDS,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    # options for analysis
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "still",
    **options: Any,
) -> Path:
    """Quick-look at a still observation.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to bad MKID IDs found on 2023-11-19.
        data_type: Data type of the input DEMS file.
            Defaults to the ``long_name`` attribute in it.
        chan_weight: Weighting method along the channel axis.
            uniform: Uniform weight (i.e. no channel dependence).
            std: Inverse square of temporal standard deviation of sky.
            std/tx: Same as std but std is divided by the atmospheric
            transmission calculated by the ATM model.
        pwv: PWV in units of mm. Only used for the calculation of
            the atmospheric transmission when chan_weight is std/tx.
        format: Output data format of the quick-look result.
        outdir: Output directory for the quick-look result.
        suffix: Suffix that precedes the file extension.
        overwrite: Whether to overwrite the output if it exists.

    Keyword Args:
        options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with xr.set_options(keep_attrs=True):
        da = load_dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            data_type=data_type,
        )

        # make continuum series
        da_off = select.by(da, "state", exclude=["ON", "SCAN"])
        weight = get_chan_weight(da_off, method=chan_weight, pwv=pwv)
        series = da.weighted(weight.fillna(0)).mean("chan")

        # save result
        suffixes = f".{suffix}.{format}"
        file = Path(outdir) / Path(dems).with_suffix(suffixes).name

        if format in DATA_FORMATS:
            return save_qlook(series, file, overwrite=overwrite, **options)

        fig, axes = plt.subplots(1, 2, figsize=DEFAULT_FIGSIZE)

        ax = axes[0]
        plot.state(da, add_colorbar=False, add_legend=False, ax=ax)

        ax = axes[1]
        plot.data(series, add_colorbar=False, ax=ax)

        for ax in axes:
            ax.set_title(Path(dems).name)
            ax.grid(True)

        fig.tight_layout()
        return save_qlook(fig, file, overwrite=overwrite, **options)


def zscan(
    dems: Path,
    /,
    *,
    # options for loading
    include_mkid_ids: Optional[Sequence[int]] = DEFAULT_INCL_MKID_IDS,
    exclude_mkid_ids: Optional[Sequence[int]] = DEFAULT_EXCL_MKID_IDS,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    # options for analysis
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "zscan",
    **options: Any,
) -> Path:
    """Quick-look at an observation of subref axial focus scan.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to bad MKID IDs found on 2023-11-19.
        data_type: Data type of the input DEMS file.
            Defaults to the ``long_name`` attribute in it.
        chan_weight: Weighting method along the channel axis.
            uniform: Uniform weight (i.e. no channel dependence).
            std: Inverse square of temporal standard deviation of sky.
            std/tx: Same as std but std is divided by the atmospheric
            transmission calculated by the ATM model.
        pwv: PWV in units of mm. Only used for the calculation of
            the atmospheric transmission when chan_weight is std/tx.
        format: Output image format of quick-look result.
        outdir: Output directory for the quick-look result.
        suffix: Suffix that precedes the file extension.
        overwrite: Whether to overwrite the output if it exists.

    Keyword Args:
        options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with xr.set_options(keep_attrs=True):
        da = load_dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            data_type=data_type,
        )

        # make continuum series
        da_on = select.by(da, "state", include="ON")
        da_off = select.by(da, "state", exclude="ON")
        weight = get_chan_weight(da_off, method=chan_weight, pwv=pwv)
        series = da_on.weighted(weight.fillna(0)).mean("chan")

        # save result
        suffixes = f".{suffix}.{format}"
        file = Path(outdir) / Path(dems).with_suffix(suffixes).name

        if format in DATA_FORMATS:
            return save_qlook(series, file, overwrite=overwrite, **options)

        fig, axes = plt.subplots(1, 2, figsize=DEFAULT_FIGSIZE)

        ax = axes[0]
        plot.data(series, hue="aste_subref_z", ax=ax)

        ax = axes[1]
        plot.data(series, x="aste_subref_z", ax=ax)

        for ax in axes:
            ax.set_title(Path(dems).name)
            ax.grid(True)

        fig.tight_layout()
        return save_qlook(fig, file, overwrite=overwrite, **options)


def mean_in_time(dems: xr.DataArray) -> xr.DataArray:
    """Similar to DataArray.mean but keeps middle time."""
    middle = dems[len(dems) // 2 : len(dems) // 2 + 1]
    return xr.zeros_like(middle) + dems.mean("time")


def subtract_per_scan(dems: xr.DataArray) -> xr.DataArray:
    """Apply source-sky subtraction to a single-scan DEMS."""
    t_amb = 273.15
    if len(states := np.unique(dems.state)) != 1:
        raise ValueError("State must be unique.")

    if (state := states[0]) == "ON":
        src = select.by(dems, "beam", include="A")
        sky = select.by(dems, "beam", include="B")
        return (src.mean("time") - sky.mean("time").data) * t_amb / ((t_amb - sky.mean("time")))

    if state == "OFF":
        src = select.by(dems, "beam", include="B")
        sky = select.by(dems, "beam", include="A")
        return (src.mean("time") - sky.mean("time").data)* t_amb/ ((t_amb - sky.mean("time")))

    raise ValueError("State must be either ON or OFF.")


def get_chan_weight(
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
        with catch_warnings():
            simplefilter("ignore")
            return dems.std("time") ** -2

    if method == "std/tx":
        tx = load.atm(type="eta").sel(pwv=float(pwv))
        freq = convert.units(dems.d2_mkid_frequency, tx.freq)

        with catch_warnings():
            simplefilter("ignore")
            return (dems.std("time") / tx.interp(freq=freq)) ** -2

    raise ValueError("Method must be either uniform, std, or std/tx.")


def get_robust_lim(da: xr.DataArray) -> tuple[float, float]:
    """Calculate a robust limit for plotting."""
    sigma = SIGMA_OVER_MAD * utils.mad(da)

    return (
        float(np.nanpercentile(da.data, 1) - sigma),
        float(np.nanpercentile(da.data, 99) + sigma),
    )


def load_dems(
    dems: Path,
    /,
    *,
    include_mkid_ids: Optional[Sequence[int]] = DEFAULT_INCL_MKID_IDS,
    exclude_mkid_ids: Optional[Sequence[int]] = DEFAULT_EXCL_MKID_IDS,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    frequency_units: str = DEFAULT_FREQUENCY_UNITS,
    skycoord_units: str = DEFAULT_SKYCOORD_UNITS,
) -> xr.DataArray:
    """Load a DEMS with given conversions and selections.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to bad MKID IDs found on 2023-11-19.
        data_type: Data type of the input DEMS file.
            Defaults to the ``long_name`` attribute in it.
        frequency_units: Units of the frequency.
        skycoord_units: Units of the sky coordinate axes.

    Returns:
        DEMS as a DataArray with given conversion and selections.

    """
    da = load.dems(dems, chunks=None)

    if da.frame == "altaz":
        z = np.pi / 2 - convert.units(da.lat, "rad")
        secz = cast(xr.DataArray, 1 / np.cos(z))

        da = da.assign_coords(
            secz=secz.assign_attrs(
                long_name="sec(Z)",
                units="dimensionless",
            )
        )

    da = assign.scan(da, by="state")
    da = convert.frame(da, "relative")
    da = select.by(da, "d2_mkid_type", "filter")
    da = select.by(
        da,
        "d2_mkid_id",
        include=include_mkid_ids,
        exclude=exclude_mkid_ids,
    )
    da = convert.coord_units(
        da,
        ["d2_mkid_frequency", "frequency"],
        frequency_units,
    )
    da = convert.coord_units(
        da,
        ["lat", "lat_origin", "lon", "lon_origin"],
        skycoord_units,
    )

    if data_type == "auto" and "units" in da.attrs:
        return da

    if data_type == "brightness":
        return da.assign_attrs(long_name="Brightness", units="K")

    if data_type == "df/f":
        return da.assign_attrs(long_name="df/f", units="dimensionless")

    raise ValueError("Data type could not be inferred.")


def save_qlook(
    qlook: Union[Figure, xr.DataArray],
    file: Path,
    /,
    *,
    overwrite: bool = False,
    **options: Any,
) -> Path:
    """Save a quick look result to a file with given format.

    Args:
        qlook: Matplotlib figure or DataArray to be saved.
        file: Path of the saved file.
        overwrite: Whether to overwrite the file if it exists.

    Keyword Args:
        options: Other options to be used when saving the file.

    Returns:
        Absolute path of the saved file.

    """
    path = Path(file).expanduser().resolve()

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists.")

    if isinstance(qlook, Figure):
        qlook.savefig(path, **options)
        plt.close(qlook)
        return path

    if path.name.endswith(".csv"):
        name = qlook.attrs["data_type"]
        ds = qlook.to_dataset(name=name)
        ds.to_pandas().to_csv(path, **options)
        return path

    if path.name.endswith(".nc"):
        qlook.to_netcdf(path, **options)
        return path

    if path.name.endswith(".zarr"):
        qlook.to_zarr(path, mode="w", **options)
        return path

    if path.name.endswith(".zarr.zip"):
        qlook.to_zarr(path, mode="w", **options)
        return path

    raise ValueError("Extension of filename is not valid.")


def main() -> None:
    """Entry point of the decode-qlook command."""

    Fire(
        {
            "auto": auto,
            "pswsc": pswsc,
            "raster": raster,
            "skydip": skydip,
            "still": still,
            "zscan": zscan,
        }
    )
