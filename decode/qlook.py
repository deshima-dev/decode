__all__ = [
    "auto",
    "daisy",
    "pswsc",
    "raster",
    "skydip",
    "still",
    "xscan",
    "yscan",
    "zscan",
]


# standard library
import copy
from contextlib import contextmanager
from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union, cast
from warnings import catch_warnings, simplefilter


# dependencies
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from astropy.units import Quantity
from fire import Fire
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from . import assign, convert, load, make, plot, select, utils


# constants
DATA_FORMATS = "csv", "nc", "zarr", "zarr.zip"
DEFAULT_DATA_TYPE = "auto"
DEFAULT_DEBUG = False
DEFAULT_FIGSIZE = 12, 4
DEFAULT_FORMAT = "png"
DEFAULT_FREQUENCY_UNITS = "GHz"
DEFAULT_EXCL_MKID_IDS = None
DEFAULT_INCL_MKID_IDS = None
DEFAULT_MIN_FREQUENCY = None
DEFAULT_MAX_FREQUENCY = None
DEFAULT_ROLLING_TIME = 200
DEFAULT_OUTDIR = Path()
DEFAULT_OVERWRITE = False
DEFAULT_SKYCOORD_GRID = "6 arcsec"
DEFAULT_SKYCOORD_UNITS = "arcsec"
SIGMA_OVER_MAD = 1.4826
LOGGER = getLogger(__name__)


@contextmanager
def set_logger(debug: bool):
    level = LOGGER.level

    if debug:
        LOGGER.setLevel(DEBUG)

    try:
        yield
    finally:
        LOGGER.setLevel(level)


def auto(dems: Path, /, **options: Any) -> Path:
    """Quick-look at an observation with auto-selected command.

    The used command will be selected based on the observation name
    stored as the ``observation`` attribute in an input DEMS file.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        **options: Options for the selected command.
            See the command help for all available options.

    Returns:
        Absolute path of the saved file.

    """
    with xr.set_options(keep_attrs=True):
        da = load.dems(dems, chunks=None)
        obs: str = da.attrs["observation"]

        if "daisy" in obs:
            return daisy(dems, **options)

        if "pswsc" in obs:
            return pswsc(dems, **options)

        if "raster" in obs:
            return raster(dems, **options)

        if "skydip" in obs:
            return skydip(dems, **options)

        if "still" in obs:
            return still(dems, **options)

        if "xscan" in obs:
            return xscan(dems, **options)

        if "yscan" in obs:
            return yscan(dems, **options)

        if "zscan" in obs:
            return zscan(dems, **options)

        raise ValueError(
            f"Could not infer the command to be used from {obs!r}: "
            "Observation name must include one of the command names. "
        )


def daisy(
    dems: Path,
    /,
    *,
    # options for loading
    include_mkid_ids: Optional[Sequence[int]] = DEFAULT_INCL_MKID_IDS,
    exclude_mkid_ids: Optional[Sequence[int]] = DEFAULT_EXCL_MKID_IDS,
    min_frequency: Optional[str] = DEFAULT_MIN_FREQUENCY,
    max_frequency: Optional[str] = DEFAULT_MAX_FREQUENCY,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    # options for analysis
    rolling_time: int = DEFAULT_ROLLING_TIME,
    source_radius: str = "60 arcsec",
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    skycoord_grid: str = DEFAULT_SKYCOORD_GRID,
    skycoord_units: str = DEFAULT_SKYCOORD_UNITS,
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "daisy",
    # other options
    debug: bool = DEFAULT_DEBUG,
    **options: Any,
) -> Path:
    """Quick-look at a daisy scan observation.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to no MKID IDs.
        min_frequency: Minimum frequency to be included in analysis.
            Defaults to no minimum frequency bound.
        max_frequency: Maximum frequency to be included in analysis.
            Defaults to no maximum frequency bound.
        data_type: Data type of the input DEMS file.
            Defaults to the ``long_name`` attribute in it.
        rolling_time: Moving window size.
        source_radius: Radius of the on-source area.
            Other areas are considered off-source in sky subtraction.
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
        overwrite: Whether to overwrite the output if it exists.
        suffix: Suffix that precedes the file extension.
        debug: Whether to print detailed logs for debugging.
        **options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with set_logger(debug):
        for key, val in locals().items():
            LOGGER.debug(f"{key}: {val!r}")

    with xr.set_options(keep_attrs=True):
        da = load_dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            data_type=data_type,
            skycoord_units=skycoord_units,
        )
        da = select.by(da, "state", exclude="GRAD")

        ### Rolling
        da_rolled = da.rolling(time=int(rolling_time), center=True).mean()
        da = da - da_rolled

        # fmt: off
        is_source = (
            (da.lon**2 + da.lat**2)
            < Quantity(source_radius).to(skycoord_units).value ** 2
        )
        # fmt: on
        da.coords["state"][is_source] = "SCAN@ON"
        assign.scan(da, by="state", inplace=True)

        # subtract temporal baseline
        da_on = select.by(da, "state", include="SCAN@ON")
        da_off = select.by(da, "state", exclude="SCAN@ON")
        da_base = (
            da_off.groupby("scan")
            .map(mean_in_time)
            .interp_like(
                da_on,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
        )
        t_atm = da_on.temperature
        da_sub = t_atm * (da_on - da_base) / (t_atm - da_base)

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

        ### GaussFit (cont)
        try:
            data = np.array(copy.deepcopy(cont).data)
            data[np.isnan(data)] = 0.0
            x, y = np.meshgrid(np.array(cube["lon"]), np.array(cube["lat"]))
            initial_guess = (1, 0, 0, 1, 10, 0, 0)
            bounds = (
                [0, -np.inf, -np.inf, 1, 0, -np.pi, -np.inf],
                [np.inf, np.inf, np.inf, np.inf, np.inf, 0, np.inf],
            )
            popt, pcov = curve_fit(
                gaussian_2d, (x, y), data.ravel(), p0=initial_guess, bounds=bounds
            )
            perr = np.sqrt(np.diag(pcov))
            data_fitted = gaussian_2d((x, y), *popt).reshape(x.shape)
            is_gaussfit_successful = True
        except:
            is_gaussfit_successful = False

        # save result
        suffixes = f".{suffix}.{format}"
        file = Path(outdir) / Path(dems).with_suffix(suffixes).name

        if format in DATA_FORMATS:
            return save_qlook(cont, file, overwrite=overwrite, **options)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        ax = axes[0]  # type: ignore
        plot.data(series, ax=ax)
        ax.set_title(f"{Path(dems).name}\n({da.observation})")

        ax = axes[1]  # type: ignore
        map_lim = max(abs(cube.lon).max(), abs(cube.lat).max())
        max_pix = cont.where(cont == cont.max(), drop=True)

        cont.plot(ax=ax)  # type: ignore
        if is_gaussfit_successful:
            ax.contour(
                data_fitted,
                extent=(x.min(), x.max(), y.min(), y.max()),
                origin="lower",
                levels=np.linspace(0, popt[0], 9) + popt[6],
                colors="k",
                linewidths=[0.75, 0.75, 0.75, 0.75, 1.50, 0.75, 0.75, 0.75, 0.75],
                linestyles="-",
            )
            ax.set_title(
                f"Peak = {popt[0]:+.2e} [{cont.units}], "
                f"dAz = {popt[1]:+.2f} [{cont.lon.attrs['units']}], "
                f"dEl = {popt[2]:+.2f} [{cont.lat.attrs['units']}],\n"
                f"FWHM_x = {popt[3]*popt[4]*2.354820:+.2f} [{skycoord_units}], "
                f"FWHM_y = {popt[4]*2.354820:+.2f} [{skycoord_units}], "
                f"P.A. = {np.rad2deg(popt[5]+np.pi/2.0):+.1f} [deg],\n"
                f"min_frequency = {min_frequency}, "
                f"max_frequency = {max_frequency}",
                fontsize=10,
            )
        else:
            ax.set_title(
                f"Peak = {cont.max():.2e} [{cont.units}], "
                f"dAz = {float(max_pix.lon):+.1f} [{cont.lon.attrs['units']}], "
                f"dEl = {float(max_pix.lat):+.1f} [{cont.lat.attrs['units']}],\n"
                f"min_frequency = {min_frequency}, "
                f"max_frequency = {max_frequency}\n"
                "(Gaussian fit failed: dAz and dEl are peak pixel based)",
                fontsize=10,
            )
        ax.set_xlim(map_lim, -map_lim)
        ax.set_ylim(-map_lim, map_lim)
        ax.axes.set_aspect("equal", "datalim")

        for ax in axes:  # type: ignore
            ax.grid(True)

        fig.tight_layout()
        return save_qlook(fig, file, overwrite=overwrite, **options)


def pswsc(
    dems: Path,
    /,
    *,
    # options for loading
    include_mkid_ids: Optional[Sequence[int]] = DEFAULT_INCL_MKID_IDS,
    exclude_mkid_ids: Optional[Sequence[int]] = DEFAULT_EXCL_MKID_IDS,
    min_frequency: Optional[str] = DEFAULT_MIN_FREQUENCY,
    max_frequency: Optional[str] = DEFAULT_MAX_FREQUENCY,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    frequency_units: str = DEFAULT_FREQUENCY_UNITS,
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "pswsc",
    # other options
    debug: bool = DEFAULT_DEBUG,
    **options: Any,
) -> Path:
    """Quick-look at a PSW observation with sky chopper.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to no MKID IDs.
        min_frequency: Minimum frequency to be included in analysis.
            Defaults to no minimum frequency bound.
        max_frequency: Maximum frequency to be included in analysis.
            Defaults to no maximum frequency bound.
        data_type: Data type of the input DEMS file.
            Defaults to the ``long_name`` attribute in it.
        frequency_units: Units of the frequency axis.
        format: Output data format of the quick-look result.
        outdir: Output directory for the quick-look result.
        overwrite: Whether to overwrite the output if it exists.
        suffix: Suffix that precedes the file extension.
        debug: Whether to print detailed logs for debugging.
        **options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with set_logger(debug):
        for key, val in locals().items():
            LOGGER.debug(f"{key}: {val!r}")

    with xr.set_options(keep_attrs=True):
        da = load_dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            data_type=data_type,
            frequency_units=frequency_units,
        )

        da_despiked = despike(da)

        # make spectrum
        da_scan = select.by(da_despiked, "state", ["ON", "OFF"])
        da_sub = da_scan.groupby("scan").map(subtract_per_scan)
        spec = da_sub.mean("scan")

        # save result
        suffixes = f".{suffix}.{format}"
        file = Path(outdir) / Path(dems).with_suffix(suffixes).name

        if format in DATA_FORMATS:
            return save_qlook(spec, file, overwrite=overwrite, **options)

        fig, axes = plt.subplots(1, 2, figsize=DEFAULT_FIGSIZE)

        ax = axes[0]  # type: ignore
        plot.data(da_despiked.scan, ax=ax)

        ax = axes[1]  # type: ignore
        plot.data(spec, x="frequency", s=5, hue=None, ax=ax)
        ax.set_ylim(get_robust_lim(spec))

        for ax in axes:  # type: ignore
            ax.set_title(f"{Path(dems).name}\n({da.observation})")
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
    min_frequency: Optional[str] = DEFAULT_MIN_FREQUENCY,
    max_frequency: Optional[str] = DEFAULT_MAX_FREQUENCY,
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
    # other options
    debug: bool = DEFAULT_DEBUG,
    **options: Any,
) -> Path:
    """Quick-look at a raster scan observation.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to no MKID IDs.
        min_frequency: Minimum frequency to be included in analysis.
            Defaults to no minimum frequency bound.
        max_frequency: Maximum frequency to be included in analysis.
            Defaults to no maximum frequency bound.
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
        overwrite: Whether to overwrite the output if it exists.
        suffix: Suffix that precedes the file extension.
        debug: Whether to print detailed logs for debugging.
        **options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with set_logger(debug):
        for key, val in locals().items():
            LOGGER.debug(f"{key}: {val!r}")

    with xr.set_options(keep_attrs=True):
        da = load_dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
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
        t_atm = da_on.temperature
        da_sub = t_atm * (da_on - da_base) / (t_atm - da_base)

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

        ### GaussFit (cont)
        try:
            data = np.array(copy.deepcopy(cont).data)
            data[np.isnan(data)] = 0.0
            x, y = np.meshgrid(np.array(cube["lon"]), np.array(cube["lat"]))
            initial_guess = (1, 0, 0, 1, 10, 0, 0)
            bounds = (
                [0, -np.inf, -np.inf, 1, 0, -np.pi, -np.inf],
                [np.inf, np.inf, np.inf, np.inf, np.inf, 0, np.inf],
            )
            popt, pcov = curve_fit(
                gaussian_2d, (x, y), data.ravel(), p0=initial_guess, bounds=bounds
            )
            perr = np.sqrt(np.diag(pcov))
            data_fitted = gaussian_2d((x, y), *popt).reshape(x.shape)
            is_gaussfit_successful = True
        except:
            is_gaussfit_successful = False

        # save result
        suffixes = f".{suffix}.{format}"
        file = Path(outdir) / Path(dems).with_suffix(suffixes).name

        if format in DATA_FORMATS:
            return save_qlook(cont, file, overwrite=overwrite, **options)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        ax = axes[0]  # type: ignore
        plot.data(series, ax=ax)
        ax.set_title(f"{Path(dems).name}\n({da.observation})")

        ax = axes[1]  # type: ignore
        map_lim = max(abs(cube.lon).max(), abs(cube.lat).max())
        max_pix = cont.where(cont == cont.max(), drop=True)

        cont.plot(ax=ax)  # type: ignore
        if is_gaussfit_successful:
            ax.contour(
                data_fitted,
                extent=(x.min(), x.max(), y.min(), y.max()),
                origin="lower",
                levels=np.linspace(0, popt[0], 9) + popt[6],
                colors="k",
                linewidths=[0.75, 0.75, 0.75, 0.75, 1.50, 0.75, 0.75, 0.75, 0.75],
                linestyles="-",
            )
            ax.set_title(
                f"Peak = {popt[0]:+.2e} [{cont.units}], "
                f"dAz = {popt[1]:+.2f} [{cont.lon.attrs['units']}], "
                f"dEl = {popt[2]:+.2f} [{cont.lat.attrs['units']}],\n"
                f"FWHM_x = {popt[3]*popt[4]*2.354820:+.2f} [{skycoord_units}], "
                f"FWHM_y = {popt[4]*2.354820:+.2f} [{skycoord_units}], "
                f"P.A. = {np.rad2deg(popt[5]+np.pi/2.0):+.1f} [deg],\n"
                f"min_frequency = {min_frequency}, "
                f"max_frequency = {max_frequency}",
                fontsize=10,
            )
        else:
            ax.set_title(
                f"Peak = {cont.max():.2e} [{cont.units}], "
                f"dAz = {float(max_pix.lon):+.1f} [{cont.lon.attrs['units']}], "
                f"dEl = {float(max_pix.lat):+.1f} [{cont.lat.attrs['units']}],\n"
                f"min_frequency = {min_frequency}, "
                f"max_frequency = {max_frequency}\n"
                "(Gaussian fit failed: dAz and dEl are peak pixel based)",
                fontsize=10,
            )
        ax.set_xlim(map_lim, -map_lim)
        ax.set_ylim(-map_lim, map_lim)
        ax.axes.set_aspect("equal", "datalim")

        for ax in axes:  # type: ignore
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
    min_frequency: Optional[str] = DEFAULT_MIN_FREQUENCY,
    max_frequency: Optional[str] = DEFAULT_MAX_FREQUENCY,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    # options for analysis
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "skydip",
    # other options
    debug: bool = DEFAULT_DEBUG,
    **options: Any,
) -> Path:
    """Quick-look at a skydip observation.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to no MKID IDs.
        min_frequency: Minimum frequency to be included in analysis.
            Defaults to no minimum frequency bound.
        max_frequency: Maximum frequency to be included in analysis.
            Defaults to no maximum frequency bound.
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
        overwrite: Whether to overwrite the output if it exists.
        suffix: Suffix that precedes the file extension.
        debug: Whether to print detailed logs for debugging.
        **options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with set_logger(debug):
        for key, val in locals().items():
            LOGGER.debug(f"{key}: {val!r}")

    with xr.set_options(keep_attrs=True):
        da = load_dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
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

        ax = axes[0]  # type: ignore
        plot.data(series, hue="secz", ax=ax)

        ax = axes[1]  # type: ignore
        plot.data(series, x="secz", ax=ax)

        for ax in axes:  # type: ignore
            ax.set_title(f"{Path(dems).name}\n({da.observation})")
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
    min_frequency: Optional[str] = DEFAULT_MIN_FREQUENCY,
    max_frequency: Optional[str] = DEFAULT_MAX_FREQUENCY,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    # options for analysis
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "still",
    # other options
    debug: bool = DEFAULT_DEBUG,
    **options: Any,
) -> Path:
    """Quick-look at a still observation.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to no MKID IDs.
        min_frequency: Minimum frequency to be included in analysis.
            Defaults to no minimum frequency bound.
        max_frequency: Maximum frequency to be included in analysis.
            Defaults to no maximum frequency bound.
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
        overwrite: Whether to overwrite the output if it exists.
        suffix: Suffix that precedes the file extension.
        debug: Whether to print detailed logs for debugging.
        **options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with set_logger(debug):
        for key, val in locals().items():
            LOGGER.debug(f"{key}: {val!r}")

    with xr.set_options(keep_attrs=True):
        da = load_dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
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

        ax = axes[0]  # type: ignore
        plot.state(da, add_colorbar=False, add_legend=False, ax=ax)

        ax = axes[1]  # type: ignore
        plot.data(series, add_colorbar=False, ax=ax)

        for ax in axes:  # type: ignore
            ax.set_title(f"{Path(dems).name}\n({da.observation})")
            ax.grid(True)

        fig.tight_layout()
        return save_qlook(fig, file, overwrite=overwrite, **options)


def xscan(
    dems: Path,
    /,
    *,
    # options for loading
    include_mkid_ids: Optional[Sequence[int]] = DEFAULT_INCL_MKID_IDS,
    exclude_mkid_ids: Optional[Sequence[int]] = DEFAULT_EXCL_MKID_IDS,
    min_frequency: Optional[str] = DEFAULT_MIN_FREQUENCY,
    max_frequency: Optional[str] = DEFAULT_MAX_FREQUENCY,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    # options for analysis
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "zscan",
    # other options
    debug: bool = DEFAULT_DEBUG,
    **options: Any,
) -> Path:
    """Quick-look at an observation of subref X-axis focus scan.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to no MKID IDs.
        min_frequency: Minimum frequency to be included in analysis.
            Defaults to no minimum frequency bound.
        max_frequency: Maximum frequency to be included in analysis.
            Defaults to no maximum frequency bound.
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
        overwrite: Whether to overwrite the output if it exists.
        suffix: Suffix that precedes the file extension.
        debug: Whether to print detailed logs for debugging.
        **options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with set_logger(debug):
        for key, val in locals().items():
            LOGGER.debug(f"{key}: {val!r}")

    return _scan(
        dems,
        "x",
        # options for loading
        include_mkid_ids=include_mkid_ids,
        exclude_mkid_ids=exclude_mkid_ids,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        data_type=data_type,
        # options for analysis
        chan_weight=chan_weight,
        pwv=pwv,
        # options for saving
        format=format,
        outdir=outdir,
        overwrite=overwrite,
        suffix=suffix,
        **options,
    )


def yscan(
    dems: Path,
    /,
    *,
    # options for loading
    include_mkid_ids: Optional[Sequence[int]] = DEFAULT_INCL_MKID_IDS,
    exclude_mkid_ids: Optional[Sequence[int]] = DEFAULT_EXCL_MKID_IDS,
    min_frequency: Optional[str] = DEFAULT_MIN_FREQUENCY,
    max_frequency: Optional[str] = DEFAULT_MAX_FREQUENCY,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    # options for analysis
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "zscan",
    # other options
    debug: bool = DEFAULT_DEBUG,
    **options: Any,
) -> Path:
    """Quick-look at an observation of subref Y-axis focus scan.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to no MKID IDs.
        min_frequency: Minimum frequency to be included in analysis.
            Defaults to no minimum frequency bound.
        max_frequency: Maximum frequency to be included in analysis.
            Defaults to no maximum frequency bound.
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
        overwrite: Whether to overwrite the output if it exists.
        suffix: Suffix that precedes the file extension.
        debug: Whether to print detailed logs for debugging.
        **options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with set_logger(debug):
        for key, val in locals().items():
            LOGGER.debug(f"{key}: {val!r}")

    return _scan(
        dems,
        "y",
        # options for loading
        include_mkid_ids=include_mkid_ids,
        exclude_mkid_ids=exclude_mkid_ids,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        data_type=data_type,
        # options for analysis
        chan_weight=chan_weight,
        pwv=pwv,
        # options for saving
        format=format,
        outdir=outdir,
        overwrite=overwrite,
        suffix=suffix,
        **options,
    )


def zscan(
    dems: Path,
    /,
    *,
    # options for loading
    include_mkid_ids: Optional[Sequence[int]] = DEFAULT_INCL_MKID_IDS,
    exclude_mkid_ids: Optional[Sequence[int]] = DEFAULT_EXCL_MKID_IDS,
    min_frequency: Optional[str] = DEFAULT_MIN_FREQUENCY,
    max_frequency: Optional[str] = DEFAULT_MAX_FREQUENCY,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    # options for analysis
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "zscan",
    # other options
    debug: bool = DEFAULT_DEBUG,
    **options: Any,
) -> Path:
    """Quick-look at an observation of subref Z-axis focus scan.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to no MKID IDs.
        min_frequency: Minimum frequency to be included in analysis.
            Defaults to no minimum frequency bound.
        max_frequency: Maximum frequency to be included in analysis.
            Defaults to no maximum frequency bound.
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
        overwrite: Whether to overwrite the output if it exists.
        suffix: Suffix that precedes the file extension.
        debug: Whether to print detailed logs for debugging.
        **options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with set_logger(debug):
        for key, val in locals().items():
            LOGGER.debug(f"{key}: {val!r}")

    return _scan(
        dems,
        "z",
        # options for loading
        include_mkid_ids=include_mkid_ids,
        exclude_mkid_ids=exclude_mkid_ids,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        data_type=data_type,
        # options for analysis
        chan_weight=chan_weight,
        pwv=pwv,
        # options for saving
        debug=debug,
        format=format,
        outdir=outdir,
        overwrite=overwrite,
        suffix=suffix,
        **options,
    )


def _scan(
    dems: Path,
    axis: Literal["x", "y", "z"],
    /,
    *,
    # options for loading
    include_mkid_ids: Optional[Sequence[int]] = DEFAULT_INCL_MKID_IDS,
    exclude_mkid_ids: Optional[Sequence[int]] = DEFAULT_EXCL_MKID_IDS,
    min_frequency: Optional[str] = DEFAULT_MIN_FREQUENCY,
    max_frequency: Optional[str] = DEFAULT_MAX_FREQUENCY,
    data_type: Literal["auto", "brightness", "df/f"] = DEFAULT_DATA_TYPE,
    # options for analysis
    chan_weight: Literal["uniform", "std", "std/tx"] = "std/tx",
    pwv: Literal["0.5", "1.0", "2.0", "3.0", "4.0", "5.0"] = "5.0",
    # options for saving
    format: str = DEFAULT_FORMAT,
    outdir: Path = DEFAULT_OUTDIR,
    overwrite: bool = DEFAULT_OVERWRITE,
    suffix: str = "_scan",
    # other options
    debug: bool = DEFAULT_DEBUG,
    **options: Any,
) -> Path:
    """Quick-look at an observation of subref axial/radial focus scan.

    Args:
        dems: Input DEMS file (netCDF or Zarr).
        axis: Axis of the scan (either ``'x'``, ``'y'``, or ``'z'``).
        include_mkid_ids: MKID IDs to be included in analysis.
            Defaults to all MKID IDs.
        exclude_mkid_ids: MKID IDs to be excluded in analysis.
            Defaults to no MKID IDs.
        min_frequency: Minimum frequency to be included in analysis.
            Defaults to no minimum frequency bound.
        max_frequency: Maximum frequency to be included in analysis.
            Defaults to no maximum frequency bound.
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
        overwrite: Whether to overwrite the output if it exists.
        suffix: Suffix that precedes the file extension.
        debug: Whether to print detailed logs for debugging.
        **options: Other options for saving the output (e.g. dpi).

    Returns:
        Absolute path of the saved file.

    """
    with set_logger(debug):
        for key, val in locals().items():
            LOGGER.debug(f"{key}: {val!r}")

    with xr.set_options(keep_attrs=True):
        da = load_dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            data_type=data_type,
        )

        # make continuum series
        da_on = select.by(da, "state", include="ON")
        da_off = select.by(da, "state", exclude="ON")
        weight = get_chan_weight(da_off, method=chan_weight, pwv=pwv)
        series = da_on.weighted(weight.fillna(0)).mean("chan")

        # save result
        suffixes = f".{suffix.replace('_', axis)}.{format}"
        file = Path(outdir) / Path(dems).with_suffix(suffixes).name

        if format in DATA_FORMATS:
            return save_qlook(series, file, overwrite=overwrite, **options)

        fig, axes = plt.subplots(1, 2, figsize=DEFAULT_FIGSIZE)

        ax = axes[0]  # type: ignore
        plot.data(series, hue=f"aste_subref_{axis}", ax=ax)

        ax = axes[1]  # type: ignore
        plot.data(series, x=f"aste_subref_{axis}", ax=ax)

        for ax in axes:  # type: ignore
            ax.set_title(f"{Path(dems).name}\n({da.observation})")
            ax.grid(True)

        fig.tight_layout()
        return save_qlook(fig, file, overwrite=overwrite, **options)


def despike(dems: xr.DataArray, /) -> xr.DataArray:
    is_spike = (
        xr.zeros_like(dems.time, bool)
        .reset_coords(drop=True)
        .groupby(utils.phaseof(dems.beam))
        .map(flag_spike)
    )
    return dems.where(~is_spike, drop=True)


def flag_spike(index: xr.DataArray, /) -> xr.DataArray:
    index[:1] = index[-1:] = True
    return index


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
        return (
            t_amb
            * (src.mean("time") - sky.mean("time").data)
            / ((t_amb - sky.mean("time")))
        )

    if state == "OFF":
        src = select.by(dems, "beam", include="B")
        sky = select.by(dems, "beam", include="A")
        return (
            t_amb
            * (src.mean("time") - sky.mean("time").data)
            / ((t_amb - sky.mean("time")))
        )

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
    min_frequency: Optional[str] = DEFAULT_MIN_FREQUENCY,
    max_frequency: Optional[str] = DEFAULT_MAX_FREQUENCY,
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
            Defaults to no MKID IDs.
        min_frequency: Minimum frequency to be included in analysis.
            Defaults to no minimum frequency bound.
        max_frequency: Maximum frequency to be included in analysis.
            Defaults to no maximum frequency bound.
        data_type: Data type of the input DEMS file.
            Defaults to the ``long_name`` attribute in it.
        frequency_units: Units of the frequency.
        skycoord_units: Units of the sky coordinate axes.

    Returns:
        DEMS as a DataArray with given conversion and selections.

    """
    da = load.dems(dems, chunks=None)

    if min_frequency is not None:
        min_frequency = Quantity(min_frequency).to(frequency_units).value

    if max_frequency is not None:
        max_frequency = Quantity(max_frequency).to(frequency_units).value

    if da.frame == "altaz":
        z = np.pi / 2 - convert.units(da.lat, "rad")
        secz = cast(xr.DataArray, 1 / np.cos(z))
        da = da.assign_coords(
            secz=secz.assign_attrs(long_name="sec(Z)", units="dimensionless")
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
    da = assign.scan(da, by="state")
    da = convert.frame(da, "relative")
    da = select.by(da, "d2_mkid_type", "filter")
    da = select.by(
        da,
        "chan",
        include=include_mkid_ids,
        exclude=exclude_mkid_ids,
    )
    da = select.by(
        da,
        "frequency",
        min=min_frequency,
        max=max_frequency,
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


def gaussian_2d(xy, amp, x0, y0, sigma_x_over_y, sigma_y, theta, offset):
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    sigma_x = sigma_y * sigma_x_over_y
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = offset + amp * np.exp(
        -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))
    )
    return g.ravel()


def main() -> None:
    """Entry point of the decode-qlook command."""

    basicConfig(
        datefmt="%Y-%m-%d %H:%M:%S",
        format="[%(asctime)s %(name)s %(funcName)s %(levelname)s] %(message)s",
    )

    Fire(
        {
            "auto": auto,
            "daisy": daisy,
            "pswsc": pswsc,
            "raster": raster,
            "skydip": skydip,
            "still": still,
            "xscan": xscan,
            "yscan": yscan,
            "zscan": zscan,
        }
    )
