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
import tomli_w as toml
from contextlib import contextmanager
from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union, Iterable, Dict
from warnings import catch_warnings, simplefilter
from datetime import datetime


# dependencies
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from astropy.units import Quantity
from fire import Fire
from matplotlib.figure import Figure
from scipy.optimize import curve_fit, minimize
from iminuit import Minuit
from . import assign, convert, load, make, plot, select, utils


# constants
ABBA_PHASES = {0, 1, 2, 3}
DATA_FORMATS = "csv", "nc", "zarr", "zarr.zip"
TEXT_FORMATS = ("toml",)
DEFAULT_DATA_SCALING = None
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
        obs: str = da.aste_obs_file

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
    data_scaling: Optional[Literal["brightness", "df/f"]] = DEFAULT_DATA_SCALING,
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
        data_scaling: Data scaling to be used in analysis.
            Defaults to the data scaling of the input DEMS file.
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
        da = load.dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            data_scaling=data_scaling,
            skycoord_units=skycoord_units,
            skycoord_frame="relative",
            chunks=None,
        )
        da = da.sel(time=da.state != "GRAD")

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
        da_on = da.sel(time=da.state == "SCAN@ON")
        da_off = da.sel(time=da.state != "SCAN@ON")
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
            chi2, reduced_chi2 = calc_chi2(
                data, data_fitted, sigma=1.0, num_params=len(initial_guess)
            )
            fit_res_params_dict = make_fit_res_params_dict(
                popt, perr, chi2, reduced_chi2
            )
            is_gaussfit_successful = True
        except Exception as error:
            LOGGER.warning(f"An error occurred on 2D Gaussian fitting: {error}")
            is_gaussfit_successful = False

        # save result
        suffixes = f".{suffix}.{format}"
        file = Path(outdir) / Path(dems).with_suffix(suffixes).name

        if format in DATA_FORMATS:
            return save_qlook(cont, file, overwrite=overwrite, **options)

        if format in TEXT_FORMATS:
            toml_string = make_pointing_toml_string(da, fit_res_params_dict, weight)
            return save_qlook(toml_string, file, overwrite=overwrite, **options)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        ax = axes[0]  # type: ignore
        plot.data(series, ax=ax)
        ax.set_title(f"{Path(dems).name}\n({da.aste_obs_file})")

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
                f"Peak = {fit_res_params_dict['peak']:+.2e} [{cont.units}], "
                f"dAz = {fit_res_params_dict['offset_az']:+.2f} [{cont.lon.attrs['units']}], "
                f"dEl = {fit_res_params_dict['offset_el']:+.2f} [{cont.lat.attrs['units']}],\n"
                f"FWHM_maj = {fit_res_params_dict['hpbw_major']:+.2f} [{skycoord_units}], "
                f"FWHM_min = {fit_res_params_dict['hpbw_minor']:+.2f} [{skycoord_units}], "
                f"P.A. = {fit_res_params_dict['position_angle']:+.1f} [deg],\n"
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
        ax.set_xlim(-map_lim, map_lim)
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
    data_scaling: Optional[Literal["brightness", "df/f"]] = DEFAULT_DATA_SCALING,
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
        data_scaling: Data scaling to be used in analysis.
            Defaults to the data scaling of the input DEMS file.
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
        da = load.dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            data_scaling=data_scaling,
            frequency_units=frequency_units,
            skycoord_frame="relative",
            chunks=None,
        )

        da_despiked = despike(da)

        # calculate ABBA cycles and phases
        da_onoff = da_despiked.sel(time=da_despiked.state.isin(["ON", "OFF"]))
        scan_onoff = utils.phaseof(da_onoff.state)
        chop_per_scan = da_onoff.beam.groupby(scan_onoff).apply(utils.phaseof)
        is_second_half = chop_per_scan.groupby(scan_onoff).apply(
            lambda group: (group >= group.mean())
        )
        abba_cycle = (scan_onoff * 2 + is_second_half - 1) // 4
        abba_phase = (scan_onoff * 2 + is_second_half - 1) % 4

        # make spectrum
        spec = (
            da_onoff.assign_coords(abba_cycle=abba_cycle, abba_phase=abba_phase)
            .groupby("abba_cycle")
            .map(subtract_per_abba_cycle)
            .mean("abba_cycle")
        )

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
            ax.set_title(f"{Path(dems).name}\n({da.aste_obs_file})")
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
    data_scaling: Optional[Literal["brightness", "df/f"]] = DEFAULT_DATA_SCALING,
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
        data_scaling: Data scaling to be used in analysis.
            Defaults to the data scaling of the input DEMS file.
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
        da = load.dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            data_scaling=data_scaling,
            skycoord_units=skycoord_units,
            skycoord_frame="relative",
            chunks=None,
        )

        # subtract temporal baseline
        da_on = da.sel(time=da.state == "SCAN")
        da_off = da.sel(time=da.state != "SCAN")
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
            chi2, reduced_chi2 = calc_chi2(
                data, data_fitted, sigma=1.0, num_params=len(initial_guess)
            )
            fit_res_params_dict = make_fit_res_params_dict(
                popt, perr, chi2, reduced_chi2
            )
            is_gaussfit_successful = True
        except Exception as error:
            LOGGER.warning(f"An error occurred on 2D Gaussian fitting: {error}")
            is_gaussfit_successful = False

        # save result
        suffixes = f".{suffix}.{format}"
        file = Path(outdir) / Path(dems).with_suffix(suffixes).name

        if format in DATA_FORMATS:
            return save_qlook(cont, file, overwrite=overwrite, **options)

        if format in TEXT_FORMATS:
            toml_string = make_pointing_toml_string(da, fit_res_params_dict, weight)
            return save_qlook(toml_string, file, overwrite=overwrite, **options)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        ax = axes[0]  # type: ignore
        plot.data(series, ax=ax)
        ax.set_title(f"{Path(dems).name}\n({da.aste_obs_file})")

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
                f"Peak = {fit_res_params_dict['peak']:+.2e} [{cont.units}], "
                f"dAz = {fit_res_params_dict['offset_az']:+.2f} [{cont.lon.attrs['units']}], "
                f"dEl = {fit_res_params_dict['offset_el']:+.2f} [{cont.lat.attrs['units']}],\n"
                f"FWHM_maj = {fit_res_params_dict['hpbw_major']:+.2f} [{skycoord_units}], "
                f"FWHM_min = {fit_res_params_dict['hpbw_minor']:+.2f} [{skycoord_units}], "
                f"P.A. = {fit_res_params_dict['position_angle']:+.1f} [deg],\n"
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
        ax.set_xlim(-map_lim, map_lim)
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
    data_scaling: Optional[Literal["brightness", "df/f"]] = DEFAULT_DATA_SCALING,
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
        data_scaling: Data scaling to be used in analysis.
            Defaults to the data scaling of the input DEMS file.
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
        da = load.dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            data_scaling=data_scaling,
            skycoord_frame="relative",
            chunks=None,
        )

        # add airmass as a coordinate
        # fmt: off
        airmass = (
            xr.DataArray(1 / np.sin(convert.units(da.lat, "rad")))
            .assign_attrs(
                long_name="Airmass",
                units="dimensionless",
            )
        )
        da = da.assign_coords(airmass=airmass)
        # fmt: on

        # make continuum series
        da_on = da.sel(time=da.state == "SCAN")
        da_off = da.sel(time=da.state != "SCAN")
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
            ax.set_title(f"{Path(dems).name}\n({da.aste_obs_file})")
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
    data_scaling: Optional[Literal["brightness", "df/f"]] = DEFAULT_DATA_SCALING,
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
        data_scaling: Data scaling to be used in analysis.
            Defaults to the data scaling of the input DEMS file.
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
        da = load.dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            data_scaling=data_scaling,
            skycoord_frame="relative",
            chunks=None,
        )

        # make continuum series
        da_off = da.sel(time=~da.state.isin(["ON", "OFF"]))
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
            ax.set_title(f"{Path(dems).name}\n({da.aste_obs_file})")
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
    data_scaling: Optional[Literal["brightness", "df/f"]] = DEFAULT_DATA_SCALING,
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
        data_scaling: Data scaling to be used in analysis.
            Defaults to the data scaling of the input DEMS file.
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
        data_scaling=data_scaling,
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
    data_scaling: Optional[Literal["brightness", "df/f"]] = DEFAULT_DATA_SCALING,
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
        data_scaling: Data scaling to be used in analysis.
            Defaults to the data scaling of the input DEMS file.
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
        data_scaling=data_scaling,
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
    data_scaling: Optional[Literal["brightness", "df/f"]] = DEFAULT_DATA_SCALING,
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
        data_scaling: Data scaling to be used in analysis.
            Defaults to the data scaling of the input DEMS file.
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
        data_scaling=data_scaling,
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
    data_scaling: Optional[Literal["brightness", "df/f"]] = DEFAULT_DATA_SCALING,
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
        data_scaling: Data scaling to be used in analysis.
            Defaults to the data scaling of the input DEMS file.
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
        da = load.dems(
            dems,
            include_mkid_ids=include_mkid_ids,
            exclude_mkid_ids=exclude_mkid_ids,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            data_scaling=data_scaling,
            skycoord_frame="relative",
            chunks=None,
        )

        # make continuum series
        da_on = da.sel(time=da.state == "ON")
        da_off = da.sel(time=da.state != "ON")
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
            ax.set_title(f"{Path(dems).name}\n({da.aste_obs_file})")
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


def subtract_per_abba_cycle(dems: xr.DataArray, /) -> xr.DataArray:
    """Subtract sky from source with atmospheric correction for each ABBA cycle.

    Args:
        dems: 2D DataArray (time x chan) of DEMS per ABBA cycle.

    Returns:
        1D DataArray (chan) of the mean spectrum after subtraction and correction.
        If ABBA phases per cycle are incomplete, i.e., some phases are missing,
        a spectrum filled with NaN will be returned instead.

    """
    if not set(np.unique(dems.abba_phase)) == ABBA_PHASES:
        return dems.mean("time") * np.nan

    return dems.groupby("abba_phase").map(subtract_per_abba_phase).mean("abba_phase")


def subtract_per_abba_phase(dems: xr.DataArray, /) -> xr.DataArray:
    """Subtract sky from source with atmospheric correction for each ABBA phase.

    Args:
        dems: 2D DataArray (time x chan) of DEMS per ABBA phase.

    Returns:
        1D DataArray (chan) of the mean spectrum after subtraction and correction.

    Raises:
        ValueError: Raised if ``dems.state`` is not ON-only nor OFF-only.

    """
    t_amb = 273.15

    if len(states := np.unique(dems.state)) != 1:
        raise ValueError("State must be unique.")

    if states[0] == "ON":
        src = dems.sel(time=dems.beam == "A").mean("time")
        sky = dems.sel(time=dems.beam == "B").mean("time")
    elif states[0] == "OFF":
        src = dems.sel(time=dems.beam == "B").mean("time")
        sky = dems.sel(time=dems.beam == "A").mean("time")
    else:
        raise ValueError("State must be either ON or OFF.")

    return t_amb * (src - sky.data) / (t_amb - sky)


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


def save_qlook(
    qlook: Union[Figure, xr.DataArray, str],
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
    elif isinstance(qlook, xr.DataArray):
        if path.name.endswith(".csv"):
            name = qlook.attrs["data_scaling"]
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
    elif isinstance(qlook, str):
        if path.name.endswith(".toml"):
            with open(path, "wt") as f:
                f.write(qlook)
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


def calc_chi2(data_obs, data_fit, sigma, num_params) -> tuple[float, float]:
    """Calculate chi^2 and reduced chi^2 of a 2D Gaussian fitting."""
    chi2 = np.sum(((data_obs - data_fit) / sigma) ** 2)
    dof = len(data_obs) - num_params
    reduced_chi2 = chi2 / dof
    return chi2, reduced_chi2


def make_fit_res_params_dict(popt, perr, chi2, reduced_chi2) -> dict[str, float]:
    """Aggregate 2D Gaussian fitting results as a dictionary."""
    res = {}
    res["offset_az"] = popt[1]
    res["offset_el"] = popt[2]
    res["offset_az_error"] = perr[1]
    res["offset_el_error"] = perr[2]
    res["hpbw_major"] = popt[3] * popt[4] * 2.354820
    res["hpbw_minor"] = popt[4] * 2.354820
    res["hpbw_major_error"] = popt[3] * perr[4] * 2.354820
    res["hpbw_minor_error"] = perr[4] * 2.354820
    res["position_angle"] = -np.rad2deg(popt[5] + np.pi / 2.0)
    res["position_angle_error"] = np.rad2deg(perr[5])
    res["peak"] = popt[0]
    res["peak_error"] = perr[0]
    res["floor"] = popt[6]
    res["floor_error"] = perr[6]
    res["chi2"] = chi2
    res["reduced_chi2"] = reduced_chi2
    return res


def make_pointing_toml_string(da, fit_res_params_dict, weight) -> str:
    """
    Args:
        dems: Input DEMS Object
        Dict: 2D Gaussian fitting results
        DataArray: weight

    Returns:
        str
    """
    fit_result = {k: v.item() for k, v in fit_res_params_dict.items()}
    freq_mean = np.sum(da.d2_mkid_frequency * weight) / np.sum(weight)
    unit = da.units
    if unit == "dimensionless":
        unit = ""

    result = {
        "analyses": [
            {
                "ana_datetime": datetime.strptime(da.name, "%Y%m%d%H%M%S"),
                "pwv": np.nan,
                "pwv_error": np.nan,
                "kid_infos": [
                    {
                        "unit": unit,
                        "frequency": freq_mean.item(),
                        "bandwidth": (
                            da.d2_mkid_frequency.max() - da.d2_mkid_frequency.min()
                        ).item(),
                        "pointings": [fit_result],
                        "coadd_kid_infos": [],
                    }
                ],
            }
        ]
    }
    for master_id, mkid_type, w in zip(
        da.d2_mkid_id.values, da.d2_mkid_type.values, weight.values
    ):
        result["analyses"][0]["kid_infos"][0]["coadd_kid_infos"].append(
            {
                "master_id": master_id.item(),
                "kid_type": mkid_type.item(),
                "weight": w.item(),
            }
        )
    return toml.dumps(result)

def shift_coords(da: xr.DataArray,
                 time_coords: Iterable[str],
                 time_offset: np.timedelta64 = np.timedelta64(0, "ms")
                 ) -> xr.DataArray:
    """Shifts the time coordinates of an Xarray object by a specified offset and re-interpolates specified coordinate variables onto the original time grid.

    Args:
        da (xr.DataArray): The input Xarray DataArray to be shifted.
        time_coords (Iterable[str]): A list of coordinate names to be re-interpolated (e.g., ['lat', 'lon']).
        time_offset (np.timedelta64): The time offset amount to shift. Defaults to 0 ms.

    Returns:
        xr.DataArray: A new DataArray with shifted time and re-interpolated coordinates.
    """
    shifted_time = da.time + time_offset
    temp_obj = da.assign_coords(time=shifted_time)
    interpolated_vars = {}
    for var_name in time_coords:
        var_to_interp = temp_obj.coords[var_name]
        interpolated_var = var_to_interp.interp_like(
            da,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )
        interpolated_vars[var_name] = interpolated_var
    result_obj = da.assign_coords(**interpolated_vars)
    return result_obj

def gaussfit(cont: xr.DataArray) -> Dict[str, Any]:
    """Performs a 2D Gaussian fitting on the input 2D map data.

    Args:
        cont (xr.DataArray): The input 2D map data array (e.g., intensity map).

    Returns:
        Dict[str, Any]: A dictionary containing fitting parameters (peak, x0, y0, etc.), chi-squared values, and DOF. Returns an empty dict if fitting fails.
    """
    try:
        mad = utils.mad(cont).item()
        sigma = mad * SIGMA_OVER_MAD
        
        data = np.array(copy.deepcopy(cont).data)
        data[np.isnan(data)] = 0.0
        
        x, y = np.meshgrid(np.array(cont["lon"]), np.array(cont["lat"]))
        
        amp_guess = np.nanmax(data)
        
        max_idx = np.argmax(data)
        x0_guess = x.ravel()[max_idx]
        y0_guess = y.ravel()[max_idx]
        
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        y_min, y_max = np.nanmin(y), np.nanmax(y)
        sigma_x_guess = (x_max - x_min) / 10.0
        sigma_y_guess = (y_max - y_min) / 10.0
        
        if sigma_x_guess <= 0: sigma_x_guess = 1e-4
        if sigma_y_guess <= 0: sigma_y_guess = 1e-4

        def fixed_offset_gaussian(xy, amp, x0, y0, sigma_x, sigma_y, theta):
            return gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, theta, 0)

        initial_guess = (amp_guess, x0_guess, y0_guess, sigma_x_guess, sigma_y_guess, 0)
        bounds = (
            [0, -np.inf, -np.inf, 1e-10, 1e-10, -np.pi],
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.pi],
        )
        
        popt, pcov = curve_fit(
            fixed_offset_gaussian, (x, y), data.ravel(), p0=initial_guess, bounds=bounds
        )
        perr = np.sqrt(np.diag(pcov))
        
        valid = np.isfinite(cont.data)
        n_data = np.sum(valid) 
        n_params = len(initial_guess)
        dof = int(n_data - n_params)
        data_fitted = fixed_offset_gaussian((x, y), *popt).reshape(x.shape)
        
        chi2, reduced_chi2 = calc_chi2(
            data, data_fitted, sigma, num_params=len(initial_guess)
        )
        
        popt_full = np.append(popt, 0)
        perr_full = np.append(perr, 0)
        
        fit_res_params_dict = make_fit_res_params_dict(
            popt_full, perr_full, chi2, reduced_chi2
        )
        
        param_names = ['peak', 'x0', 'y0', 'sigma_x', 'sigma_y', 'theta', 'offset']
        for i, name in enumerate(param_names):
            fit_res_params_dict[name] = popt_full[i]
            fit_res_params_dict[f"{name}_err"] = perr_full[i]
            
        fit_res_params_dict['popt'] = popt_full #type: ignore
        fit_res_params_dict['DOF'] = dof
        
        return fit_res_params_dict
    except Exception as error:
        return {}

def get_result(da: xr.DataArray,
               time_offset: float,
               ) -> Dict[str, Any]:
    """Calculates the Gaussian fitting result for a given time offset.

    Args:
        da (xr.DataArray): The input DataArray.
        time_offset (float): The time offset in milliseconds to be applied.

    Returns:
        Dict[str, Any]: A dictionary containing the Gaussian fitting results (parameters, chi2, etc.).
    """

    time_offset_ns = int(time_offset * 1000000)
    time_offset_np = np.timedelta64(int(np.round(time_offset_ns)), 'ns')

    da_shifted = shift_coords(da,['lat', 'lon'], time_offset_np)
    da_scan = da_shifted[da_shifted.state == "SCAN"]
    time_profile = da_scan.mean("chan", skipna=True)
    peak_idx = time_profile.argmax(dim="time")

    peak_lon = da_scan.lon[peak_idx].item()
    peak_lat = da_scan.lat[peak_idx].item()

    mask_spatial = (abs(da_shifted.lon - peak_lon) <= 60) & \
                   (abs(da_shifted.lat - peak_lat) <= 60)
    mask_scan = (da_shifted.state == "SCAN")
    mask_off = (da_shifted.state != "SCAN")
    da_cropped = da_shifted.where((mask_scan & mask_spatial) | mask_off, drop=True)

    map_data = mapping(da_cropped, da.frequency.min().item(), da.frequency.max().item())

    result = gaussfit(map_data)
    return result

def mapping(da: xr.DataArray,
            min_freq: float,
            max_freq: float
            ) -> xr.DataArray:
    
    """Generates a 2D map from the input DataArray by performing sky subtraction and gridding.

    Args:
        da (xr.DataArray): The input DataArray containing ON/OFF scan data.
        min_freq (float): Minimum frequency for the map integration (GHz).
        max_freq (float): Maximum frequency for the map integration (GHz).

    Returns:
        xr.DataArray: The generated 2D map averaged over the frequency range. Returns None if data is insufficient.
    """

    da_on = da[da.state == "SCAN"]
    da_off = da[da.state != "SCAN"]
    da_base = (
       da_off.groupby("scan")
       .map(mean_in_time)
       .interp_like(
          da_on,method="linear",kwargs={"fill_value": "extrapolate"},
          )
    )
    with catch_warnings():
       simplefilter('ignore')
       cube = make.cube(da_on - da_base.data, skycoord_grid='3 arcsec')
    condition = (cube.frequency >= min_freq) & (cube.frequency <= max_freq)
    map = cube.where(condition, drop=True).mean("chan")
    return map

def visualize_correction(da: xr.DataArray,
                         min_freq: float,
                         max_freq: float,
                         offset_val: float,
                         save_path: str
                         ) -> None:
    
    """Visualizes the corrected map, best-fit model, and residuals for a specific time offset.

    Args:
        da (xr.DataArray): The input DataArray.
        offset_val (float): The optimized time offset in milliseconds.
        min_freq (float): Minimum frequency for visualization (GHz).
        max_freq (float): Maximum frequency for visualization (GHz).
        save_path (Optional[str]): File path to save the plot image. If None, the plot is displayed but not saved.

    Returns:
        None
    """
    if np.isnan(offset_val):
        return

    offset_ns = int(offset_val * 1_000_000)
    offset_td = np.timedelta64(int(np.round(offset_ns)), 'ns')
    da_shifted = shift_coords(da, ['lat', 'lon'], offset_td)
    
    data_map = mapping(da_shifted, min_freq, max_freq)
    
    fit_result = gaussfit(data_map)
    
    if 'popt' not in fit_result:
        return

    popt = fit_result['popt']
    
    Z = np.array(data_map.data)
    Z[np.isnan(Z)] = 0.0
    
    X, Y = np.meshgrid(np.array(data_map["lon"]), np.array(data_map["lat"]))
    
    model_data = gaussian_2d((X, Y), *popt).reshape(X.shape)
    residual_data = Z - model_data

    model_da = xr.DataArray(model_data, coords=data_map.coords, dims=data_map.dims, name='Model')
    residual_da = xr.DataArray(residual_data, coords=data_map.coords, dims=data_map.dims, name='Residual')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    
    data_map.plot.pcolormesh(ax=axes[0], cmap='coolwarm', cbar_kwargs={'label': 'Intensity (K)'})
    axes[0].set_title(f'Data (Shifted: {offset_val:.2f} ms)')
    axes[0].set_aspect('equal')
    
    model_da.plot.pcolormesh(ax=axes[1], cmap='coolwarm', cbar_kwargs={'label': 'Intensity (K)'})
    axes[1].set_title('Best-fit Model')
    axes[1].set_aspect('equal')
    axes[1].set_ylabel('')
    
    vmax_res = np.nanmax(np.abs(residual_data))
    residual_da.plot.pcolormesh(ax=axes[2], cmap='viridis', vmin=-vmax_res, vmax=vmax_res, cbar_kwargs={'label': 'Residual (K)'})
    axes[2].set_title('Residual (Data - Model)')
    axes[2].set_aspect('equal')
    axes[2].set_ylabel('')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def make_toml(file_prefix: str,
              min_freq: float,
              max_freq: float,
              opt_offset: float,
              hesse_err: float,
              init_val: float,
              init_peak: float,
              final_res: Dict[str, Any]) -> str:
    
    """Generates a TOML formatted string from the analysis results.

    Args:
        file_prefix (str): Prefix used for file identification.
        min_freq (float): Minimum frequency used.
        max_freq (float): Maximum frequency used.
        opt_offset (float): Optimized time offset.
        hesse_err (float): Hessian error of the time offset.
        init_val (float): Initial chi-squared value (at 0 offset).
        opt_val (float): Optimized chi-squared value.
        init_peak (float): Initial peak intensity.
        opt_peak (float): Optimized peak intensity.
        final_res (Dict[str, Any]): Dictionary containing final fitting parameters.

    Returns:
        str: A TOML formatted string.
    """

    def to_native(val):
        if isinstance(val, (np.integer, int)):
            return int(val)
        elif isinstance(val, (np.floating, float)):
            return float(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        return val

    data_dict = {
        "file_id": file_prefix,
        "min_freq": to_native(min_freq),
        "max_freq": to_native(max_freq),
        "optimal_offset": to_native(opt_offset),
        "hesse_error": to_native(hesse_err),
        "initial_chi2": to_native(init_val),
        "initial_peak": to_native(init_peak),
    }

    for k, v in final_res.items():
        if k != 'popt':
            data_dict[k] = to_native(v)
            
    return toml.dumps(data_dict)

def timeoffset_search(dems: str,
                      min_freq: float,
                      max_freq: float,
                      key: str = 'chi2',
                      save_dir=".")->None:

    """Searches for the optimal time offset to minimize the specified key (e.g., chi2) and saves the results.

    Args:
        dems (str): Path to the DEMS file (Zarr/Zip).
        min_freq (float): Minimum frequency for analysis (GHz).
        max_freq (float): Maximum frequency for analysis (GHz).
        key (str, optional): Key to minimize (default is 'chi2').
        save_dir (str, optional): Directory to save output files (default is current directory).

    Returns:
        Tuple[Optional[int], float, Dict[str, Any]]: A tuple containing (Rounded optimal offset, Hesse error, Final result dictionary). Returns (None, nan, nan, {}) if failed.
    """

    da = load.dems(dems, skycoord_frame='relative', data_scaling = 'brightness').compute()
    da = da.where((da.frequency >= min_freq) & (da.frequency <= max_freq),drop=True)
    
    basename = dems
    if basename.endswith(".zarr.zip"):
        file_prefix = basename.replace(".zarr.zip", "")
    elif basename.endswith(".zarr"):
        file_prefix = basename.replace(".zarr", "")
    else:
        file_prefix = basename

    toml_path = f"{save_dir}/{file_prefix}_result.toml"
    plot_path = f"{save_dir}/{file_prefix}_profile.png"
    map_path = f"{save_dir}/{file_prefix}_maps.png"
    init_res = get_result(da, 0.0)

    init_val = init_res.get(key, np.nan)
    init_peak = init_res.get('peak', np.nan)

    def objective(offset:float)->float:
        if isinstance(offset, np.ndarray): offset_val = offset[0]
        else: offset_val = offset
        try: offset_val = float(offset_val)
        except: return np.inf
        if np.isnan(offset_val) or np.isinf(offset_val): return np.inf
        
        try:
            result_dict = get_result(da, offset_val)
            current_val = result_dict.get(key, np.inf)
        except Exception: return np.inf
        
        if np.isnan(current_val): return np.inf
        return current_val

    initial_guess = [50.0]
    bounds = [(-50, 100)]
    
    res_scipy = minimize(lambda x: objective(x[0]), initial_guess, method='Powell', bounds=bounds, options={'ftol': 1e-3})
    scipy_offset = res_scipy.x[0] if res_scipy.success else 50.0

    m = Minuit(objective, offset=scipy_offset)# type: ignore
    m.errordef = Minuit.LEAST_SQUARES
    m.limits['offset'] = (0, 100)
    m.errors["offset"] = 1.0 
    m.tol = 1.0
    
    m.simplex() 
    
    try: m.hesse()
    except: pass

    opt_offset = m.values["offset"]
    hesse_err = m.errors["offset"]
    opt_val = m.fval

    final_res = get_result(da, opt_offset)
    
    try:
        offset_grid, val_grid = m.profile("offset", bound=3, subtract_min=False)
        plt.figure(figsize=(10, 6))
        plt.plot(offset_grid, val_grid, label=f'{key} Profile', color='blue', linewidth=2)
        plt.plot(opt_offset, opt_val, 'ro', markersize=8, label='Minimum')# type: ignore
        
        lower_bound = opt_offset - hesse_err
        upper_bound = opt_offset + hesse_err
        
        plt.axvspan(lower_bound, upper_bound, color='green', alpha=0.2, label='Hesse Error Range')
        plt.axvline(lower_bound, color='green', linestyle='--', alpha=0.5)
        plt.axvline(upper_bound, color='green', linestyle='--', alpha=0.5)
        
        plt.xlabel("Time Offset (ms)", fontsize=12)
        plt.ylabel(key, fontsize=12)
        plt.title(f"{key} Profile ({file_prefix})\nOffset = {opt_offset:.2f} $\\pm$ {hesse_err:.2f} ms", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if plot_path:
            plt.savefig(plot_path, dpi=100)
            plt.close()
        else:
            plt.show()
    except:
        pass

        visualize_correction(da, min_freq, max_freq, opt_offset, save_path=map_path)
    if np.isnan(opt_offset):
        return None
        
    if toml_path:
        try:
            toml_str = make_toml(file_prefix, min_freq, max_freq, opt_offset, hesse_err, init_val, init_peak, final_res)
            
            with open(toml_path, "w") as f:
                f.write(toml_str)
            print(f"Results saved to {toml_path}")
        except Exception as e:
            print(f"Failed to save TOML: {e}")

    return None

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
