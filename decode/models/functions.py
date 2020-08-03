# coding: utf-8


# public items
__all__ = ["pca", "chopper_calibration", "r_division", "gauss_fit"]


# standard library
from logging import getLogger


# dependent packages
import decode as dc
import numpy as np
from astropy.modeling import fitting, models
from sklearn.decomposition import TruncatedSVD


def pca(onarray, offarray, n=10, exchs=None, pc=False, mode="mean"):
    """Apply Principal Component Analysis (PCA) method to estimate baselines at each time.

    Args:
        onarray (decode.array): Decode array of on-point observations.
        offarray (decode.array): Decode array of off-point observations.
        n (int): The number of pricipal components.
        pc (bool): When True, this function also returns
            eigen vectors and their coefficients.
        mode (None or str): The way of correcting offsets.
            'mean': Mean.
            'median': Median.
            None: No correction.

    Returns:
        filtered (decode.array): Baseline-subtracted array.
        When pc is True:
            Ps (list(np.ndarray)): Eigen vectors.
            Cs (list(np.ndarray)): Coefficients.
    """
    logger = getLogger("decode.models.pca")
    logger.info("n_components exchs mode")
    if exchs is None:
        exchs = [16, 44, 46]
    logger.info("{} {} {}".format(n, exchs, mode))

    offid = np.unique(offarray.scanid)
    onid = np.unique(onarray.scanid)

    onarray = onarray.copy()  # Xarray
    onarray[:, exchs] = 0
    onvalues = onarray.values
    onscanid = onarray.scanid.values
    offarray = offarray.copy()  # Xarray
    offarray[:, exchs] = 0
    offvalues = offarray.values
    offscanid = offarray.scanid.values

    Ps, Cs = [], []
    Xatm = dc.full_like(onarray, onarray)
    Xatmvalues = Xatm.values
    model = TruncatedSVD(n_components=n)
    for i in onid:
        leftid = np.searchsorted(offid, i) - 1
        rightid = np.searchsorted(offid, i)

        Xon = onvalues[onscanid == i]
        if leftid == -1:
            Xoff = offvalues[offscanid == offid[rightid]]
            Xoff_m = getattr(np, "nan" + mode)(Xoff, axis=0) if mode is not None else 0
            Xon_m = Xoff_m
            model.fit(Xoff - Xoff_m)
        elif rightid == len(offid):
            Xoff = offvalues[offscanid == offid[leftid]]
            Xoff_m = getattr(np, "nan" + mode)(Xoff, axis=0) if mode is not None else 0
            Xon_m = Xoff_m
            model.fit(Xoff - Xoff_m)
        else:
            Xoff_l = offvalues[offscanid == offid[leftid]]
            Xoff_lm = (
                getattr(np, "nan" + mode)(Xoff_l, axis=0) if mode is not None else 0
            )
            Xoff_r = offvalues[offscanid == offid[rightid]]
            Xoff_rm = (
                getattr(np, "nan" + mode)(Xoff_r, axis=0) if mode is not None else 0
            )
            Xon_m = (
                getattr(np, "nan" + mode)(np.vstack([Xoff_l, Xoff_r]), axis=0)
                if mode is not None
                else 0
            )
            model.fit(np.vstack([Xoff_l - Xoff_lm, Xoff_r - Xoff_rm]))
        P = model.components_
        C = model.transform(Xon - Xon_m)

        Xatmvalues[onscanid == i] = C @ P + Xon_m
        # Xatms.append(dc.full_like(Xon, C @ P + Xon_m.values))
        Ps.append(P)
        Cs.append(C)

    if pc:
        return Xatm, Ps, Cs
    else:
        return Xatm


def chopper_calibration(onarray, offarray, rarray, Tamb, mode="mean"):
    logger = getLogger("decode.models.chopper_calibration")
    logger.info("mode")
    logger.info("{}".format(mode))

    onarray, offarray = r_division(onarray, offarray, rarray, mode=mode)

    offid = np.unique(offarray.scanid)
    onid = np.unique(onarray.scanid)

    onarray = onarray.copy()  # Xarray
    onvalues = onarray.values
    onscanid = onarray.scanid.values
    offarray = offarray.copy()  # Xarray
    offvalues = offarray.values
    offscanid = offarray.scanid.values
    for i in onid:
        oleftid = np.searchsorted(offid, i) - 1
        orightid = np.searchsorted(offid, i)

        Xon = onvalues[onscanid == i]
        if oleftid == -1:
            Xoff = offvalues[offscanid == offid[orightid]]
            Xoff_m = getattr(np, "nan" + mode)(Xoff, axis=0)
        elif orightid == len(offid):
            Xoff = offvalues[offscanid == offid[oleftid]]
            Xoff_m = getattr(np, "nan" + mode)(Xoff, axis=0)
        else:
            Xoff_l = offvalues[offscanid == offid[oleftid]]
            Xoff_r = offvalues[offscanid == offid[orightid]]
            Xoff_m = getattr(np, "nan" + mode)(np.vstack([Xoff_l, Xoff_r]), axis=0)
        onvalues[onscanid == i] = Tamb * (Xon - Xoff_m) / (1 - Xoff_m)

    for j in offid:
        Xoff = offvalues[offscanid == j]
        Xoff_m = getattr(np, "nan" + mode)(Xoff, axis=0)
        offvalues[offscanid == j] = Tamb * (Xoff - Xoff_m) / (1 - Xoff_m)

    return onarray, offarray


def r_division(onarray, offarray, rarray, mode="mean"):
    """Apply R division.

    Args:
        onarray (decode.array): Decode array of on-point observations.
        offarray (decode.array): Decode array of off-point observations.
        rarray (decode.array): Decode array of R observations.
        mode (str): Method for the selection of nominal R value.
            'mean': Mean.
            'median': Median.

    Returns:
        onarray_cal (decode.array): Calibrated array of on-point observations.
        offarray_cal (decode.array): Calibrated array of off-point observations.
    """
    logger = getLogger("decode.models.r_division")
    logger.info("mode")
    logger.info("{}".format(mode))

    offid = np.unique(offarray.scanid)
    onid = np.unique(onarray.scanid)
    rid = np.unique(rarray.scanid)

    onarray = onarray.copy()  # Xarray
    onvalues = onarray.values
    onscanid = onarray.scanid.values
    offarray = offarray.copy()  # Xarray
    offvalues = offarray.values
    offscanid = offarray.scanid.values
    rarray = rarray.copy()  # Xarray
    rvalues = rarray.values
    rscanid = rarray.scanid.values
    for i in onid:
        rleftid = np.searchsorted(rid, i) - 1
        rrightid = np.searchsorted(rid, i)

        if rleftid == -1:
            Xr = rvalues[rscanid == rid[rrightid]]
            Xr_m = getattr(np, "nan" + mode)(Xr, axis=0)
        elif rrightid == len(rid):
            Xr = rvalues[rscanid == rid[rleftid]]
            Xr_m = getattr(np, "nan" + mode)(Xr, axis=0)
        else:
            Xr_l = rvalues[rscanid == rid[rleftid]]
            Xr_r = rvalues[rscanid == rid[rrightid]]
            Xr_m = getattr(np, "nan" + mode)(np.vstack([Xr_l, Xr_r]), axis=0)
        onvalues[onscanid == i] /= Xr_m

    for j in offid:
        rleftid = np.searchsorted(rid, j) - 1
        rrightid = np.searchsorted(rid, j)

        if rleftid == -1:
            Xr = rvalues[rscanid == rid[rrightid]]
            Xr_m = getattr(np, "nan" + mode)(Xr, axis=0)
        elif rrightid == len(rid):
            Xr = rvalues[rscanid == rid[rleftid]]
            Xr_m = getattr(np, "nan" + mode)(Xr, axis=0)
        else:
            Xr_l = rvalues[rscanid == rid[rleftid]]
            Xr_r = rvalues[rscanid == rid[rrightid]]
            Xr_m = getattr(np, "nan" + mode)(np.vstack([Xr_l, Xr_r]), axis=0)
        offvalues[offscanid == j] /= Xr_m

    Xon_rdiv = dc.full_like(onarray, onarray)
    Xoff_rdiv = dc.full_like(offarray, offarray)
    Xonoff_rdiv = dc.concat([Xon_rdiv, Xoff_rdiv], dim="t")
    Xonoff_rdiv_sorted = Xonoff_rdiv[np.argsort(Xonoff_rdiv.time.values)]

    scantype = Xonoff_rdiv_sorted.scantype.values
    newscanid = np.cumsum(np.hstack([False, scantype[1:] != scantype[:-1]]))
    onmask = np.in1d(Xonoff_rdiv_sorted.scanid, onid)
    offmask = np.in1d(Xonoff_rdiv_sorted.scanid, offid)
    Xon_rdiv = Xonoff_rdiv_sorted[onmask]
    Xoff_rdiv = Xonoff_rdiv_sorted[offmask]
    Xon_rdiv.coords.update({"scanid": ("t", newscanid[onmask])})
    Xoff_rdiv.coords.update({"scanid": ("t", newscanid[offmask])})

    return Xon_rdiv, Xoff_rdiv


def gauss_fit(
    map_data,
    chs=None,
    mode="deg",
    amplitude=1,
    x_mean=0,
    y_mean=0,
    x_stddev=None,
    y_stddev=None,
    theta=None,
    cov_matrix=None,
    noise=0,
    **kwargs
):
    """make a 2D Gaussian model and fit the observed data with the model.

    Args:
        map_data (xarray.Dataarray): Dataarray of cube or single chs.
        chs (list of int): in prep.
        mode (str): Coordinates for the fitting
            'pix'
            'deg'
        amplitude (float or None): Initial amplitude value of Gaussian fitting.
        x_mean (float): Initial value of mean of the fitting Gaussian in x.
        y_mean (float): Initial value of mean of the fitting Gaussian in y.
        x_stddev (float or None): Standard deviation of the Gaussian
            in x before rotating by theta.
        y_stddev  (float or None): Standard deviation of the Gaussian
            in y before rotating by theta.
        theta (float, optional or None): Rotation angle in radians.
        cov_matrix (ndarray, optional): A 2x2 covariance matrix. If specified,
            overrides the ``x_stddev``, ``y_stddev``, and ``theta`` defaults.

    Returns:
        decode cube (xarray cube) with fitting results in array and attrs.
    """

    if chs is None:
        chs = np.ogrid[0:63]  # the number of channels would be changed

    if len(chs) > 1:
        for n, ch in enumerate(chs):
            subdata = np.transpose(
                np.full_like(map_data[:, :, ch], map_data.values[:, :, ch])
            )
            subdata[np.isnan(subdata)] = 0

            if mode == "deg":
                mX, mY = np.meshgrid(map_data.x, map_data.y)

            elif mode == "pix":
                mX, mY = np.mgrid[0 : len(map_data.y), 0 : len(map_data.x)]

            g_init = models.Gaussian2D(
                amplitude=np.nanmax(subdata),
                x_mean=x_mean,
                y_mean=y_mean,
                x_stddev=x_stddev,
                y_stddev=y_stddev,
                theta=theta,
                cov_matrix=cov_matrix,
                **kwargs
            ) + models.Const2D(noise)
            fit_g = fitting.LevMarLSQFitter()
            g = fit_g(g_init, mX, mY, subdata)

            g_init2 = models.Gaussian2D(
                amplitude=np.nanmax(subdata - g.amplitude_1),
                x_mean=x_mean,
                y_mean=y_mean,
                x_stddev=x_stddev,
                y_stddev=y_stddev,
                theta=theta,
                cov_matrix=cov_matrix,
                **kwargs
            )
            fit_g2 = fitting.LevMarLSQFitter()
            g2 = fit_g2(g_init2, mX, mY, subdata)

            if n == 0:
                results = np.array([g2(mX, mY)])
                peaks = np.array([g2.amplitude.value])
                x_means = np.array([g2.x_mean.value])
                y_means = np.array([g2.y_mean.value])
                x_stddevs = np.array([g2.x_stddev.value])
                y_stddevs = np.array([g2.y_stddev.value])
                thetas = np.array([g2.theta.value])
                if fit_g2.fit_info["param_cov"] is None:
                    uncerts = np.array([0])
                else:
                    error = np.diag(fit_g2.fit_info["param_cov"]) ** 0.5
                    uncerts = np.array([error[0]])

            else:
                results = np.append(results, [g2(mX, mY)], axis=0)
                peaks = np.append(peaks, [g2.amplitude.value], axis=0)
                x_means = np.append(x_means, [g2.x_mean.value], axis=0)
                y_means = np.append(y_means, [g2.y_mean.value], axis=0)
                x_stddevs = np.append(x_stddevs, [g2.x_stddev.value], axis=0)
                y_stddevs = np.append(y_stddevs, [g2.y_stddev.value], axis=0)
                thetas = np.append(thetas, [g2.theta.value], axis=0)

                if fit_g2.fit_info["param_cov"] is None:
                    uncerts = np.append(uncerts, [0], axis=0)
                else:
                    error = np.diag(fit_g2.fit_info["param_cov"]) ** 0.5
                    uncerts = np.append(uncerts, [error[0]], axis=0)

        result = map_data.copy()
        result.values = np.transpose(results)
        result.attrs.update(
            {
                "peak": peaks,
                "x_mean": x_means,
                "y_mean": y_means,
                "x_stddev": x_stddevs,
                "y_stddev": y_stddevs,
                "theta": thetas,
                "uncert": uncerts,
            }
        )

    else:
        subdata = np.transpose(
            np.full_like(map_data[:, :, 0], map_data.values[:, :, 0])
        )
        subdata[np.isnan(subdata)] = 0

        if mode == "deg":
            mX, mY = np.meshgrid(map_data.x, map_data.y)

        elif mode == "pix":
            mX, mY = np.mgrid[0 : len(map_data.y), 0 : len(map_data.x)]

        g_init = models.Gaussian2D(
            amplitude=np.nanmax(subdata),
            x_mean=x_mean,
            y_mean=y_mean,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
            theta=theta,
            cov_matrix=cov_matrix,
            **kwargs
        ) + models.Const2D(noise)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, mX, mY, subdata)

        g_init2 = models.Gaussian2D(
            amplitude=np.nanmax(subdata - g.amplitude_1),
            x_mean=x_mean,
            y_mean=y_mean,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
            theta=theta,
            cov_matrix=cov_matrix,
            **kwargs
        )
        fit_g2 = fitting.LevMarLSQFitter()
        g2 = fit_g2(g_init2, mX, mY, subdata)

        results = np.array([g2(mX, mY)])
        peaks = np.array([g2.amplitude.value])
        x_means = np.array([g2.x_mean.value])
        y_means = np.array([g2.y_mean.value])
        x_stddevs = np.array([g2.x_stddev.value])
        y_stddevs = np.array([g2.y_stddev.value])
        thetas = np.array([g2.theta.value])
        error = np.diag(fit_g2.fit_info["param_cov"]) ** 0.5
        uncerts = np.array(error[0])

        result = map_data.copy()
        result.values = np.transpose(results)
        result.attrs.update(
            {
                "peak": peaks,
                "x_mean": x_means,
                "y_mean": y_means,
                "x_stddev": x_stddevs,
                "y_stddev": y_stddevs,
                "theta": thetas,
                "uncert": uncerts,
            }
        )

    return result
