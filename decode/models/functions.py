# coding: utf-8

# public items
__all__ = [
    'skewgauss',
    'savgol_filter',
    'pca',
    'rsky_calibration',
    'chopper_calibration',
    'r_subtraction',
    'r_division',
]

# standard library
from logging import getLogger

# dependent packages
import decode as dc
import numpy as np
from scipy.special import erf
from scipy import signal
from sklearn.decomposition import TruncatedSVD
import xarray as xr


##### skewed Gaussian
def skewgauss(x, sigma, mu, alpha, a):
    """Skewed Gaussian.

    Args:
        x (np.ndarray): Dataset.
        sigma (float): Standard deviation.
        mu (float): Mean.
        alpha (float): Skewness.
        a (float): Normalization factor.

    Returns:
        skewed gaussian (np.ndarray): Skewed Gaussian.
    """
    normpdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    normcdf = 0.5 * (1 + erf((alpha * ((x - mu) / sigma)) / (np.sqrt(2))))

    return 2 * a * normpdf * normcdf


@dc.xarrayfunc
def savgol_filter(array, win=2001, polyn=3, iteration=1, threshold=1):
    """Apply scipy.signal.savgol_filter iteratively.

    Args:
        array (decode.array): Decode array which will be filtered.
        win (int): Length of window.
        polyn (int): Order of polynominal function.
        iteration (int): The number of iterations.
        threshold (float): Threshold above which the data will be used as signals
            in units of sigma.

    Returns:
        fitted (decode.array): Fitted results.
    """
    logger = getLogger('decode.models.savgol_filter')
    logger.warning('Do not use this function. We recommend you to use dc.models.pca instead.')
    logger.info('win polyn iteration threshold')
    logger.info('{} {} {} {}'.format(win, polyn, iteration, threshold))
    ### 1st estimation
    array    = array.copy()
    fitted   = signal.savgol_filter(array, win, polyn, axis=0)
    filtered = array - fitted
    ### nth iteration
    for i in range(iteration):
        sigma  = filtered.std(axis=0)
        mask   = (filtered >= threshold * sigma)
        masked = np.ma.array(filtered, mask=mask)
        ### maskされた点についてはthreshold * sigmaのデータで埋める
        filled = masked.filled(threshold * sigma)
        ### fitted dataを足し、このデータに対して再度savgol_filterをかける
        clipped  = filled + fitted
        fitted   = signal.savgol_filter(clipped, win, polyn, axis=0)
        filtered = array - fitted

    return fitted


def pca(onarray, offarray, n=10, exchs=None, pc=False, mode='mean'):
    """Apply Principal Component Analysis (PCA) method to estimate baselines at each time.

    Args:
        onarray (decode.array): Decode array of on-point observations.
        offarray (decode.array): Decode array of off-point observations.
        n (int): The number of pricipal components.
        pc (bool): When True, this function also returns eigen vectors and their coefficients.
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
    logger = getLogger('decode.models.pca')
    logger.info('n_components exchs mode')
    if exchs is None:
        exchs = [16, 44, 46]
    logger.info('{} {} {}'.format(n, exchs, mode))

    offid = np.unique(offarray.scanid)
    onid  = np.unique(onarray.scanid)

    onarray   = onarray.copy() # Xarray
    onarray[:, exchs] = 0
    onvalues  = onarray.values
    onscanid  = onarray.scanid.values
    offarray  = offarray.copy() # Xarray
    offarray[:, exchs] = 0
    offvalues = offarray.values
    offscanid = offarray.scanid.values

    Ps, Cs     = [], []
    Xatm       = dc.full_like(onarray, onarray)
    Xatmvalues = Xatm.values
    model = TruncatedSVD(n_components=n)
    for i in onid:
        leftid  = np.searchsorted(offid, i) - 1
        rightid = np.searchsorted(offid, i)

        Xon = onvalues[onscanid == i]
        if leftid == -1:
            Xoff   = offvalues[offscanid == offid[rightid]]
            Xoff_m = getattr(np, 'nan'+mode)(Xoff, axis=0) if mode is not None else 0
            Xon_m  = Xoff_m
            model.fit(Xoff - Xoff_m)
        elif rightid == len(offid):
            Xoff   = offvalues[offscanid == offid[leftid]]
            Xoff_m = getattr(np, 'nan'+mode)(Xoff, axis=0) if mode is not None else 0
            Xon_m  = Xoff_m
            model.fit(Xoff - Xoff_m)
        else:
            Xoff_l  = offvalues[offscanid == offid[leftid]]
            Xoff_lm = getattr(np, 'nan'+mode)(Xoff_l, axis=0) if mode is not None else 0
            Xoff_r  = offvalues[offscanid == offid[rightid]]
            Xoff_rm = getattr(np, 'nan'+mode)(Xoff_r, axis=0) if mode is not None else 0
            Xon_m   = getattr(np, 'nan'+mode)(np.vstack([Xoff_l, Xoff_r]), axis=0) if mode is not None else 0
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


def rsky_calibration(onarray, offarray, rarray, Tamb, mode='mean'):
    """Apply R-sky calibrations.

    Args:
        onarray (decode.array): Decode array of on-point observations.
        offarray (decode.array): Decode array of off-point observations.
        rarray (decode.array): Decode array of R observations.
        Tamb (float): Ambient temperature [K].
        mode (str): The way of correcting offsets.
            'mean': Mean.
            'median': Median.

    Returns:
        onarray_cal (decode.array): Calibrated array of on-point observations.
        offarray_cal (decode.array): Calibrated array of off-point observations.
    """
    logger = getLogger('decode.models.rsky_calibration')
    logger.info('mode')
    logger.info('{}'.format(mode))

    offid = np.unique(offarray.scanid)
    onid  = np.unique(onarray.scanid)
    rid   = np.unique(rarray.scanid)

    onarray   = onarray.copy() # Xarray
    onvalues  = onarray.values
    onscanid  = onarray.scanid.values
    offarray  = offarray.copy() # Xarray
    offvalues = offarray.values
    offscanid = offarray.scanid.values
    rarray    = rarray.copy() # Xarray
    rvalues   = rarray.values
    rscanid   = rarray.scanid.values
    for i in onid:
        oleftid  = np.searchsorted(offid, i) - 1
        orightid = np.searchsorted(offid, i)
        rleftid  = np.searchsorted(rid, i) - 1
        rrightid = np.searchsorted(rid, i)

        Xon = onvalues[onscanid == i]
        if oleftid == -1:
            Xoff   = offvalues[offscanid == offid[orightid]]
            Xoff_m = getattr(np, 'nan'+mode)(Xoff, axis=0)
        elif orightid == len(offid):
            Xoff   = offvalues[offscanid == offid[oleftid]]
            Xoff_m = getattr(np, 'nan'+mode)(Xoff, axis=0)
        else:
            Xoff_l  = offvalues[offscanid == offid[oleftid]]
            Xoff_r  = offvalues[offscanid == offid[orightid]]
            Xoff_m  = getattr(np, 'nan'+mode)(np.vstack([Xoff_l, Xoff_r]), axis=0)

        if rleftid == -1:
            Xr   = rvalues[rscanid == rid[rrightid]]
            Xr_m = getattr(np, 'nan'+mode)(Xr, axis=0)
        elif rrightid == len(rid):
            Xr   = rvalues[rscanid == rid[rleftid]]
            Xr_m = getattr(np, 'nan'+mode)(Xr, axis=0)
        else:
            Xr_l  = rvalues[rscanid == rid[rleftid]]
            Xr_r  = rvalues[rscanid == rid[rrightid]]
            Xr_m  = getattr(np, 'nan'+mode)(np.vstack([Xr_l, Xr_r]), axis=0)
        onvalues[onscanid == i] = Tamb * (Xon - Xoff_m) / (Xr_m - Xoff_m)

    for j in offid:
        rleftid  = np.searchsorted(rid, j) - 1
        rrightid = np.searchsorted(rid, j)

        Xoff   = offvalues[offscanid == j]
        Xoff_m = getattr(np, 'nan'+mode)(Xoff, axis=0)
        if rleftid == -1:
            Xr   = rvalues[rscanid == rid[rrightid]]
            Xr_m = getattr(np, 'nan'+mode)(Xr, axis=0)
        elif rrightid == len(rid):
            Xr   = rvalues[rscanid == rid[rleftid]]
            Xr_m = getattr(np, 'nan'+mode)(Xr, axis=0)
        else:
            Xr_l  = rvalues[rscanid == rid[rleftid]]
            Xr_r  = rvalues[rscanid == rid[rrightid]]
            Xr_m  = getattr(np, 'nan'+mode)(np.vstack([Xr_l, Xr_r]), axis=0)
        offvalues[offscanid == j] = Tamb * (Xoff - Xoff_m) / (Xr_m - Xoff_m)

    return onarray, offarray


def chopper_calibration(onarray, offarray, rarray, Tamb, mode='mean'):
    logger = getLogger('decode.models.chopper_calibration')
    logger.info('mode')
    logger.info('{}'.format(mode))

    onarray, offarray = r_division(onarray, offarray, rarray, mode=mode)

    offid = np.unique(offarray.scanid)
    onid  = np.unique(onarray.scanid)

    onarray   = onarray.copy() # Xarray
    onvalues  = onarray.values
    onscanid  = onarray.scanid.values
    offarray  = offarray.copy() # Xarray
    offvalues = offarray.values
    offscanid = offarray.scanid.values
    for i in onid:
        oleftid  = np.searchsorted(offid, i) - 1
        orightid = np.searchsorted(offid, i)

        Xon = onvalues[onscanid == i]
        if oleftid == -1:
            Xoff   = offvalues[offscanid == offid[orightid]]
            Xoff_m = getattr(np, 'nan'+mode)(Xoff, axis=0)
        elif orightid == len(offid):
            Xoff   = offvalues[offscanid == offid[oleftid]]
            Xoff_m = getattr(np, 'nan'+mode)(Xoff, axis=0)
        else:
            Xoff_l  = offvalues[offscanid == offid[oleftid]]
            Xoff_r  = offvalues[offscanid == offid[orightid]]
            Xoff_m  = getattr(np, 'nan'+mode)(np.vstack([Xoff_l, Xoff_r]), axis=0)
        onvalues[onscanid == i] = Tamb * (Xon - Xoff_m) / (1 - Xoff_m)

    for j in offid:
        Xoff   = offvalues[offscanid == j]
        Xoff_m = getattr(np, 'nan'+mode)(Xoff, axis=0)
        offvalues[offscanid == j] = Tamb * (Xoff - Xoff_m) / (1 - Xoff_m)

    return onarray, offarray


def r_subtraction(onarray, offarray, rarray, mode='mean'):
    """Apply R subtraction.

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

    Warnings:
         Current implementation is too slow!
    """
    logger = getLogger('decode.models.r_subtraction')
    logger.info('mode')
    logger.info('{}'.format(mode))

    offid = np.unique(offarray.scanid)
    onid  = np.unique(onarray.scanid)
    rid   = np.unique(rarray.scanid)

    onarray  = onarray.copy()
    offarray = offarray.copy()

    Xon_rsub  = []
    Xoff_rsub = []
    # print(len(onid))
    for n, i in enumerate(onid):
        # print('{}'.format(n), end=', ')
        rleftid  = np.searchsorted(rid, i) - 1
        rrightid = np.searchsorted(rid, i)

        Xon = onarray[onarray.scanid == i]
        if rleftid == -1:
            Xr   = rarray[rarray.scanid == rid[rrightid]]
            Xr_m = getattr(Xr, mode)(dim='t')
        elif rrightid == len(rid):
            Xr   = rarray[rarray.scanid == rid[rleftid]]
            Xr_m = getattr(Xr, mode)(dim='t')
        else:
            Xr_l  = rarray[rarray.scanid == rid[rleftid]]
            Xr_r  = rarray[rarray.scanid == rid[rrightid]]
            Xr_m  = getattr(dc.concat([Xr_l, Xr_r], dim='t'), mode)(dim='t')
        Xon_rsub.append(Xon - Xr_m)
        # Xon -= Xr_m

    # print(len(offid))
    for n, j in enumerate(offid):
        # print('{}'.format(n), end=', ')
        rleftid  = np.searchsorted(rid, j) - 1
        rrightid = np.searchsorted(rid, j)

        Xoff   = offarray[offarray.scanid == j]
        Xoff_m = getattr(Xoff, mode)(dim='t')
        if rleftid == -1:
            Xr   = rarray[rarray.scanid == rid[rrightid]]
            Xr_m = getattr(Xr, mode)(dim='t')
        elif rrightid == len(rid):
            Xr   = rarray[rarray.scanid == rid[rleftid]]
            Xr_m = getattr(Xr, mode)(dim='t')
        else:
            Xr_l  = rarray[rarray.scanid == rid[rleftid]]
            Xr_r  = rarray[rarray.scanid == rid[rrightid]]
            Xr_m  = getattr(dc.concat([Xr_l, Xr_r], dim='t'), mode)(dim='t')
        Xoff_rsub.append(Xoff - Xr_m)
        # Xoff -= Xr_m
    Xonoff_rsub = dc.concat(Xon_rsub + Xoff_rsub, dim='t')
    # Xonoff_rsub = dc.concat([onarray, offarray], dim='t')
    Xonoff_rsub_sorted = Xonoff_rsub[np.argsort(Xonoff_rsub.time.values)]

    scantype    = Xonoff_rsub_sorted.scantype.values
    newscanid   = np.cumsum(np.hstack([False, scantype[1:] != scantype[:-1]]))
    onmask      = np.in1d(Xonoff_rsub_sorted.scanid, onid)
    offmask     = np.in1d(Xonoff_rsub_sorted.scanid, offid)
    Xon_rsub    = Xonoff_rsub_sorted[onmask]
    Xoff_rsub   = Xonoff_rsub_sorted[offmask]
    Xon_rsub.coords.update({'scanid': ('t', newscanid[onmask])})
    Xoff_rsub.coords.update({'scanid': ('t', newscanid[offmask])})

    return Xon_rsub, Xoff_rsub


def r_division(onarray, offarray, rarray, mode='mean'):
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
    logger = getLogger('decode.models.r_division')
    logger.info('mode')
    logger.info('{}'.format(mode))

    offid = np.unique(offarray.scanid)
    onid  = np.unique(onarray.scanid)
    rid   = np.unique(rarray.scanid)

    onarray   = onarray.copy() # Xarray
    onvalues  = onarray.values
    onscanid  = onarray.scanid.values
    offarray  = offarray.copy() # Xarray
    offvalues = offarray.values
    offscanid = offarray.scanid.values
    rarray    = rarray.copy() # Xarray
    rvalues   = rarray.values
    rscanid   = rarray.scanid.values
    for i in onid:
        rleftid  = np.searchsorted(rid, i) - 1
        rrightid = np.searchsorted(rid, i)

        if rleftid == -1:
            Xr   = rvalues[rscanid == rid[rrightid]]
            Xr_m = getattr(np, 'nan'+mode)(Xr, axis=0)
        elif rrightid == len(rid):
            Xr   = rvalues[rscanid == rid[rleftid]]
            Xr_m = getattr(np, 'nan'+mode)(Xr, axis=0)
        else:
            Xr_l  = rvalues[rscanid == rid[rleftid]]
            Xr_r  = rvalues[rscanid == rid[rrightid]]
            Xr_m  = getattr(np, 'nan'+mode)(np.vstack([Xr_l, Xr_r]), axis=0)
        onvalues[onscanid == i] /= Xr_m

    for j in offid:
        rleftid  = np.searchsorted(rid, j) - 1
        rrightid = np.searchsorted(rid, j)

        Xoff_m = getattr(np, 'nan'+mode)(offvalues[offscanid == j], axis=0)
        if rleftid == -1:
            Xr   = rvalues[rscanid == rid[rrightid]]
            Xr_m = getattr(np, 'nan'+mode)(Xr, axis=0)
        elif rrightid == len(rid):
            Xr   = rvalues[rscanid == rid[rleftid]]
            Xr_m = getattr(np, 'nan'+mode)(Xr, axis=0)
        else:
            Xr_l  = rvalues[rscanid == rid[rleftid]]
            Xr_r  = rvalues[rscanid == rid[rrightid]]
            Xr_m  = getattr(np, 'nan'+mode)(np.vstack([Xr_l, Xr_r]), axis=0)
        offvalues[offscanid == j] /= Xr_m

    Xon_rdiv    = dc.full_like(onarray, onarray)
    Xoff_rdiv   = dc.full_like(offarray, offarray)
    Xonoff_rdiv = dc.concat([Xon_rdiv, Xoff_rdiv], dim='t')
    Xonoff_rdiv_sorted = Xonoff_rdiv[np.argsort(Xonoff_rdiv.time.values)]

    scantype    = Xonoff_rdiv_sorted.scantype.values
    newscanid   = np.cumsum(np.hstack([False, scantype[1:] != scantype[:-1]]))
    onmask      = np.in1d(Xonoff_rdiv_sorted.scanid, onid)
    offmask     = np.in1d(Xonoff_rdiv_sorted.scanid, offid)
    Xon_rdiv    = Xonoff_rdiv_sorted[onmask]
    Xoff_rdiv   = Xonoff_rdiv_sorted[offmask]
    Xon_rdiv.coords.update({'scanid': ('t', newscanid[onmask])})
    Xoff_rdiv.coords.update({'scanid': ('t', newscanid[offmask])})

    return Xon_rdiv, Xoff_rdiv
