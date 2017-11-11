# coding: utf-8

# public items
__all__ = [
    'skewgauss',
    'savgol_filter',
    'pca',
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
    normpdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    normcdf = 0.5 * (1 + erf((alpha * ((x - mu) / sigma)) / (np.sqrt(2))))

    return 2 * a * normpdf * normcdf


def savgol_filter(array, win=2001, polyn=3, iteration=1, threshold=1):
    logger = getLogger('decode.models.savgol_filter')
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


def pca(onarray, offarray, n=10, exchs=None, pc=True):
    logger = getLogger('decode.models.pca')
    logger.info('n_components exchs')
    if exchs is None:
        exchs = [16, 44, 46]
    logger.info('{} {}'.format(n, exchs))

    onarray  = onarray.copy()
    onarray[:, exchs] = 0
    offarray = offarray.copy()
    offarray[:, exchs] = 0

    offid = np.unique(offarray.scanid)
    onid  = np.unique(onarray.scanid)
    Xatms = []
    Ps = []
    Cs = []
    for i in onid:
        leftid  = np.searchsorted(offid, i) - 1
        rightid = np.searchsorted(offid, i)

        Xon  = onarray[onarray.scanid == i]
        if leftid == -1:
            Xoff  = offarray[offarray.scanid == offid[rightid]]
            Xoff -= Xoff.mean(dim='t')
        elif rightid == len(offid):
            Xoff  = offarray[offarray.scanid == offid[leftid]]
            Xoff -= Xoff.mean(dim='t')
        else:
            Xoff_l  = offarray[offarray.scanid == offid[leftid]]
            Xoff_l -= Xoff_l.mean(dim='t')
            Xoff_r  = offarray[offarray.scanid == offid[rightid]]
            Xoff_r -= Xoff_r.mean(dim='t')
            Xoff    = dc.concat([Xoff_l, Xoff_r], dim='t')
        model = TruncatedSVD(n_components=n)
        model.fit(Xoff)
        P = model.components_
        Ps.append(P)
        C = model.transform(Xon - Xon.mean(dim='t'))
        Cs.append(C)

        Xatm = Xon.copy()
        Xatm.data = C @ P + Xon.mean(dim='t').values
        Xatms.append(Xatm)

    if pc:
        return dc.concat(Xatms, dim='t'), Ps, Cs
    else:
        return dc.concat(Xatms, dim='t')
