# coding: utf-8

# public items
__all__ = []

# standard library
from collections import OrderedDict
from logging import getLogger
from pkgutil import get_data
from pytz import timezone
from datetime import datetime

# dependent packages
import astropy.units as u
import decode as dc
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from astropy.io import fits
from .. import BaseAccessor

# local constants
XCOORDS = lambda array: OrderedDict([
    ('x', ('x', np.zeros(array.shape[0], dtype=float))),
])

YCOORDS = lambda array: OrderedDict([
    ('y', ('y', np.zeros(array.shape[1], dtype=float))),
])

CHCOORDS = lambda array: OrderedDict([
    ('masterid', ('ch', np.zeros(array.shape[2], dtype=int))),
    ('kidid', ('ch', np.zeros(array.shape[2], dtype=int))),
    ('kidfq', ('ch', np.zeros(array.shape[2], dtype=float))),
    ('kidtp', ('ch', np.zeros(array.shape[2], dtype=int)))
])

DATACOORDS = lambda array: OrderedDict([
    ('noise', (('x', 'y', 'ch'), np.ones(array.shape, dtype=float)))
])

SCALARCOORDS = OrderedDict([
    ('coordsys', 'RADEC'),
    ('datatype', 'temperature'),
    ('xref', 0.0),
    ('yref', 0.0),
    ('type', 'dcc'),
])


# classes
@xr.register_dataarray_accessor('dcc')
class DecodeCubeAccessor(BaseAccessor):
    def __init__(self, array):
        """Initialize the Decode accessor of an array.

        Note:
            This method is only for the internal use.
            Users can create an array with Decode accessor using dc.array.

        Args:
            array (xarray.DataArray): Array to which Decode accessor is added.
        """
        super().__init__(array)

    def _initcoords(self):
        """Initialize coords with default values.

        Warning:
            Do not use this method after an array is created.
            This forcibly replaces all vaules of coords with default ones.
        """
        self.coords.update(XCOORDS(self))
        self.coords.update(YCOORDS(self))
        self.coords.update(CHCOORDS(self))
        self.coords.update(DATACOORDS(self))
        self.coords.update(SCALARCOORDS)

    @property
    def xcoords(self):
        """Dictionary of arrays that label x axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('x',)}

    @property
    def ycoords(self):
        """Dictionary of arrays that label y axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('y',)}

    @property
    def chcoords(self):
        """Dictionary of arrays that label channel axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('ch',)}

    @property
    def datacoords(self):
        """Dictionary of arrays that label x, y, and channel axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('x', 'y', 'ch')}

    @staticmethod
    def savefits(cube, fitsname, **kwargs):
        logger = getLogger('decode.io.savefits')

        ### pick up kwargs
        dropdeg = kwargs.pop('dropdeg', False)
        ndim    = len(cube.dims)

        ### load yaml
        FITSINFO = get_data('decode', 'data/fitsinfo.yaml')
        hdrdata = yaml.load(FITSINFO, dc.utils.OrderedLoader)

        ### default header
        if ndim == 2:
            header = fits.Header(hdrdata['dcube_2d'])
            data   = cube.values.T
        elif ndim == 3:
            if dropdeg:
                header = fits.Header(hdrdata['dcube_2d'])
                data   = cube.values[:, :, 0].T
            else:
                header = fits.Header(hdrdata['dcube_3d'])
                data   = cube.values.T
        else:
            raise TypeError(ndim)

        ### update Header
        if cube.coordsys == 'AZEL':
            header.update({'CTYPE1': 'dAZ', 'CTYPE2': 'dEL'})
        elif cube.coordsys == 'RADEC':
            header.update({'OBSRA': float(cube.xref), 'OBSDEC': float(cube.yref)})
        else:
            pass
        header.update({'CRVAL1': float(cube.x[0]),
                       'CDELT1': float(cube.x[1] - cube.x[0]),
                       'CRVAL2': float(cube.y[0]),
                       'CDELT2': float(cube.y[1] - cube.y[0]),
                       'DATE': datetime.now(timezone('UTC')).isoformat()})
        if (ndim == 3) and (not dropdeg):
            header.update({'CRVAL3': float(cube.kidid[0])})

        fits.writeto(fitsname, data, header, **kwargs)
        logger.info('{} has been created.'.format(fitsname))

    @staticmethod
    def plotspectrum(cube, ax, xtick, ytick, aperture, **kwargs):
        logger = getLogger('decode.plot.plotspectrum')

        ### pick up kwargs
        xc     = kwargs.pop('xc', None)
        yc     = kwargs.pop('yc', None)
        width  = kwargs.pop('width', None)
        height = kwargs.pop('height', None)
        xmin   = kwargs.pop('xmin', None)
        xmax   = kwargs.pop('xmax', None)
        ymin   = kwargs.pop('ymin', None)
        ymax   = kwargs.pop('ymax', None)
        radius = kwargs.pop('radius', None)
        exchs  = kwargs.pop('exchs', None)

        ### labels
        xlabeldict = {'freq': 'frequency [GHz]', 'id': 'kidid'}

        cube     = cube.copy()
        datatype = cube.datatype
        if aperture == 'box':
            if None not in [xc, yc, width, height]:
                xmin, xmax = int(xc - width / 2), int(xc + width / 2)
                ymin, ymax = int(yc - width / 2), int(yc + width / 2)
            elif None not in [xmin, xmax, ymin, ymax]:
                pass
            else:
                raise KeyError('Invalid arguments.')
            value = getattr(cube[xmin:xmax, ymin:ymax, :], ytick)(dim=('x', 'y'))
        elif aperture == 'circle':
            if None not in [xc, yc, radius]:
                pass
            else:
                raise KeyError('Invalid arguments.')
            x, y   = np.ogrid[0:len(cube.x), 0:len(cube.y)]
            mask   = ((x - xc)**2 + (y - yc)**2 < radius**2)
            mask   = np.broadcast_to(mask[:, :, np.newaxis], cube.shape)
            masked = np.ma.array(cube.values, mask=~mask)
            value  = getattr(np, 'nan'+ytick)(masked, axis=(0, 1))
        else:
            raise KeyError(aperture)

        if xtick == 'freq':
            kidfq     = cube.kidfq.values
            freqrange = ~np.isnan(kidfq)
            if exchs is not None:
                freqrange[exchs] = False
            x = kidfq[freqrange]
            y = value[freqrange]
            ax.step(x[np.argsort(x)], y[np.argsort(x)], where='mid', **kwargs)
        elif xtick == 'id':
            ax.step(cube.kidid.values, value, where='mid', **kwargs)
        else:
            raise KeyError(xtick)
        ax.set_xlabel('{}'.format(xlabeldict[xtick]), fontsize=20, color='grey')
        ax.set_ylabel('{} ({})'.format(datatype.values, ytick), fontsize=20, color='grey')
        ax.set_title('spectrum', fontsize=20, color='grey')
