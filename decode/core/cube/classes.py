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
