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
import decode as dc
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
yaml.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node))
)
from astropy.io import fits
import astropy.units as u
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
    def fromcube(cube, array):
        logger = getLogger('decode.fromcube')
        pass

    @staticmethod
    def tocube(array, **kwargs):
        logger = getLogger('decode.tocube')
        array  = array.copy()

        ### pick up kwargs
        unit     = kwargs.pop('unit', 'deg')
        unit2deg = getattr(u, unit).to('deg')

        xc   = kwargs.pop('xc', float(array.xref)) * unit2deg
        yc   = kwargs.pop('yc', float(array.yref)) * unit2deg
        xarr = kwargs.pop('xarr', None)
        yarr = kwargs.pop('yarr', None)
        xmin = kwargs.pop('xmin', None)
        xmax = kwargs.pop('xmax', None)
        ymin = kwargs.pop('ymin', None)
        ymax = kwargs.pop('ymax', None)
        gx   = kwargs.pop('gx', None)
        gy   = kwargs.pop('gy', None)
        nx   = kwargs.pop('nx', None)
        ny   = kwargs.pop('ny', None)
        if None not in [xarr, yarr]:
            x_grid = xr.DataArray(xarr * unit2deg, dims='grid')
            y_grid = xr.DataArray(yarr * unit2deg, dims='grid')
        else:
            if None not in [xmin, xmax, ymin, ymax]:
                xmin *= unit2deg
                xmax *= unit2deg
                ymin *= unit2deg
                ymax *= unit2deg
            else:
                xmin = array.x.min()
                xmax = array.x.max()
                ymin = array.y.min()
                ymax = array.y.max()
            logger.info('xmin xmax ymin ymax')
            logger.info('{} {} {} {}'.format(xmin, xmax, ymin, ymax))

            if None not in [gx, gy]:
                gx *= unit2deg
                gy *= unit2deg
                logger.info('xc yc gx gy')
                logger.info('{} {} {} {}'.format(xc, yc, gx, gy))

                gxmin = np.floor((xmin - xc) / gx)
                gxmax = np.ceil((xmax - xc) / gx)
                gymin = np.floor((ymin - yc) / gy)
                gymax = np.ceil((ymax - yc) / gy)
                xmin  = gxmin * gx + xc
                xmax  = gxmax * gx + xc
                ymin  = gymin * gy + yc
                ymax  = gymax * gy + yc

                x_grid = xr.DataArray(np.arange(xmin, xmax+gx, gx), dims='grid')
                y_grid = xr.DataArray(np.arange(ymin, ymax+gy, gy), dims='grid')
            elif None not in [nx, ny]:
                logger.info('nx ny')
                logger.info('{} {}'.format(nx, ny))
                ### nx/ny does not support xc/yc
                xc = 0
                yc = 0

                x_grid = xr.DataArray(np.linspace(xmin, xmax, nx), dims='grid')
                y_grid = xr.DataArray(np.linspace(ymin, ymax, ny), dims='grid')
            else:
                raise KeyError('Arguments are wrong.')

        ### reverse the direction of x when coordsys == 'RADEC'
        if array.coordsys == 'RADEC':
            x_grid = x_grid[::-1]

        nx_grid = len(x_grid)
        ny_grid = len(y_grid)
        nz_grid = len(array.ch)

        ### define coordinates because groupby() will remove previous coordinates
        xcoords      = {'x': x_grid.values}
        ycoords      = {'y': y_grid.values}
        chcoords     = {'masterid': array.masterid.values, 'kidid': array.kidid.values,
                        'kidfq': array.kidfq.values, 'kidtp': array.kidtp.values}
        scalarcoords = {'coordsys': array.coordsys.values, 'datatype': array.datatype.values,
                        'xref': array.xref.values, 'yref': array.yref.values}

        i = np.abs(array.x - x_grid).argmin('grid')
        j = np.abs(array.y - y_grid).argmin('grid')
        index = i + j * nx_grid

        array.coords.update({'index': index})
        griddedarray   = array.groupby('index').mean('t')
        noisearray     = array.groupby('index').std('t')
        numberarray    = dc.ones_like(array).groupby('index').sum('t')
        noisearray    /= np.sqrt(numberarray)

        template         = np.full([nx_grid*ny_grid, nz_grid], np.nan)
        mask             = griddedarray.index.values
        template[mask]   = griddedarray.values
        template_n       = np.full([nx_grid*ny_grid, nz_grid], np.nan)
        template_n[mask] = noisearray.values
        cubedata         = template.reshape((ny_grid, nx_grid, nz_grid)).swapaxes(0, 1)
        noisedata        = template_n.reshape((ny_grid, nx_grid, nz_grid)).swapaxes(0, 1)

        datacoords = {'noise': noisedata}

        return dc.cube(cubedata, xcoords=xcoords, ycoords=ycoords, chcoords=chcoords,
                       scalarcoords=scalarcoords, datacoords=datacoords)

    @staticmethod
    def makecontinuum(cube, **kwargs):
        logger = getLogger('decode.makecontinuum')

        ### pick up kwargs
        inchs = kwargs.pop('inchs', None)
        exchs = kwargs.pop('exchs', None)

        if inchs is not None:
            logger.info('inchs')
            logger.info('{}'.format(inchs))
            subcube = cube[:, :, inchs]
        else:
            mask = np.full(len(cube.ch), True)
            if exchs is not None:
                logger.info('exchs')
                logger.info('{}'.format(exchs))
                mask[exchs] = False
            subcube = cube[:, :, mask]
        cont = (subcube * (1 / subcube.noise**2)).sum(dim='ch') / (1 / subcube.noise**2).sum(dim='ch')
        cont = cont.expand_dims(dim='ch', axis=2)

        ### define coordinates
        xcoords      = {'x': cube.x.values}
        ycoords      = {'y': cube.y.values}
        chcoords     = {'masterid': np.array([int(subcube.masterid.mean(dim='ch'))]),
                        'kidid': np.array([int(subcube.kidid.mean(dim='ch'))]),
                        'kidfq': np.array([float(subcube.kidfq.mean(dim='ch'))]),
                        'kidtp': np.array([1])}
        scalarcoords = {'coordsys': cube.coordsys.values, 'datatype': cube.datatype.values,
                        'xref': cube.xref.values, 'yref': cube.yref.values}

        return dc.cube(cont.values, xcoords=xcoords, ycoords=ycoords, chcoords=chcoords,
                       scalarcoords=scalarcoords)

    @staticmethod
    def savefits(cube, fitsname, **kwargs):
        logger = getLogger('decode.io.savefits')

        ### pick up kwargs
        dropdeg = kwargs.pop('dropdeg', False)
        ndim    = len(cube.dims)

        ### load yaml
        FITSINFO = get_data('decode', 'data/fitsinfo.yaml')
        hdrdata = yaml.load(FITSINFO)

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
