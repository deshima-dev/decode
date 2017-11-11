# coding: utf-8

# public items
__all__ = []

# standard library
from collections import OrderedDict
from logging import getLogger

# dependent packages
import decode as dc
import matplotlib.pyplot as plt
try:
    plt.style.use('seaborn-darkgrid')
    plt.style.use('seaborn-pastel')
except:
    pass
import numpy as np
import xarray as xr
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
    ('weight', (('x', 'y', 'ch'), np.ones(array.shape, dtype=float)))
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
            array (xarray.DataArray): An array to which Decode accessor is added.

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

        xc = kwargs['xc'] if ('xc' in kwargs) else 0
        yc = kwargs['yc'] if ('yc' in kwargs) else 0

        if 'xarr' in kwargs and 'yarr' in kwargs:
            x_grid = xr.DataArray(kwargs['xarr'], dims='grid')
            y_grid = xr.DataArray(kwargs['yarr'], dims='grid')
        else:
            xmin = kwargs['xmin'] if ('xmin' in kwargs) else array.x.min()
            xmax = kwargs['xmax'] if ('xmax' in kwargs) else array.x.max()
            ymin = kwargs['ymin'] if ('ymin' in kwargs) else array.y.min()
            ymax = kwargs['ymax'] if ('ymax' in kwargs) else array.y.max()
            logger.info('xmin xmax ymin ymax')
            logger.info('{} {} {} {}'.format(xmin, xmax, ymin, ymax))

            if 'gx' in kwargs and 'gy' in kwargs:
                gx = kwargs['gx']
                gy = kwargs['gy']
                logger.info('xc yc gx gy')
                logger.info('{} {} {} {}'.format(xc, yc, gx, gy))

                gxmin = np.floor((xmin - xc) / gx)
                gxmax = np.ceil((xmax - xc) / gx)
                gymin = np.floor((ymin - yc) / gy)
                gymax = np.ceil((ymax - yc) / gy)

                xmin = gxmin * gx
                xmax = gxmax * gx
                ymin = gymin * gy
                ymax = gymax * gy

                x_grid = xr.DataArray(np.arange(xmin, xmax+gx, gx), dims='grid')
                y_grid = xr.DataArray(np.arange(ymin, ymax+gy, gy), dims='grid')
            elif 'nx' in kwargs and 'ny' in kwargs:
                nx = kwargs['nx']
                ny = kwargs['ny']
                logger.info('nx ny')
                logger.info('{} {}'.format(nx, ny))

                ### nx/ny does not support xc/yc
                xc = 0
                yc = 0

                x_grid = xr.DataArray(np.linspace(xmin, xmax, nx), dims='grid')
                y_grid = xr.DataArray(np.linspace(ymin, ymax, ny), dims='grid')
            else:
                raise KeyError('Arguments are wrong.')

        if array.coordsys == 'RADEC':
            x_grid = x_grid[::-1]

        nx_grid = len(x_grid)
        ny_grid = len(y_grid)
        nz_grid = len(array.ch)

        xcoords      = {'x': x_grid.values}
        ycoords      = {'y': y_grid.values}
        chcoords     = {'masterid': array.masterid.values, 'kidid': array.kidid.values,
                        'kidfq': array.kidfq.values, 'kidtp': array.kidtp.values}
        scalarcoords = {'datatype': array.datatype.values}

        i = np.abs((array.x - xc) - x_grid).argmin('grid')
        j = np.abs((array.y - yc) - y_grid).argmin('grid')
        index = i + j * nx_grid

        array.coords.update({'index': index})
        griddedarray   = array.groupby('index').mean('t')
        template       = np.full([nx_grid*ny_grid, nz_grid], np.nan)
        mask           = griddedarray.index.values
        template[mask] = griddedarray.values
        cubedata       = template.reshape((ny_grid, nx_grid, nz_grid)).swapaxes(0, 1)

        return dc.cube(cubedata, xcoords=xcoords, ycoords=ycoords, chcoords=chcoords, scalarcoords=scalarcoords)

    @staticmethod
    def makecontinuum(cube, kidtp, **kwargs):
        logger = getLogger('decode.makecontinuum')
        logger.info('kidtp')
        logger.info('{}'.format(kidtp))

        ### some weighting procedure
        mask = (cube.kidtp == kidtp).values
        if 'exchs' in kwargs:
            logger.info('exchs')
            logger.info('{}'.format(kwargs['exchs']))
            mask[kwargs['exchs']] = False
        cont = cube[:, :, mask].mean(dim='ch')

        return cont

    @staticmethod
    def savefits(cube, fitsname, **kwargs):
        logger = getLogger('decode.io.savefits')
        # should be modified in the future
        cdelt1 = float(cube.x[1] - cube.x[0])
        crval1 = float(cube.x[0])
        cdelt2 = float(cube.y[1] - cube.y[0])
        crval2 = float(cube.y[0])
        header = fits.Header(OrderedDict([('CTYPE1', 'deg'), ('CDELT1', cdelt1), ('CRVAL1', crval1), ('CRPIX1', 1),
                                          ('CTYPE2', 'deg'), ('CDELT2', cdelt2), ('CRVAL2', crval2), ('CRPIX2', 1)]))

        if cube.dims == ('x', 'y', 'ch'):
            try:
                cdelt3 = float(cube.kidid[1] - cube.kidid[0])
            except IndexError:
                cdelt3 = 0
            crval3 = float(cube.kidid[0])
            header.update(OrderedDict([('CTYPE3', 'ID'),  ('CDELT3', cdelt3), ('CRVAL3', crval3), ('CRPIX3', 1)]))

        fits.writeto(fitsname, cube.values.T, header, **kwargs)
        logger.info('{} has been created.'.format(fitsname))

    @staticmethod
    def plotspectrum(cube, ax, xtick, ytick, aperture, **kwargs):
        logger = getLogger('decode.plot.plotspectrum')

        cube     = cube.copy()
        datatype = cube.datatype
        if aperture == 'box':
            if 'xc' in kwargs and 'yc' in kwargs and 'width' in kwargs and 'height' in kwargs:
                xc     = kwargs['xc']
                yc     = kwargs['yc']
                width  = kwargs['width']
                height = kwargs['height']

                xmin, xmax = int(xc - width / 2), int(xc + width / 2)
                ymin, ymax = int(yc - width / 2), int(yc + width / 2)
            elif 'xmin' in kwargs and 'xmax' in kwargs and 'ymin' in kwargs and 'ymax' in kwargs:
                xmin, xmax = kwargs['xmin'], kwargs['xmax']
                ymin, ymax = kwargs['ymin'], kwargs['ymax']
            else:
                raise KeyError('Arguments are wrong.')

            if ytick == 'sum':
                value = cube[xmin:xmax, ymin:ymax, :].sum(dim=('x', 'y'))
            elif ytick == 'peak':
                value = cube[xmin:xmax, ymin:ymax, :].max(axis=(0, 1))
        elif aperture == 'circle':
            if 'xc' in kwargs and 'yc' in kwargs and 'radius' in kwargs:
                xc     = kwargs['xc']
                yc     = kwargs['yc']
                radius = kwargs['radius']
            else:
                raise KeyError('Arguments are wrong.')

            x, y   = np.ogrid[0:len(cube.x), 0:len(cube.y)]
            mask   = ((x - xc)**2 + (y - yc)**2 < radius**2)
            mask   = np.broadcast_to(mask[:, :, np.newaxis], cube.shape)
            masked = np.ma.array(cube.values, mask=~mask)

            if ytick == 'sum':
                value = np.nansum(np.nansum(masked, axis=0), axis=0).data
                if datatype == 'temperature':
                    ax.set_ylabel(r'$T_\mathrm{sum}$ [K]', fontsize=20, color='grey')
                elif datatype == 'power':
                    ax.set_ylabel(r'$P_\mathrm{sum}$ [W]', fontsize=20, color='grey')
            elif ytick == 'peak':
                value = np.nanmax(masked, axis=(0, 1))
                if datatype == 'temperature':
                    ax.set_ylabel(r'$T_\mathrm{peak}$ [K]', fontsize=20, color='grey')
                elif datatype == 'power':
                    ax.set_ylabel(r'$P_\mathrm{peak}$ [W]', fontsize=20, color='grey')

        if xtick == 'freq':
            freqrange = ~np.isnan(cube.kidfq.values)
            if 'exchs' in kwargs:
                freqrange[kwargs['exchs']] = False
            x = cube.kidfq.values[freqrange]
            y = value[freqrange]
            ax.step(x[np.argsort(x)], y[np.argsort(x)], where='mid')
            ax.set_xlabel('Frequency [GHz]', fontsize=20, color='grey')
        elif xtick == 'id':
            x = cube.kidid.values
            y = value
            ax.step(x, y, where='mid')
            ax.set_xlabel('kidid', fontsize=20, color='grey')
        ax.set_title('spectrum', fontsize=20, color='grey')
