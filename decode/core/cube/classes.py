# coding: utf-8

# public items
__all__ = []

# standard library
import sys
from collections import OrderedDict

# dependent packages
import decode as dc
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
    ('kidid', ('ch', np.zeros(array.shape[2], dtype=float))),
    ('kidfq', ('ch', np.zeros(array.shape[2], dtype=float))),
])

SCALARCOORDS = OrderedDict([
    ('coordsys', 'RADEC'),
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

    @staticmethod
    def fromcube(cube, array):
        pass

    @staticmethod
    def tocube(array, **kwargs):
        array = array.copy()

        if 'xarr' in kwargs and 'yarr' in kwargs:
            x_grid = xr.DataArray(kwargs['xarr'], dims='grid')
            y_grid = xr.DataArray(kwargs['yarr'], dims='grid')
        elif 'nx' in kwargs and 'ny' in kwargs:
            if 'xmin' in kwargs and 'xmax' in kwargs:
                xmin, xmax = kwargs['xmin'], kwargs['xmax']
                if 'ymin' in kwargs and 'ymax' in kwargs:
                    ymin, ymax = kwargs['ymin'], kwargs['ymax']
                else:
                    ymin, ymax = array.y.min(), array.y.max()
            else:
                xmin, xmax = array.x.min(), array.x.max()
                if 'ymin' in kwargs and 'ymax' in kwargs:
                    ymin, ymax = kwargs['ymin'], kwargs['ymax']
                else:
                    ymin, ymax = array.y.min(), array.y.max()

            x_grid = xr.DataArray(np.linspace(xmin, xmax, kwargs['nx']), dims='grid')
            y_grid = xr.DataArray(np.linspace(ymin, ymax, kwargs['ny']), dims='grid')
        else:
            print('Arguments are wrong.')
            sys.exit(1)

        nx_grid = len(x_grid)
        ny_grid = len(y_grid)
        nz_grid = len(array.ch)

        xcoords  = {'x': x_grid.values}
        ycoords  = {'y': y_grid.values}
        chcoords = {'kidid': array.kidid, 'kidfq': array.kidfq}

        i     = np.abs(array.x - x_grid).argmin('grid')
        j     = np.abs(array.y - y_grid).argmin('grid')
        index = i + j * nx_grid

        array.coords.update({'index': index})
        griddedarray   = array.groupby('index').mean('t')
        template       = np.full([nx_grid*ny_grid, nz_grid], np.nan)
        mask           = griddedarray.index.values
        template[mask] = griddedarray.values
        cubedata       = template.reshape((ny_grid, nx_grid, nz_grid)).swapaxes(0, 1)

        return dc.cube(cubedata, xcoords=xcoords, ycoords=ycoords, chcoords=chcoords)

    @staticmethod
    def savefits(cube, fitsname, **kwargs):
        # should be modified in the future
        cdelt1 = float(cube.x[1] - cube.x[0])
        crval1 = float(cube.x[0])
        cdelt2 = float(cube.y[1] - cube.y[0])
        crval2 = float(cube.y[0])
        cdelt3 = float(cube.kidfq[1] - cube.kidfq[0])
        crval3 = float(cube.kidfq[0])

        header = fits.Header(OrderedDict([('CTYPE1', 'deg'), ('CDELT1', cdelt1), ('CRVAL1', crval1), ('CRPIX1', 1),
                                          ('CTYPE2', 'deg'), ('CDELT2', cdelt2), ('CRVAL2', crval2), ('CRPIX2', 1),
                                          ('CTYPE3', 'Hz'),  ('CDELT3', cdelt3), ('CRVAL3', crval3), ('CRPIX3', 1)]))
        fits.writeto(fitsname, cube.values.T, header, **kwargs)
