# coding: utf-8

# public items
__all__ = []

# standard library
from collections import OrderedDict

# dependent packages
import decode as dc
import numpy as np
import xarray as xr
from astropy.io import fits
from .. import BaseAccessor

# local constants
XCOORDS = lambda array: OrderedDict([
    ('ra', ('x', np.zeros(array.shape[0], dtype=float))),
])

YCOORDS = lambda array: OrderedDict([
    ('dec', ('y', np.zeros(array.shape[1], dtype=float))),
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
        """A dictionary of arrays that label x axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('x',)}

    @property
    def ycoords(self):
        """A dictionary of arrays that label y axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('y',)}

    @property
    def chcoords(self):
        """A dictionary of arrays that label channel axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('ch',)}

    @staticmethod
    def fromcube(cube):
        pass

    @staticmethod
    def tocube(array, x_grid, y_grid):
        nx_grid = len(x_grid)
        ny_grid = len(y_grid)
        nz_grid = len(array.ch)

        if isinstance(x_grid, list):
            x_grid = xr.DataArray(np.array(x_grid), dims='grid')
        elif isinstance(x_grid, np.ndarray):
            x_grid = xr.DataArray(x_grid, dims='grid')

        if isinstance(y_grid, list):
            y_grid = xr.DataArray(np.array(y_grid), dims='grid')
        elif isinstance(y_grid, np.ndarray):
            y_grid = xr.DataArray(y_grid, dims='grid')

        i     = np.abs(array.x - x_grid).argmin('grid')
        j     = np.abs(array.y - y_grid).argmin('grid')
        index = i + j * nx_grid

        array.coords.update({'index': index})
        griddedarray = array.groupby('index').mean('t')
        template     = np.zeros([nx_grid*ny_grid, nz_grid])
        template[griddedarray.index.values] = griddedarray.values
        cubedata     = template.reshape((ny_grid, nx_grid, nz_grid)).swapaxes(0, 1)

        xcoords  = {'ra': x_grid.values}
        ycoords  = {'dec': y_grid.values}

        return dc.cube(cubedata, xcoords=xcoords, ycoords=ycoords)

    @staticmethod
    def savefits(cube, fitsname, clobber):
        cdelt1 = float((cube.ra[1] - cube.ra[0]).values)
        crval1 = float(cube.ra[0].values)
        cdelt2 = float((cube.dec[1] - cube.dec[0]).values)
        crval2 = float(cube.dec[0].values)

        header = fits.Header(OrderedDict([('CTYPE1', 'deg'), ('CDELT1', cdelt1),
                                          ('CRVAL1', crval1), ('CRPIX1', 1), ('CTYPE2', 'deg'), ('CDELT2', cdelt2),
                                          ('CRVAL2', crval2), ('CRPIX2', 1), ('CTYPE3', 'freq')]))
        fits.writeto(fitsname, cube.values.T, header, clobber=clobber)
