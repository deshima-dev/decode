# coding: utf-8

# public items
__all__ = []

# standard library
from collections import OrderedDict
from datetime import datetime
from logging import getLogger

# dependent packages
import decode as dc
import numpy as np
import matplotlib.pyplot as plt
try:
    plt.style.use('seaborn-darkgrid')
    plt.style.use('seaborn-pastel')
except:
    pass
import xarray as xr
from .. import BaseAccessor

# local constants
TCOORDS = lambda array: OrderedDict([
    ('vrad', ('t', np.zeros(array.shape[0], dtype=float))),
    ('x',    ('t', np.zeros(array.shape[0], dtype=float))),
    ('y',    ('t', np.zeros(array.shape[0], dtype=float))),
    ('time', ('t', np.zeros(array.shape[0], dtype=float))),
    ('temp', ('t', np.zeros(array.shape[0], dtype=float))),
    ('pressure', ('t', np.zeros(array.shape[0], dtype=float))),
    ('vapor-pressure', ('t', np.zeros(array.shape[0], dtype=float))),
    ('windspd', ('t', np.zeros(array.shape[0], dtype=float))),
    ('winddir', ('t', np.zeros(array.shape[0], dtype=float))),
    ('scantype', ('t', np.full(array.shape[0], 'GRAD'))),
    ('scanid', ('t', np.zeros(array.shape[0], dtype=int))),
])

CHCOORDS = lambda array: OrderedDict([
    ('masterid', ('ch', np.zeros(array.shape[1], dtype=int))),
    ('kidid', ('ch', np.zeros(array.shape[1], dtype=int))),
    ('kidfq', ('ch', np.zeros(array.shape[1], dtype=float))),
    ('kidtp', ('ch', np.zeros(array.shape[1], dtype=int)))
])

DATACOORDS = lambda array: OrderedDict([
    ('weight', (('t', 'ch'), np.ones(array.shape, dtype=float)))
])

SCALARCOORDS = OrderedDict([
    ('coordsys', 'RADEC'),
    ('datatype', 'Temperature'),
    ('xref', 0.0),
    ('yref', 0.0),
    ('type', 'dca'),
])


@xr.register_dataarray_accessor('dca')
class DecodeArrayAccessor(BaseAccessor):
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
        self.coords.update(TCOORDS(self))
        self.coords.update(CHCOORDS(self))
        self.coords.update(DATACOORDS(self))
        self.coords.update(SCALARCOORDS)

    @property
    def tcoords(self):
        """Dictionary of arrays that label time axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('t',)}

    @property
    def chcoords(self):
        """Dictionary of arrays that label channel axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('ch',)}

    @property
    def datacoords(self):
        """Dictionary of arrays that label time and channel axis."""
        return {k: v.values for k, v in self.coords.items() if v.dims==('t', 'ch')}

    @staticmethod
    def plotcoords(array, coords, scantypes=None, save=True, **kwargs):
        logger = getLogger('decode.plot.plotcoords')

        fig, ax = plt.subplots(1, 1, **kwargs)
        if scantypes is None:
            ax.plot(array[coords[0]], array[coords[1]], label='ALL')
        else:
            for scantype in scantypes:
                ax.plot(array[coords[0]][array.scantype == scantype],
                        array[coords[1]][array.scantype == scantype], label=scantype)
        ax.set_xlabel(coords[0])
        ax.set_ylabel(coords[1])
        ax.legend()
        fig.tight_layout()
        if save:
            filename = '{}_vs_{}.png'.format(coords[1], coords[0])
            fig.savefig(filename)
            logger.info('{} has been created.'.format(filename))

    @staticmethod
    def plotweather(array, save=True, **kwargs):
        logger = getLogger('decode.plot.plotweather')

        infos  = ['temp', 'pressure', 'vapor-pressure', 'windspd', 'winddir']
        labels = ['external temperature [C]', 'pressure [hPa]', 'vapor pressure [hPa]',
                  'wind speed [m/s]', 'wind direction [deg]']
        fig = plt.figure(**kwargs)
        for n, (info, label) in enumerate(zip(infos, labels)):
            ax = fig.add_subplot(3, 2, n+1)
            ax.plot(array['time'], array[info])
            ax.set_xlabel('time')
            ax.set_ylabel(label)
        fig.tight_layout()
        if save:
            filename = 'weather_info.png'
            fig.savefig(filename)
            logger.info('{} has been created.'.format(filename))
