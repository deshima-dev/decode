# coding: utf-8

"DESHIMA Code for data analysis"

# standard library
from setuptools import setup

# module constants
INSTALL_REQUIRES = [
    'astropy',
    'numpy',
    'scipy',
    'pyyaml',
    'xarray',
    'matplotlib',
]

PACKAGES = [
    'decode',
    'decode.core',
    'decode.core.array',
    'decode.core.cube',
    'decode.io',
    'decode.logging',
    'decode.plot',
    'decode.utils',
    'decode.utils.misc',
]


# main
setup(
    name = 'decode',
    description = __doc__,
    version = '0.2.8',
    author = 'snoopython',
    author_email = 'taniguchi@ioa.s.u-tokyo.ac.jp',
    url = 'https://github.com/deshima-dev/decode',
    # install_requires = INSTALL_REQUIRES,
    packages = PACKAGES,
)
