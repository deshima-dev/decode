# coding: utf-8

"DESHIMA Code for data analysis"

# standard library
from setuptools import setup, find_packages

install_requires = [
    'astropy',
    'numpy',
    'scipy',
    'pyyaml',
    'xarray',
]

packages = [
    'decode',
    'decode.core',
    'decode.core.array',
    'decode.core.cube',
    'decode.io',
    'decode.logging',
    'decode.utils',
    'decode.utils.misc',
]

# main
setup(
    name = 'decode',
    description = __doc__,
    version = '0.1.3',
    author = 'snoopython',
    author_email = 'taniguchi@ioa.s.u-tokyo.ac.jp',
    url = 'https://github.com/deshima-dev/decode',
    install_requires = install_requires,
    packages = packages,
)
