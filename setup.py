# coding: utf-8

"DESHIMA Code for data analysis"

# standard library
from setuptools import setup, find_packages

# dependent packages
import decode as dc


# main
setup(
    name = 'decode',
    description = __doc__,
    version = '0.1.1',
    author = 'snoopython',
    author_email = 'taniguchi@ioa.s.u-tokyo.ac.jp',
    url = 'https://github.com/deshima-dev/decode',
    install_requires = ['astropy', 'numpy', 'xarray'],
    packages = find_packages(),
)
