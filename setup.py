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
    version = dc.__version__,
    author = dc.__author__,
    author_email = dc.__email__,
    url = 'https://github.com/deshima-dev/decode',
    install_requires = ['astropy', 'numpy', 'xarray'],
    packages = find_packages(),
)
