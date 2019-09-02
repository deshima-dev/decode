# standard library
from setuptools import setup

# module constants
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'astropy',
    'xarray',
    'matplotlib',
    'scikit-learn',
    'pyyaml',
    'tqdm',
    'netCDF4',
]

PACKAGES = [
    'decode',
    'decode.core',
    'decode.core.array',
    'decode.core.cube',
    'decode.io',
    'decode.logging',
    'decode.plot',
    'decode.models',
    'decode.utils',
    'decode.utils.misc',
    'decode.utils.ndarray'
]


# main
setup(
    name = 'decode',
    description = 'DESHIMA code for data analysis',
    version = '0.5.4',
    author = 'DESHIMA software team',
    author_email = 'taniguchi@a.phys.nagoya-u.ac.jp',
    url = 'https://github.com/deshima-dev/decode',
    install_requires = INSTALL_REQUIRES,
    packages = PACKAGES,
    package_data = {'decode': ['data/*.yaml']},
)
