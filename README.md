# de:code

[![Release](https://img.shields.io/pypi/v/decode?label=Release&color=cornflowerblue&style=flat-square)](https://pypi.org/project/decode/)
[![Python](https://img.shields.io/pypi/pyversions/decode?label=Python&color=cornflowerblue&style=flat-square)](https://pypi.org/project/decode/)
[![Downloads](https://img.shields.io/pypi/dm/decode?label=Downloads&color=cornflowerblue&style=flat-square)](https://pepy.tech/project/decode)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.3384216-cornflowerblue?style=flat-square)](https://doi.org/10.5281/zenodo.3384216)
[![Tests](https://img.shields.io/github/actions/workflow/status/deshima-dev/decode/tests.yaml?label=Tests&style=flat-square)](https://github.com/deshima-dev/decode/actions)

DESHIMA code for data analysis

## Installation

```shell
pip install decode==2024.9.1
```

## Quick look

de:code ships with a quick look command `decode-qlook`, which will be available from the CUI after installation. It has several subcommands for each observation type. For example, to quick-look at a raster observation:
```shell
$ decode-qlook raster /path/to/dems.zarr.zip
```
where `dems.zarr.zip` is the merged observation data ([DESHIMA measurement set: DEMS](https://github.com/deshima-dev/dems)) to be checked. By default, it will output an image of the result plots by a simple analysis (e.g. continuum map, etc). You can also get the result data themselves by changing the output format:
```shell
$ decode-qlook raster /path/to/dems.zarr.zip --format zarr.zip
```
See the command help for all available options:
```shell
# list of the subcommands and descriptions
$ decode-qlook --help

# list of the available command options
$ decode-qlook raster --help
```

If you are not sure about the observation type, the `auto` subcommand may be useful to automatically select the appropriate command to use:
```shell
$ decode-qlook auto /path/to/dems.zarr.zip
```

Finally, all subcommands are available as functions in the `qlook` submodule. For example, the `raster` command corresponds to `decode.qlook.raster` and the following Python code is equivalent to the CUI:
```python
import decode

decode.qlook.raster("/path/to/dems.zarr.zip")
```
See [the qlook module documentation](https://deshima-dev.github.io/decode/_apidoc/decode.qlook.html) for more information.
