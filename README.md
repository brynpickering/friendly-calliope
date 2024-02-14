# friendly-calliope
Toolkit for converting Calliope inputs/outputs to friendly data.

By default, Calliope stores data as multidimensional NetCDFs, which can be handled with Calliope functionality built on xarray. To communicate this data easily with other models, this toolkit enables anybody to re-package parts of a NetCDF (or collection of them) in to the friendly_data format.

For more information on Calliope, see [here](https://github.com/calliope-project/calliope).

For more information on friendly data, see [here](https://github.com/sentinel-energy/friendly_data).

At present, this is heavily tailored to generate outputs relevant to the SENTINEL Horizon-2020 project.

## Install

For development:

```shell
mamba env create -f environment.yml
mamba activate friendly-calliope
pip install --no-deps -e .
```

For use:

```shell
pip install .
```
