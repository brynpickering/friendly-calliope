import glob
import os
from copy import deepcopy

import calliope

from friendly_data.converters import from_df
from friendly_data.dpkg import create_pkg, write_pkg


def get_models_from_file(indir, baseline_filename=None, use_filename_as_scenario=False):
    input_files = glob.glob(os.path.join(indir, "*.nc"))
    models = {}
    _scenario = 0
    cost_optimal_model = None
    for file in input_files:
        if baseline_filename is not None and os.path.basename(file) == baseline_filename:
            cost_optimal_model = calliope.read_netcdf(file)
        else:
            if use_filename_as_scenario:
                models[os.path.basename(file).replace(".nc", "")] = calliope.read_netcdf(file)
            else:
                models[_scenario] = calliope.read_netcdf(file)
        _scenario += 1
    return models, cost_optimal_model


def write_dpkg(
    data_dict, outdir, meta,
    name_mapping=None, iamc_mapping=None,
    include_timeseries_data=False
):
    """
    Emulates possible friendly_data functionality by saving our dicts of pandas Series
    to file, including relevant metadata and
    """
    alias = {"locs": "region", "techs": "technology"}
    resources = []
    idx = []
    for name, data in data_dict.items():
        if include_timeseries_data is False and "timesteps" in data.index.names:
            continue
        data = data.to_frame(name).dropna()
        resource = from_df(
            data,
            outdir,
            os.path.join("data", name + ".csv"),
            alias=alias,
        )
        resources.append(resource)
        index_meta = {
            "path": resource["path"],
            "alias": deepcopy(alias),
            "idxcols": resource["schema"]["primaryKey"],
            "name": name_mapping[name] if name_mapping is not None else name,
            "iamc": "Foo|Bar|{technology}"
        }
        if iamc_mapping is not None:
            index_meta["iamc"] = iamc_mapping.format(name_mapping[name])
        idx.append(index_meta)

    package = create_pkg(meta, resources, basepath=outdir, infer=False)
    write_pkg(package, outdir, idx=idx)
