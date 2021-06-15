import glob
import os
import yaml

import calliope


def get_models_from_file(indir):
    input_files = glob.glob(os.path.join(indir, "*.nc"))
    models = {}
    _scenario = 0
    cost_optimal_model = None
    for file in input_files:
        if os.path.basename(file) == "spore_0.nc":
            cost_optimal_model = calliope.read_netcdf(file)
        else:
            models[_scenario] = calliope.read_netcdf(file)
        _scenario += 1
    return models, cost_optimal_model


def dict_to_csvs(data_dict, outdir, meta, name_mapping=None, iamc_mapping=None):
    """
    Emulates possible friendly_data functionality by saving our dicts of pandas Series
    to file, including relevant metadata and
    """
    index = []
    for subdir, data in data_dict.items():
        os.makedirs(os.path.join(outdir, "data"), exist_ok=True)
        outfile = os.path.join(outdir, "data", subdir + ".csv")
        data.to_frame(subdir).dropna().to_csv(outfile)
        index_meta = {
            "path": os.path.join("data", subdir + ".csv"),
            "alias": {"locs": "region", "techs": "technology"},
            "idxcols": list(data.index.names),
            "name": name_mapping[subdir] if name_mapping is not None else subdir
        }
        if iamc_mapping is not None:
            index_meta["iamc"] = iamc_mapping.format(name_mapping[subdir])
        index.append(index_meta)

    with open(os.path.join(outdir, "conf.yaml"), "w") as f:
        yaml.safe_dump(meta, f)
    with open(os.path.join(outdir, "index.yaml"), "w") as f:
        yaml.safe_dump(index, f)
