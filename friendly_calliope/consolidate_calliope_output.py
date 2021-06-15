import argparse

import pandas as pd
import numpy as np
import xarray as xr

from calliope.core.util.dataset import split_loc_techs

from io import get_models_from_file, dict_to_csvs

EU28 = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU",
    "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "NLD", "POL", "PRT", "ROU",
    "SVK", "SVN", "ESP", "SWE", "GBR"
]
ZERO_THRESHOLD = 1e-6  # capacities lower than this will be considered as effectively zero
COST_NAME_MAPPING = {
    'cost_energy_cap': "cost_per_nameplate_capacity",
    'cost_om_prod': "cost_per_flow_out",
    'cost_om_annual_investment_fraction': "annual_cost_per_unit_invested",
    'cost_storage_cap': "cost_per_storage_capacity",
    'cost_om_annual': "annual_cost_per_nameplate_capacity",
    'cost_depreciation_rate': "depreciation_rate",
    'cost_om_con': "cost_per_flow_in",
    'cost': 'total_system_cost',
    'cost_investment': 'total_investment_cost',
    'cost_var': 'total_operation_cost',
}

NEW_TECH_NAMES = {
    "eu_import": "Transmission between countries inside and outside the EU",
    "external_transmission": "Transmission between and within countries outside the EU",
    "heat_storage_big": "District-scale heat storage vessels",
    "heat_storage_small": "Building-scale heat storage vessels",
    "internal_eu_transmission": "Transmission between and within countries inside the EU",
    "wind_onshore": "Onshore wind",
}

TRANSMISSION_NAMES = ["eu_import", "external_transmission", "internal_eu_transmission"]


def combine_scenarios_to_one_dict(model_dict, cost_optimal_model=None, new_dimension_name="scenario"):
    """
    Take in a dictionary of Calliope models and create a dictionary of processed data ready to share with the world
    """
    all_data_dict = {}
    if cost_optimal_model is None:
        cost_optimal_model = list(model_dict.values())[0]

    names = cost_optimal_model.inputs.names.to_series()
    all_data_dict.update(get_input_costs(cost_optimal_model.inputs))
    if isinstance(new_dimension_name, str):
        new_dimension_name = [new_dimension_name]

    output_costs = pd.concat(
        [get_output_costs(model) for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name
    )
    energy_caps = pd.concat(
        [get_caps(model) for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name,
    )
    energy_flows = pd.concat(
        [get_flows(model, None) for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name,
    )
    energy_flows_sum = strip_small_techs(energy_caps, pd.concat(
        [get_flows(model, "sum") for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name,
    ))
    energy_flows_max = strip_small_techs(energy_caps, pd.concat(
        [get_flows(model, "max") for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name,
    ))
    energy_caps = add_units_to_caps(energy_caps, energy_flows_max, cost_optimal_model)

    names = names.reindex(energy_caps.index.get_level_values("techs").unique()).fillna(NEW_TECH_NAMES)
    assert names.isna().sum() == 0

    for df in [output_costs, energy_caps, energy_flows, energy_flows_sum, energy_flows_max]:
        dataframe_to_dict_elements(df, all_data_dict)

    all_data_dict["names"] = names

    return all_data_dict


def dataframe_to_dict_elements(df, data_dict):
    """DF to dict of series, with key names == column names"""
    data_dict.update({col_name: df.loc[:, col_name] for col_name in df.columns})


def get_caps(model):
    """Get storage and energy capacity"""
    caps = []
    for cap_type in ["energy", "storage"]:
        mapped_da = map_da(model._model_data[f"{cap_type}_cap"], keep_demand=False)
        series = clean_series(mapped_da, zero_threshold=ZERO_THRESHOLD, halve_transmission=True)
        if series is None:
            continue
        else:
            caps.append(
                series
                .sum(level=["techs", "locs"])
                .to_frame("{}_capacity".format("nameplate" if cap_type == "energy" else cap_type))
                .div(10)
            )
    return pd.concat(caps, axis=1)


def get_a_flow(model, flow_direction, timeseries_agg):
    direction = "out" if flow_direction == "prod" else "in"
    mapped_da = map_da(
        model._model_data[f"carrier_{flow_direction}"], timeseries_agg=timeseries_agg
    )
    name = f"flow_{direction}_{timeseries_agg}" if timeseries_agg is not None else f"flow_{direction}"
    return (
        clean_series(mapped_da, halve_transmission=True)
        .div(10)
        .abs()
        .to_frame(name)
        .assign(unit='twh')
        .reset_index()
    )


def get_flows(model, timeseries_agg):
    prod = get_a_flow(model, "prod", timeseries_agg)
    con = get_a_flow(model, "con", timeseries_agg)

    prod.loc[prod.techs.str.find('transport_') > -1, 'unit'] = 'billion_km'
    con.loc[con.techs.isin(['demand_heavy_transport', 'demand_light_transport']), 'unit'] = 'billion_km'
    for df in [prod, con]:
        df.loc[df.carriers.isna(), "unit"] = np.nan
        df.loc[df.carriers == 'co2', "unit"] = '100kt'
    if timeseries_agg is None:
        levels = ["techs", "locs", "carriers", "unit", "timesteps"]
    else:
        levels = ["techs", "locs", "carriers", "unit"]
    flow = pd.concat(
        [i.set_index(levels).iloc[:, 0] for i in [prod, con]],
        axis=1
    )
    return flow


def map_da(da, keep_demand=True, timeseries_agg="sum", loc_tech_agg="sum"):
    loc_tech_dim = [i for i in da.dims if "loc_tech" in i][0]
    mapped_da = get_cleaned_dim_mapping(da, loc_tech_dim, keep_demand, loc_tech_agg)
    if "timesteps" in mapped_da.dims and timeseries_agg is not None:
        return agg_da(mapped_da, timeseries_agg, "timesteps")
    else:
        return mapped_da


def clean_series(da, zero_threshold=0, halve_transmission=False):
    """
    Get series from dataarray in which we clean out useless info.

    Parameters
    ----------
    da : xr DataArray
        DataArray that has already been prepared for turning into a clean series using "map_da"
    zero_threshold : float
        Value above which all valid data lies.
        Anything below this in absolute magnitude will be deemed as invalid and removed from the data.
        This exists to remove the small values left over from the Barrier LP method.
    halve_transmission : bool
        If True, halve the value associated with transmission technologies.
        This is required when dealing with Calliope models as there are always two technologies associated
        with a single transmission link (so summing them will erroneously double the value).
        Should be False when passing in averaged data.
    """
    series = da.to_series()
    clean_series = (
        series
        .where(lambda x: (~np.isinf(x)) & (abs(x) > zero_threshold))
        .dropna()
    )
    if halve_transmission is True:
        clean_series[clean_series.index.get_level_values("techs").isin(TRANSMISSION_NAMES)] /= 2
    if clean_series.empty:
        return None
    else:
        return clean_series


def add_units_to_caps(energy_caps, energy_flows, cost_optimal_model):
    """
    Get units for nameplate capacities and add additional capacities for multi-carrier
    technologies, to give an approximate maximum capacity for non-primary carriers.
    """
    multicarrier_info = split_loc_techs(
        cost_optimal_model.inputs.lookup_primary_loc_tech_carriers_out,
        return_as="Series"
    )
    multicarrier_info = (
        multicarrier_info
        .str
        .split("::", expand=True)[2]
        .to_frame("carriers")
    )
    multicarrier_info = (
        multicarrier_info
        .replace({
            x: x.split("_")[1] if "_" in x else x
            for x in np.unique(multicarrier_info.values)
        })
        .groupby(level="techs").first()
        .set_index("carriers", append=True)
    )
    non_multi_carrier = (
        energy_flows.drop(
            multicarrier_info
            .index
            .levels[0],
            level='techs'
        )
        .dropna(subset=["flow_out_max"])
        .reset_index(["carriers", "unit"])
    )
    levels_to_move = [i for i in energy_flows.index.names if i not in multicarrier_info.index.names]
    multi_carrier = (
        energy_flows
        .unstack(levels_to_move)
        .reindex(multicarrier_info.index)
        .stack(levels_to_move)
        .dropna(subset=["flow_out_max"])
        .reorder_levels(energy_flows.index.names)
        .reset_index(["carriers", "unit"])
    )
    all_carrier_info = (
        pd.concat([non_multi_carrier, multi_carrier])
        .loc[:, ["carriers", "unit"]]
    )
    all_carrier_info.loc[all_carrier_info.unit == "twh", "unit"] = "tw"
    all_carrier_info.loc[all_carrier_info.unit != "tw", "unit"] = all_carrier_info.loc[all_carrier_info.unit != "tw", "unit"] + "_per_hour"

    energy_caps_with_units = (
        pd.concat([energy_caps, all_carrier_info.reindex(energy_caps.index)], axis=1, sort=True)
        .set_index(["carriers", "unit"], append=True)
    )
    assert len(energy_caps_with_units) == len(energy_caps)

    secondary_carrier_caps = energy_caps_with_units.nameplate_capacity.align(
        energy_flows["flow_out_max"].dropna().droplevel("unit")
    )
    energy_caps_with_units_and_secondary = secondary_carrier_caps[0].fillna(secondary_carrier_caps[1]).dropna()

    assert len(energy_caps_with_units_and_secondary) == len(energy_flows["flow_out_max"].dropna())

    assert energy_caps_with_units_and_secondary.reindex(energy_caps_with_units.index).equals(energy_caps_with_units.nameplate_capacity)

    energy_caps_with_units = energy_caps_with_units.reindex(energy_caps_with_units_and_secondary.index)
    energy_caps_with_units["nameplate_capacity"] = energy_caps_with_units_and_secondary

    return energy_caps_with_units


def get_output_costs(model):
    """
    Get costs associated with model results
    """
    costs = {}
    for _cost in ["cost", "cost_investment", "cost_var"]:
        cost_da = model._model_data[_cost].loc[{"costs": "monetary"}]
        mapped_da = map_da(cost_da, keep_demand=False)

        cost_series = clean_series(mapped_da)
        if cost_series is None:
            continue
        else:
            costs[COST_NAME_MAPPING[_cost]] = (
                cost_series
                .to_frame("billion_2015eur")
                .rename_axis(columns="unit")
                .stack()
            )
    return pd.concat(costs.values(), keys=costs.keys(), axis=1)


def get_input_costs(inputs):
    """
    Get costs used as model inputs
    """
    costs = {}
    for var_name, var_data in inputs.data_vars.items():
        if "costs" not in var_data.dims or not var_name.startswith("cost"):
            continue

        if "cap" in var_name:
            unit = "billion_2015eur_per_tw"
        elif var_name == "cost_om_annual":
            unit = "billion_2015eur_per_tw_per_year"
        elif var_name == "cost_om_annual_investment_fraction":
            unit = "fraction_of_total_investment"
        elif var_name == "cost_depreciation_rate":
            unit = "fraction"
        elif "om_" in var_name:
            unit = "billion_2015eur_per_twh"
        _name = COST_NAME_MAPPING[var_name]
        mapped_da = map_da(var_data.loc[{"costs": "monetary"}], loc_tech_agg="mean")
        costs[_name] = (
            clean_series(mapped_da)
            .to_frame(unit)
            .rename_axis(columns="unit")
            .stack()
        )
        costs[_name].loc[costs[_name].index.get_level_values("unit").str.find("2015_eur") > -1] *= 10

    return costs


def strip_small_techs(energy_caps, energy_flows):
    """
    Reindex energy flows based on nameplate capacity index,
    which has a higher threshold given for `zero_threshold` in `clean_series`
    """
    levels_to_move = [i for i in energy_flows.index.names if i not in energy_caps.index.names]
    return (
        energy_flows
        .unstack(levels_to_move)
        .reindex(energy_caps.index)
        .stack(levels_to_move)
        .reorder_levels(energy_flows.index.names)
        .append(energy_flows.filter(regex="demand", axis=0))
    )


def get_eu_imports(loc_tech_mapping):
    """
    Separate transmission links into those which are and are not inside the EU28
    """
    is_transmission = loc_tech_mapping[1].str.find(":") > -1
    if len(loc_tech_mapping[is_transmission]) == 0:
        return loc_tech_mapping
    in_eu = loc_tech_mapping[0].isin(EU28)
    remote_in_eu = loc_tech_mapping[1].str.split(":", expand=True)[1].str.split("_", expand=True)[0].isin(EU28)
    loc_tech_mapping.loc[is_transmission & in_eu & ~remote_in_eu, 1] = "eu_import"
    loc_tech_mapping.loc[is_transmission & ~in_eu & remote_in_eu, 1] = "eu_import"
    loc_tech_mapping.loc[is_transmission & ~in_eu & ~remote_in_eu, 1] = "external_transmission"
    loc_tech_mapping.loc[is_transmission & in_eu & remote_in_eu, 1] = "internal_eu_transmission"
    assert len(loc_tech_mapping[loc_tech_mapping[1].str.find(":") > -1]) == 0
    return loc_tech_mapping


def get_cleaned_dim_mapping(da, dim, keep_demand=True, dim_agg="sum"):
    """
    Go from a concatenated location::technology(::carrier) list into an unconcatenated
    list, including summing up sub-national regions to the national level and technologies
    which are model artefacts to technologies which are have a real 'equivalent'.

    Parameters
    ----------
    da : xarray DataArray
    dim : string
        The concatenated dimension in `da` to unconcatenate
    keep_demand : bool, default = True
        If true, keep demand 'technologies' in the output
    dim_agg : string, default = "sum"
        How to group the unconcatenated and cleaned up data.
        Any xarray groupby method can be passed.
    """
    split_dim = da[dim].to_series().str.split("::", expand=True)
    split_dim[0] = split_dim[0].str.split("_", expand=True)[0]
    split_dim = get_eu_imports(split_dim)
    split_dim = split_dim.loc[split_dim[1].str.find("tech_heat") == -1]
    if keep_demand is False:
        split_dim = split_dim.loc[split_dim[1].str.find("demand") == -1]

    heat_storage = split_dim.loc[split_dim[1].str.find("heat_storage") > -1, 1]
    if not heat_storage.empty:
        split_dim.loc[split_dim[1].str.find("_heat_storage") > -1, 1] = (
            "heat_storage" + heat_storage.str.partition("heat_storage")[2]
        )
    split_dim.loc[split_dim[1].str.startswith("wind_onshore"), 1] = "wind_onshore"
    split_dim.loc[split_dim[1].str.startswith("ac_"), 1] = "ac_transmission"
    split_dim.loc[split_dim[1].str.startswith("dc_"), 1] = "dc_transmission"

    if "carriers" in dim:
        split_dim.loc[split_dim[2].str.endswith("_heat"), 2] = "heat"
        split_dim.loc[split_dim[2].str.endswith("_transport"), 2] = "transport"
        new_dim = ("locs", "techs", "carriers")
    else:
        new_dim = ("locs", "techs")

    split_dim["new_dim"] = list(split_dim.itertuples(index=False, name=None))
    new_da = da.loc[{dim: split_dim.index}]
    new_da = agg_da(new_da.groupby(xr.DataArray.from_series(split_dim["new_dim"])), dim_agg)
    new_da.coords["new_dim"] = pd.MultiIndex.from_tuples(new_da.coords["new_dim"].to_index(), names=new_dim)

    return new_da


def agg_da(da, agg_method, agg_dim=None):
    """
    Aggregate `agg_dim` dimension of an xarray DataArray `da`,
    including setting 'min_count=1' if `agg_method` is "sum"
    """
    kwargs = {"keep_attrs": True}
    if agg_method == "sum":
        kwargs.update({"min_count": 1})
    return getattr(da, agg_method)(agg_dim, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="Directory containing all scenario files")
    parser.add_argument("outdir", help="Filepath to dump friendly data")
    args = parser.parse_args()

    model_dict, cost_optimal_model = get_models_from_file(args.indir)
    data_dict = combine_scenarios_to_one_dict(model_dict, cost_optimal_model)
    meta = {
        "name": "calliope-sentinel-data",
        "description": "Calliope output dataset",
        "keywords": ["calliope"],
        "license": "CC-BY-4.0"
    }
    dict_to_csvs(data_dict, args.outdir, meta)
