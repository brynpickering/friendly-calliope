import argparse

import pandas as pd
import numpy as np
import xarray as xr

from calliope.core.util.dataset import split_loc_techs

from friendly_calliope.io import get_models_from_file, write_dpkg

EU28 = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU",
    "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "NLD", "POL", "PRT", "ROU",
    "SVK", "SVN", "ESP", "SWE", "GBR"
]
ZERO_THRESHOLD = 5e-6  # capacities lower than this will be considered as effectively zero
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

EMISSIONS_NAME_MAPPING = {
    'cost_energy_cap': "emissions_per_nameplate_capacity",
    'cost_om_prod': "emissions_per_flow_out",
    'cost_om_annual_investment_fraction': "annual_emissions_per_unit_invested",
    'cost_storage_cap': "emissions_per_storage_capacity",
    'cost_om_annual': "annual_emissions_per_nameplate_capacity",
    'cost_depreciation_rate': "depreciation_rate",
    'cost_om_con': "emissions_per_flow_in",
    'cost': 'total_system_emissions',
    'cost_investment': 'total_investment_emissions',
    'cost_var': 'total_operation_emissions',
}

NEW_TECH_NAMES = {
    "internal_to_external_transmission": "Transmission from countries inside the region of interest to those outside it",
    "external_to_internal_transmission": "Transmission from countries outside the region of interest to those inside it",
    "external_transmission": "Transmission between and within countries outside the region of interest",
    "heat_storage_big": "District-scale heat storage vessels",
    "heat_storage_small": "Building-scale heat storage vessels",
    "internal_transmission": "Transmission between and within countries inside the region of interest",
    "wind_onshore": "Onshore wind",
}

TRANSMISSION_NAMES = ["internal_to_external_transmission", "external_to_internal_transmission", "external_transmission", "internal_transmission"]


def combine_scenarios_to_one_dict(
    model_dict, cost_optimal_model=None, new_dimension_name="scenario", **kwargs
):
    """
    Take in a dictionary of Calliope models and create a dictionary of processed data ready to share with the world
    """
    all_data_dict = {}
    if cost_optimal_model is None:
        cost_optimal_model = list(model_dict.values())[0]
    if isinstance(new_dimension_name, str):
        new_dimension_name = [new_dimension_name]

    names = cost_optimal_model.inputs.names.to_series()
    kwargs["timestep_resolution"] = cost_optimal_model.inputs.timestep_resolution

    all_data_dict.update(get_input_costs(cost_optimal_model.inputs, **kwargs))
    energy_caps = pd.concat(
        [get_energy_caps(model, **kwargs) for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name,
    )
    valid_loc_techs = get_valid_loc_techs(energy_caps)
    kwargs["valid_loc_techs"] = valid_loc_techs

    output_costs = pd.concat(
        [get_output_costs(model, **kwargs) for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name
    )
    if "co2" in cost_optimal_model._model_data.costs.values:
        co2_kwargs = {"cost_class": "co2", "unit": "MtCO2", "mapping": EMISSIONS_NAME_MAPPING, **kwargs}
        all_data_dict.update(get_input_costs(cost_optimal_model.inputs, **co2_kwargs))
        _output_emissions = {}
        for model_name, model in model_dict.items():
            if "co2" in model._model_data.costs.values:
                _output_emissions[model_name] = get_output_costs(model, **co2_kwargs)
        output_emissions = pd.concat(
            _output_emissions.values(), keys=_output_emissions.keys(), names=new_dimension_name
        )
        dataframe_to_dict_elements(output_emissions, all_data_dict)

    storage_caps = pd.concat(
        [get_storage_caps(model, **kwargs) for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name,
    )
    energy_flows = pd.concat(
        [get_flows(model, None, **kwargs) for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name,
    )
    energy_flows_sum = pd.concat(
        [get_flows(model, "sum", **kwargs) for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name,
    )
    energy_flows_max = pd.concat(
        [get_flows(model, "max", **kwargs) for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name,
    )
    transmission_flows = pd.concat(
        [get_transmission_flows(model, "sum", **kwargs) for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name,
    )
    transmission_caps = pd.concat(
        [get_transmission_caps(model, **kwargs) for model in model_dict.values()],
        keys=model_dict.keys(), names=new_dimension_name,
    )
    energy_caps = add_units_to_caps(energy_caps, energy_flows_max, cost_optimal_model)

    names = names.reindex(energy_caps.index.get_level_values("techs").unique()).fillna(NEW_TECH_NAMES)
    assert names.isna().sum() == 0

    for df in [
        output_costs, energy_caps, energy_flows, energy_flows_sum, energy_flows_max,
        storage_caps, transmission_flows, transmission_caps
    ]:
        dataframe_to_dict_elements(df, all_data_dict)

    all_data_dict["names"] = names

    return all_data_dict


def dataframe_to_dict_elements(df, data_dict):
    """DF to dict of series, with key names == column names"""
    data_dict.update({col_name: df.loc[:, col_name] for col_name in df.columns})


def get_energy_caps(model, **kwargs):
    """Get energy capacity"""
    mapped_da = map_da(model._model_data["energy_cap"], keep_demand=False, **kwargs)
    series = clean_series(mapped_da, zero_threshold=ZERO_THRESHOLD)
    if series is None:
        return None
    else:
        return (
            series
            .sum(level=["techs", "locs"])
            .to_frame("nameplate_capacity")
            .div(10)
        )


def get_storage_caps(model, **kwargs):
    """Get storage capacity"""
    mapped_da = map_da(model._model_data["storage_cap"], keep_demand=False, **kwargs)
    series = clean_series(mapped_da, **kwargs)
    if series is None:
        return None
    else:
        return (
            series
            .sum(level=["techs", "locs"])
            .to_frame("storage_capacity")
            .assign(unit="twh")
            .set_index("unit", append=True)
            .div(10)
        )


def get_valid_loc_techs(df):
    return df.index.get_level_values("locs") + "::" + df.index.get_level_values("techs")


def get_a_flow(model, flow_direction, timeseries_agg, **kwargs):
    mapped_da = map_da(
        model._model_data[f"carrier_{flow_direction}"], timeseries_agg=timeseries_agg, **kwargs
    )
    return (
        clean_series(mapped_da, **kwargs)
        .div(10)
        .abs()
        .to_frame(flow_name(flow_direction, timeseries_agg))
        .assign(unit='twh')
        .reset_index()
    )


def flow_name(flow_direction, timeseries_agg):
    direction = "out" if flow_direction == "prod" else "in"
    return f"flow_{direction}_{timeseries_agg}" if timeseries_agg is not None else f"flow_{direction}"


def get_flows(model, timeseries_agg, **kwargs):
    prod = get_a_flow(model, "prod", timeseries_agg, **kwargs)
    con = get_a_flow(model, "con", timeseries_agg, **kwargs)

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


def get_transmission_flows(model, timeseries_agg, **kwargs):
    kwargs["valid_loc_techs"] = None
    region_group = kwargs.get("region_group", "countries")
    flows = get_flows(
        model, timeseries_agg, transmission_only=True, halve_transmission=False,
        zero_threshold=ZERO_THRESHOLD, **kwargs
    )
    _from = "exporting_region"
    _to = "importing_region"
    index_names = flows.rename_axis(index={"techs": _from, "locs": _to}).index.names
    summed_flows = []
    for flow in ["prod", "con"]:
        _flow = flows[flow_name(flow, timeseries_agg)]
        remote_loc = _to if flow == "con" else _from
        loc = _from if flow == "con" else _to
        _flow = (
            _flow
            .rename(lambda x: rename_locations(pd.Series([x.split(":")[1]]), region_group).item(), level="techs")
            .rename_axis(index={"techs": remote_loc, "locs": loc})
        )
        _flow_summed_across_all_transmission_techs = _flow.groupby(level=index_names).sum()
        summed_flows.append(_flow_summed_across_all_transmission_techs)

    # take the mean, which averages over transmission losses in either direction along a link
    transmission = pd.concat(summed_flows, axis=1).mean(axis=1)
    import_export = pd.concat(
        [
            transmission,
            transmission
            .rename_axis(index={_from: _to, _to: _from})
            .reorder_levels(transmission.index.names)
        ],
        axis=1, keys=["import", "export"], sort=True
    )

    net_import = import_export["import"] - import_export["export"]
    # Don't keep links that have the same importand export location
    return net_import[
        net_import.index.get_level_values("exporting_region") !=
        net_import.index.get_level_values("importing_region")
    ].to_frame("net_import")


def get_transmission_caps(model, **kwargs):
    region_group = kwargs.get("region_group", "countries")
    _from = "exporting_region"
    _to = "importing_region"
    mapped_da = map_da(model._model_data["energy_cap"], keep_demand=False, transmission_only=True, **kwargs)
    df = clean_series(mapped_da, zero_threshold=ZERO_THRESHOLD).unstack("locs")
    df.index = df.index.str.split(":", expand=True).rename(["techs", _from])
    series = (
        df
        .rename(lambda x: rename_locations(pd.Series([x]), region_group).item(), level=_from)
        .rename(lambda x: "ac_transmission" if x.startswith("ac") else "dc_transmission", level="techs")
        .rename_axis(columns=_to)
        .stack()
    )
    return series.groupby(level=series.index.names).sum().to_frame("net_transfer_capacity")


def map_da(da, keep_demand=True, timeseries_agg="sum", loc_tech_agg="sum", **kwargs):
    loc_tech_dim = [i for i in da.dims if "loc_tech" in i][0]
    mapped_da = get_cleaned_dim_mapping(da, loc_tech_dim, keep_demand, loc_tech_agg, **kwargs)
    if "timesteps" in mapped_da.dims and timeseries_agg is not None:
        return agg_da(mapped_da, timeseries_agg, "timesteps", **kwargs)
    else:
        return mapped_da


def clean_series(da, zero_threshold=0, halve_transmission=True, valid_loc_techs=None, **kwargs):
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
    if valid_loc_techs is not None:
        clean_series = clean_series.where(
            (clean_series.index.get_level_values("locs") + "::" + clean_series.index.get_level_values("techs"))
            .isin(valid_loc_techs)
        ).dropna()
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
            level='techs',
            errors="ignore"
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

    energy_caps_with_units_and_secondary = (
        secondary_carrier_caps[0]
        .fillna(secondary_carrier_caps[1])
        .dropna()
        .rename({np.nan: "tw"}, level="unit")
    )

    assert len(energy_caps_with_units_and_secondary) == len(energy_flows["flow_out_max"].dropna())

    assert energy_caps_with_units_and_secondary.reindex(energy_caps_with_units.index).equals(energy_caps_with_units.nameplate_capacity)

    energy_caps_with_units = energy_caps_with_units.reindex(energy_caps_with_units_and_secondary.index)
    energy_caps_with_units["nameplate_capacity"] = energy_caps_with_units_and_secondary

    return energy_caps_with_units


def get_output_costs(
    model, cost_class="monetary", unit="billion_2015eur", mapping=COST_NAME_MAPPING, **kwargs
):
    """
    Get costs associated with model results
    """
    costs = {}
    for _cost in ["cost", "cost_investment", "cost_var"]:
        cost_da = model._model_data[_cost].loc[{"costs": cost_class}]
        mapped_da = map_da(cost_da, keep_demand=False, **kwargs)

        cost_series = clean_series(mapped_da, **kwargs)
        if cost_series is None:
            continue
        else:
            costs[mapping[_cost]] = (
                cost_series
                .to_frame(unit)
                .rename_axis(columns="unit")
                .stack()
            )
    return pd.concat(costs.values(), keys=costs.keys(), axis=1)


def get_input_costs(
    inputs, cost_class="monetary", unit="billion_2015eur", mapping=COST_NAME_MAPPING,
    **kwargs
):
    """
    Get costs used as model inputs
    """
    costs = {}
    for var_name, var_data in inputs.data_vars.items():
        if "costs" not in var_data.dims or not var_name.startswith("cost"):
            continue

        if "cap" in var_name:
            _unit = f"{unit}_per_tw"
        elif var_name == "cost_om_annual":
            _unit = f"{unit}_per_tw_per_year"
        elif var_name == "cost_om_annual_investment_fraction":
            _unit = "fraction_of_total_investment"
        elif var_name == "cost_depreciation_rate":
            _unit = "fraction"
        elif "om_" in var_name:
            _unit = f"{unit}_per_twh"
        _name = mapping[var_name]
        mapped_da = map_da(var_data.loc[{"costs": cost_class}], loc_tech_agg="mean", **kwargs)
        series = clean_series(mapped_da, halve_transmission=False)
        if series is not None:
            costs[_name] = (
                series
                .to_frame(_unit)
                .rename_axis(columns="unit")
                .stack()
            )
            costs[_name].loc[costs[_name].index.get_level_values("unit").str.find("per_tw") > -1] *= 10

    return costs


def get_imports(loc_tech_mapping, group=EU28, groupname="eu", region_group="countries"):
    """
    Separate transmission links into those which are and are not inside a given
    set of regions. These regions default to the EU28 (i.e. including the UK)

    Parameters
    ----------
    import_group : array of strings, default = all countries of the EU28
        List of regions (post-grouping based on `region_group`) to consider as the
        'region of interest' for imports/exports across the border
    import_groupname : string, default = 'eu'
        Name of the 'region of interest'
    region_group : string or mapping, default = "countries"
        If "countries", group locations to a country level, otherwise group based on
        a mapping from model locations to group names. Mapping allows e.g. one country to
        remain as sub-country regions, while all others are grouped to countries or 'rest'
    """
    is_transmission = loc_tech_mapping[1].str.find(":") > -1
    if len(loc_tech_mapping[is_transmission]) == 0:
        return loc_tech_mapping
    _in = loc_tech_mapping[0].isin(group)
    _remote_in = rename_locations(loc_tech_mapping[1].str.split(":", expand=True)[1], region_group).isin(group)
    loc_tech_mapping.loc[is_transmission & _in & ~_remote_in, 1] = "external_to_internal_transmission"
    loc_tech_mapping.loc[is_transmission & ~_in & _remote_in, 1] = "internal_to_external_transmission"
    loc_tech_mapping.loc[is_transmission & ~_in & ~_remote_in, 1] = "external_transmission"
    loc_tech_mapping.loc[is_transmission & _in & _remote_in, 1] = "internal_transmission"
    assert len(loc_tech_mapping[loc_tech_mapping[1].str.find(":") > -1]) == 0
    return loc_tech_mapping


def rename_locations(locations, region_group):
    if region_group == "countries":
        return locations.str.split("_", expand=True)[0]
    else:
        return locations.replace(region_group)


def get_cleaned_dim_mapping(
    da, dim, keep_demand=True, dim_agg="sum", import_group=EU28, import_groupname="eu",
    region_group="countries", transmission_only=False, **kwargs
):
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
    import_group : array of strings, default = all countries of the EU28
        List of regions (post-grouping based on `region_group`) to consider as the
        'region of interest' for imports/exports across the border
    import_groupname : string, default = 'eu'
        Name of the 'region of interest'
    region_group : string or mapping, default = "countries"
        If "countries", group locations to a country level, otherwise group based on
        a mapping from model locations to group names. Mapping allows e.g. one country to
        remain as sub-country regions, while all others are grouped to countries or 'rest'
    """
    split_dim = da[dim].to_series().str.split("::", expand=True)
    split_dim[0] = rename_locations(split_dim[0], region_group)
    if transmission_only:
        split_dim = split_dim[split_dim[1].str.find(":") > -1]
    else:
        split_dim = get_imports(split_dim, import_group, import_groupname, region_group)
    split_dim = split_dim.loc[split_dim[1].str.find("tech_heat") == -1]
    if keep_demand is False:
        split_dim = split_dim.loc[split_dim[1].str.find("demand") == -1]

    heat_storage = split_dim.loc[split_dim[1].str.find("heat_storage") > -1, 1]
    if not heat_storage.empty:
        split_dim.loc[split_dim[1].str.find("_heat_storage") > -1, 1] = (
            "heat_storage" + heat_storage.str.partition("heat_storage")[2]
        )
    split_dim.loc[split_dim[1].str.startswith("wind_onshore"), 1] = "wind_onshore"

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


def agg_da(da, agg_method, agg_dim=None, **kwargs):
    """
    Aggregate `agg_dim` dimension of an xarray DataArray `da`,
    including setting 'min_count=1' if `agg_method` is "sum"
    """
    if agg_dim == "timesteps" and "timestep_resolution" in kwargs.keys() and agg_method != "sum":
        da = da / kwargs["timestep_resolution"]
    agg_kwargs = {"keep_attrs": True}
    if agg_method == "sum":
        agg_kwargs.update({"min_count": 1})
    return getattr(da, agg_method)(agg_dim, **agg_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", help="Directory containing all scenario files")
    parser.add_argument("outdir", help="Filepath to dump friendly data")
    parser.add_argument(
        "--import_group", default=EU28,
        help="list of regions defining a `region of interest` to track imports/exports"
    )
    parser.add_argument(
        "--import_groupname", default="eu",
        help="Name for 'region of interest' for grouping imports/exports"
    )
    args = parser.parse_args()

    model_dict, cost_optimal_model = get_models_from_file(args.indir)
    kwargs = {
        k: getattr(args, k) for k in ["import_group", "import_groupname"]
        if getattr(args, k) is not None
    }
    data_dict = combine_scenarios_to_one_dict(model_dict, cost_optimal_model, **kwargs)
    meta = {
        "name": "calliope-sentinel-data",
        "description": "Calliope output dataset",
        "keywords": ["calliope"],
        "license": "CC-BY-4.0"
    }
    write_dpkg(data_dict, args.outdir, meta)
