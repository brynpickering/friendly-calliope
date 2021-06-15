import pandas as pd

from friendly_calliope.consolidate_calliope_output import EU28
from friendly_calliope.io import dict_to_csvs

IAMC_GROUP_MAPPING = {
    "nameplate_capacity": "Installed capacity",
    "total_investment_cost": "Investments",
    "flow_in_25": "Hourly power consumption|Percentile 25",
    "flow_in_50": "Hourly power consumption|Percentile 50",
    "flow_in_summer_peak": "Hourly power consumption|Summer Peak",
    "flow_in_winter_peak": "Hourly power consumption|Winter Peak",
    "flow_in_sum": "Hourly power consumption|Yearly",
    "flow_out_25": "Generation|Percentile 25",
    "flow_out_50": "Generation|Percentile 50",
    "flow_out_summer_peak": "Generation|Summer Peak",
    "flow_out_winter_peak": "Generation|Winter Peak",
    "flow_out_sum": "Generation|Yearly",
}

IAMC_MAPPING_GEN = {
    "Electricity|Gas": {"techs": "ccgt", "carriers": "electricity"},
    "Heat|Gas": {"techs": ["methane_boiler", "gas_hob", "chp_methane_extraction"], "carriers": "heat"},
    "Electricity|Solar|Open field": {"techs": "open_field_pv", "carriers": "electricity"},
    "Electricity|Solar|Rooftop PV": {"techs": "roof_mounted_pv", "carriers": "electricity"},
    "Electricity|Wind|Onshore": {"techs": "wind_onshore", "carriers": "electricity"},
    "Electricity|Wind|Offshore": {"techs": "wind_offshore", "carriers": "electricity"},
    "Electricity|Hydro|dam": {"techs": "hydro_reservoir", "carriers": "electricity"},
    "Electricity|Hydro|river": {"techs": "hydro_run_of_river", "carriers": "electricity"},
    "Electricity|Nuclear": {"techs": "nuclear", "carriers": "electricity"},
    "Electricity|Solid bio and waste|Primary solid biomass": {"techs": "chp_biofuel_extraction", "carriers": "electricity"},
    "Heat|Solid bio and waste|Primary solid biomass": {"techs": ["biofuel_boiler", "chp_biofuel_extraction"], "carriers": "heat"},
    "Electricity|Solid bio and waste|Waste": {"techs": "chp_wte_back_pressure", "carriers": "electricity"},
    "Heat|Solid bio and waste|Waste": {"techs": "chp_wte_back_pressure", "carriers": "heat"},
    "Flexibility|Electricity Storage": {"techs": ["battery", "pumped_hydro"], "carriers": "electricity"},
    "Flexibility|Heat Storage": {"techs": ["heat_storage_big", "heat_storage_small"], "carriers": "heat"},
    "Flexibility|Hydrogen Storage": {"techs": "hydrogen_storage", "carriers": "hydrogen"},
    "Flexibility|Interconnect Importing Capacity": {"techs": "eu_import", "carriers": "electricity"},
    "Heat|Electricity|Heat Pumps": {"techs": "hp", "carriers": "heat"},
    "Heat|Electricity|Others": {"techs": ["electric_heater", "electric_hob"], "carriers": "heat"},
    "Hydrogen|Electricity": {"techs": "electrolysis", "carriers": "hydrogen"},
    "P2G|Electricity": {"techs": "hydrogen_to_methane", "carriers": "methane"},
    "P2L|Electricity": {"techs": ["hydrogen_to_methanol", "hydrogen_to_liquids"], "carriers": ["methanol", "diesel", "kerosene"]},
}

IAMC_MAPPING_CON = {
    "Electricity|Heating": {"techs": ["electric_heater", "hp", "electric_hob"], "carriers": "electricity"},
    "Electricity|Road": {"techs": ["light_transport_ev", "heavy_transport_ev"], "carriers": "electricity"}
}


def prep_data_groups(in_data_dict):
    out_data_dict = {i: {} for i in IAMC_GROUP_MAPPING.keys()}

    add_capacity_investment(in_data_dict, out_data_dict)
    add_generation(in_data_dict, out_data_dict)
    add_consumption(in_data_dict, out_data_dict)

    return out_data_dict


def add_capacity_investment(in_data_dict, out_data_dict):
    for iamc_group, grouping in IAMC_MAPPING_GEN.items():
        out_data_dict["nameplate_capacity"][iamc_group] = slice_series(
            in_data_dict["nameplate_capacity"],
            locs=EU28, **grouping
        )
        out_data_dict["total_investment_cost"][iamc_group] = slice_series(
            in_data_dict["total_investment_cost"],
            locs=EU28, **grouping
        )


def add_generation(in_data_dict, out_data_dict):
    for iamc_group, grouping in IAMC_MAPPING_GEN.items():
        for input_data in ["flow_out_25", "flow_out_50", "flow_out_winter_peak", "flow_out_summer_peak", "flow_out_sum"]:
            out_data_dict[input_data][iamc_group] = get_flow_agg(
                slice_series(
                    in_data_dict["flow_out"],
                    locs=EU28, **grouping
                ),
                input_data
            )


def add_consumption(in_data_dict, out_data_dict):
    for iamc_group, grouping in IAMC_MAPPING_CON.items():
        for input_data in ["flow_in_25", "flow_in_50", "flow_in_winter_peak", "flow_in_summer_peak", "flow_in_sum"]:
            out_data_dict[input_data][iamc_group] = get_flow_agg(
                slice_series(
                    in_data_dict["flow_in"],
                    locs=EU28, **grouping
                ),
                input_data
            )


def get_flow_agg(flow, agg):
    flow = flow.unstack("timesteps")
    if agg.endswith("25"):
        flow = flow.quantile(0.25, axis=1)
    elif agg.endswith("50"):
        flow = flow.quantile(0.50, axis=1)
    elif agg.endswith("winter_peak"):
        flow = flow.loc[:, flow.columns.month.isin([12, 1, 2])].max(axis=1)
    elif agg.endswith("summer_peak"):
        flow = flow.loc[:, flow.columns.month.isin([6, 7, 8])].max(axis=1)
    elif agg.endswith("sum"):
        flow = flow.sum(axis=1, min_count=1)
    return flow


def sum_multiple_items_in_dim(series, dim, items):
    if isinstance(items, list):
        series = series[series.index.get_level_values(dim).isin(items)]
    else:
        series = series.xs(items, level=dim)
    return series


def slice_series(series, **kwargs):
    levels = [i for i in series.index.names if i not in ["techs", "carriers"]]
    for dim_name, dim_slice in kwargs.items():
        if dim_slice is not None and dim_name in series.index.names:
            series = sum_multiple_items_in_dim(series, dim_name, dim_slice)
    return series.sum(level=levels, min_count=1)


def iamc_dict_to_csvs(data_dict, outdir):
    meta = {
        "name": "calliope-sentinel-free-model-run",
        "description": "Calliope output dataset for IAMC conversion",
        "keywords": ["calliope", "iamc", "free-model-runs"],
        "license": "CC-BY-4.0"
    }
    data_dict = {
        indicator: pd.concat(data.values(), keys=data.keys(), names=["techs"])
        for indicator, data in data_dict.items()
    }
    dict_to_csvs(
        data_dict,
        outdir,
        meta,
        name_mapping=IAMC_GROUP_MAPPING,
        iamc_mapping="{}|{{technology}}"
    )
