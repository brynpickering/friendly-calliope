import pandas as pd
import numpy as np

from friendly_calliope.consolidate_calliope_output import EU28
from friendly_calliope.io import write_dpkg

IAMC_GROUP_MAPPING = {
    "nameplate_capacity": "Installed capacity",
    "total_investment_cost": "Investments",
    "flow_in_25": "Hourly power consumption|Percentile 25",
    "flow_in_50": "Hourly power consumption|Percentile 50",
    "flow_in_summer_peak": "Hourly power consumption|Summer Peak",
    "flow_in_winter_peak": "Hourly power consumption|Winter Peak",
    "flow_in_sum": "Final energy consumption",
    "service_demand": "Service demand",
    "flow_out_25": "Generation|Percentile 25",
    "flow_out_50": "Generation|Percentile 50",
    "flow_out_summer_peak": "Generation|Summer Peak",
    "flow_out_winter_peak": "Generation|Winter Peak",
    "flow_out_sum": "Generation|Yearly",
    "total_system_emissions": "Emissions|Kyoto gases|Fossil CO2",
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
    #"Flexibility|Hydrogen Storage": {"techs": "hydrogen_storage", "carriers": "hydrogen"},
    "Heat|Electricity|Heat Pumps": {"techs": "hp", "carriers": "heat"},
    "Heat|Electricity|Others": {"techs": ["electric_heater", "electric_hob"], "carriers": "heat"},
    "Hydrogen|Electricity": {"techs": "electrolysis", "carriers": "hydrogen"},
    "P2G|Electricity": {"techs": "hydrogen_to_methane", "carriers": "methane"},
    "P2L|Electricity": {"techs": ["hydrogen_to_methanol", "hydrogen_to_liquids"], "carriers": ["methanol", "diesel", "kerosene"]},
}

IAMC_MAPPING_CON = {
    "Electricity|Heating": {"techs": ["electric_heater", "hp", "electric_hob"], "carriers": "electricity"},
    "Electricity|Road": {"techs": ["light_transport_ev", "heavy_transport_ev"], "carriers": "electricity"},
    "Diesel|Road": {"techs": ["light_transport_ice", "heavy_transport_ice"], "carriers": "diesel"},
}

IAMC_MAPPING_FINAL_CON = {
    "Heating|Undefined|Gases|Methane": {"techs": ["methane_boiler", "gas_hob", "chp_methane_extraction"], "carriers": "methane"},
    "Heating|Undefined|Solid bio and waste|Primary solid biomass": {"techs": ["biofuel_boiler", "chp_biofuel_extraction"], "carriers": "biofuel"},
    "Heating|Undefined|Solid bio and waste|Waste": {"techs": "chp_wte_back_pressure", "carriers": "waste"},
    "Heating|Undefined|Electricity": {"techs": ["electric_heater", "hp", "electric_hob"], "carriers": "electricity"},
    "Heating|Undefined|District heating": {"techs": ["chp_methane_extraction", "chp_biofuel_extraction", "chp_wte_back_pressure"], "carriers": "heat"},
    "Transportation|Road|Diesel": {"techs": ["light_transport_ice", "heavy_transport_ice"], "carriers": "diesel"},
    "Transportation|Road|Electricity": {"techs": ["light_transport_ev", "heavy_transport_ev"], "carriers": "electricity"},
}

IAMC_MAPPING_FINAL_CON_FROM_ANNUAL_DEMAND = {
    "Transportation|Rail|Passenger|Electricity": {"dataset": "transport_demand", "cat_name": "rail", "end_use": "electricity"},
    "Transportation|Rail|Freight|Electricity": {"dataset": "industry_demand", "cat_name": "rail", "end_use": "electricity"},
    "Transportation|Navigation|Liquids|Diesel": {"dataset": "industry_demand", "cat_name": "marine", "end_use": "diesel"},
    "Transportation|Aviation|Liquids|Kerosene": {"dataset": "industry_demand", "cat_name": "air", "end_use": "kerosene"},
    "Industry|Liquids|Diesel": {"dataset": "industry_demand", "cat_name": "industry", "end_use": "diesel"},
    "Industry|Liquids|Methanol": {"dataset": "industry_demand", "cat_name": "industry", "end_use": "methanol"},
    "Industry|Gases|Methane": {"dataset": "industry_demand", "cat_name": "industry", "end_use": "methane"},
    "Industry|Electricity": {"dataset": "industry_demand", "cat_name": "industry", "end_use": "electricity"},
}

IAMC_MAPPING_SERVICE_DEMAND = {
    "Road|Passenger|PC0": {"dataset": "transport_demand", "cat_name": "road", "end_use": ["passenger_car", "motorcycle"]},
    "Road|Passenger|BS0": {"dataset": "transport_demand", "cat_name": "road", "end_use": "bus"},
    "Road|Freight|LDV": {"dataset": "transport_demand", "cat_name": "road", "end_use": "ldv"},
    "Road|Freight|HDV": {"dataset": "industry_demand", "cat_name": "road", "end_use": "hdv"},
    "Heating|Space heating|Residential": {"dataset": "heat_demand", "cat_name": "household", "end_use": "space_heat"},
    "Heating|Space heating|Services": {"dataset": "heat_demand", "cat_name": "commercial", "end_use": "space_heat"},
    "Heating|Water heating and cooking|Residential": {"dataset": "heat_demand", "cat_name": "household", "end_use": ["water_heat", "cooking"]},
    "Heating|Water heating and cooking|Services": {"dataset": "heat_demand", "cat_name": "commercial", "end_use": ["water_heat", "cooking"]},
}

IAMC_MAPPING_FOSSILS = {
    "Liquids|Fossil|Diesel": {"techs": "diesel_supply"},
    "Transportation|Aviation|Liquids|Fossil|Kerosene": {"techs": "kerosene_supply"},
    "Fossil|Gas|Natural gas": {"techs": "methane_supply"},
    "Heavy industries|Chemicals and petrochemicals|Fossil|Methanol": {"techs": "methanol_supply"},
    "Electricity|Fossil|Coal": {"techs": "coal_power_plant"},
}


def prep_data_groups(in_data_dict, annual_demand_2030, annual_demand_2050):
    out_data_dict = {i: {} for i in IAMC_GROUP_MAPPING.keys()}

    print("add_capacity_investment")
    add_capacity_investment(in_data_dict, out_data_dict)
    print("add_import_capacity_investment")
    add_import_capacity_investment(in_data_dict, out_data_dict)
    print("add_generation")
    add_generation(in_data_dict, out_data_dict)
    print("add_hourly_consumption")
    add_hourly_consumption(in_data_dict, out_data_dict)
    print("add_final_consumption")
    add_final_consumption(in_data_dict, out_data_dict)
    print("add_final_consumption_from_annual_demand")
    add_final_consumption_from_annual_demand(in_data_dict, out_data_dict, annual_demand_2030, annual_demand_2050)
    print("add_service_demand")
    add_service_demand(in_data_dict, out_data_dict, annual_demand_2030, annual_demand_2050)
    print("add_fossils")
    add_fossils(in_data_dict, out_data_dict)

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

def add_import_capacity_investment(in_data_dict, out_data_dict):
    out_data_dict["total_investment_cost"]["Flexibility|Interconnect Importing Capacity"] = (
        in_data_dict["net_transfer_capacity"]
        .unstack("importing_region")
        [EU28]
        .drop(EU28, level="exporting_region")
        .rename_axis(columns="locs")
        .stack()
        .sum(level=list(out_data_dict["total_investment_cost"].values())[0].index.names)
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


def add_hourly_consumption(in_data_dict, out_data_dict):
    for iamc_group, grouping in IAMC_MAPPING_CON.items():
        for input_data in ["flow_in_25", "flow_in_50", "flow_in_winter_peak", "flow_in_summer_peak"]:
            out_data_dict[input_data][iamc_group] = get_flow_agg(
                slice_series(
                    in_data_dict["flow_in"],
                    locs=EU28, **grouping
                ),
                input_data
            )

def add_final_consumption(in_data_dict, out_data_dict):
    for iamc_group, grouping in IAMC_MAPPING_FINAL_CON.items():
        out_data_dict["flow_in_sum"][iamc_group] = get_flow_agg(
            slice_series(
                in_data_dict["flow_in"],
                locs=EU28, **grouping
            ),
            "sum"
        )


def add_final_consumption_from_annual_demand(
    in_data_dict, out_data_dict, annual_demand_2030, annual_demand_2050
):
    basis_series = list(out_data_dict["flow_in_sum"].values())[0]
    weather_year = in_data_dict["flow_in"].index.get_level_values("timesteps").year.unique().item()
    scenarios = basis_series.index.get_level_values("scenario").unique()
    levels = basis_series.index.names
    for iamc_group, grouping in IAMC_MAPPING_FINAL_CON_FROM_ANNUAL_DEMAND.items():
        out_data_dict["flow_in_sum"][iamc_group] = annual_demand_to_friendly_format(
        annual_demand_2030, annual_demand_2050, grouping, weather_year, scenarios, levels
    )


def add_service_demand(
    in_data_dict, out_data_dict, annual_demand_2030, annual_demand_2050
):
    basis_series = list(out_data_dict["flow_in_sum"].values())[0]
    weather_year = in_data_dict["flow_in"].index.get_level_values("timesteps").year.unique().item()
    scenarios = basis_series.index.get_level_values("scenario").unique()
    levels = basis_series.index.names
    for iamc_group, grouping in IAMC_MAPPING_SERVICE_DEMAND.items():
        out_data_dict["service_demand"][iamc_group] = annual_demand_to_friendly_format(
        annual_demand_2030, annual_demand_2050, grouping, weather_year, scenarios, levels
    )


def add_fossils(in_data_dict, out_data_dict):
    for iamc_group, grouping in IAMC_MAPPING_FOSSILS.items():
        out_data_dict["total_system_emissions"][iamc_group] = slice_series(
            in_data_dict["total_system_emissions"],
            locs=EU28, **grouping
        )
        if iamc_group != "Coal":
            out_data_dict["flow_in_sum"][iamc_group] =  get_flow_agg(
                slice_series(
                    in_data_dict["flow_out"],
                    locs=EU28, **grouping
                ),
                "flow_out_sum"
            )
        else:
            out_data_dict["flow_in_sum"][iamc_group] =  get_flow_agg(
                slice_series(
                    in_data_dict["flow_out"],
                    locs=EU28, **grouping
                ),
                "flow_out_sum"
            ) / 0.4  # efficiency of coal plant


def get_flow_agg(flow, agg):
    flow = flow.unstack("timesteps")
    resolution = int(round((8760 / len(flow.columns)), 0))
    if agg.endswith("25"):
        flow = flow.div(resolution).quantile(0.25, axis=1)
    elif agg.endswith("50"):
        flow = flow.div(resolution).quantile(0.50, axis=1)
    elif agg.endswith("winter_peak"):
        flow = flow.div(resolution).loc[:, flow.columns.month.isin([12, 1, 2])].max(axis=1)
    elif agg.endswith("summer_peak"):
        flow = flow.div(resolution).loc[:, flow.columns.month.isin([6, 7, 8])].max(axis=1)
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


def slice_annual_demand(df, slice_dict):
    return np.logical_and.reduce([
        df.index.get_level_values(level).isin(value)
        if (hasattr(value, '__iter__') and not isinstance(value, str))
        else df.index.get_level_values(level) == value
        for level, value in slice_dict.items()
    ])


def data_from_annual_demand(annual_demand, slice_dict, weather_year):
    year_and_id_slice = {"id": EU28, "year": weather_year}
    _slice = slice_annual_demand(annual_demand, {**slice_dict, **year_and_id_slice})
    if _slice.sum() == 0:
        return None
    sliced_demand = annual_demand[_slice]
    unit = sliced_demand.index.get_level_values("unit").unique().item()
    scale, new_unit = unit.split(" ")
    scaled_demand = sliced_demand * float(scale)
    scaled_demand = scaled_demand.rename({unit: new_unit}, level="unit")
    return scaled_demand.sum(level=["id", "unit"]).rename_axis(index=["locs", "unit"])


def annual_demand_to_friendly_format(
    annual_demand_2030, annual_demand_2050, slice_dict, weather_year, scenarios, levels
):
    _2030 = data_from_annual_demand(annual_demand_2030, slice_dict, weather_year)
    _2050 = data_from_annual_demand(annual_demand_2050, slice_dict, weather_year)

    all_data = []
    for year, demand_data in {"2030": _2030, "2050": _2050}.items():
        if demand_data is None:
            print(f"Skipping slice {slice_dict} for year {year}")
            continue
        for scenario in scenarios:
            all_data.append(
                demand_data.to_frame((scenario, year))
                .rename_axis(columns=["scenario", "year"])
                .stack(["scenario", "year"])
            )
    return pd.concat(all_data).reorder_levels(levels).sort_index()


def write_dpkg_with_iamc(data_dict, outdir):
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
    write_dpkg(
        data_dict,
        outdir,
        meta,
        name_mapping=IAMC_GROUP_MAPPING,
        iamc_mapping="{}|{{technology}}"
    )
