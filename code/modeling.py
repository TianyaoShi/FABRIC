import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from load_data import *

DRAM_SSD_CPU_PACKAGING_WEIGHT_FACTOR = 2.0
GPU_PACKAGING_WEIGHT_FACTOR = 1.5
PO4_TO_P_CONVERSION_FACTOR = 0.33

# HDD Modeling -- handled specially

results_per_gb = {}
metrics = ['AP', 'freshwater EP', 'marine EP', 'freshwater ecotoxicity']

for year in seagate_hdd_lca_results.keys():
    if year == '2019':
        # Skip 2019 as it only considers the MA part, not the entire HDD
        continue
    results_per_gb[int(year)] = {}
    capacity_gb = seagate_hdd_lca_results[year]['capacity (TB)'] * 1000  # Convert TB to GB

    # Calculate US-grid based SOx, NOx, and NH3 emissions
    for pollutant in ['SOx', 'NOx', 'NH3']:
        idx = 0 if int(year) <=2016 else int(year)-2016
        lifelong_emissions = seagate_hdd_lca_results[year]['lifecycle electricity consumption (kWh)'] * unified_emission_factors[pollutant]['US'][idx] / 1000  # Convert to kg
        # AP
        if pollutant in impact_factors['AP'].keys():
            seagate_hdd_lca_results[year]['AP'] -= lifelong_emissions * impact_factors['AP'][pollutant]
        # EP
        if pollutant in impact_factors['EP'].keys() and year !='2014':
           seagate_hdd_lca_results[year]['freshwater EP'] -= lifelong_emissions * impact_factors['EP'][pollutant]
        # FETox
        if pollutant in impact_factors['FETox'].keys():
            seagate_hdd_lca_results[year]['freshwater ecotoxicity'] -= lifelong_emissions * impact_factors['FETox'][pollutant] 
    for metric in metrics:
        if not np.isnan(seagate_hdd_lca_results[year][metric]):
            results_per_gb[int(year)][metric] = seagate_hdd_lca_results[year][metric] / capacity_gb

years = []
carbon_per_gb = []

for hdd in hdd_carbon_footprint_records.values():
    years.append(hdd['year'])
    carbon_per_gb.append(hdd['carbon footprint (kg CO2e)'] / (hdd['capacity (TB)'] * 1000))  # Convert TB to GB

# Convert to numpy arrays for linear regression
X = np.array(years).reshape(-1, 1)
y = np.array(carbon_per_gb)

# Apply log transform to y values (adding small constant to avoid log(0))
y_log = np.log(y + 1e-10)

# Fit linear regression
model = LinearRegression()
model.fit(X, y_log)

# Get the predicted values for 2016 and 2023
y_2016_log = model.predict([[2016]])[0]
y_2023_log = model.predict([[2023]])[0]

# Transform back to original scale
y_2016 = np.exp(y_2016_log)
y_2023 = np.exp(y_2023_log)

# Calculate percentage improvement
percent_improvement = (y_2016 - y_2023) / y_2016 * 100

# Start with 2016 values from results_per_gb
base_values_2016 = {
    'AP': results_per_gb[2016]['AP'],
    'freshwater EP': results_per_gb[2016]['freshwater EP'],
    'freshwater ecotoxicity': results_per_gb[2016]['freshwater ecotoxicity']
}

# Calculate annual improvement rate from carbon footprint analysis
annual_improvement_rate = 1-((y_2023/y_2016)**(1/7))

# FIXME: Extrapolate only on the embodied part -- subtracting power consumption induced AP,EP and assume a similar reduction rario for FETP

# Create years array
years = np.array(range(2016, 2024))

# Calculate predicted values for each metric
predicted_values = {}
for metric, base_value in base_values_2016.items():
    predicted_values[metric] = [
        base_value * (1 - annual_improvement_rate) ** (year - 2016) for year in years
    ]

# Get original 2016 data point from results_per_gb
hdd_impact_factors_by_year = {
    2016: {
        'AP': results_per_gb[2016]['AP'],
        'EP': results_per_gb[2016]['freshwater EP'],
        'FETox': results_per_gb[2016]['freshwater ecotoxicity']
    }
}

# Add predicted values for 2017-2023
for year, idx in enumerate(range(2017, 2024), 1):
    hdd_impact_factors_by_year[idx] = {
        'AP': predicted_values['AP'][idx-2016],
        'EP': predicted_values['freshwater EP'][idx-2016],
        'FETox': predicted_values['freshwater ecotoxicity'][idx-2016]
    }

# Modeling process of storage weight density -- only use as a archive, already loaded from load_data.py

# # Data based on different timelines for each component
# hdd_years = ['2016', '2017', '2019', '2021', '2023']
# ssd_years = ['2016', '2018', '2020', '2021', '2023']
# dram_years = ['2016', '2018', '2020', '2022']

# # Common positions for plotting
# all_years = sorted(list(set(hdd_years + ssd_years + dram_years)))
# position_map = {year: i for i, year in enumerate(all_years)}

# # Parse raw HDD data to get g/GB values
# hdd_ranges_data = {
#     '2016': [605/2000, 635/4000, 780/5000, 780/6000],  # Raw weight/capacity data
#     '2017': [422/1000, 354/1000],  # From EXOS 15E900 and Skyhawk data
#     '2019': [205/1000, 83/1000, 178/1000, 87/1000, 46/1000],  # Various EXOS models
#     '2021': [56/1000, 100/1000, 81/1000, 41/1000],  # EXOS models
#     '2023': [39/1000]  # Latest EXOS models
# }

# # DRAM data in g/GB based on module weights
# dram_ranges_data = {
#     '2016': [45.4/8, 45.4/16, 45.4/32],  # Early DDR4 0.1lb converted
#     '2018': [45.4/16, 45.4/32, 45.4/64, 90.8/64],  # Mix of 0.1lb and 0.2lb modules
#     '2020': [25/32, 45.4/32, 45.4/64],  # Mixed DDR4/DDR5 data
#     '2022': [20/64, 45.4/128, 90.8/128]  # Latest DDR5 data
# }

# # SSD data in g/GB based on form factors and capacities
# ssd_ranges_data = {
#     '2016': [15/240, 80/240, 80/3840],  # M.2 and 2.5" SATA
#     '2018': [15/480, 80/480, 80/3840],  # M.2 and 2.5" SATA
#     '2020': [15/960, 140/960, 140/7680, 105/7680],  # M.2, U.2/3, E1.S
#     '2021': [15/1920, 140/7680, 105/7680],  # M.2, U.2/3, E1.S
#     '2023': [15/1920, 250/15360, 105/7680]  # Latest M.2, E1.L, E1.S
# }

# def prepare_bxp_stats(ranges_data, year_list, packaging_factor=1.0):
#     stats = []
#     positions = []
#     for year in year_list:
#         if year in ranges_data:
#             data = np.array(ranges_data[year]) * packaging_factor
#             stat = {
#                 'med': np.median(data),
#                 'q1': np.percentile(data, 25),
#                 'q3': np.percentile(data, 75),
#                 'whislo': np.min(data),
#                 'whishi': np.max(data),
#                 'fliers': []
#             }
#             stats.append(stat)
#             positions.append(position_map[year])
#     return stats, positions

# # Prepare stats with packaging factors
# ssd_bxp_stats, ssd_positions = prepare_bxp_stats(ssd_ranges_data, ssd_years, DRAM_SSD_CPU_PACKAGING_WEIGHT_FACTOR)
# dram_bxp_stats, dram_positions = prepare_bxp_stats(dram_ranges_data, dram_years, DRAM_SSD_CPU_PACKAGING_WEIGHT_FACTOR)
# hdd_bxp_stats, hdd_positions = prepare_bxp_stats(hdd_ranges_data, hdd_years)
# ssd_medians = [stat['med'] for stat in ssd_bxp_stats]
# dram_medians = [stat['med'] for stat in dram_bxp_stats]
# hdd_medians = [stat['med'] for stat in hdd_bxp_stats]
# components = {
#     'SSD': {'years': [2016, 2018, 2020, 2021, 2023], 'values': ssd_medians},
#     'DRAM': {'years': [2016, 2018, 2020, 2022], 'values': dram_medians},
#     'HDD': {'years': [2016, 2017, 2019, 2021, 2023], 'values': hdd_medians}
# }

# # Function to fill missing years using linear regression on log-transformed data
# def fill_missing_years(years, values, target_years):
#     model = LinearRegression()
#     X = np.array(years).reshape(-1, 1)
#     y = np.log10(values)  # Log transform the values
#     model.fit(X, y)
    
#     X_pred = np.array(target_years).reshape(-1, 1)
#     log_predictions = model.predict(X_pred)
#     return 10**log_predictions  # Transform back to original scale

# # Target years
# all_years = list(range(2016, 2024))

# # Create DataFrame with filled values
# df_storage_weight_density = pd.DataFrame(index=all_years)

# for component, data in components.items():
#     filled_values = fill_missing_years(data['years'], data['values'], all_years)
#     df_storage_weight_density[f'{component}_g/GB'] = filled_values

  

def calculate_cpu_manufacturing_impacts(cpu_specs, prodcution_yield=0.875):
    """
    Calculate the CPU manufacturing impacts based on the provided specifications. 
        - Gas emissions
        - Wastewater discharge
        - Electricity consumption
    
    Args:
        cpu_specs (dict): A dictionary containing CPU specifications. Expected keys are:
            - 'die_size_mm2': Die size in mm^2
            - 'technology_node_nm': Technology node in nm
            - 'production_year': Year of production
            - 'io_die_size_mm2': IO die size in mm^2 (optional, default is 0)
            - 'io_technology_node_nm': IO technology node in nm (optional, default is 0)
        
    Returns:
        impact (dict): A dictionary containing the calculated manufacturing impacts.
    """
    node_to_layer_masks_map = {
        16: 60,
        14: 60,
        12: 60,
        10: 78,
        7: 87,
        6: 87,
        5: 81,
        3: 81
    }

    year = cpu_specs['production_year']-2016  # Adjust year to match the index in the data
    if year < 0 or year > 7:
        raise ValueError("Year must be between 2016 and 2023.")

    impacts = {'AP':0.0, 'EP':0.0, 'FETox':0.0}

    total_produce_units = (node_to_layer_masks_map[cpu_specs['technology_node_nm']] * (cpu_specs['die_size_mm2'] / 300**2) + 
                           node_to_layer_masks_map[cpu_specs['io_die_technology_node_nm']] * (cpu_specs['io_die_size_mm2'] / 300**2)) / prodcution_yield

    # Acid gas emissions 
    for acid in tsmc_acid_emission_mix_ratio["2016"].keys():
        emission_mass = tsmc_acid_emission_mix_ratio[str(year+2016)][acid] * tsmc_emissions_macro_16thro23['per_unit_acid_g/wafer-mask-layer'][year] * total_produce_units / 1000 # kg/device
        for impact_type in ['AP', 'EP', 'FETox']:
            if acid in impact_factors[impact_type].keys():
                impacts[impact_type] += impact_factors[impact_type][acid] * emission_mass

    # SOx and NOx emissions
    for gas in ['SOx', 'NOx']:
        per_unit_emission = tsmc_emissions_macro_16thro23[gas + '_mt'][year]*1e6 / (tsmc_emissions_macro_16thro23['total_acid_mt'][year] * 1e6 / tsmc_emissions_macro_16thro23['per_unit_acid_g/wafer-mask-layer'][year])
        emission_mass = per_unit_emission * total_produce_units / 1000 # kg/device
        for impact_type in ['AP', 'EP', 'FETox']:
            if gas in impact_factors[impact_type].keys():
                impacts[impact_type] += impact_factors[impact_type][gas] * emission_mass
    
    # Wastewater discharge - Cu2+, NH4-N, and COD
    for pollutant in ['Cu2+', 'NH4-N', 'COD']:
        discharge_liter = tsmc_emissions_macro_16thro23['per_unit_wastewater_L/wafer-mask-layer'][year] * total_produce_units # L/device
        discharge_mass = discharge_liter * tsmc_emissions_macro_16thro23[pollutant + '_ppm'][year] * 1e-6 # kg/device
        for impact_type in ['AP', 'EP', 'FETox']:
            if pollutant in impact_factors[impact_type].keys():
                impacts[impact_type] += impact_factors[impact_type][pollutant] * discharge_mass

    # Electricity consumption
    electricity_consumption = tsmc_electricity_consumption['unit_consumption_kWh/wafer-mask-layer'][year] * total_produce_units *(1- tsmc_electricity_consumption['renewable_energy_ratio_%'][year] * 1e-2) # Non-renewable kWh/device
    for pollutant in ['SOx', 'NOx']:
        indirect_emission = electricity_consumption * unified_emission_factors[pollutant]['Taiwan_g/kWh_report'][year] / 1000 # kg/device
        for impact_type in ['AP', 'EP', 'FETox']:
            if pollutant in impact_factors[impact_type].keys():
                impacts[impact_type] += impact_factors[impact_type][pollutant] * indirect_emission
    indirect_emission_NH3 = electricity_consumption * unified_emission_factors['NH3']['Taiwan'][year] / 1000 # kg/device
    for impact_type in ['AP', 'EP', 'FETox']:
        if 'NH3' in impact_factors[impact_type].keys():
            impacts[impact_type] += impact_factors[impact_type]['NH3'] * indirect_emission_NH3

    return impacts

def calculate_storage_manufacturing_impacts(storage_type, production_year, capacity, production_yield=0.875, unit='GB', HBM_edition=None):
    """
    Calculate the manufacturing impacts for different storage types based on the provided specifications.
    
    Args:
        storage_type (str): Type of storage ('DRAM', 'NAND', 'SSD', 'HDD').
        production_year (int): Year of production.
        capacity (float): Storage capacity in GB.
        
    Returns:
        impact (dict): A dictionary containing the calculated manufacturing impacts.
    """
    if storage_type not in ['DRAM', 'SSD', 'HBM', 'HDD']:
        raise ValueError("Invalid storage type. Must be one of ['DRAM', 'SSD', 'HBM', 'HDD'].")
    
    if unit not in ['GB', 'TB']:
        raise ValueError("Invalid unit. Must be either 'GB' or 'TB'.")
    if unit == 'TB':
        capacity *= 1000
    
    if production_year < 2016 or production_year > 2023:
        raise ValueError("Year must be between 2016 and 2023.")
    
    if capacity <= 0:
        raise ValueError("Capacity must be a positive number.")
    
    # Initialize impacts
    impacts = {'AP':0.0, 'EP':0.0, 'FETox':0.0}
    
    # Get the year index
    year_idx = production_year - 2016

    if storage_type == 'DRAM':
        # manufacturing emissions for DRAM
        for pollutant in dram_emissions[production_year].keys():
            pollutant_mass = dram_emissions[production_year][pollutant] * (capacity / (dram_bit_density_by_year[year_idx] * 300**2 / 8 * production_yield)) # kg/device
            for impact_type in ['AP', 'EP', 'FETox']:
                if pollutant in impact_factors[impact_type].keys():
                    impacts[impact_type] += impact_factors[impact_type][pollutant] * pollutant_mass
                    
        # electricity emissions for DRAM
        dram_ratio = hynix_production_data['dram_revenue_ratio'][year_idx]
        dram_wafers = hynix_production_data['estimated_dram_k_wafers'][year_idx] * 1000
        wafer_electricity_consumption = hynix_production_data['electricity_consumption_GWh'][year_idx] * 1e6 * dram_ratio / dram_wafers # kWh/wafer
        electricity_consumption = wafer_electricity_consumption * (capacity / (dram_bit_density_by_year[year_idx] * 300**2 / 8 * production_yield)) # kWh/device
        for pollutant in ['SOx', 'NOx']:
            indirect_emission = electricity_consumption * unified_emission_factors[pollutant]['Korea_g/kWh_report'][year_idx] / 1000
            for impact_type in ['AP', 'EP', 'FETox']:
                if pollutant in impact_factors[impact_type].keys():
                    impacts[impact_type] += impact_factors[impact_type][pollutant] * indirect_emission
        indirect_emission_NH3 = electricity_consumption * unified_emission_factors['NH3']['Korea'][year_idx] / 1000 # kg/device
        for impact_type in ['AP', 'EP', 'FETox']:
            if 'NH3' in impact_factors[impact_type].keys():
                impacts[impact_type] += impact_factors[impact_type]['NH3'] * indirect_emission_NH3

    elif storage_type == 'HBM':
        for pollutant in dram_emissions[production_year].keys():
            pollutant_mass = dram_emissions[production_year][pollutant] * (capacity / (vram_bit_density[HBM_edition] * 300**2 / 8 * production_yield)) # kg/device
            for impact_type in ['AP', 'EP', 'FETox']:
                if pollutant in impact_factors[impact_type].keys():
                    impacts[impact_type] += impact_factors[impact_type][pollutant] * pollutant_mass

        dram_ratio = hynix_production_data['dram_revenue_ratio'][year_idx]
        dram_wafers = hynix_production_data['estimated_dram_k_wafers'][year_idx] * 1000
        wafer_electricity_consumption = hynix_production_data['electricity_consumption_GWh'][year_idx] * 1e6 * dram_ratio / dram_wafers # kWh/wafer
        electricity_consumption = wafer_electricity_consumption * (capacity / (dram_bit_density_by_year[year_idx] * 300**2 / 8 * production_yield)) # kWh/device
        for pollutant in ['SOx', 'NOx']:
            indirect_emission = electricity_consumption * unified_emission_factors[pollutant]['Korea_g/kWh_report'][year_idx] / 1000
            for impact_type in ['AP', 'EP', 'FETox']:
                if pollutant in impact_factors[impact_type].keys():
                    impacts[impact_type] += impact_factors[impact_type][pollutant] * indirect_emission
        indirect_emission_NH3 = electricity_consumption * unified_emission_factors['NH3']['Korea'][year_idx] / 1000 # kg/device
        for impact_type in ['AP', 'EP', 'FETox']:
            if 'NH3' in impact_factors[impact_type].keys():
                impacts[impact_type] += impact_factors[impact_type]['NH3'] * indirect_emission_NH3

    elif storage_type == 'SSD':
        for pollutant in nand_emissions[production_year].keys():
            pollutant_mass = nand_emissions[production_year][pollutant] * (capacity / (ssd_bit_density_by_year[year_idx] * 300**2 / 8 * production_yield))
            for impact_type in ['AP', 'EP', 'FETox']:
                if pollutant in impact_factors[impact_type].keys():
                    impacts[impact_type] += impact_factors[impact_type][pollutant] * pollutant_mass

        # electricity emissions for SSD
        ssd_ratio = hynix_production_data['nand_revenue_ratio'][year_idx]
        ssd_wafers = hynix_production_data['nand_k_wafers'][year_idx] * 1000
        wafer_electricity_consumption = hynix_production_data['electricity_consumption_GWh'][year_idx] * 1e6 * ssd_ratio / ssd_wafers
        electricity_consumption = wafer_electricity_consumption * (capacity / (ssd_bit_density_by_year[year_idx] * 300**2 / 8 * production_yield))
        for pollutant in ['SOx', 'NOx']:
            indirect_emission = electricity_consumption * unified_emission_factors[pollutant]['Korea_g/kWh_report'][year_idx] / 1000
            for impact_type in ['AP', 'EP', 'FETox']:
                if pollutant in impact_factors[impact_type].keys():
                    impacts[impact_type] += impact_factors[impact_type][pollutant] * indirect_emission
        indirect_emission_NH3 = electricity_consumption * unified_emission_factors['NH3']['Korea'][year_idx] / 1000
        for impact_type in ['AP', 'EP', 'FETox']:
            if 'NH3' in impact_factors[impact_type].keys():
                impacts[impact_type] += impact_factors[impact_type]['NH3'] * indirect_emission_NH3

    else:
        for impact_type in ['AP', 'EP', 'FETox']:
            impacts[impact_type] += hdd_impact_factors_by_year[production_year][impact_type] * capacity

    return impacts

def calculate_gpu_manufacturing_impacts(gpu_specs, memory_production_yield=0.875):
    """
    Calculate the GPU manufacturing impacts based on the provided specifications. 
    
    GPU is basically a special CPU die plus HBM. We will use the existing functions to calculate the overall impact.
    Args:
        gpu_specs (dict): A dictionary containing GPU specifications. Superset of CPU specs.
            - 'die_size_mm2': Die size in mm^2
            - 'technology_node_nm': Technology node in nm
            - 'production_year': Year of production
            - 'die_production_yield': Production yield for the die. Could be very low for A100 / H100.
            - 'hbm_capacity_GB': HBM capacity in GB 
            - 'hbm_type': HBM type (e.g., 'HBM2', 'HBM3')
    """
    gpu_specs['io_die_size_mm2'] = 0
    gpu_specs['io_die_technology_node_nm'] = 14
    die_impacts = calculate_cpu_manufacturing_impacts(gpu_specs, prodcution_yield=gpu_specs['die_production_yield'])
    hbm_impacts = calculate_storage_manufacturing_impacts('HBM', gpu_specs['production_year'], gpu_specs['hbm_capacity_GB'], production_yield=memory_production_yield, HBM_edition=gpu_specs['hbm_type'])
    impacts = {
        'AP': die_impacts['AP'] + hbm_impacts['AP'],
        'EP': die_impacts['EP'] + hbm_impacts['EP'],
        'FETox': die_impacts['FETox'] + hbm_impacts['FETox']
    }
    return impacts

distance = {
    'default': {
        # East Asia manufacture, US-CA usage & recycling, marine transport
        'Truck': 200,  # km, 100 for fab-to-shipyard, 100 for ship-to-warehouse
        'Ship': 14000,  # km, East Asia to US west coast
        'Air': 0,    # km
    },
}

def calculate_transport_impact(mass, distance=distance['default'], mass_unit='g', year=2023):
    """
    Calculate transportation impact given mass and distance
    Source: Ecoinvent v2.0, v3.9 Transportation
    
    Parameters:
    mass: float - mass of the component
    distance: dict - distances for different transport modes
    mass_unit: str - unit of mass ('g', 'kg', 't', default 'g')
    year: int - year for which to calculate impact (2016-2023)
    
    Returns:
    dict - impact values for each transport mode and impact type
    """
    # Convert mass to tonnes
    conversion = {
        'g': 1e-6,
        'kg': 1e-3,
        't': 1
    }
    
    if mass_unit not in conversion:
        raise ValueError("Mass unit must be 'g', 'kg', or 't'")
    
    mass_in_tonnes = mass * conversion[mass_unit]
    
    # Get correct year's factors
    if year <= 2017:
        factors = cf_transportation['old']
    elif year >= 2020:
        factors = cf_transportation['new']
    else:
        factors = cf_transportation[str(year)]
    
    # Calculate impacts
    impacts = {'AP':0.0, 'EP':0.0, 'FETox':0.0}
    for transport_means, distances in distance.items():
        for metric in ['AP', 'EP', 'FETox']:
            if transport_means in factors and metric in factors[transport_means]:
                impacts[metric] += factors[transport_means][metric] * mass_in_tonnes * distances 
    
    return impacts

# def calculate_recycling_impact(mass, mass_unit='kg', recyling_rate=0.9, inceneration_rate=0.08, landfill_rate=0.02):
#     """
#     Calculate recycling impact given mass and recycling rates
    
#     Parameters:
#     mass: float - mass of the component
#     mass_unit: str - unit of mass ('g', 'kg', 't', default 'kg')
#     recyling_rate: float - recycling rate (default 0.9)
#     inceneration_rate: float - inceneration rate (default 0.08)
#     landfill_rate: float - landfill rate (default 0.02)
    
#     Returns:
#     dict - impact values for each impact type
#     """

#     def quick_impact(mass, recycling_rate, inceneration_rate, landfill_rate, impact_type):
#         impact_factor_values = {
#             'AP': {
#                 'Metal': 1e-3,
#                 'Plastic': 4e-3,
#                 'glass': 2e-4,
#                 'inceneration_credit': 1e-3,
#                 'landfill': 2e-5,
#                 'recycling_credit': 2e-3
#             },
#             'EP': {
#                 'Metal': 2e-4,
#                 'Plastic': 7e-4,
#                 'glass': 4e-5,
#                 'inceneration_credit': 1.5e-4,
#                 'landfill': 3e-6,
#                 'recycling_credit': 3e-4
#             },
#             'FETox': {
#                 'Metal': 30,
#                 'Plastic': 1.9,
#                 'glass': 0.2,
#                 'inceneration_credit': 3e-4,
#                 'landfill': 1e-4,
#                 'recycling_credit': 20
#             }
#         }
#         return mass * (0.45 * impact_factor_values[impact_type]['Metal'] + 0.25 * impact_factor_values[impact_type]['Plastic'] + 0.15 * impact_factor_values[impact_type]['glass']) + \
#             mass * inceneration_rate * impact_factor_values[impact_type]['inceneration_credit'] + \
#             mass * landfill_rate * impact_factor_values[impact_type]['landfill'] - \
#             mass * 0.45 * recycling_rate * impact_factor_values[impact_type]['recycling_credit']

#     # Convert mass to kg
#     conversion = {
#         'g': 1e-3,
#         'kg': 1,
#         't': 1e3
#     }
    
#     if mass_unit not in conversion:
#         raise ValueError("Mass unit must be 'g', 'kg', or 't'")
    
#     mass_in_kg = mass * conversion[mass_unit]
    
#     # Calculate impacts
#     impacts = {
#         'AP': quick_impact(mass_in_kg, recyling_rate, inceneration_rate, landfill_rate, 'AP'),
#         'EP': quick_impact(mass_in_kg, recyling_rate, inceneration_rate, landfill_rate, 'EP'),
#         'FETox': quick_impact(mass_in_kg, recyling_rate, inceneration_rate, landfill_rate, 'FETox')
#     }
    
#     return impacts

def calculate_recycling_impact(mass, mass_unit='kg', **kwargs):
    """
    Calculate recycling impact given mass, assuming linear extrapolated impact based on Fairphone 5 LCA data.
    https://www.fairphone.com/wp-content/uploads/2024/09/Fairphone5_LCA_Report_2024.pdf
    
    Parameters:
    mass: float - mass of the component
    mass_unit: str - unit of mass ('g', 'kg', 't', default 'g')
    
    Returns:
    dict - impact values for each impact type
    """
    # Convert mass to kg
    conversion = {
        'g': 1e-3,
        'kg': 1,
        't': 1e3
    }
    
    if mass_unit not in conversion:
        raise ValueError("Mass unit must be 'g', 'kg', or 't'")
    
    mass_in_kg = mass * conversion[mass_unit]
    
    fairphone_recylcing_impact = {
        'AP': 7e-4/0.212, # Estimation, 1 magnitude higher than EP
        'EP': 6.82e-5/0.212,
        'FETox': 7.45e-4/0.212
    }

    return {
        'AP': mass_in_kg * fairphone_recylcing_impact['AP'],
        'EP': mass_in_kg * fairphone_recylcing_impact['EP'],
        'FETox': mass_in_kg * fairphone_recylcing_impact['FETox']
    }

def midpoint_to_endpoint(midpoint_impact):
    """
    Convert midpoint impact to endpoint impact.
    
    Args:
        midpoint_impact (dict): A dictionary containing midpoint impacts. Expected keys are:
            - 'AP': Acidification Potential
            - 'EP': Eutrophication Potential
            - 'FETox': Freshwater Ecotoxicity Potential
    
    Returns:
        endpoint_impact (dict): A dictionary containing endpoint impacts.
    """
    # Conversion factors from midpoint to endpoint - Source: ReCiPe 2016
    conversion_factors = {
        'AP': 2.12e-7,
        'EP': 6.1e-7 * PO4_TO_P_CONVERSION_FACTOR,
        'FETox': 6.95e-10 / TO_CTUe_CONVERSION_FACTOR_FRESHWATER # Convert CTUe to 1,4-DCB eq
    }
    
    endpoint_impact = {key+'_to_endpoint': value * conversion_factors[key] for key, value in midpoint_impact.items()}
    
    return endpoint_impact

def calculate_total_impact(device_specs, occupy_ratio=1.0):
    """
    Calculate the total impact of a device based on its specifications.
    
    Args:
        device_specs (dict): A dictionary containing device specifications. Expected keys are:
            - 'component_type': Type of component ('CPU', 'GPU', 'SSD', 'HDD', 'DRAM')
            - 'mass': Mass of the component in grams 
            - 'distance': dict with transport distances for different modes (Truck, Ship, Air)
            - 'production_year': Year of production (2016-2023)
            - 'recycling_rate': Recycling rate (default 0.9)
            - 'inceneration_rate': Inceneration rate (default 0.08)
            - 'landfill_rate': Landfill rate (default 0.02)
    
    Returns:
        impact (dict): A dictionary containing the calculated impacts breakdown.
    """
    # Calculate manufacturing impacts
    if device_specs['component_type'] == 'CPU':
        manufacturing_impacts = calculate_cpu_manufacturing_impacts(device_specs, prodcution_yield=device_specs.get('production_yield', 0.875))
    elif device_specs['component_type'] == 'GPU':
        manufacturing_impacts = calculate_gpu_manufacturing_impacts(device_specs, memory_production_yield=device_specs.get('memory_production_yield', 0.875))
    elif device_specs['component_type'] == 'SSD':
        manufacturing_impacts = calculate_storage_manufacturing_impacts('SSD', device_specs['production_year'], device_specs['capacity'], production_yield=device_specs.get('production_yield', 0.875))
    elif device_specs['component_type'] == 'HDD':
        manufacturing_impacts = calculate_storage_manufacturing_impacts('HDD', device_specs['production_year'], device_specs['capacity'], production_yield=device_specs.get('production_yield', 0.875))
    elif device_specs['component_type'] == 'DRAM':
        manufacturing_impacts = calculate_storage_manufacturing_impacts('DRAM', device_specs['production_year'], device_specs['capacity'], production_yield=device_specs.get('production_yield', 0.875))
    else:
        raise ValueError("Invalid component type. Must be one of ['CPU', 'GPU', 'SSD', 'HDD', 'DRAM'].")
    
    # Calculate mass for transportation and recycling
    if device_specs['component_type'] in ['DRAM', 'SSD', 'HDD']:
        device_specs['mass'] = df_storage_weight_density[f"{device_specs['component_type']}_g/GB"].loc[str(device_specs['production_year'])] * device_specs['capacity']
        device_specs['packaging_mass'] = DRAM_SSD_CPU_PACKAGING_WEIGHT_FACTOR * device_specs['mass']
    else:
        if device_specs['component_type'] == 'GPU':
            device_specs['packaging_mass'] = GPU_PACKAGING_WEIGHT_FACTOR * device_specs['mass']
        else: # CPU
            device_specs['packaging_mass'] = device_specs['gross_weight']

    # Calculate transportation and recycling impacts
    transport_impacts = calculate_transport_impact(device_specs['packaging_mass'], device_specs['distance'], mass_unit='g', year=device_specs['production_year']) if device_specs['component_type'] != 'HDD' else {'AP':0.0, 'EP':0.0, 'FETox':0.0}
    recycling_impacts = calculate_recycling_impact(device_specs['packaging_mass'], mass_unit='g', recyling_rate=device_specs.get('recycling_rate', 0.9), inceneration_rate=device_specs.get('inceneration_rate', 0.08), landfill_rate=device_specs.get('landfill_rate', 0.02)) if device_specs['component_type'] != 'HDD' else {'AP':0.0, 'EP':0.0, 'FETox':0.0}

    # Calculate total impacts
    manufacturing_endpoint = midpoint_to_endpoint(manufacturing_impacts)
    transport_endpoint = midpoint_to_endpoint(transport_impacts)
    recycling_endpoint = midpoint_to_endpoint(recycling_impacts)
    total_impacts = {
        'manufacturing': {
            'midpoint': {k: manufacturing_impacts[k]* occupy_ratio for k in manufacturing_impacts.keys()},
            'endpoint': {k: manufacturing_endpoint[k]* occupy_ratio for k in manufacturing_endpoint.keys()}
        },
        'transportation': {
            'midpoint': {k: transport_impacts[k]* occupy_ratio for k in transport_impacts.keys()},
            'endpoint': {k: transport_endpoint[k]* occupy_ratio for k in transport_endpoint.keys()}
        },
        'recycling': {
            'midpoint': {k: recycling_impacts[k]* occupy_ratio for k in recycling_impacts.keys()},
            'endpoint': {k: recycling_endpoint[k]* occupy_ratio for k in recycling_endpoint.keys()}
        },
        'total': {
            'midpoint': {
                'AP': manufacturing_impacts['AP'] + transport_impacts['AP'] + recycling_impacts['AP'],
                'EP': manufacturing_impacts['EP'] + transport_impacts['EP'] + recycling_impacts['EP'],
                'FETox': manufacturing_impacts['FETox'] + transport_impacts['FETox'] + recycling_impacts['FETox']
            }
        }
    }
    
    # Calculate total endpoint impacts
    total_impacts['total']['endpoint'] = midpoint_to_endpoint(total_impacts['total']['midpoint'])
    
    return total_impacts

# Calculate operational impacts for different load scenarios
def calculate_operational_impacts(device_specs, load_ratio, years=5):
    """
    Calculate operational impacts based on TDP and load ratio
    
    Args:
        device_specs: Device specifications including TDP
        load_ratio: Ratio of actual power to TDP
        years: Operating years
    """
    if device_specs['component_type'] not in ['CPU', 'GPU']:
        achieved_power = (load_ratio-0.3)/0.7 * (device_specs['peak_power']-device_specs['idle_power']) + device_specs['idle_power']
        annual_energy = achieved_power * 24 * 365 / 1000
    else:
        tdp = device_specs['TDP']  # in watts
        annual_energy = tdp * load_ratio * 24 * 365 / 1000  # kWh/year
    
    # Initialize impacts dictionary
    impacts = {
        'AP': 0,
        'EP': 0, 
        'FETox': 0
    }
    
    # Calculate year by year impacts
    for year in range(device_specs['production_year'], device_specs['production_year'] + years):
        # Get emission factors for that year (use 2023 if beyond)
        year_idx = min(year, 2023) - 2019  # 2019 is index 0
        sox_ef = unified_emission_factors['SOx']['US'][year_idx]  # g/kWh
        nox_ef = unified_emission_factors['NOx']['US'][year_idx]  # g/kWh
        nh3_ef = unified_emission_factors['NH3']['US'][year_idx]  # g/kWh
        
        # Convert g to kg and calculate impacts for this year
        impacts['AP'] += (sox_ef * impact_factors['AP']['SOx'] + 
                         nox_ef * impact_factors['AP']['NOx'] + 
                         nh3_ef * impact_factors['AP']['NH3']) * annual_energy / 1000
        impacts['EP'] += (nox_ef * impact_factors['EP']['NOx'] + 
                         nh3_ef * impact_factors['EP']['NH3']) * annual_energy / 1000
        impacts['FETox'] += 0  # Assuming negligible direct ecotoxicity impact from electricity use
    
    return impacts