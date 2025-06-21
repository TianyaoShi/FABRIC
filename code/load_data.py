import json
import os
import pandas as pd
import numpy as np

DATA_DIR = "../data"

def load_json_file(file_path):
    """
    Load a JSON file and return its content.
    
    :param file_path: Path to the JSON file.
    :return: Parsed JSON content as a dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return data

cf_data = load_json_file(os.path.join(DATA_DIR, "characterization_factors.json"))

impact_factors = cf_data['impact_factors']['manufacturing']
TO_CTUe_CONVERSION_FACTOR_FRESHWATER = 9.84e2 # 1,4‑DCB emitted to continental freshwater 
TO_CTUe_CONVERSION_FACTOR_MARINE = 8.02e-1 # 1,4‑DCB emitted to marine water
unified_emission_factors = cf_data['impact_factors']['emission_factors']
cf_transportation = cf_data['impact_factors']['transportation']

me_data = load_json_file(os.path.join(DATA_DIR, "manufacturing_emissions.json"))

tsmc_acid_emissions_raw_pixels = me_data['tsmc_acid_emissions_raw_pixels']
tsmc_acid_emission_mix_ratio = {year:{} for year in tsmc_acid_emissions_raw_pixels.keys()}
for year in tsmc_acid_emissions_raw_pixels.keys():
    total_acid = tsmc_acid_emissions_raw_pixels[year]['total']
    for acid in tsmc_acid_emissions_raw_pixels[year].keys():
        if acid != 'total':
            tsmc_acid_emission_mix_ratio[year][acid] = tsmc_acid_emissions_raw_pixels[year][acid] / total_acid

tsmc_emissions_macro_16thro23 = me_data['tsmc_emissions_macro_16thro23']
tsmc_electricity_consumption = me_data['tsmc_electricity_consumption']

hynix_emissions_macro_16thro23 = me_data['hynix_emissions_macro_16thro23']
hynix_production_data = me_data['hynix_production_data']

# Create dictionaries to store the allocated emissions
dram_emissions = {}
nand_emissions = {}

# List of emission types to process
emission_types = ['SOx_mt', 'NOx_mt', 'VOC_mt', 'HF_mt', 'HCl_mt', 'NH3_mt', 'total_wastewater_nitrogen_mt', 
                 'total_wastewater_phosphorus_mt', 'wastewater_fluoride_mt', 'ammonia_nitrogen_mt']

# Calculate allocated emissions for each year and type
for year_idx, year in enumerate(range(2016, 2024)):
    dram_emissions[year] = {}
    nand_emissions[year] = {}
    
    dram_ratio = hynix_production_data['dram_revenue_ratio'][year_idx]
    nand_ratio = hynix_production_data['nand_revenue_ratio'][year_idx]
    
    dram_wafers = hynix_production_data['estimated_dram_k_wafers'][year_idx] * 1000  # Convert to wafers
    nand_wafers = hynix_production_data['nand_k_wafers'][year_idx] * 1000  # Convert to wafers
    
    for emission_type in emission_types:
        total_emission = hynix_emissions_macro_16thro23[emission_type][year_idx]
        
        # Calculate allocated emissions (convert mt to kg)
        dram_allocated = (total_emission * dram_ratio * 1000) / dram_wafers
        nand_allocated = (total_emission * nand_ratio * 1000) / nand_wafers
        
        dram_emissions[year][emission_type] = dram_allocated
        nand_emissions[year][emission_type] = nand_allocated

# Create DataFrames for better visualization

# Create DRAM DataFrame
dram_df = pd.DataFrame({
    '_'.join(emission_type.split('_')[:-1]): [dram_emissions[year][emission_type] 
                                 for year in range(2016, 2024)] 
    for emission_type in emission_types
}, index=range(2016, 2024))

# Create NAND DataFrame
nand_df = pd.DataFrame({
    '_'.join(emission_type.split('_')[:-1]): [nand_emissions[year][emission_type] 
                                 for year in range(2016, 2024)] 
    for emission_type in emission_types
}, index=range(2016, 2024))

dram_bit_density_by_year = me_data['dram_bit_density_by_year']
vram_bit_density = me_data['vram_bit_density']
ssd_bit_density_by_year = me_data['ssd_bit_density_by_year']

for idx, year in enumerate(range(2016, 2024)):
    # For DRAM
    dram_ratio = hynix_production_data['dram_revenue_ratio'][idx]
    dram_wafers = hynix_production_data['estimated_dram_k_wafers'][idx] * 1000
    wastewater_per_wafer = hynix_emissions_macro_16thro23['wastewater_discharge_1000_m3'][idx] * dram_ratio * 1e6 / dram_wafers  # L/wafer
    dram_emissions[year]['Cu2+'] = wastewater_per_wafer * tsmc_emissions_macro_16thro23['Cu2+_ppm'][idx] * 1e-6  # kg/wafer
    
    # For NAND
    nand_ratio = hynix_production_data['nand_revenue_ratio'][idx]
    nand_wafers = hynix_production_data['nand_k_wafers'][idx] * 1000
    wastewater_per_wafer = hynix_emissions_macro_16thro23['wastewater_discharge_1000_m3'][idx] * nand_ratio * 1e6 / nand_wafers  # L/wafer
    nand_emissions[year]['Cu2+'] = wastewater_per_wafer * tsmc_emissions_macro_16thro23['Cu2+_ppm'][idx] * 1e-6  # kg/wafer

seagate_hdd_lca_results = me_data['seagate_hdd_lca_results']
for year in seagate_hdd_lca_results.keys():
    for key in seagate_hdd_lca_results[year].keys():
        if seagate_hdd_lca_results[year][key] is None:
            seagate_hdd_lca_results[year][key] = np.nan

hdd_carbon_footprint_records = me_data['hdd_carbon_footprint_records']

ds_data = load_json_file(os.path.join(DATA_DIR, "device_specs.json"))
epyc_7b12_specs = ds_data['CPUs']['EPYC 7B12']
epyc_7443_specs = ds_data['CPUs']['EPYC 7443']
epyc_7b13_specs = ds_data['CPUs']['EPYC 7B13']
epyc_9b14_specs = ds_data['CPUs']['EPYC 9B14']

t4_specs = ds_data['GPUs']['T4']
v100_specs = ds_data['GPUs']['V100']
l40_specs = ds_data['GPUs']['L40']
a100_40g_specs = ds_data['GPUs']['A100']
h100_specs = ds_data['GPUs']['H100']

df_storage_weight_density = pd.DataFrame.from_dict(ds_data['storage_weight_density'])