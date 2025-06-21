# FABRIC: When Servers Meet Species

This repository contains the code and data for the HotCarbon'25 paper "When Servers Meet Species: A Fab-to-Grave Lens on Computing's Biodiversity Impact" by Tianyao Shi, Ritbik Kumar, Inez Hua, and Yi Ding from Purdue University.

## Overview

FABRIC (Fab-to-grave Analysis of Biodiversity-Related Impact of Computing) is a comprehensive framework for assessing the environmental impacts of computing hardware throughout their entire lifecycle, with a particular focus on biodiversity-related impact categories including:

- **Acidification Potential (AP)** - Impacts on ecosystems from acid rain and soil acidification
- **Eutrophication Potential (EP)** - Nutrient pollution effects on aquatic ecosystems  
- **Freshwater Ecotoxicity (FETox)** - Toxic effects on freshwater species and ecosystems

The framework covers the complete lifecycle from semiconductor fabrication to end-of-life recycling, analyzing CPUs, GPUs, memory (DRAM/HBM), and storage devices (SSDs/HDDs).

## Repository Structure

```
FABRIC/
├── code/
│   ├── load_data.py              # Data loading and preprocessing utilities
│   ├── modeling.py               # Core impact assessment models
│   ├── process_hpc_results.py    # HPC benchmark result processing
│   ├── visualization.ipynb       # Paper visualization reproduction
│   ├── cpu_monitor.py            # AMD CPU power monitoring using MSR registers
│   ├── cpu_monitor_gcp.py        # GCP CPU power estimation using utilization models
│   └── profile_all.sh            # Batch script for running HPC benchmark profiling
├── data/
│   ├── characterization_factors.json    # Environmental impact factors
│   ├── manufacturing_emissions.json     # Semiconductor manufacturing data
│   └── device_specs.json               # Hardware specifications database
├── FABRIC-flowchart.pdf         # Framework diagram
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
└── README.md
```

## Framework Diagram

![FABRIC Framework](FABRIC-flowchart.pdf)

## Reproducibility

All visualizations presented in the paper can be reproduced using the code available in `code/visualization.ipynb`.

The raw profiling results of selected HPC workloads can be accessed at ![Google Drive](https://drive.google.com/file/d/1Fbo7hSWu_e1V3M2UxoZWJCrphymfk_0K/view?usp=sharing).


## Data Sources

The framework integrates data from multiple authoritative sources:

- **TSMC**: Semiconductor manufacturing emissions and resource consumption
- **SK Hynix**: Memory manufacturing environmental data
- **Seagate**: HDD lifecycle assessment reports
- **Ecoinvent**: Transportation and recycling impact factors
- **EPA eGRID**: Regional electricity emission factors
- **ReCiPe 2016**: Environmental impact characterization factors

## Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{shi2025fabric,
  title={When Servers Meet Species: A Fab-to-Grave Lens on Computing's Biodiversity Impact},
  author={Shi, Tianyao and Kumar, Ritbik and Hua, Inez and Ding, Yi},
  booktitle={Proceedings of the 6th Workshop on
Sustainable Computer Systems (HotCarbon'25)},
  year={2025},
  organization={ACM}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

