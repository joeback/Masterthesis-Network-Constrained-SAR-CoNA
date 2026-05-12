# Master's Thesis Code

This repository contains the code accompanying the Master's thesis by Joe Dylan Back in Statistics:

**A Simultaneous Autoregressive Model for Composition-Valued Trajectory Data on Network Structures with Application to the Berlin Public Transport System**

The project analyses public transport delay patterns in the Berlin bus network using GTFS structure data, realtime VBB delay data, graph-based representations, compositional data transformations, and spatial autoregressive modelling.

## Repository Structure

```text
Masterthesis_Hand_In_Joe_Back/
├── Code/
│   ├── thesis_EDA.ipynb
│   ├── Models.ipynb
│   ├── Simulations.ipynb
│   ├── thesis_core.py
│   └── Data_Processing/
│       └── Workfiles/
│           ├── preprocessing_busdata.py
│           ├── Data_Collecting.py
│           └── Data_Collecting_New.py
└── Data/
    ├── data_raw/
    ├── GTFS_Structure_Data/
    └── logs/
```

## Main Files

Note that all graphics included in the text can be found and reproduced in the following notebooks. They are separated by topic.

- `Code/thesis_EDA.ipynb`: exploratory analysis of GTFS and realtime delay data in the compositional space
- `Code/Models.ipynb`: main modeling notebook, with applications and results from the SAR modeling
- `Code/Simulations.ipynb`: simulation studies to observe the robustness of the approach
- `Code/thesis_core.py`: core functions for loading GTFS data, graph construction, SAR modelling, and simulations
- `Code/Data_Processing/Workfiles/preprocessing_busdata.py`: preprocessing functions for realtime and compositional bus delay data

## Data

The data are not included in this GitHub repository due to file size.
The code expects the following local data structure:

```text
Data/
├── data_raw/
│   └── vbb_realtime_delays_buses_*.csv
├── GTFS_Structure_Data/
│   └── GTFS_160126/
│       ├── stops.txt
│       ├── trips.txt
│       ├── stop_times.txt
│       └── routes.txt
└── logs/
```

The data is submitted separately via WeTransfer.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

## Running the Code

Start Jupyter Notebook or JupyterLab from the project root:

```bash
jupyter notebook
```

Recommended order:

1. `Code/thesis_EDA.ipynb`
2. `Code/Models.ipynb`
3. `Code/Simulations.ipynb`

## Notes

The paths in `thesis_core.py` are defined relative to the project root. Therefore, the project folder can be moved to another machine as long as the `Code/` and `Data/` folders keep the same structure.

## Author

Joe Dylan Back 639075
M.Sc. Statistics
Humboldt-Universität zu Berlin