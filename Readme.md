Load Forecasting, Thermal Modeling, and Power Flow Analysis Pipeline

Overview

This repository contains the complete computational workflow used to preprocess weather and load data, train machine learning load forecasting models, generate scenario-based predictions, construct weather-dependent thermal model inputs, and perform downstream power flow analyses.

The pipeline is implemented in Python using Jupyter notebooks, with supporting configuration files and custom source packages developed specifically for this project.

The code is provided to support transparency and reproducibility of the analyses.

Repository Structure
.
├── machine_learning_demand_forecasting/
│   ├── config/
│   │   └── *.yaml
│   ├── src/
│   │   └── *.py
│   ├── 1. DP Prepare input data for training (resstock load and weather).ipynb
│   ├── 2. DP Prepare input data for prediction- TGW weather.ipynb
│   ├── 3. DP Create resstock dataframes.ipynb
│   ├── 4. DP Create cooling heating and non-CH data for dissaggregation process.ipynb
│   ├── 5. TRAIN load prediction model.ipynb
│   ├── 6. PREDICT load with TGW.ipynb
│   └── 7. POST DP Disaggregate feeder to buildings.ipynb
│
├── thermal_models_and_power_flow/
│   ├── config/
│   │   └── *.yaml
│   ├── src/
│   │   └── *.py
│   ├── 1. DP2 create weather-dependent smartds files (multiple snapshot).ipynb
│   ├── 2. PF5 multiple snapshot PF - derating and MLP demand growth.ipynb
│   └── 3. PP create pf summary statistics across timesteps.ipynb
│
├── data/                # Not included; see Data Availability
└── README.md

High-Level Workflow

The analysis is organized into two main modules:

1. Machine Learning Demand Forecasting

2. Thermal Modeling and Power Flow Analysis

Each module contains:

1. A config/ directory with YAML files specifying user inputs and scenario parameters.

2. A src/ directory with project-specific Python custom packages and helper functions.

3. Jupyter notebooks that orchestrate each stage of the workflow.

Module 1: Machine Learning Demand Forecasting

machine_learning_demand_forecasting/

This module handles data preparation, feature engineering, model training, and load prediction.

Workflow Steps

Data Preparation (Training)

1. DP Prepare input data for training (resstock load and weather).ipynb
Loads and preprocesses ResStock load data and historical weather data for model training.

Data Preparation 

2. DP Prepare input data for prediction- TGW weather.ipynb
Prepares weather inputs for TGW (climate dataset) prediction runs.

Dataset Construction

3. DP Create resstock dataframes.ipynb
Constructs structured dataframes from ResStock inputs.

4. DP Create cooling heating and non-CH data for dissaggregation process.ipynb
Creates data stuctures of cooling, heating, and non-cooling/heating components for model trainig.

Model Training

5. TRAIN load prediction model.ipynb
Trains the machine learning load prediction model(s).

Prediction

6. PREDICT load with TGW.ipynb
Generates load predictions under TGW weather scenarios.

Post-processing and Disaggregation

7. POST DP Disaggregate feeder to buildings.ipynb
Disaggregates feeder-level loads to individual buildings.

Configuration

All user inputs (e.g., file paths, scenario parameters, model settings) are specified in:

machine_learning_demand_forecasting/config/*.yaml

Custom Source Code

Reusable helper functions and project-specific logic are implemented in:

machine_learning_demand_forecasting/src/

Module 2: Thermal Models and Power Flow

thermal_models_and_power_flow/

This module constructs weather-dependent thermal model inputs, runs multi-snapshot power flow simulations, and post-processes results.

Workflow Steps

Thermal Model Input Generation

1. DP2 create weather-dependent smartds files (multiple snapshot).ipynb
Creates weather-dependent SmartDS input files across multiple snapshots.

Power Flow Simulation

2. PF5 multiple snapshot PF - derating and MLP demand growth.ipynb
Executes multiple snapshot power flow analyses, considering equipment derating, electrical losses and demand growth.

Post-processing

3. PP create pf summary statistics across timesteps.ipynb
Aggregates and summarizes power flow results across time steps.

Configuration

All user inputs and scenario settings are specified in:

thermal_models_and_power_flow/config/*.yaml

Custom Source Code

Project-specific helper functions and utilities are implemented in:

thermal_models_and_power_flow/src/

Requirements:

Python ≥ 3.9

Core packages (not exhaustive):

numpy

pandas

scikit-learn

opendssdirect

matplotlib

jupyter

pyyaml

Usage

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git


Set up the Python environment (see Requirements).

Populate the required input paths in the relevant config/*.yaml files.

Execute the notebooks in each module in the order listed above.

Reproducibility Notes

All model hyperparameters and scenario assumptions are explicitly defined in the YAML configuration files.


License

This repository is released under the MIT License, which permits reuse, modification, and distribution of the code for both academic and commercial purposes, provided that the original copyright notice and license terms are included. See the LICENSE file for full details.

