import pandas as pd # to create data frames
import os
import datetime
import re
import yaml
from collections import Counter
from typing import List

# Load YAML configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Adjust building types based on aggregation level
    if config["aggregation_level"] == "building":
        config["building_types"] = ["_"]  # At building scale, res/com distinction isn't needed
        
    # Format paths dynamically based on chosen parameters
    config["input_data_training_path"] = f"main_folder/load_prediction/results/data/training/{config['smart_ds_years'][0]}/months_{config['start_month']}_{config['end_month']}/ml_input_data/resstock/amy2018/{config['aggregation_level']}/"
    
    config["output_data_training_path"] = f"main_folder/load_prediction/results/data/training/{config['smart_ds_years'][0]}/months_{config['start_month']}_{config['end_month']}/ml_output_data/{config['Y_column']}/{config['X_columns_set']}/{config['aggregation_level']}/"
    
    config["input_data_prediction_path"] = f"main_folder/load_prediction/results/data/prediction/input/TGW_weather"
           
    config["output_data_prediction_path"] = f"main_folder/load_prediction/results/data/prediction/output/{config['smart_ds_years'][0]}/months_{config['start_month']}_{config['end_month']}/{config['Y_column']}/{config['X_columns_set']}/{config['aggregation_level']}/" 

    config["output_pf_path"] = f"main_folder/OpenDSS/results/data/network_performance/{config['demand_mode']}/{config['solution_mode']}/{config['smart_ds_years'][0]}/" # /city/region/climate_scenario/year/time_mode(e.g., regional_peak,99p,100p)
    
    return config

def add_feeder_upper_folder(s):
    match = re.match(r"(.*?--)", s)
    if match:
        prefix = match.group(1)[:-2]  # Remove the '--' from the match
        return f"{prefix}/{s}"
    return s  # Return as is if no match

# convert building id name from Load.dss to match building_id name in parquet files
def convert_yearly_expression(expression):
    """
    Converts expressions like 'res_kw_366_pu' to 'res_366' 
    and 'com_kw_14536_pu' to 'com_14536'.
    """
    match = re.match(r"(\w+)_kw_(\d+)_pu", expression)
    if match:
        prefix, number = match.groups()
        return f"{prefix}_{number}"
    else:
        return expression  # Return as-is if format doesn't match

# Count number of times each Res/Com Stock building type exists in a Load.dss file 
# Note that in Load.dss building loads are sometimes split per # of phases. Here we count each phase seperatly, e.g., if there's a single building with type res_1 but it's split to 2 phases than it will be counted twice. 
# Return a data frame with columns "building_id" (e.g., res_366) and column "count" with number of times it exists in Load.dss
def count_building_id_occurrences(load_dss_file_paths: List[str]):
    """
    Counts occurrences of building IDs across multiple OpenDSS Load.dss files.

    This function reads a list of Load.dss file paths, extracts the 'yearly' load profile  
    expressions, 
    converts them into standardized building ID format using `convert_yearly_expression`, and
    returns 
    a DataFrame summarizing the frequency of each unique building ID across all files.

    Inputs:
    -------
    load_dss_file_paths : List[str]
        A list of file paths to Load.dss files. Each file is expected to contain lines with 
        'yearly=' expressions indicating building load Res/Com stock profiles.

    Outputs:
    --------
    df_building_id_count : pd.DataFrame
        A DataFrame with two columns:
        - 'building_id': standardized building ID extracted from 'yearly' expressions
        - 'count': the number of times each building ID appears across all files

    Example:
    --------
    >>> count_building_id_occurrences(['path/to/Load1.dss', 'path/to/Load2.dss'])
        building_id     count
        com_12345          17
        res_67890          13
        ...
    """
    total_building_ids = []
    for folder_path in load_dss_file_paths:
        file_path = folder_path + "/Loads.dss"
        with open(file_path, 'r') as file:
            content = file.read()
            raw_building_ids = re.findall(r'yearly=([^\s]+)', content)
            converted_building_ids = [convert_yearly_expression(expr) for expr in raw_building_ids]
            total_building_ids.extend(converted_building_ids)

    # Count all occurrences across all files
    building_id_counter = Counter(total_building_ids)

    # Create and return DataFrame
    df_building_id_count = pd.DataFrame(building_id_counter.items(), columns=['building_id', 'count'])
    df_building_id_count = df_building_id_count.sort_values(by='count', ascending=False).reset_index(drop=True)

    return df_building_id_count

# Function to get city and region names given index numbers
# inputs: CITY_REGIONS - dictionary with names of cities and regionsf 
# output:
def get_city_and_region(CITY_REGIONS, city_num, region_num):
    if city_num not in range(1, len(CITY_REGIONS) + 1):
        raise ValueError("Chosen city number is invalid")

    city = list(CITY_REGIONS.keys())[city_num - 1]
    regions = CITY_REGIONS[city]

    if region_num not in range(1, len(regions) + 1):
        raise ValueError("Chosen region number is invalid")

    return city, regions[region_num - 1]


def compute_peak_hour_and_second(city, region, year):
    """
    Computes the hour and second that correlate with the annual peak day, hour, and minute data in SMART-DS analysis folder summary statistics,
    as well as the time step corresponding to the peak day, hour, and minute (assuming 15-minute resolution).

    Args:
        city (str): City name.
        region (str): Region name.
        year (str): Year.

    Returns:
        tuple: The computed hour, second, the day, hour, minute of annual peak, and the corresponding peak time step.
    """
    file_path = f'main_folder/SMART-DS/v1.0/{year}/{city}/{region}/scenarios/base_timeseries/opendss/analysis/Summary_data.csv'

    try:
        # Read the CSV file
        df = pd.read_csv(file_path, header=None)

        # Extract day, hour, and minute from the 2nd row (index 1)
        peak_day = int(df.iloc[1, 6])  # 7th column (index 6)
        peak_hour = int(df.iloc[1, 7])  # 8th column (index 7)
        peak_minute = int(df.iloc[1, 8])  # 9th column (index 8)

        # Compute hour and second
        hours_to_peak = (peak_day - 1) * 24 + peak_hour
        seconds_to_peak = peak_minute * 60

        # Compute the time step (15-minute resolution)
        peak_time_step = (peak_day - 1) * 24 * 4 + peak_hour * 4 + peak_minute // 15

        return hours_to_peak, seconds_to_peak, peak_day, peak_hour, peak_minute, peak_time_step

    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
    except ValueError:
        print("Error: Invalid data format in the specified file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
        
def create_building_multiplier_dict(path_to_profiles, peak_time_step):
    """
    Creates a dictionary with building names as keys and their kW and kVar multipliers as values
    for a specific time step.

    Args:
        path_to_profiles (str): Path to the folder containing the CSV files.
        peak_time_step (int): Row number (time step) to retrieve values from.

    Returns:
        dict: Dictionary with building names as keys and a tuple (kW, kVar) as values.
    """
    # Initialize the dictionary to store building multipliers
    building_multipliers = {}

    # Regex pattern to extract building name and type (com or res)
    pattern = r'^(com|res)_\w+_(\d+)_pu\.csv$'

    # Iterate through all files in the folder
    for file_name in os.listdir(path_to_profiles):
        match = re.match(pattern, file_name)
        if match:
            # Extract building type (com or res) and ID
            building_type = match.group(1)
            building_id = match.group(2)

            # Determine whether the file is for kW or kVar
            if "kw" in file_name:
                metric = "kw"
            elif "kvar" in file_name:
                metric = "kvar"
            else:
                continue

            # Build the full path to the file
            file_path = os.path.join(path_to_profiles, file_name)

            # Read the specific row for the given time step
            try:
                df = pd.read_csv(file_path, header=None)
                value = df.iloc[peak_time_step - 1, 0]  # Subtract 1 because row index starts at 0
            except (IndexError, FileNotFoundError, pd.errors.EmptyDataError):
                print(f"Error reading {file_path} at time step {peak_time_step}")
                continue

            # Construct the building name
            building_name = f"{building_type}_{building_id}"

            # Add or update the dictionary entry for the building
            if building_name not in building_multipliers:
                building_multipliers[building_name] = {"kw": None, "kvar": None}
            building_multipliers[building_name][metric] = value

    # Convert the dictionary values from dict to tuple (kw, kvar)
    return {k: (v["kw"], v["kvar"]) for k, v in building_multipliers.items()}


def modify_building_multiplier_dict(building_multipliers, T1, T2, GP):
    """
    Modifies the kW and kVar values in the building multipliers dictionary based on temperature change and growth percentage.

    Args:
        building_multipliers (dict): Dictionary with building names as keys and (kW, kVar) tuples as values.
        T1 (float): Initial temperature.
        T2 (float): Final temperature.
        GP (float): Growth percentage.

    Returns:
        dict: Updated dictionary with modified kW and kVar values.
    """
    factor = 1 + ((T2 - T1) * GP) / 100
    updated_multipliers = {}
    for building, (kw, kvar) in building_multipliers.items():
        updated_kw = kw * factor if kw is not None else None
        updated_kvar = kvar * factor if kvar is not None else None
        updated_multipliers[building] = (updated_kw, updated_kvar)
    return updated_multipliers  

def extract_temperature_from_csv(csv_path, month, day, hour, minute):
    """
    Extracts the temperature from the 10th column of a CSV file for a given time (month, day, hour, minute).

    Args:
        csv_path (str): Path to the CSV file.
        month (int): Month value.
        day (int): Day value.
        hour (int): Hour value.
        minute (int): Minute value.

    Returns:
        float: Temperature value from the 10th column.
    """
    try:
        df = pd.read_csv(csv_path, header=None)
        match = df[(df.iloc[:, 1] == month) & (df.iloc[:, 2] == day) & (df.iloc[:, 3] == hour) & (df.iloc[:, 4] == minute)]
        if not match.empty:
            return match.iloc[0, 8]
        else:
            print("No matching time entry found.")
            return None
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} does not exist.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
    
