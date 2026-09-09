import pandas as pd 
import os
import datetime
import re
import yaml
import joblib
from collections import Counter
from typing import List

from src import df_ops

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

def expand_TGW_years_scenarios(config):
    """
    Return TGW_years_scenarios as:
        {
            "1990": ["historical"],
            ...
            "2059": ["rcp45hotter"]
        }

    Supports either:
      1. TGW_years_scenarios_ranges
      2. existing TGW_years_scenarios
    """
    if "TGW_years_scenarios_ranges" in config:
        TGW_years_scenarios = {}

        for range_spec in config["TGW_years_scenarios_ranges"]:
            start_year = int(range_spec["start_year"])
            end_year = int(range_spec["end_year"])
            scenarios = range_spec["scenarios"]

            for year in range(start_year, end_year + 1):
                TGW_years_scenarios[str(year)] = scenarios

        return TGW_years_scenarios

    elif "TGW_years_scenarios" in config:
        return config["TGW_years_scenarios"]

    else:
        raise KeyError(
            "Config must contain either 'TGW_years_scenarios_ranges' "
            "or 'TGW_years_scenarios'."
        )
    
    
def build_TGW_scenarios_filename_suffix(TGW_years_scenarios):
    """
    Build a filename suffix describing the years and scenarios in
    TGW_years_scenarios.

    Example:
        {
            "1990": ["historical"],
            ...
            "2019": ["historical"],
            "2030": ["rcp45hotter"],
            ...
            "2059": ["rcp45hotter"],
        }

    returns:
        "_1990_2019_historical_2030_2059_rcp45hotter"

    Contiguous year ranges are represented by their starting and ending years.
    A single year is represented by that year only.
    """

    # Invert year -> scenarios into scenario -> years.
    scenario_to_years = {}

    for year, scenarios in TGW_years_scenarios.items():
        year_int = int(year)

        for scenario in scenarios:
            scenario_to_years.setdefault(scenario, []).append(year_int)

    suffix_parts = []

    # Sort by earliest year for stable suffix ordering.
    for scenario, years in sorted(
        scenario_to_years.items(),
        key=lambda item: min(item[1]),
    ):
        years = sorted(set(years))

        # Detect contiguous year ranges.
        start_year = years[0]
        previous_year = years[0]

        for current_year in years[1:] + [None]:
            if current_year is not None and current_year == previous_year + 1:
                previous_year = current_year
                continue

            # Close the current contiguous range.
            if start_year == previous_year:
                suffix_parts.append(f"{start_year}_{scenario}")
            else:
                suffix_parts.append(
                    f"{start_year}_{previous_year}_{scenario}"
                )

            # Start the next range, when present.
            if current_year is not None:
                start_year = current_year
                previous_year = current_year

    return f"{'_'.join(suffix_parts)}" if suffix_parts else ""
    


def build_regional_demand_weather_filename(TGW_years_scenarios):
    """
    Build dynamic output filename based on TGW_years_scenarios.

    Example:
        {
            "1990": ["historical"],
            ...
            "2019": ["historical"],
            "2030": ["rcp45hotter"],
            ...
            "2059": ["rcp45hotter"]
        }

    becomes:
        regional_demand_weather_all_cities_1990_2019_historical_2030_2059_rcp45hotter.joblib
    """

    # Invert year -> scenarios into scenario -> years
    scenario_to_years = {}

    for year, scenarios in TGW_years_scenarios.items():
        year_int = int(year)

        for scenario in scenarios:
            if scenario not in scenario_to_years:
                scenario_to_years[scenario] = []

            scenario_to_years[scenario].append(year_int)

    filename_parts = ["regional_demand_weather_all_cities"]

    # Sort by earliest year for stable filename ordering
    for scenario, years in sorted(
        scenario_to_years.items(),
        key=lambda item: min(item[1])
    ):
        years = sorted(years)

        # Detect contiguous year ranges
        start_year = years[0]
        previous_year = years[0]

        for current_year in years[1:] + [None]:
            if current_year is not None and current_year == previous_year + 1:
                previous_year = current_year
            else:
                # Close current contiguous range
                end_year = previous_year

                if start_year == end_year:
                    filename_parts.append(f"{start_year}_{scenario}")
                else:
                    filename_parts.append(f"{start_year}_{end_year}_{scenario}")

                # Start next range if there is one
                if current_year is not None:
                    start_year = current_year
                    previous_year = current_year

    filename = "_".join(filename_parts) + ".joblib"

    return filename    



def load_and_sort_regional_demand(config):
    """
    Load dictionary with regional/city aggregated demand/weather/ampacity data,
    sort by aggregated total buildings demand, and return the sorted dictionary.
    """

    smart_ds_year = config["smart_ds_years"][0]
    smart_ds_load_path = config["smart_ds_load_path"] + f"/{smart_ds_year}"
    
    TGW_years_scenarios = config['TGW_years_scenarios']
    regional_demand_weather_filename = build_regional_demand_weather_filename(TGW_years_scenarios)
    city = "all_cities"
    regional_demand_weather_path = (
        smart_ds_load_path
        + f"/{city}/aggregated_demand/"
        + regional_demand_weather_filename
    )

    regional_demand_weather_ampacity_all_cities = joblib.load(regional_demand_weather_path)

    # Sort dictionary by aggregated total demand
    regional_demand_weather_ampacity_all_cities_sorted = df_ops.sort_nested_dict_dfs(regional_demand_weather_ampacity_all_cities, "aggregated_predicted_buildings_total_kw", ascending=False)

    return regional_demand_weather_ampacity_all_cities_sorted


def load_city_weather_inputs(
    config,
    city,
    TGW_scenario,
    TGW_weather_year,
    regional_demand_weather_ampacity_all_cities_sorted,
    smart_ds_year,
    near_worst_stat,
    top_n_hours,
):
    """
    Load city-level TGW weather data, near-worst historical temperature,
    and create the list of top demand mdh values for the city.
    """

    # Load TGW weather data for TGW location (city)
    TGW_location = {
        "GSO": "Greensboro",
        "AUS": "Austin",
        "SFO": "SanFrancisco",
    }.get(city, city)

    TGW_weather_df_save_path = (
        f"{config['input_data_prediction_path']}/"
        f"{TGW_location}/{TGW_scenario}/"
    )

    TGW_weather_df = joblib.load(
        os.path.join(
            TGW_weather_df_save_path,
            f"TGW_weather_{TGW_weather_year}.joblib",
        )
    )

    # Load near-worst historical temperature for TGW location (city)
    # to set default ambient temp of power lines ampacity
    TGW_stats_dir = (
        "main_folder/TGW/"
    )

    loaded_temp_stats = joblib.load(
        os.path.join(TGW_stats_dir, "temperature_stats.joblib")
    )

    Ta_near_worst = loaded_temp_stats[TGW_location].loc[
        (
            loaded_temp_stats[TGW_location]["scenario"] == "historical"
        )
        & (
            loaded_temp_stats[TGW_location]["year"] == int(smart_ds_year)
        ),
        near_worst_stat,
    ].values[0]

    # Create list of mdh for top % hours
    df_city = regional_demand_weather_ampacity_all_cities_sorted[
        (TGW_weather_year, TGW_scenario)
    ][city]

    list_of_mdh = df_ops.get_top_n_mdh(
        df_city,
        top_n_hours,
        config["start_month_mdh"],
        config["end_month_mdh"],
    )

    return {
        "TGW_location": TGW_location,
        "TGW_weather_df": TGW_weather_df,
        "Ta_near_worst": Ta_near_worst,
        "list_of_mdh": list_of_mdh,
    }
    

def build_solar_battery_scenario_folder(solar_share, battery_share):
    """
    Build SMART-DS scenario folder name from solar and battery share parameters.

    Examples:
        solar_share='none', battery_share='none'
            -> 'base_timeseries'

        solar_share='high', battery_share='low'
            -> 'solar_high_batteries_low_timeseries'
    """

    valid_solar_shares = {"none", "low", "medium", "high", "extreme"}
    valid_battery_shares = {"none", "low", "high"}

    if solar_share not in valid_solar_shares:
        raise ValueError(
            f"solar_share must be one of {valid_solar_shares}, got {solar_share!r}"
        )

    if battery_share not in valid_battery_shares:
        raise ValueError(
            f"battery_share must be one of {valid_battery_shares}, got {battery_share!r}"
        )

    if solar_share == "none" and battery_share == "none":
        return "base_timeseries"

    return f"solar_{solar_share}_batteries_{battery_share}_timeseries"

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
    
    
