import numpy as np
import pandas as pd
import time
import os
import re
import glob
import yaml
import pytz
from collections import Counter
from typing import List
from timezonefinder import TimezoneFinder

# Load YAML configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Adjust building types based on aggregation level
    if config["aggregation_level"] == "building":
        config["building_types"] = ["_"]  # At building scale, res/com distinction isn't needed
        
    config['X_columns'] = config[config['X_columns_set']]

    # Format paths dynamically based on chosen parameters
    config["input_data_training_path"] = f"main_folder/load_prediction/results/data/training/{config['smart_ds_years'][0]}/months_{config['start_month']}_{config['end_month']}/ml_input_data/resstock/amy2018/{config['aggregation_level']}/"
    
    config["output_data_training_path"] = f"main_folder/load_prediction/results/data/training/{config['smart_ds_years'][0]}/months_{config['start_month']}_{config['end_month']}/ml_output_data/{config['Y_column']}/{config['X_columns_set']}/{config['aggregation_level']}/"
    
    config["input_data_prediction_path"] = f"main_folder/load_prediction/results/data/prediction/input/TGW_weather"
           
    config["output_data_prediction_path"] = f"main_folder/load_prediction/results/data/prediction/output/{config['smart_ds_years'][0]}/months_{config['start_month']}_{config['end_month']}/{config['Y_column']}/{config['X_columns_set']}/{config['aggregation_level']}/" 

   
    # Check if directoris for X_columns_set exist. If not,create them. 
    output_data_base_dir = f"main_folder/load_prediction/results/data/training/{config['smart_ds_years'][0]}/months_{config['start_month']}_{config['end_month']}/ml_output_data/{config['Y_column']}/{config['X_columns_set']}/"

    if not os.path.exists(output_data_base_dir):
        required_dirs = ["building", "feeder", "regional"]
        sub_dirs = ["metrics", "models", "scalers"]

        for dir_name in required_dirs:
            main_dir = os.path.join(output_data_base_dir, dir_name)
            os.makedirs(main_dir, exist_ok=True)
            for sub in sub_dirs:
                sub_dir = os.path.join(main_dir, sub)
                os.makedirs(sub_dir, exist_ok=True)

    return config

# Make a list of paths to existing joblib files
def find_results_files(directory, endswith_str, aggregation_level):
    matching_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(endswith_str):
                # Get the relative path and split into parts
                relative_path = os.path.relpath(root, directory)
                path_parts = relative_path.split(os.sep)
                
                # Ensure the grandparent directory is named "regional"
                if len(path_parts) >= 2 and path_parts[-2] == aggregation_level:
                    matching_files.append(os.path.join(root, file))
    
    return matching_files

def find_results_joblib_files_df(directory, N):
    """
    Given a directory, finds all files ending with 'results.joblib' recursively,
    extracts the last N elements of each path, and returns a DataFrame.

    :param directory: str, path to the directory
    :param N: int, number of last elements from the path to include as columns
    :return: pandas DataFrame
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Invalid directory: {directory}")

    data = []
    full_paths = []

    for root, _, files in os.walk(directory):  # Recursively walk through directories
        for file in files:
            if file.endswith("results.joblib"):
                full_path = os.path.join(root, file)
                full_paths.append(full_path)
                path_parts = full_path.split(os.sep)[-N:]  # Get the last N elements
                
                # Ensure all rows have exactly N elements by padding at the start if necessary
                path_parts = [''] * (N - len(path_parts)) + path_parts
                
                data.append(path_parts)

    # Dynamically generate column names based on N
    col_names = [f"{i+1}st element" if i == 0 else f"{i+1}th element" for i in range(N)]

    # Create DataFrame
    df = pd.DataFrame(data, columns=col_names)

    return df, full_paths



def get_parquet_load_data(parquet_folder_path, load_model,start_month,end_month):
    file_prefix = load_model.split('_')[0] 
    parquet_file_path = os.path.join(parquet_folder_path, load_model)
    load_df = pd.read_parquet(parquet_file_path).iloc[::4]  # Select every 4th row
    load_df.rename(columns={load_df.columns[0]: 'date_time'}, inplace=True)
    if load_df['date_time'].dtype == 'object' or pd.api.types.is_string_dtype(load_df['date_time']):
        load_df['date_time'] = load_df['date_time'].str.replace(r':00$', '', regex=True)
        load_df['date_time'] = pd.to_datetime(load_df['date_time'])
    load_df["month"] = load_df['date_time'].dt.month
    load_df = load_df[(load_df['month'] >= start_month) & (load_df['month'] <= end_month)] # Filter data to only include rows within the specified month range
    load_df['date_time'] = load_df['date_time'] - pd.Timedelta(minutes=15)     # Shift load_df timestamps back by 15 minutes (to match weather data)
    return load_df

def find_folders_with_file(base_path, file_name):
    """
    Finds all folders and subfolders under the given path that contain a specific file.

    Parameters:
        base_path (str): The path to the folder to search.
        file_name (str): The name of the file to search for.

    Returns:
        list: A list of folder paths containing the specified file.
    """
    matching_folders = []

    for root, _, files in os.walk(base_path):
        if file_name in files:
            # Normalize the root path to use '/' as the delimiter
            matching_folders.append(root.replace('\\', '/').replace('\\', '/'))

    return matching_folders

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

def aggregate_parquet_data(parquet_data_path, loads_dss_path, file_prefix, aggregation_level, start_month, end_month):
    """
    Reads parquet files from a folder and aggregates them.
    Keeps columns 1, 3, and 4 from the first file read.
    Sums values from columns 2 and 4-31 across all files. Use count of occurrences of building IDs across multiple OpenDSS Load.dss files.
    Reads every 4th row to reduce resolution from 15 min to 1 hour.

    :param parquet_data_path: Path to the folder containing parquet files
    :param file_prefix: Prefix of parquet files to filter
    :return: Aggregated pandas DataFrame
    """
    
    # Create dataframe of the count of each building type in feeder
    df_building_id_count = count_building_id_occurrences(loads_dss_path)

    ### Get all parquet files from load_data folder (all res/com stock profiles in the region) ###
    parquet_files = sorted([f for f in os.listdir(parquet_data_path) if f.startswith(file_prefix) and f.endswith(".parquet")])
    
    if not parquet_files:
        print("No parquet files found in the given folder with the specified prefix.")
        return pd.DataFrame()   # Return an empty DataFrame 
        
    ### If it's feeder level - filter parquet_files list and keep only res/com stock profiles that exist in the feeder   ### 
    if aggregation_level == 'feeder':
        feeder_path = loads_dss_path[0]
        # Extract valid names from Loads.dss
        valid_names = set()
        file_path = feeder_path + "/Loads.dss"
        with open(file_path, "r") as f:
            for line in f:
                match = re.search(r"yearly=(res|com)_kw_(\d+)", line)
                if match:
                    valid_names.add(f"{match.group(1)}_{match.group(2)}")  # e.g., Extract "res_376" or "com_10111"
        # Filter parquet_files based on valid_names
        filtered_files = [f for f in parquet_files if f.replace(".parquet", "") in valid_names]
        parquet_files = filtered_files
        if not parquet_files:
            print(f"Feeder {feeder_path} skipped since no parquet files found in the given folder with the specified prefix.")
            return pd.DataFrame()   # Return an empty DataFrame 
    
    ### Create dataframe of first file in parquet_files, convert 15min resolution to 1h, convert time column name, add month column and filter so start_month - end_month ###
    first_file = os.path.join(parquet_data_path, parquet_files[0])
    df = pd.read_parquet(first_file).iloc[::4]  # Select every 4th row
    df.rename(columns={df.columns[0]: 'date_time'}, inplace=True) # Rename Time column as date_time to match weather file 
    if file_prefix == 'com':
        df['date_time'] = pd.to_datetime(df['date_time'].astype(str).str.replace(r':00$', '', regex=True),errors='coerce')
    else:
        df['date_time'] = pd.to_datetime(df['date_time'])
    df["month"] = df['date_time'].dt.month # Add a month column
    df = df[(df['month'] >= start_month) & (df['month'] <= end_month)] # Filter months
   
    # Keep first, third, and fourth identical to first parquet file (date, building id and power factor), sum the rest (Electricity consumption variables in kw and kvar)
    dont_sum_columns = [df.columns[0], df.columns[2], df.columns[3]]
    sum_columns = [col for col in df.columns if col not in dont_sum_columns]
    df_final = df[dont_sum_columns]
    df_sums = df[sum_columns]
    
    # Aggregate sum_columns (Electricity consumption variables) from parquet_files
    for file in parquet_files[1:]:
        temp_df = pd.read_parquet(os.path.join(parquet_data_path, file)).iloc[::4] # Select every 4th row
        temp_df.rename(columns={temp_df.columns[0]: 'date_time'}, inplace=True) # Rename Time column as date_time to match weather file 
        if file_prefix == 'com': # Convert Time column to date_time object
            temp_df['date_time'] = pd.to_datetime(temp_df['date_time'].astype(str).str.replace(r':00$', '', regex=True),errors='coerce')
        else:
            temp_df['date_time'] = pd.to_datetime(temp_df['date_time'])
        temp_df["month"] = temp_df['date_time'].dt.month # Add a month column
        temp_df = temp_df[(temp_df['month'] >= start_month) & (temp_df['month'] <= end_month)] # Filter months
        temp_numpy_array = temp_df[sum_columns].values  # Convert to NumPy array to avoid index issues
        
        ### Multiply values by the number of times the building type exists in the feeder (use df_building_id_count)
        parquet_building_id = file.split('.')[0] # e.g., extracts "com_12941" from "com_12941.parquet"
        matching_row = df_building_id_count[df_building_id_count['building_id'] == parquet_building_id]
        building_count = int(matching_row['count'].values[0]) if not matching_row.empty else 0
        temp_numpy_array = temp_numpy_array * building_count
        if temp_numpy_array.shape != df_sums.shape:
            raise ValueError(f"Shape mismatch for file {file}: expected {df_sums.shape}, got {temp_numpy_array.shape}")
        df_sums += temp_numpy_array  # Sum the remaining columns
    df_final = pd.concat([df_final, pd.DataFrame(df_sums, columns=sum_columns)], axis=1)  # Combine Electricity consumption columns with non-consumption columns (date, building id and power factor)

    # Shift load_df timestamps back by 15 minutes (to match weather data)
    df_final['date_time'] = df_final['date_time'] - pd.Timedelta(minutes=15)
       
    return df_final



def import_resstock_weather_data(weather_data_path, start_month, end_month):
    """
    Reads csv resstock weather data files from a folder and processes it (shift -1h, converts datetime to numerical values)
    :param start_month: Start month for filtering (inclusive)
    :param end_month: End month for filtering (inclusive)
    :param weather_data_path: Path to the folder containing csv weather file
    :return: processesd weather DataFrame
    """
    ## Import weather data into data frame and preprocess (shift time and convert datatime to numerical values)
    weather_df = pd.read_csv(weather_data_path)

    # shift time -1h to match load data
    weather_df = weather_df.iloc[:-1]  # Remove last row
    weather_df_first_row = weather_df.iloc[0].copy()  # Copy first row
    weather_df_first_row.iloc[0] = pd.to_datetime(weather_df_first_row.iloc[0]) # Convert first column to datetime 
    weather_df_first_row.iloc[0] = weather_df_first_row.iloc[0].replace(hour=0, minute=0, second=0) # Modify the first row's first column value to set the time to '00:00:00'
    weather_df = pd.concat([pd.DataFrame([weather_df_first_row]), weather_df], ignore_index=True)  # Duplicate first row with changes

    # Convert the column to datetime format
    weather_df["date_time"] = pd.to_datetime(weather_df["date_time"])

    # Convert datetime column to useful numerical features
    weather_df["year"] = weather_df["date_time"].dt.year
    weather_df["month"] = weather_df["date_time"].dt.month
    weather_df["day"] = weather_df["date_time"].dt.day
    weather_df["hour"] = weather_df["date_time"].dt.hour
    weather_df["weekday"] = weather_df["date_time"].dt.weekday  # Monday = 0, Sunday = 6
    weather_df["weekend"] = (weather_df["date_time"].dt.weekday >= 5).astype(int)  # 1 if weekend, else 0
    weather_df = weather_df.reset_index(drop=True) # Reset indices before merging with load data frame

    # Put time columns first
    cols_to_move = ['year', 'month', 'day', 'hour', 'weekday', 'weekend'] # Columns to move to the front
    remaining_cols = [col for col in weather_df.columns if col not in cols_to_move] # Get the remaining columns (i.e., all except those in cols_to_move)
    new_order = cols_to_move + remaining_cols # Define new column order
    weather_df = weather_df[new_order] # Reorder DataFrame
    
    # # Filter data by month range ***
    weather_df = weather_df[(weather_df['month'] >= start_month) & (weather_df['month'] <= end_month)]
    
    return weather_df

def import_TGW_weather_data(weather_data_path,TGW_weather_year, TGW_location, start_month, end_month):
    """
    Reads csv TGW weather data files from a folder and processes it (converts index numbers to datetime)
    :param start_month: Start month for filtering (inclusive)
    :param end_month: End month for filtering (inclusive)
    :param weather_data_path: Path to the folder containing csv weather file
    :return: processesd weather DataFrame
    """
    ## Import weather data into data frame and preprocess (shift time and convert datatime to numerical values)
    weather_df = pd.read_csv(weather_data_path)

    ### Rename columns to match training dataframe (with weather data from resstock) ###
    weather_df.rename(columns={weather_df.columns[0]: "Index"}, inplace=True)
    weather_df.rename(columns={weather_df.columns[1]: "Relative Humidity [%]"}, inplace=True)
    weather_df.rename(columns={weather_df.columns[2]: "Dry Bulb Temperature [°C]"}, inplace=True)
    weather_df.rename(columns={weather_df.columns[3]: "Global Horizontal Radiation [W/m2]"}, inplace=True)
    weather_df.rename(columns={weather_df.columns[4]: "Wind Speed [m/s]"}, inplace=True)
    weather_df.rename(columns={weather_df.columns[5]: "Wind Direction [Deg]"}, inplace=True)
    weather_df["Dry Bulb Temperature [°C]"] = weather_df["Dry Bulb Temperature [°C]"] - 273.15 # Convert from Kelvin to Celcuis

    weather_df = weather_df.drop('Index', axis=1) 

    ### Get the time zone for TGW locations coordinates ###
    #  latitude and longitude for TGW cities for time zone correction 
    location_coordinates = {
        "Greensboro": (36.0608, -80.0003),
        "Austin": (30.4196, -97.8095),
        "SanFrancisco": (37.7083, -122.4074),
        "Concord": (37.7083, -122.4074),
    }
    # Define the location (either a city name or coordinates)
    latitude, longitude = location_coordinates[TGW_location]
    # Get the time zone for the given coordinates
    tf = TimezoneFinder(); timezone_str = tf.timezone_at(lng=longitude, lat=latitude)


    # Generate datetime values for the entire year with hourly resolution
    datetime_index = pd.date_range(start=f"{TGW_weather_year}-01-01 00:00:00",periods=len(weather_df),freq='H',tz=timezone_str) 
    # Replace the first column with the generated datetime values
    weather_df["date_time"] = datetime_index
    # Extract UTC offset in hours 
    utc_offset_hours = int(weather_df["date_time"].iloc[0].utcoffset().total_seconds() / 3600)

    # cyclically shift all columns except date_time
    # Make a copy of the DataFrame to avoid modifying the original
    weather_df_shifted = weather_df.copy()
    # Get list of all columns except 'date_time'
    columns_to_shift = [col for col in weather_df.columns if col != "date_time"]

    # Apply cyclic shift to each of those columns
    weather_df_shifted[columns_to_shift] = np.roll(weather_df_shifted[columns_to_shift], utc_offset_hours,axis=0)

    weather_df = weather_df_shifted

    ### Add useful time numerical features ###
    # Convert datetime column to useful numerical features
    weather_df["year"] = weather_df["date_time"].dt.year
    weather_df["month"] = weather_df["date_time"].dt.month
    weather_df["day"] = weather_df["date_time"].dt.day
    weather_df["hour"] = weather_df["date_time"].dt.hour
    weather_df["weekday"] = weather_df["date_time"].dt.weekday  # Monday = 0, Sunday = 6
    weather_df["weekend"] = (weather_df["date_time"].dt.weekday >= 5).astype(int)  # 1 if weekend, else 0
    weather_df = weather_df.reset_index(drop=True) # Reset indices before merging with load data frame
    # Put time columns first
    cols_to_move = ['date_time', 'year', 'month', 'day', 'hour', 'weekday', 'weekend'] # Columns to move to the front
    remaining_cols = [col for col in weather_df.columns if col not in cols_to_move] # Get the remaining columns (i.e., all except those in cols_to_move)
    new_order = cols_to_move + remaining_cols # Define new column order
    weather_df = weather_df[new_order] # Reorder DataFrame
       
    weather_df = weather_df[(weather_df['month'] >= start_month) & (weather_df['month'] <= end_month)] # Filter data to only include rows within the specified month range

    return weather_df


def merge_load_weather(load_df, weather_df):
    """
    Merges load and weather dataframes

    :param load_df: data frame with load data for ML training
    :param weather_df: data frame with weather data for ML training
    :return: processesd merged weather and load DataFrame
    """
    # load_df_trimmed = load_df.iloc[:, 1:] # Drop the first column of load_df [both dataframes have a time column, so one can be removed]
    # input_df = pd.concat([weather_df, load_df_trimmed], axis=1) # Merge dataframes
    # Merge the dataframes using date_time as the key
    # **Set both DataFrames' indices to datetime before merging**
    load_df = load_df.set_index('date_time')
    weather_df = weather_df.set_index('date_time')
      
    #  Use `pd.merge()` instead of `pd.concat()` to align on `date_time`**
    input_df = pd.merge(
        weather_df,
        load_df.iloc[:, 1:],  # Drop the first column (time) as it is now the index
        left_index=True,
        right_index=True,
        how='inner'  # Only keep timestamps that exist in both
    )
    ## Index data frame by date_time
    # print(f'input df is {input_df}')
    # print(f'index of df is {input_df.index}')
    # input_df['date_time'] = pd.to_datetime(input_df['date_time']) #  Convert the 'date_time' column to datetime format
    # input_df = input_df.set_index('date_time') # Set 'date_time' as the DataFrame index
    # input_df.sort_index(inplace=True) # Sort the DataFrame by its index
    return input_df


def make_feeder_list(root_path):
    '''
    Finds all feeder folders in an opendss regional path

    :param root_path: path to opendss regional path
    :return: list of paths to all of the sub-sub-folders within those folders that contain the string "_1247". The paths will only include the last two folders in each path.
    '''
    matching_paths = []
    
    # Iterate through sub-folders in the root directory
    for sub_folder in os.listdir(root_path):
        sub_folder_path = os.path.join(root_path, sub_folder)
        
        # Check if it's a directory and contains "_1247"
        if os.path.isdir(sub_folder_path) and "_1247" in sub_folder:
            # Iterate through sub-sub-folders
            for sub_sub_folder in os.listdir(sub_folder_path):
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_sub_folder)
                
                # Check if it's a directory and contains "_1247"
                if os.path.isdir(sub_sub_folder_path) and "_1247" in sub_sub_folder:
                    # Append only the last two folder names in the path
                    matching_paths.append(f"{sub_sub_folder}")
    
    return matching_paths

def add_feeder_upper_folder(s):
    match = re.match(r"(.*?--)", s)
    if match:
        prefix = match.group(1)[:-2]  # Remove the '--' from the match
        return f"{prefix}/{s}"
    return s  # Return as is if no match



def convert_columns_to_CH_and_non_CH(load_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Gets a data frame with end-use consumption columns and does the following:
    1. sums cooling columns (AC, fans, refrigaration)
    2. removes non cooling or heating columns
    3. creates a non_cooling_and_heating column by substracting cooling and heating from total
    4. removes month and building id columns
    '''
    df = load_df.copy()

    # Add cooling sum kvar
    df['cooling_sum_kvar'] = df['cooling_kvar'] + df['refrigeration_kvar'] + df['fans_kvar']

    # Drop non_cooling_heating_kw columns
    non_cooling_heating_kw_cols = [
        'lighting_kw', 'pumps_kw', 'water_systems_kw', 'motors_kw',
        'plug_loads_kw', 'clothes_dryer_kw', 'clothes_washer_kw',
        'stove_kw', 'dishwasher_kw'
    ]
    df = df.drop(columns=non_cooling_heating_kw_cols, errors='ignore')

    # Drop non_cooling_heating_kvar columns
    non_cooling_heating_kvar_cols = [
        'lighting_kvar', 'pumps_kvar', 'water_systems_kvar', 'motors_kvar',
        'plug_loads_kvar', 'clothes_dryer_kvar', 'clothes_washer_kvar',
        'stove_kvar', 'dishwasher_kvar'
    ]
    df = df.drop(columns=non_cooling_heating_kvar_cols, errors='ignore')

    # Drop 'month' and 'building' if they exist
    df = df.drop(columns=['month', 'building'], errors='ignore')

    # Add non cooling and heating columns
    df['non_cool_n_heat_kw'] = df['total_site_electricity_kw'] - (
        df['heating_kw'] + df['cooling_kw'] + df['refrigeration_kw'] + df['fans_kw']
    )
    df['non_cool_n_heat_kvar'] = df['total_site_electricity_kvar'] - (
        df['heating_kvar'] + df['cooling_kvar'] + df['refrigeration_kvar'] + df['fans_kvar']
    )

    # Add cooling sum kw
    df['cooling_sum_kw'] = df['cooling_kw'] + df['refrigeration_kw'] + df['fans_kw']
    
    # columns_to_keep = ['date_time', 'total_site_electricity_kw', 'total_site_electricity_kvar', 'pf','cooling_sum_kw', 'cooling_sum_kvar','heating_kw', 'heating_kvar',
       # 'non_cool_n_heat_kw', 'non_cool_n_heat_kvar', 'cooling_kw', 'cooling_kvar', 'fans_kw', 'fans_kvar','refrigeration_kw', 'refrigeration_kvar' ]
    columns_to_keep = ['date_time', 'total_site_electricity_kw', 'pf', 'cooling_sum_kw' ,'heating_kw', 'non_cool_n_heat_kw' ]
    
    df = df[columns_to_keep]

    return df
