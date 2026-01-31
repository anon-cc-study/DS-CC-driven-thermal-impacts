import os
import glob
import numpy as np
import pandas as pd # to create data frames
import shutil #To enable duplicating files
from opendssdirect import dss
from src import physics_ops

## Create a numpy array of temperatures from TGW data (csv file) for a city, year and RCP scenario
# Input: city_name (e.g.,'SanFrancisco'), scenario (e.g., 'rcp45hotter'), and year (e.g., 1980)
# Output: a numpy array of temperatures (year, 1h res, 8760 elements)
def load_temperature_data(city_name, scenario, year):
    # Base directory where the files are stored
    base_dir = f"main_folder/TGW/TGW_Distribution/Yearly/{city_name}/{scenario}/"
    
    # File pattern to match the required CSV files
    # file_pattern = f"tgw_wrf_{scenario}_hourly_{year}*.csv"
    file_pattern = f"tgw_wrf_{scenario}_hourly_{year}.csv"

    # Find all matching files for the given scenario
    file_paths = glob.glob(os.path.join(base_dir, file_pattern))
        
    # Check if any files were found
    if not file_paths:
        raise FileNotFoundError(f"No files found for {city_name}, {scenario}, {year}")

    # Initialize an empty list to hold temperature values
    temperature_values = []
    
    # Loop through all matching files
    for file_path in file_paths:
        # Read the CSV file
        df = pd.read_csv(file_path, header=None, skiprows=1, usecols=[2])  # Read the CSV file, skip the first row, and extract the third column (index 2)
        # Convert from Kelvin to Celsius (C = K - 273.15)
        df_celsius = df[2] - 273.15
        # Append values to the list
        temperature_values.extend(df_celsius.values)
           
    # Convert the list of temperature values to a numpy array
    temperature_array = np.array(temperature_values)
    
    return temperature_array

def compute_temperature_percentiles(city_name, years_historical, years_rcp45hotter, percentile):
    """
    Computes the specified percentile of temperature data for a given city across historical and future scenario years.

    Parameters:
    - city_name (str): Name of the city.
    - years_historical (list): List of historical years.
    - years_rcp45hotter (list): List of future scenario years.
    - percentile (float): Percentile to compute (default is 99.9).

    Returns:
    - list: List of computed percentiles for each year in the order of years_historical followed by years_rcp45hotter.
            If data is missing for a year, None is appended for that year.
    """
    percentiles = []

    # Compute percentiles for historical years
    for year in years_historical:
        try:
            temperatures = load_temperature_data(city_name, 'historical', year)
            percentiles.append(np.percentile(temperatures, percentile))
        except FileNotFoundError:
            percentiles.append(None)  # Handle missing data

    # Compute percentiles for future scenario years
    for year in years_rcp45hotter:
        try:
            temperatures = load_temperature_data(city_name, 'rcp45hotter', year)
            percentiles.append(np.percentile(temperatures, percentile))
        except FileNotFoundError:
            percentiles.append(None)  # Handle missing data

    # return percentiles
    # Convert the list to a NumPy array and return
    return np.array(percentiles, dtype=object)  # Use dtype=object to accommodate None values