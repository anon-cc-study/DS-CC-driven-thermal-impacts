import os
import numpy as np
import pandas as pd # to create data frames

def get_max_parquet_column(parquet_file_path,col_num):
    """
    Reads a Parquet file and returns the maximum value in its desired column.

    Parameters:
        parquet_file_path (str): Path to the Parquet file.
        col_num: column number in Parquet file to take max from
    Returns:
        max_value: Maximum value in the required column of the Parquet file.
    """
    try:
        # Read the Parquet file into a Pandas DataFrame
        df = pd.read_parquet(parquet_file_path)

        # Check if the DataFrame has enough columns
        if df.shape[1] < col_num:
            raise ValueError("The DataFrame has fewer than the expected number of columns.")

        # Get the maximum value in the column
        column_name = df.columns[col_num - 1]
        max_value = df[column_name].max()
        return max_value

    except Exception as e:
        raise RuntimeError(f"Error processing Parquet file: {e}")

def get_column_values(parquet_file_path, column_number):
    """
    Reads a Parquet file and returns the array of values in the specified column number.

    Parameters:
        parquet_file_path (str): Path to the Parquet file.
        column_number (int): The column number (0-indexed) to extract values from.

    Returns:
        values_array: Array of values in the specified column.
    """
    try:
        # Read the Parquet file into a Pandas DataFrame
        df = pd.read_parquet(parquet_file_path)

        # Check if the column number is valid
        if column_number >= df.shape[1] or column_number < 0:
            raise ValueError("Invalid column number.")

        # Extract the values from the specified column
        column_name = df.columns[column_number - 1]
        values_array = df[column_name].values

        return values_array

    except Exception as e:
        raise RuntimeError(f"Error processing Parquet file: {e}")
        
def sum_max_values_in_folder(folder_path,col_num):
    """
    Iterates over all Parquet files in a folder, calculates the sum of maximum values from the second column
    of each file using the get_max_parquet_column() function, and returns the total sum.

    Parameters:
        folder_path (str): Path to the folder containing Parquet files.

    Returns:
        total_sum (float): Sum of all maximum values from the specified column of each Parquet file.
    """
    total_sum = 0
    iterations = 0
    try:
        # Iterate over all files in the folder
        for file_name in os.listdir(folder_path):
            iterations += 1
            file_path = os.path.join(folder_path, file_name)

            # Check if the file is a Parquet file
            if file_name.endswith('.parquet'):
                max_value = get_max_parquet_column(file_path, col_num)  
                total_sum += max_value
        print(f'Iterations:{iterations}')
        return total_sum

    except Exception as e:
        raise RuntimeError(f"Error processing files in folder: {e}")
        


def modify_profiles_linear_approx(path_to_csv_folder, temp_array1, temp_array2, cooling_threshold, load_temp_percentage_change):
    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(path_to_csv_folder) if f.endswith('.csv')]

    for file_name in csv_files:
        file_path = os.path.join(path_to_csv_folder, file_name)

        # Read the CSV file
        df = pd.read_csv(file_path, header=None)  # Assuming no header in CSV

        # Ensure the CSV file has the expected number of rows
        if len(df) != 35040:
            print(f"Skipping {file_name}: unexpected row count ({len(df)} rows).")
            continue

        # Modify the first column by adding temperature values
        for i in range(35040):
            temp_index = i // 4  # Get the corresponding hourly temperature index
            temp_change = temp_array2[temp_index] - temp_array1[temp_index]
            if temp_change > 0:
                if temp_array2[temp_index] > cooling_threshold:
                    print(f'temp change is {temp_change}')
                    print(f'load value before change: { df.iloc[i, 0]}')
                    df.iloc[i, 0] *= (1 + load_temp_percentage_change * temp_change)
                    print(f'load value after change: { df.iloc[i, 0]}')

        # Save the modified file, overwriting the original
        df.to_csv(file_path, index=False, header=False)



def modify_profiles_linear_approx_v2(path_to_csv_folder, temp_array1, temp_array2, cooling_threshold, load_temp_percentage_change):
    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(path_to_csv_folder) if f.endswith('.csv')]

    for file_name in csv_files:
        file_path = os.path.join(path_to_csv_folder, file_name)

        # Read the CSV file
        df = pd.read_csv(file_path, header=None)  # Assuming no header in CSV

        # Ensure the CSV file has the expected number of rows
        if len(df) != 35040:
            print(f"Skipping {file_name}: unexpected row count ({len(df)} rows).")
            continue

        # **Vectorized Computation**
        temp_indices = np.arange(35040) // 4  # Compute all indices at once
        temp_changes = temp_array2[temp_indices] - temp_array1[temp_indices]

        # Mask for valid modifications (where temp_change > 0 and temp_array2 is above threshold)
        valid_mask = (temp_changes > 0) & (temp_array2[temp_indices] > cooling_threshold)

        # Apply changes using vectorized operations
        df.loc[valid_mask, 0] *= (1 + load_temp_percentage_change * temp_changes[valid_mask])

        # Save the modified file, overwriting the original
        df.to_csv(file_path, index=False, header=False)
