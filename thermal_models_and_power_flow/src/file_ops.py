import os
import numpy as np
import shutil #To enable duplicating files
import pandas as pd # to create data frames
import json
import pyarrow as pa
import joblib
import glob
from collections.abc import Mapping
from typing import Any, List, Tuple

def export_dict_to_joblib(data_dict, filename="exported_dict.joblib"):
    """
    Saves a dictionary to a .joblib file in the current working directory.
    
    Parameters:
    - data_dict (dict): The dictionary to save.
    - filename (str): Name of the output file (default: 'exported_dict.joblib')
    """
    cwd = os.getcwd()
    output_path = os.path.join(cwd, filename)
    joblib.dump(data_dict, output_path)
    print(f"Dictionary saved to: {output_path}")

def print_nested_keys_structure(d):
    """
    Recursively print keys at each level of a nested dictionary.
    - Level 1: print all keys.
    - Deeper levels: only print keys from the first key of the previous level.
    - Stop when a non-dictionary value is reached.
    """
    level = 1
    current = d
    path = []

    while isinstance(current, dict):
        keys = list(current.keys())
        print(f"level {level}:")
        print(f"dict_keys({keys})\n")

        if not keys:
            break  # Empty dict

        # Go one level deeper using the first key
        path.append(keys[0])
        current = current[keys[0]]
        level += 1

def return_leaf_dataframe(nested: Mapping, key_number: int, n: int = 2) -> pd.DataFrame:
    """
    Flatten a nested dict whose leaves are pandas DataFrames, select the leaf by
    its ordinal index (key_number), print the path and head(n), and return head(n).

    Args:
        nested: Arbitrarily nested dict-like object with DataFrames at leaves.
        key_number: Zero-based index into the list of leaf DataFrames (DFS order).
                    Negative indices are supported (like Python lists).
        n: Number of rows to show from the selected DataFrame (default: 2).

    Returns:
        pd.DataFrame: The head(n) of the selected leaf DataFrame.

    Raises:
        ValueError: If no leaf DataFrames are found or index is out of range.
        TypeError:  If a leaf is not a pandas DataFrame.
    """
    def _iter_leaves(obj: Any, path: Tuple[Any, ...]) -> List[Tuple[Tuple[Any, ...], pd.DataFrame]]:
        leaves: List[Tuple[Tuple[Any, ...], pd.DataFrame]] = []
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                leaves.extend(_iter_leaves(v, path + (k,)))
        else:
            if not isinstance(obj, pd.DataFrame):
                raise TypeError(f"Leaf at path {path} is not a pandas DataFrame (got {type(obj).__name__}).")
            leaves.append((path, obj))
        return leaves

    leaves = _iter_leaves(nested, ())
    if not leaves:
        raise ValueError("No DataFrame leaves were found in the provided nested dictionary.")

    # Normalize index (supports negatives)
    idx = key_number if key_number >= 0 else len(leaves) + key_number
    if not (0 <= idx < len(leaves)):
        raise ValueError(f"key_number {key_number} out of range. Valid range: 0..{len(leaves)-1} (or negatives).")

    path, df = leaves[idx]
    # Pretty path display
    path_str = " -> ".join(repr(k) for k in path)

    print(f"\nSelected leaf #{idx} at path: {path_str}")
    display(df.head(n))

    return df        
        
# Helper to walk the structure and print sample keys and DataFrame info
def print_nested_dict_key_examples_and_dataframe_details(d, level=1, max_depth=5):
    if not isinstance(d, dict) or level > max_depth:
        return
    keys = list(d.keys())
    print(f"Level {level} - sample keys ({len(keys)} total): {keys[:3]}")
    
    if isinstance(d[keys[0]], dict):
        print_nested_dict_key_examples_and_dataframe_details(d[keys[0]], level + 1, max_depth)
    elif isinstance(d[keys[0]], pd.DataFrame):
        df = d[keys[0]]
        print(f"\nReached DataFrame at level {level}")
        print(f"Shape: {df.shape}")
        print(f"Index type: {type(df.index)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Dtypes:\n{df.dtypes}")

def load_region_network_data(base_path,results_folder_name, notebook_code, climate_mode, smart_ds_year, city, region):
    # Set paths to results
    line_results_name = f"{notebook_code}_{city}_{region}_{climate_mode}_{smart_ds_year}_line_data"
    transformer_results_name = f"{notebook_code}_{city}_{region}_{climate_mode}_{smart_ds_year}_transformer_data"
    line_results_path = f'{base_path}{line_results_name}.parquet'
    transformer_results_path = f'{base_path}{transformer_results_name}.parquet'

    # Load DataFrame from Parquet
    lines_df = pd.read_parquet(line_results_path, engine='pyarrow')
    transformer_df = pd.read_parquet(transformer_results_path, engine='pyarrow')

    # Load metadata
    with open(f"{base_path}{line_results_name}_metadata.json", 'r') as f:
        metadata = json.load(f)
    return lines_df, transformer_df, metadata

def find_folders_with_file(base_path, file_name, max_depth=3):
    """
    Finds all folders under base_path that contain a specific file, up to a given depth.

    Parameters:
        base_path (str): The base path to search from, e.g., SMART-DS region folder
        file_name (str): The name of the file to search for (e.g., 'LineCodes.dss').
        max_depth (int): Maximum depth of subfolders to search into from base_path.

    Returns:
        list: List of folder paths containing the specified file within max_depth levels.
    """
    matching_folders = []
    base_depth = base_path.rstrip(os.path.sep).count(os.path.sep)

    for root, _, files in os.walk(base_path):
        current_depth = root.count(os.path.sep)
        if current_depth - base_depth <= max_depth:
            if file_name in files:
                matching_folders.append(root.replace('\\', '/'))

    return matching_folders

def copy_csv_files(path1, path2):
    # Ensure the destination folder exists
    os.makedirs(path2, exist_ok=True)
    
    # Loop through all files in the source directory
    for file_name in os.listdir(path1):
        if file_name.endswith('.csv'):  # Check if the file is a CSV
            src_file = os.path.join(path1, file_name)
            dest_file = os.path.join(path2, file_name)
            shutil.copy2(src_file, dest_file)  # Copy file while preserving metadata
            
            
