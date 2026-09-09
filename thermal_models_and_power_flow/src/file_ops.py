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
import os
import json
import re
from pathlib import Path

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
           

def find_folders_with_file(
    base_path,
    file_name,
    max_depth=3,
    use_saved_paths=True,
    save_paths=True,
    cache_base_folder=None,
):
    """
    Finds all folders under base_path that contain a specific file, up to a given depth.

    Parameters:
        base_path (str or Path):
            Base path to search from, e.g., SMART-DS region folder.

        file_name (str):
            Exact file name to search for, e.g., "LineCodes.dss", "Loads.dss",
            "Transformers.dss", or "PVSystems.dss".

        max_depth (int):
            Maximum depth of subfolders to search into from base_path.

        use_saved_paths (bool):
            If True, the function first checks whether saved results already exist.
            If saved results exist, they are loaded and returned without walking the filesystem.

            If False, the function ignores saved results and walks the filesystem again.

        save_paths (bool):
            If True, results from a filesystem search are saved to JSON for future runs.

        cache_base_folder (str or Path or None):
            Folder where cached path files are saved.

            If None, the default is:
                base_path / "folder_with_file"

            The final cache path will be:
                cache_base_folder / <file_stem_lowercase> /
                folders_with_<safe_file_name>_depth_<max_depth>.json

            Example for LineCodes.dss and max_depth=3:
                region_path/folder_with_file/linecodes/
                folders_with_linecodes_dss_depth_3.json

    Returns:
        list:
            List of folder paths containing the specified file, with "/" path separators.
            This matches the output style of the original function.
    """

    base_path = Path(base_path).resolve()

    if not base_path.exists():
        raise FileNotFoundError(f"base_path does not exist: {base_path}")

    if not base_path.is_dir():
        raise NotADirectoryError(f"base_path is not a directory: {base_path}")

    # Examples:
    #   "LineCodes.dss"    -> file_stem_safe = "linecodes"
    #   "Transformers.dss" -> file_stem_safe = "transformers"
    file_stem_safe = re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        Path(file_name).stem.lower(),
    ).strip("_")

    # Examples:
    #   "LineCodes.dss"    -> safe_file_name = "linecodes_dss"
    #   "PVSystems.dss"    -> safe_file_name = "pvsystems_dss"
    safe_file_name = re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        file_name.lower(),
    ).strip("_")

    if cache_base_folder is None:
        cache_base_folder = base_path / "folder_with_file"
    else:
        cache_base_folder = Path(cache_base_folder)

    cache_folder = cache_base_folder / file_stem_safe
    cache_file = cache_folder / f"folders_with_{safe_file_name}_depth_{max_depth}.json"

    # ------------------------------------------------------------
    # 1. Load saved paths if requested and available
    # ------------------------------------------------------------
    if use_saved_paths and cache_file.exists():
        with open(cache_file, "r") as f:
            cached_data = json.load(f)

        # Support both possible cache formats:
        #   - list of folders
        #   - dict with metadata and "matching_folders"
        if isinstance(cached_data, list):
            return cached_data

        if isinstance(cached_data, dict) and "matching_folders" in cached_data:
            return cached_data["matching_folders"]

        raise ValueError(
            f"Cache file exists but has unexpected format: {cache_file}"
        )

    # ------------------------------------------------------------
    # 2. Search filesystem
    # ------------------------------------------------------------
    matching_folders = []

    base_depth = len(base_path.parts)

    for root, dirs, _ in os.walk(base_path):
        root_path = Path(root)
        current_depth = len(root_path.parts) - base_depth

        # Prevent os.walk from descending below max_depth.
        if current_depth >= max_depth:
            dirs[:] = []

        # Direct exact-file lookup.
        # This avoids scanning thousands of similarly named files like
        # LineCodes_9_7_16.dss.
        if (root_path / file_name).is_file():
            matching_folders.append(root_path.as_posix())

    # ------------------------------------------------------------
    # 3. Save paths for future runs
    # ------------------------------------------------------------
    if save_paths:
        cache_folder.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "base_path": base_path.as_posix(),
            "file_name": file_name,
            "max_depth": max_depth,
            "matching_folders": matching_folders,
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

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
            

            
