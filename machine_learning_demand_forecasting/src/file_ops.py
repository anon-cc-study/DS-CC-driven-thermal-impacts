import os
import numpy as np
import shutil #To enable duplicating files
import pandas as pd # to create data frames
import json
import pyarrow as pa
import glob
from collections.abc import Mapping
from typing import Any, List, Tuple


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
        