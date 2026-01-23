import numpy as np
import pandas as pd

## This function gets a dictionary with region-level dataframe data and returns dictionaries with city-level dataframe (all rows from region df combined)
def concat_regions_to_city(transformers_dict_summary_multi_region, lines_dict_summary_multi_region, TGW_weather_year, TGW_scenario, smart_ds_year):
    ### Add column 'region' to dataframes
    for key in transformers_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)]:
        smart_ds_year, city, region = key
        transformers_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][key]['region'] = region

    for key in lines_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)]:
        smart_ds_year, city, region = key
        lines_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][key]['region'] = region


    ### Concatenate regions to single city level dataframes (e.g., a single df with all transformers in GSO)
    transformers_dict_summary_multi_region_agg_by_city = {}
    transformers_dict_summary_multi_region_agg_by_city[(TGW_weather_year, TGW_scenario)] = {}

    lines_dict_summary_multi_region_agg_by_city = {}
    lines_dict_summary_multi_region_agg_by_city[(TGW_weather_year, TGW_scenario)] = {}



    # Greensboro (GSO)
    city = 'GSO'
    region1, region2, region3 = 'rural', 'industrial', 'urban-suburban'
    transformers_dict_summary_multi_region_agg_by_city[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city)] = pd.concat(
        [
            transformers_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region1)],
            transformers_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region2)],
            transformers_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region3)]
        ],
        ignore_index=True
    )
    lines_dict_summary_multi_region_agg_by_city[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city)] = pd.concat(
        [
            lines_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region1)],
            lines_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region2)],
            lines_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region3)]
        ],
        ignore_index=True
    )
    # San Francisco (SFO)
    city = 'SFO'
    region1, region2, region3 = 'P1R', 'P1U', 'P2U'
    transformers_dict_summary_multi_region_agg_by_city[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city)] = pd.concat(
        [
            transformers_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region1)],
            transformers_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region2)],
            transformers_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region3)]
        ],
        ignore_index=True
    )
    lines_dict_summary_multi_region_agg_by_city[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city)] = pd.concat(
        [
            lines_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region1)],
            lines_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region2)],
            lines_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region3)]
        ],
        ignore_index=True
    )
    # Austin (AUS)
    city = 'AUS'
    region1, region2, region3 = 'P1R', 'P1U', 'P2U'
    transformers_dict_summary_multi_region_agg_by_city[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city)] = pd.concat(
        [
            transformers_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region1)],
            transformers_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region2)],
            transformers_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region3)]
        ],
        ignore_index=True
    )
    lines_dict_summary_multi_region_agg_by_city[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city)] = pd.concat(
        [
            lines_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region1)],
            lines_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region2)],
            lines_dict_summary_multi_region[(TGW_weather_year, TGW_scenario)][(smart_ds_year, city, region3)]
        ],
        ignore_index=True
    )
    return transformers_dict_summary_multi_region_agg_by_city, lines_dict_summary_multi_region_agg_by_city


## ---- Define sorting and mdh extraction functions  ---- 
def sort_nested_dict_dfs(data_dict, sort_col, ascending=True):
    """
    Recursively sorts all DataFrames in a nested dictionary by a given column.

    Parameters
    ----------
    data_dict : dict
        Nested dictionary where leaf nodes are pandas DataFrames.
    sort_col : str
        Column name to sort each DataFrame by.
    ascending : bool, optional
        Sort order. Default is True (ascending).

    Returns
    -------
    dict
        A new dictionary with the same structure, but with sorted DataFrames.
    """
    sorted_dict = {}

    for key, value in data_dict.items():
        if isinstance(value, dict):  # go deeper if nested dict
            sorted_dict[key] = sort_nested_dict_dfs(value, sort_col, ascending)
        elif isinstance(value, pd.DataFrame):  # sort DataFrame
            if sort_col in value.columns:
                sorted_dict[key] = value.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)
            else:
                raise KeyError(f"Column '{sort_col}' not found in DataFrame at key {key}")
        else:
            sorted_dict[key] = value  # leave non-DF values unchanged

    return sorted_dict

def get_top_n_mdh(df, n, start_month=1, end_month=12):
    """
    Extracts (month, day, hour) tuples from the top N rows of a DataFrame,
    after filtering to only include rows where 'month' is within [start_month, end_month].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'month', 'day', and 'hour' columns.
    n : int
        Number of rows to take from the top of the filtered DataFrame.
    start_month : int, optional
        First month to include (default=1).
    end_month : int, optional
        Last month to include (default=12).

    Returns
    -------
    list of tuples
        List of (month, day, hour) tuples from the top N filtered rows.
    """
    if not all(col in df.columns for col in ["month", "day", "hour"]):
        raise KeyError("DataFrame must contain 'month', 'day', and 'hour' columns")
    
    # Filter by month range
    df_filtered = df[(df["month"] >= start_month) & (df["month"] <= end_month)]
    
    # Take top N rows after filtering
    top_rows = df_filtered.head(n)
    
    return list(zip(top_rows["month"], top_rows["day"], top_rows["hour"]))

def select_evenly_spaced_transformers(df, num_points):
    selected_transformers = []

    for kva_rating, group in df.groupby('kVA rating [kVA]'):
        group = group.sort_values(by='Loading [%]').reset_index(drop=True)  # Reset index for safe iloc
        
        if len(group) <= num_points:
            # If there are fewer transformers than needed, take them all
            selected_transformers.extend(group['Transformer'].tolist())
        else:
            # Generate evenly spaced target values
            loading_values = np.linspace(group['Loading [%]'].min(), group['Loading [%]'].max(), num_points)

            closest_transformers = []
            used_indices = set()  # Track used indices to avoid duplicates
            
            for val in loading_values:
                closest_idx = (group['Loading [%]'] - val).abs().idxmin()
                
                if closest_idx not in used_indices:
                    closest_transformers.append(group.loc[closest_idx, 'Transformer'])
                    used_indices.add(closest_idx)  # Mark this index as used

            selected_transformers.extend(closest_transformers)

    return selected_transformers


def get_top_n_loaded_transformers(df, n):
    """
    Finds the top `n` loaded transformers for each kVA rating [kVA].

    Parameters:
    df (pd.DataFrame): The dataframe containing transformer data with columns 
                       'Transformer', 'Loading [%]', and 'kVA rating [kVA]'.
    n (int): The number of top loaded transformers to select per kVA rating.

    Returns:
    list: A list of transformer names that are in the top `n` loading values for each kVA rating.
    """
    top_transformers = (
        df.groupby('kVA rating [kVA]')
        .apply(lambda x: x.nlargest(n, 'Loading [%]'))
        .reset_index(drop=True)
    )

    return top_transformers['Transformer'].tolist()


import pandas as pd

def get_regional_peak_times(building_predicted_total_dict, start_month: int, end_month: int):
    """
    Computes regional total kW and kVAR time series, restricted to the months
    [start_month, end_month] inclusive, and returns the timestamps of the
    maximum values within that window.

    Parameters
    ----------
    building_predicted_total_dict : dict
        Nested dict:
          level-1 key: (year, city, region_type, feeder_id, btype)
          level-2 key: building label
          leaf value : DataFrame with columns ['kw', 'kvar']; index is timestamps
        Frames can have differing timestamp indices.
    start_month : int
        Starting month (1-12), inclusive. E.g., 6 for June.
    end_month : int
        Ending month (1-12), inclusive. E.g., 9 for September.
        If end_month < start_month, the window wraps across year-end.
        (e.g., start_month=11, end_month=2 selects Nov–Dec–Jan–Feb.)

    Returns
    -------
    dict
        {
          'kw_regional_peak_time':  pandas.Timestamp | None,
          'kvar_regional_peak_time': pandas.Timestamp | None
        }
        None is returned if the filtered series is empty or all-NaN.

    Notes
    -----
    - Converts non-datetime indexes via `pd.to_datetime` (errors='raise').
    - Sums using alignment with `add(..., fill_value=0)` to avoid NaNs.
    """

    if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
        raise ValueError("start_month and end_month must be integers in [1, 12].")

    # Build the allowed month set, supporting wrap-around across Dec->Jan.
    if start_month <= end_month:
        allowed_months = set(range(start_month, end_month + 1))
    else:
        allowed_months = set(list(range(start_month, 13)) + list(range(1, end_month + 1)))

    regional_kw_series = None
    regional_kvar_series = None

    # Aggregate across all buildings (aligned sum; missing timestamps treated as 0)
    for _, buildings_dict in building_predicted_total_dict.items():  # feeders x types
        for _, df in buildings_dict.items():  # buildings
            if not isinstance(df.index, pd.DatetimeIndex):
                # Try to coerce to datetime index
                idx = pd.to_datetime(df.index, errors='raise')
                df = df.copy()
                df.index = idx

            # Ensure expected columns exist
            if 'kw' not in df.columns or 'kvar' not in df.columns:
                raise KeyError("Each building DataFrame must contain 'kw' and 'kvar' columns.")

            s_kw = df['kw']
            s_kvar = df['kvar']

            if regional_kw_series is None:
                # Copy; do not mutate originals
                regional_kw_series = s_kw.copy()
                regional_kvar_series = s_kvar.copy()
            else:
                regional_kw_series = regional_kw_series.add(s_kw, fill_value=0)
                regional_kvar_series = regional_kvar_series.add(s_kvar, fill_value=0)

    # If nothing was aggregated (empty dict)
    if regional_kw_series is None or regional_kw_series.empty:
        return {'kw_regional_peak_time': None, 'kvar_regional_peak_time': None}

    # Filter to the desired months
    month_mask_kw = regional_kw_series.index.month.astype(int)
    month_mask_kvar = regional_kvar_series.index.month.astype(int)
    kw_filtered = regional_kw_series[ [m in allowed_months for m in month_mask_kw] ]
    kvar_filtered = regional_kvar_series[ [m in allowed_months for m in month_mask_kvar] ]

    # Handle empty after filtering
    if kw_filtered.empty:
        kw_peak_time = None
    else:
        # If all values are NaN within the window, idxmax() would error; guard it.
        if kw_filtered.dropna().empty:
            kw_peak_time = None
        else:
            kw_peak_time = kw_filtered.idxmax()

    if kvar_filtered.empty:
        kvar_peak_time = None
    else:
        if kvar_filtered.dropna().empty:
            kvar_peak_time = None
        else:
            kvar_peak_time = kvar_filtered.idxmax()

    return {
        'kw_regional_peak_time': kw_peak_time,
        'kvar_regional_peak_time': kvar_peak_time
    }
