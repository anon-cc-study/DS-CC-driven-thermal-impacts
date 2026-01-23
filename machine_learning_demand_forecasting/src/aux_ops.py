import os

def extract_unique_dimensions(mlp_models, city_regions_to_run=None):
    """
    Extracts sorted unique values for each key dimension in the mlp_models dictionary.
    Optionally filters city and region based on city_regions_to_run, and only includes
    load_models and building_types for those combinations.

    Parameters:
    - mlp_models (dict): Dictionary with keys as 5-tuples 
                         (smart_ds_year, city, region, load_model, building_type).
    - city_regions_to_run (dict, optional): Dictionary of the form 
        {
            "City1": ["RegionA", "RegionB"],
            "City2": ["RegionC"]
        }

    Returns:
    - smart_ds_years (list)
    - cities (list)
    - regions (list)
    - load_models (list)
    - building_types (list)
    """
    smart_ds_years_set = set()
    cities_set = set()
    regions_set = set()
    load_models_set = set()
    building_types_set = set()

    for key in mlp_models.keys():
        smart_ds_year, city, region, load_model, building_type = key

        if city_regions_to_run:
            if city not in city_regions_to_run:
                continue
            if region not in city_regions_to_run[city]:
                continue

        smart_ds_years_set.add(smart_ds_year)
        cities_set.add(city)
        regions_set.add(region)
        load_models_set.add(load_model)
        building_types_set.add(building_type)

    return (
        sorted(smart_ds_years_set),
        sorted(cities_set),
        sorted(regions_set),
        sorted(load_models_set),
        sorted(building_types_set)
    )


def print_joblib_tree(folder_path, indent=""):
    """
    Recursively prints a tree of directories and .joblib files within the given folder.

    Parameters:
    folder_path (str): The root folder path to start searching from.
    indent (str): Used internally for indentation in tree view.
    """
    try:
        entries = sorted(os.listdir(folder_path))
    except PermissionError:
        print(f"{indent}Permission denied: {folder_path}")
        return

    folders = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]
    files = [entry for entry in entries if os.path.isfile(os.path.join(folder_path, entry)) and entry.endswith('.joblib')]

    print(f"{indent}{os.path.basename(folder_path)}/")
    
    for file in files:
        print(f"{indent}  └── {file}")
    
    for folder in folders:
        print_joblib_tree(os.path.join(folder_path, folder), indent + "  ")