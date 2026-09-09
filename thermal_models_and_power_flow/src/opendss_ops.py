import os
import numpy as np
import pandas as pd 
import shutil 
import re 
from opendssdirect import dss
from src import physics_ops


## Function create_dict_bus_coord() creates a dictionary of bus names and coordinates
# Input: file path to Buscoords.dss
# Output: dictionary of bus names and coordinates
def create_dict_bus_coord(file_path):
    bus_coords = {}
    with open(file_path + '/Buscoords.dss', 'r') as file:
        for line in file:
            # Split each line into parts and clean up the data
            parts = line.strip().split()
            if len(parts) >= 3:  # Ensure the line contains a bus name and two coordinates
                bus_name = parts[0]
                try:
                    lat = float(parts[1])
                    long = float(parts[2])
                    bus_coords[bus_name] = (lat, long)
                except ValueError:
                    continue  # Skip lines that don't have valid coordinate data
        return bus_coords

## Function modify_master_file() creates a new masterfile with paths to scenario-based smart-ds files 
# mdh - month, day, hour
# Inputs: path to new master file (must be a duplicate of an original master file) + parameters that define the scenario
# Output: None
def modify_master_file(new_master_file_path, solution_mode, demand_mode, line_rating_mode, transformers_rating_mode, TGW_scenario, TGW_weather_year,m,d,h):
    # Read the file content
    with open(new_master_file_path, 'r') as file:
        file_content = file.read()    

    modified_content = file_content             # Initialize modified_content with the original file content

    # ---- Define paths to scenario-based smart-ds files ---
    loadshapes_path = os.path.join('predicted_loadshapes','TGW', TGW_scenario, TGW_weather_year, 'LoadShapes.dss')
    if solution_mode == 'snapshot':
        loads_path = os.path.join('predicted_loads','TGW', TGW_scenario, TGW_weather_year, f'Loads_{m}_{d}_{h}.dss')
        linecodes_path = os.path.join('predicted_linecodes', line_rating_mode, TGW_scenario, TGW_weather_year, f'LineCodes_{m}_{d}_{h}.dss')      
        transformer_path = os.path.join('predicted_transformers', transformers_rating_mode, TGW_scenario, TGW_weather_year, f'Transformers_{m}_{d}_{h}.dss')
        pvsystems_path = os.path.join('predicted_pvsystems','TGW',TGW_scenario,TGW_weather_year, f'PVSystems_{m}_{d}_{h}.dss')
    else:
        loads_path = os.path.join('predicted_loads','TGW', TGW_scenario, TGW_weather_year, 'Loads.dss')
        linecodes_path = os.path.join('predicted_linecodes', line_rating_mode, TGW_scenario, TGW_weather_year, 'LineCodes.dss')      
        transformer_path = os.path.join('predicted_transformers', transformers_rating_mode, TGW_scenario, TGW_weather_year, 'Transformers.dss')
        pvsystems_path = os.path.join('predicted_pvsystems','TGW',TGW_scenario,TGW_weather_year,'PVSystems.dss')


    if solution_mode == 'snapshot':
        match demand_mode:
            case 'MLP':
                # modified_content = modified_content.replace('Loads.dss', loads_path)     # Replace all occurrences of "Loads.dss" with new path to predicted "Loads.dss"
                modified_content = re.sub(r"(\d+[\\/]+)Loads\.dss", r"\1" + loads_path, modified_content) # Replace all occurrences of "<number>/Loads.dss" with new path to predicted "Loads.dss" (we're adding number so that subtransmission/Loads.dss is skipped, since that single load does not have a prediction model)
                modified_content = re.sub(
                    r"(\d+[\\/]+)PVSystems\.dss",
                    r"\1" + pvsystems_path,
                    modified_content,
                    flags=re.IGNORECASE,
                )
                modified_content = modified_content.replace('LineCodes.dss', linecodes_path)     
                modified_content = modified_content.replace('Transformers.dss', transformer_path)    
                modified_content = modified_content.replace('Redirect ', 'Redirect ../../../')     # Replace all occurrences of Redirect to match the relative location of the mdh master files with the new TGE locations
                modified_content = modified_content.replace('Buscoords ', 'Buscoords ../../../')    
            case 'linear_load_growth':
                modified_content = modified_content.replace('Loads.dss', 'LoadsT1.dss')     # Replace all occurrences of "Loads.dss" with "LoadsT1.dss"
            case _:
                modified_content = modified_content.replace('Loads.dss', 'LoadsT1.dss')     # Replace all occurrences of "Loads.dss" with "LoadsT1.dss"

    if solution_mode == 'yearly':
        match demand_mode:
            case 'MLP':
                # modified_content = modified_content.replace('Loads.dss', loads_path)     # Replace all occurrences of "Loads.dss" with "Loads.dss"
                modified_content = re.sub(r"\d+/Loads\.dss",loads_path,modified_content) # Replace all occurrences of "<number>/Loads.dss" with new path to predicted "Loads.dss" (we're adding number so that subtransmission/Loads.dss is skipped, since that single load does not have a prediction model)
                modified_content = re.sub(
                    r"(\d+[\\/]+)PVSystems\.dss",
                    r"\1" + pvsystems_path,
                    modified_content,
                    flags=re.IGNORECASE,
                )
                modified_content = modified_content.replace('LoadShapes.dss', f"{loadshapes_path}")    
                modified_content = modified_content.replace('LineCodes.dss', linecodes_path)    
                modified_content = modified_content.replace('Transformers.dss', transformer_path)   
            case 'linear_load_growth':
                modified_content = modified_content.replace('Loads.dss', 'LoadsT1.dss')     # Replace all occurrences of "Loads.dss" with "Loads.dss"
                modified_content = modified_content.replace('LoadShapes.dss', 'LoadShapesT1.dss')    
            case _:
                modified_content = modified_content.replace('Loads.dss', 'LoadsT1.dss')     # Replace all occurrences of "Loads.dss" with "Loads.dss"
                modified_content = modified_content.replace('LoadShapes.dss', 'LoadShapesT1.dss')   

        if "Solve mode=yearly stepsize=15m number=35040\n" \
       "Export monitors m1\n" \
       "Plot monitor object= m1 channels=(7 9 11 )\n" \
       "Export monitors m2\n" \
       "Plot monitor object= m2 channels=(1 3 5 )\n" \
       "Plot Profile Phases=All" in file_content:
            modified_content = modified_content.replace(
                "Solve mode=yearly stepsize=15m number=35040\n"
                "Export monitors m1\n"
                "Plot monitor object= m1 channels=(7 9 11 )\n"
                "Export monitors m2\n"
                "Plot monitor object= m2 channels=(1 3 5 )\n"
                "Plot Profile Phases=All",
                "!Solve mode=yearly stepsize=15m number=35040\n"
                "!Export monitors m1\n"
                "!Plot monitor object= m1 channels=(7 9 11 )\n"
                "!Export monitors m2\n"
                "!Plot monitor object= m2 channels=(1 3 5 )\n"
                "!Plot Profile Phases=All"
        )
        else:
            print("Warning: tried to comment last  text block in master file but the text block was not found in the file.")

    # Save the modified content back to the file (or a new file if you want to keep a backup)
    with open(new_master_file_path, 'w') as file:
        file.write(modified_content)


## function modify_LineCodes with new Ampacity and Rmatrix values
# input: file_path - a path to duplicate master.dss file to modify
# mdh - month, day, hour
# factor - Rmatrix multiplier based on T-R relationship
# Ta - Ambient air temperature in Celsius
# output: None (LineCodesT1 will be created and modified)
def modify_LineCodes(folder_path, Rmatrix_factor, amp_factor, line_rating_mode, TGW_scenario, TGW_weather_year,m,d,h):
    dss.Command(f'Redirect "{folder_path}/LineCodes.dss"')
    
    ## Duplicate lindeCodes files   
    # Define the paths for the original and new linecodes files
    linecodes_dir = os.path.join(folder_path, "predicted_linecodes", line_rating_mode, TGW_scenario, TGW_weather_year)
    new_linecodes_file = os.path.join(linecodes_dir, f"LineCodes_{m}_{d}_{h}.dss")
    original_linecodes_file = os.path.join(folder_path, "LineCodes.dss")
    # Create directory if it does not exist
    os.makedirs(linecodes_dir, exist_ok=True)
    # Duplicate the LineCodes.dss file
    shutil.copyfile(original_linecodes_file, new_linecodes_file)

    # Read the original LineCodes.dss file
    with open(original_linecodes_file, "r") as file:
        lines = file.readlines()

    # Store the updated LineCodes content
    updated_lines = []

    # Loop through each line in the original file
    for line in lines:
        if line.strip().lower().startswith("new linecode."):
            # Extract the line code name between "." and " "
            try:
                start_idx = line.index(".") + 1
                end_idx = line.index(" ", start_idx)
                linecode_name = line[start_idx:end_idx].strip()
            except ValueError:
                # Skip if the line does not follow the expected format
                updated_lines.append(line)
                print('Warning: a line was skipped since it wasnt in the expected format')
                continue

            # Select the line code in OpenDSS
            dss.LineCodes.Name(linecode_name)

            # Get the linecode's Rmatrix, normap and number of phases
            Rmatrix = dss.LineCodes.Rmatrix()
            nphases = dss.LineCodes.Phases()
            NormAmps = dss.LineCodes.NormAmps()

            # Multiply the entire Rmatrix by the calculated factor
            Rmatrix_new = [r * Rmatrix_factor for r in Rmatrix] 

            # Calculate new NormAmps with an approximation of ampacity change per temp change (assuming ampacity in SMART-DS is for 25C)
            NormAmps_new = NormAmps * amp_factor

            # Format the Rmatrix based on the number of phases
            Rmatrix_str = ""
            if nphases == 1:
                # For a 1x1 Rmatrix
                Rmatrix_str = f"{Rmatrix_new[0]:.6f}"
            else:
                # For 2x2 or 3x3 Rmatrix, add the "|" delimiter for each row
                for i in range(nphases):
                    row = " ".join(f"{Rmatrix_new[i * nphases + j]:.6f}" for j in range(nphases))
                    Rmatrix_str += row + " | " if i < nphases - 1 else row

            # Replace the existing Rmatrix value in the line
            if "rmatrix=" in line.lower():
                # Find and replace the Rmatrix in the line
                start_idx = line.lower().index("rmatrix=") + len("rmatrix=")
                end_idx = line.find(")", start_idx) + 1
                line = line[:start_idx] + f"({Rmatrix_str})" + line[end_idx:]
            else:
                # Add Rmatrix entry if not present
                line = line.strip() + f" Rmatrix=({Rmatrix_str})\n"

            # Replace or add the NormAmps value in the line
            if "normamps=" in line.lower():
                start_idx = line.lower().index("normamps=") + len("normamps=")
                end_idx = line.find(" ", start_idx) 
                line = line[:start_idx] + f"{NormAmps_new:.2f}" + line[end_idx:]
            else:
                # Add NormAmps entry if not present
                line = line.strip() + f" NormAmps={NormAmps_new:.2f}\n"

        updated_lines.append(line)  # Append the modified (or unmodified) line to the updated lines list

    # Write the updated content to the new LineCodes.dss file
    with open(new_linecodes_file, "w") as file:
        file.writelines(updated_lines)


def update_kva_values(line, Ta):
    """
    Updates the numerical values of all expressions in the line containing 'kva=' (e.g., 'kva=5', 'normHkva=8').

    Parameters:
        line (str): The input string containing kva expressions.

    Returns:
        str: The updated line with numerical values multiplied by a calculated factor.
    """
    # Regular expression to match 'kva=' followed by a number, case insensitive
    pattern = re.compile(r"(\b[a-zA-Z]*kva=)(\d+(\.\d+)?)", re.IGNORECASE)

    def replace_function(match):
        # Extract the key and the numerical value
        key = match.group(1)
        kva_value = float(match.group(2))

        # Set derating factor: -1.5%/+1% per degree celcius for KVA<10,000 and -1%/+0.75% per degree celcius for KVA<10,000 (based on IEEEc57.91 Table 3)
        if kva_value <= 10000:
            if Ta >= 30:
                kva_derating_factor = 1 - (0.015 * (Ta - 30))
            else:
                kva_derating_factor = 1 + (0.01 * (30 - Ta))
        else:
            if Ta >= 30:
                kva_derating_factor = 1 - (0.01 * (Ta - 30))
            else:
                kva_derating_factor = 1 + (0.0075 * (30 - Ta))

        # Multiply the numerical value by the factor
        updated_value = kva_value * kva_derating_factor
        # Return the updated expression
        return f"{key}{updated_value:.2f}"

    # Substitute all matches in the line
    updated_line = pattern.sub(replace_function, line)
    return updated_line

## function modify_Tranformers modifies modify_Tranformers.dss with new KVA, NormHKVa and EmergHKva values
# input: file_path - a path to duplicate master.dss file to modify, e.g., "Smart-DS_folder_path/MasterT1.dss"
# mdh - month, day, hour
# Ta - Ambient air temperature in Celsius (-1.5 %/C derating for Ta>=30 and +1.5 %/C uprating for Ta<30)
# output: None (Transformer.dss will be created and modified)
def modify_tranformers(folder_path, Ta, transformers_rating_mode, TGW_scenario, TGW_weather_year,m,d,h):

    ## Duplicate Transformer files
    # Define the paths for the original and new linecodes files
    original_transformer_file = folder_path + "/Transformers.dss"
    
    # new_transformer_file = folder_path + "/TransformersT1.dss"
    transformer_dir = os.path.join(folder_path, "predicted_transformers", transformers_rating_mode, TGW_scenario, TGW_weather_year)
    # Create directory if it does not exist
    os.makedirs(transformer_dir, exist_ok=True)
    
    new_transformer_file = os.path.join(transformer_dir, f"Transformers_{m}_{d}_{h}.dss")
    
    # Duplicate the original Transformer.dss file
    shutil.copyfile(original_transformer_file, new_transformer_file) 

    # Read the original Transformer.dss file
    with open(original_transformer_file, "r") as file:
        lines = file.readlines()

    # Store the updated Transformers content
    updated_lines = []

    # Loop through each line in the original file
    for line in lines:
        if line.strip().lower().startswith("new transformer."):
            updated_line = update_kva_values(line, Ta)
            updated_lines.append(f'{updated_line}\n')  # Append the modified (or unmodified) line to the updated lines list

    # Write the updated content to the new Transformers.dss file
    with open(new_transformer_file, "w") as file:
        file.writelines(updated_lines)
        
def update_loads_file(loads_file_path, building_multipliers):
    """
    Modifies a copy of a Loads.dss file with updated kW and kvar values based on building multipliers.

    Args:
        loads_file_path (str): Path to the Loads.dss file.
        building_multipliers (dict): Dictionary with building names as keys and (kW, kVar) tuples as values.

    Returns:
        None
    """
    output_file_path = os.path.join(loads_file_path, "LoadsT1.dss")
    input_file_path = os.path.join(loads_file_path, "Loads.dss")
    with open(input_file_path, "r") as infile, open(output_file_path, "w") as outfile:
        for line in infile:
            if line.startswith("New Load."):
#                 print(f"original line:{line}")
                # Extract the building name key from the line
                match = re.search(r"!yearly=(\w+)_kw_(\d+)_pu", line)
                if match:
                    building_name = f"{match.group(1)}_{match.group(2)}"
                    # Check if the building name is in the dictionary
                    if building_name in building_multipliers:
                        kW_multiplier, kVar_multiplier = building_multipliers[building_name]

                        # Update kW and kvar values in the line
                        line = re.sub(r"kW=([0-9\.]+)", lambda m: f"kW={float(m.group(1)) * kW_multiplier}", line)
                        line = re.sub(r"kvar=([0-9\.]+)", lambda m: f"kvar={float(m.group(1)) * kVar_multiplier}", line)
#                 print(f"New line:{line}")
            outfile.write(line)
    

    
def extract_pvsystems_information(
    dict_bus_coord,
    weather_year,
    m,
    d,
    h,
    row_i,
    decimals=6,
    extraction_mode="minimal",
    enable_low_memory=True,
):
    """
    Extract PVSystem information from the active OpenDSS circuit.

    """

    valid_extraction_modes = {"minimal", "full"}
    if extraction_mode not in valid_extraction_modes:
        raise ValueError(
            f"extraction_mode must be one of {valid_extraction_modes}, "
            f"got {extraction_mode!r}"
        )

    def _round_or_none(x, ndigits=decimals):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float, np.integer, np.floating)):
                if np.isnan(x):
                    return None
                return round(float(x), ndigits)
            return x
        except Exception:
            return x

    def _safe_prop(prop_name, default=None):
        """
        Safely get an OpenDSS property from the active element.
        Returns strings because OpenDSS properties are stored as text.
        """
        try:
            dss.Properties.Name(prop_name)
            value = dss.Properties.Value()
            if value == "":
                return default
            return value
        except Exception:
            return default

    def _safe_float_prop(prop_name, default=None):
        value = _safe_prop(prop_name, default=default)
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _terminal_power_from_powers(powers, num_phases, terminal=1):
        """
        Compute total P, Q, and S for one terminal.

        OpenDSS CktElement.Powers() returns:
            [P1, Q1, P2, Q2, ...]

        ordered by conductors/terminals.
        """
        if powers is None or len(powers) == 0 or num_phases <= 0:
            return None, None, None

        start_idx = 2 * num_phases * (terminal - 1)
        end_idx = start_idx + 2 * num_phases

        if len(powers) < end_idx:
            end_idx = min(len(powers), end_idx)

        p_kw = 0.0
        q_kvar = 0.0

        for idx in range(start_idx, end_idx, 2):
            if idx + 1 < len(powers):
                p_kw += powers[idx]
                q_kvar += powers[idx + 1]

        s_kva = abs(complex(p_kw, q_kvar))
        return p_kw, q_kvar, s_kva

    # Some SMART-DS base_timeseries folders may have no PVSystem objects.
    try:
        pvsystem_names = dss.PVsystems.AllNames()
    except Exception:
        pvsystem_names = []

    if not pvsystem_names:
        return pd.DataFrame()

    pv_data = []
    default_coordinates = (0, 0)

    for pv_name in pvsystem_names:
        try:
            dss.PVsystems.Name(pv_name)
            dss.Circuit.SetActiveElement(f"PVSystem.{pv_name}")
        except Exception as e:
            print(
                f"Warning: skipping PVSystem {pv_name} because it could not be "
                f"activated. Details: {e}"
            )
            continue

        # -----------------------------
        # Bus / coordinate information
        # -----------------------------
        try:
            bus_names = dss.CktElement.BusNames()
        except Exception:
            bus_names = []

        bus_full = bus_names[0] if len(bus_names) > 0 else None
        bus_name = bus_full.split(".")[0] if bus_full is not None else None

        coordinates = dict_bus_coord.get(bus_name, None) if bus_name is not None else None

        if coordinates is None:
            if bus_name is not None:
                print(
                    f"Bus {bus_name} was not found in Buscoords.dss, "
                    f"so PVSystem {pv_name} was assigned the previous/default coordinates"
                )
            coordinates = default_coordinates
        else:
            default_coordinates = coordinates

        # -----------------------------
        # Basic circuit properties
        # -----------------------------
        try:
            num_phases = dss.CktElement.NumPhases()
        except Exception:
            num_phases = None

        try:
            powers = dss.CktElement.Powers()
        except Exception:
            powers = []

        p_kw_terminal1, q_kvar_terminal1, s_kva_terminal1 = _terminal_power_from_powers(
            powers=powers,
            num_phases=num_phases if num_phases is not None else 0,
            terminal=1,
        )

        # -----------------------------
        # Static or quasi-static PVSystem properties
        # -----------------------------
        kv = _safe_float_prop("kv")
        kva = _safe_float_prop("kVA")
        pmpp = _safe_float_prop("Pmpp")
        irradiance = _safe_float_prop("irradiance")
        pf = _safe_float_prop("pf")

        # -----------------------------
        # Minimal output
        # -----------------------------
        row = {
            "Weather year": weather_year,
            "month": m,
            "day": d,
            "hour": h,
            "row_i": row_i,

            "PVSystem": pv_name,
            "Bus": bus_name,
            "Lat": _round_or_none(coordinates[1]),
            "Long": _round_or_none(coordinates[0]),
            "Num phases": num_phases,

            "kV rating [kV]": _round_or_none(kv),
            "kVA rating [kVA]": _round_or_none(kva),
            "Pmpp [kW]": _round_or_none(pmpp),
            "Irradiance [pu]": _round_or_none(irradiance),
            "PF": _round_or_none(pf),

            "P terminal 1 [kW]": _round_or_none(p_kw_terminal1),
            "Q terminal 1 [kvar]": _round_or_none(q_kvar_terminal1),
            "S terminal 1 [kVA]": _round_or_none(s_kva_terminal1),
        }

        # -----------------------------
        # Full output: extra diagnostics
        # -----------------------------
        if extraction_mode == "full":
            pct_pmpp = _safe_float_prop("%Pmpp")
            pct_cut_in = _safe_float_prop("%Cutin")
            pct_cut_out = _safe_float_prop("%Cutout")

            try:
                currents_mag_ang = dss.CktElement.CurrentsMagAng()
                current_magnitudes = currents_mag_ang[::2]
                max_current_a = (
                    max(current_magnitudes) if len(current_magnitudes) > 0 else None
                )
            except Exception:
                current_magnitudes = []
                max_current_a = None

            try:
                voltages_mag_ang = dss.CktElement.VoltagesMagAng()
                voltage_magnitudes_v = voltages_mag_ang[::2]
                max_voltage_v = (
                    max(voltage_magnitudes_v) if len(voltage_magnitudes_v) > 0 else None
                )
                min_voltage_v = (
                    min(voltage_magnitudes_v) if len(voltage_magnitudes_v) > 0 else None
                )
            except Exception:
                voltage_magnitudes_v = []
                max_voltage_v = None
                min_voltage_v = None

            try:
                losses_w, losses_var = dss.CktElement.Losses()
                losses_kw = losses_w / 1000
                losses_kvar = losses_var / 1000
            except Exception:
                losses_kw = None
                losses_kvar = None

            row.update({
                "Bus full": bus_full,

                "%Pmpp": _round_or_none(pct_pmpp),
                "%Cutin": _round_or_none(pct_cut_in),
                "%Cutout": _round_or_none(pct_cut_out),
                "daily": _safe_prop("daily"),
                "yearly": _safe_prop("yearly"),
                "duty": _safe_prop("duty"),
                "EffCurve": _safe_prop("EffCurve"),
                "P-TCurve": _safe_prop("P-TCurve"),

                "Losses [kW]": _round_or_none(losses_kw),
                "Losses [kvar]": _round_or_none(losses_kvar),
                "Max current [A]": _round_or_none(max_current_a),
                "Min voltage [V]": _round_or_none(min_voltage_v),
                "Max voltage [V]": _round_or_none(max_voltage_v),

                "Voltage magnitudes [V]": [
                    _round_or_none(v) for v in voltage_magnitudes_v
                ],
                "Current magnitudes [A]": [
                    _round_or_none(i) for i in current_magnitudes
                ],
            })

        pv_data.append(row)

    df = pd.DataFrame(pv_data)

    if enable_low_memory:
        dtype_map = {
            "month": "uint8",
            "day": "uint8",
            "hour": "uint8",
            "Lat": "float32",
            "Long": "float32",
            "Num phases": "Int32",
            "kV rating [kV]": "float32",
            "kVA rating [kVA]": "float32",
            "Pmpp [kW]": "float32",
            "Irradiance [pu]": "float32",
            "PF": "float32",
            "P terminal 1 [kW]": "float32",
            "Q terminal 1 [kvar]": "float32",
            "S terminal 1 [kVA]": "float32",
        }

        if extraction_mode == "full":
            dtype_map.update({
                "%Pmpp": "float32",
                "%Cutin": "float32",
                "%Cutout": "float32",
                "Losses [kW]": "float32",
                "Losses [kvar]": "float32",
                "Max current [A]": "float32",
                "Min voltage [V]": "float32",
                "Max voltage [V]": "float32",
            })

        df = df.astype(dtype_map)

    return df
    
def update_loadsShapes_file(new_loadsShapes_file, climate_scenario, TGW_city, weather_year):
    """
    Creates a modified copy of a Loadsshapes.dss file with updated paths to csv multiplier profiles

    Args:
        folder_path (str): Path to the LoadsShapes.dss file.
        climate_scenario, TGW_city, weather_year - strings that define where the csv multiplier profiles are saved

    Returns:
        None
    """
    # Read the file content
    with open(new_loadsShapes_file, 'r') as file:
        lines = file.readlines()
    # Modify each line that contains 'profiles'
    modified_lines = [line.replace('profiles', f'profiles/{climate_scenario}/{TGW_city}/{weather_year}') for line in lines]
    
    # Save the modified content back to the file
    with open(new_loadsShapes_file, 'w') as file:
        file.writelines(modified_lines)
    
# Function extract_line_information() extracts line information from activated circuit
# Input: Dictionary of busses and coordinates (+ an opendss circuit should be activated beforehand) and weather year
# Output: a dataframe with information of all lines in the activated circuit 
def extract_line_information(dict_bus_coord, weather_year,m,d,h,row_i,enable_low_memory=True):
    line_names = dss.Lines.AllNames()
    line_data = []
    for line in line_names:
        dss.Lines.Name(line)              
        dss.Circuit.SetActiveElement(f"line.{line}")  # Set the line as the active element
        # Get the names of the from and to buses
        buses = dss.CktElement.BusNames()
        if len(buses) < 2:
            print(f"Warning: Line {line} does not have two connected buses.")
            continue
        frombus = buses[0].split('.')[0]  # Extract bus name without phase information
        tobus = buses[1].split('.')[0]    # Extract bus name without phase information
               
        # Get kVBase for both buses
        try:
            dss.Circuit.SetActiveBus(frombus) # ToBus of Switch Disconnect that connects bus to bus can't be activated so we skip these lines switches
            kVBase_frombus = dss.Bus.kVBase()
            if kVBase_frombus == 0.0:
                raise ValueError(f"Invalid kVBase for from bus {frombus}")
            dss.Circuit.SetActiveBus(tobus)
            kVBase_tobus = dss.Bus.kVBase()
            if kVBase_tobus == 0.0:
                raise ValueError(f"Invalid kVBase for to bus {tobus}")     
        except (dss.DSSException, ValueError) as e:
#             print(f"Warning: Skipping line {line} due to issue with buses. Details: {e}")
            continue
         
        # Get the line coordinates
        From_coordinates = dict_bus_coord.get(frombus, None) 
        To_coordinates = dict_bus_coord.get(tobus, None) 
        
        # Assign alternative coordinates if a bus is not found in the coordinates dictionary 
        if From_coordinates == None:
            print(f'Bus {frombus} was not found in	Buscoords.dss so bus {frombus} was assigned the same coordinates as bus {tobus}')
            From_coordinates = To_coordinates
        if To_coordinates == None:
            print(f'Bus {tobus} was not found in	Buscoords.dss so bus {tobus} was assigned the same coordinates as bus {frombus}')
            To_coordinates = From_coordinates

        # Get current magnitudes from even indices
        currents_mag_ang = dss.CktElement.CurrentsMagAng() # current magnitudes and angles
        current_magnitudes = currents_mag_ang[::2]  # Extract only magnitudes (even indices)

        # Compute max line phase loading
        loading = 100 * max(current / dss.Lines.NormAmps() for current in current_magnitudes) 
  
        line_data.append({
            "Weather year": weather_year,
            "month": m,
            "day": d,
            "hour": h,
            "row_i":row_i,
            "Line": line,
            "LineCode": dss.Lines.LineCode(),
            "Length [km]": dss.Lines.Length(),
            "From_Lat": From_coordinates[1],
            "To_Lat": To_coordinates[1],
            "From_Long": From_coordinates[0],
            "To_Long": To_coordinates[0],
            "NormAmps [A]": dss.Lines.NormAmps(),
            "Nominal V [kV]": kVBase_frombus,
            "Voltage [pu]": dss.CktElement.VoltagesMagAng()[0]/(1000*kVBase_tobus),
            "Loading [%]": loading,
        })

    df = pd.DataFrame(line_data)

    if enable_low_memory:
        df = df.astype({
            "month": "uint8",
            "day": "uint8",
            "hour": "uint8",
            "Length [km]": "float32",
            "From_Lat": "float32",
            "To_Lat": "float32",
            "From_Long": "float32",
            "To_Long": "float32",
            "NormAmps [A]": "float32",
            "Nominal V [kV]": "float32",
            "Voltage [pu]": "float32",
            "Loading [%]": "float32",
        })

    return df

# Function extract_transformer_information() extracts Transformer information from activated circuit 
# Input:  Dictionary of busses and coordinates (+ an opendss circuit should be activated beforehand) and weather year
# Output: a dataframe with information of all transformers in the activated circuit 
def extract_transformer_information(dict_bus_coord, weather_year,m,d,h,row_i,enable_low_memory=True):
    transformer_names = dss.Transformers.AllNames()
    transformer_data = []
    default_coordinates = (0, 0)
    for transformer in transformer_names:
        dss.Transformers.Name(transformer)

        num_windings = dss.Transformers.NumWindings() # Number of windings 
        
        # Activate the CktElement for the current transformer
        dss.Circuit.SetActiveElement(f"Transformer.{transformer}")
        
        # Get the bus names connected to the transformer
        bus_names = dss.CktElement.BusNames()  # List of bus names for all windings
        
        transformer_bus_name = bus_names[0].split('.')[0]
        
        # Get the bus coordinates
        coordinates = dict_bus_coord.get(transformer_bus_name, None) 
        
                # Assign alternative coordinates if a bus is not found in the coordinates dictionary 
        if coordinates == None:
            print(f'Bus {transformer_bus_name} was not found in	Buscoords.dss so it was assigned the coordinates of the previous bus')
            coordinates = default_coordinates
        else:
            default_coordinates = coordinates
        
        # Get power values for all terminals
        powers = dss.CktElement.Powers()
        
        # Get the number of phases from the active element
        num_phases = dss.CktElement.NumPhases()
        
        max_loading = 0  # Initialize the max loading for the transformer
        kV_rating = []   # Initialize kV rating in each winding        
        
        powers = dss.CktElement.Powers()  # [P_ph1_winding1, Q_ph1_winding1, P_nuetral_winding1, Q2_winding1, ...] P and Q per winding and phase (+neutral)
              
        kVA_rating = dss.Transformers.kVA() # kVA of winding
                              
        for winding in range(1, num_windings + 1):  # Loop through each winding 
            dss.Transformers.Wdg(winding)  # Select the winding
            kV_rating += [dss.Transformers.kV()]  # Get the voltage of the winding           
        
        # Compute the total apparent power on winding 1 (sum of phases)
        winding1_power = 0
        for i in range(num_phases):
            real_power = powers[2 * i]  # Real power (kW) for phase i
            reactive_power = powers[2 * i + 1]  # Reactive power (kvar) for phase i
            winding1_power += abs(complex(real_power, reactive_power))
            
        # Calculate transformer loading percentage on winding 1
        loading_percent = (winding1_power / (kVA_rating)) * 100

        losses = dss.CktElement.Losses()  # Losses in watts
        losses_kW = [loss / 1000 for loss in losses]  # Convert to kW
        
        voltages = dss.CktElement.VoltagesMagAng()
        voltages_kV = [voltage / 1000 for voltage in voltages]  # Convert to kV
        
        transformer_data.append({
            "Weather year": weather_year,
            "month": m,
            "day": d,
            "hour": h,
            "row_i":row_i,
            "Transformer": transformer,
            "Bus": bus_names[0].split('.')[0],
            "Lat":coordinates[1],
            "Long":coordinates[0],
            "Num windings": num_windings,
            "Num phases": num_phases,
            "KV rating [kV]": kV_rating,
            "kVA rating [kVA]": kVA_rating,
            "Loading [%]": loading_percent,
            "Voltages [kV]": voltages_kV,
        })

    df = pd.DataFrame(transformer_data)

    if enable_low_memory:
        df = df.astype({
            "month": "uint8",
            "day": "uint8",
            "hour": "uint8",
            "Lat": "float32",
            "Long": "float32",
            "Num windings": "uint8",
            "Num phases": "uint8",
            "kVA rating [kVA]": "float32",
            "Loading [%]": "float32",
        })

    return df