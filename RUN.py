import os
import random
import csv
import shutil
import numpy as np
import pandas as pd
import configparser
import subprocess


def read_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    if 'DEFAULT' not in config or 'sample_count' not in config['DEFAULT']:
        raise ValueError("Configuration file must conclude'sample_count' in 'DEFAULT' section")

    sample_count = config.getint('DEFAULT', 'sample_count')

    if 'LPF' not in config:
        raise ValueError("Configuration file must conclude'LPF' section")

    lpf_ranges = {
        'perm': {f'zone{zone}': tuple(map(float, config.get('LPF', f'perm_zone{zone}_range').split(','))) for zone in range(1, 7)},
        'stor': {f'zone{zone}': tuple(map(float, config.get('LPF', f'stor_zone{zone}_range').split(','))) for zone in range(1, 7)}
    }

    if 'RIV' not in config:
        raise ValueError("Configuration file must conclude'RIV' section")

    riv_ranges = []
    for key, value in config['RIV'].items():
        parts = value.split()
        if len(parts) == 3:
            start, end = map(int, parts[0].split('-'))
            min_val, max_val = map(float, parts[1:])
            riv_ranges.append((start, end, min_val, max_val))

    return sample_count, lpf_ranges, riv_ranges

def apply_zone_values(lines, start_index, end_index, samples, zone_definitions, dtype, line_format):
    for i in range(start_index, end_index + 1):
        line_values = [float(val) for val in lines[i].split()]
        for j, value in enumerate(line_values):
            for zone_code, zone_name in zone_definitions.items():
                if value == zone_code:
                    line_values[j] = dtype(samples[zone_name])
                    break
        lines[i] = line_format(line_values) + '\n'

def permeability_line_format(numbers):
    result = '  '  # Start with two spaces
    for i, num in enumerate(numbers):
        if i > 0:
            prev = numbers[i - 1]
            if num == 0.0 and prev == 0.0:
                result += '   '  # Three spaces between two '0.0'
            elif prev == 0.0 and num != 0.0:
                result += ' '  # One space if '0.0' followed by a number
            elif num == 0.0 and prev != 0.0:
                result += '   '  # Three spaces if number followed by '0.0'
            else:
                result += ' '  # One space between numbers
        result += f"{num:.1f}" if num != 0 else f"{num:.1f}"
    return result

def storage_line_format(numbers):
    result = ' ' 
    for i, num in enumerate(numbers):
        if i > 0:
            prev = numbers[i - 1]
            if num == 0.0 and prev == 0.0:
                result += '  ' 
            elif num == 0.0:
                result += '  '  
            elif float(prev) >= 0.11 and num == 0.0:
                result += '  '  
            elif prev == 0.0 and float(num) >= 0.11:
                result += ' '  
            else:
                result += ' '  
        result += f"{num:.2f}" if num != 0 else "0.0"
    return result

def modify_lpf_file(lpf_path, sample_index, ranges, perm_zones, stor_zones):
    zone_samples = {param: {zone: np.random.uniform(low, high) for zone, (low, high) in param_ranges.items()}
                    for param, param_ranges in ranges.items()}
    with open(lpf_path, 'r') as file:
        lines = file.readlines()
    apply_zone_values(lines, 7, 75, zone_samples['perm'], perm_zones, float, permeability_line_format)
    apply_zone_values(lines, 80, 148, zone_samples['stor'], stor_zones, float, storage_line_format)
    with open(lpf_path, 'w') as file:
        file.writelines(lines)
    return zone_samples

def get_spaces_format(line):
    parts = line.split()
    spaces = []
    start = 0
    for part in parts:
        start = line.find(part, start)
        spaces.append(start)
        start += len(part)
    return spaces

def format_columns(columns, spaces_format, line_index):
    formatted_line = "    "
    for j, col in enumerate(columns):
        if j == 4:  
            last_col_end = spaces_format[line_index][j - 1] + len(columns[j - 1])
            current_col_start = len(formatted_line) + len(col)
            spaces_needed = 13 - (current_col_start - last_col_end)
            formatted_line += ' ' * spaces_needed + col
        elif j == 0:
            formatted_line += col
        else:
            formatted_line += ' ' * (spaces_format[line_index][j] - len(formatted_line)) + col
    return formatted_line

def modify_riv_file(riv_path, ranges):
    with open(riv_path, 'r') as riv_file:
        lines = riv_file.readlines()
    spaces_format = [get_spaces_format(line) for line in lines]
    samples = []
    for start, end, min_val, max_val in ranges:
        sample_value = random.uniform(min_val, max_val)
        samples.append(sample_value)
        for i in range(start - 1, end):
            columns = lines[i].split()
            if columns[4].replace('.', '', 1).isdigit():
                original_value = float(columns[4])
                new_value = original_value * sample_value
                formatted_value = f'{new_value:.4f}'
                columns[4] = formatted_value
                lines[i] = format_columns(columns, spaces_format, i) + '\n'
    with open(riv_path, 'w') as riv_file:
        riv_file.writelines(lines)
    return samples

def parse_output(output_path):
    with open(output_path, 'r') as file:
        for line in file:
            if "SUM OF SQUARED WEIGHTED RESIDUALS (HEADS ONLY)" in line:
                return float(line.strip().split()[-1])
    return None

def clear_run_directory(base_dir):
    run_dir = os.path.join(base_dir, "run")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)

def extract_required_columns(base_dir, run_folders):
    columns_data = {'hed_name': []}
    max_length = 0
    
    for run_folder in run_folders:
        sample_index = int(run_folder.split('_')[-1])
        os_file_path = os.path.join(run_folder, 'Taoerhe._os')
        
        with open(os_file_path, 'r') as os_file:
            lines = os_file.readlines()
        
        first_column = []
        hed_names = []
        for line in lines:
            parts = line.split()
            if parts[3].startswith('hed'):
                hed_names.append(parts[3])
                first_column.append(parts[0])
        
        if sample_index == 1:
            columns_data['hed_name'] = hed_names
        
        columns_data[f'run_{sample_index}'] = first_column
        max_length = max(max_length, len(first_column))
    

    for key in columns_data:
        columns_data[key] += [''] * (max_length - len(columns_data[key]))
    
    df = pd.DataFrame(columns_data)
    df.to_csv(os.path.join(base_dir, 'first_columns_os.csv'), index=False)

def calculate_p_r_factors(simulated_df, observed_df, output_path):
    p_factors = []
    r_factors = []
    lower_bounds_all = []
    upper_bounds_all = []


    groups = [
        list(range(1, 13)),
        list(range(49, 61)),
        list(range(97, 109)),
        list(range(145, 157)),
        list(range(193, 205)),
        list(range(241, 253)),
        list(range(289, 301)),
        list(range(337, 349)),
        list(range(385, 397)),
    ]

    for group in groups:
        well_names = [f'hed{i}' for i in group]

        well_simulated_data = simulated_df[simulated_df['hed_name'].isin(well_names)]
        well_observed_data = observed_df[observed_df['hed_name'].isin(well_names)]

        if not well_simulated_data.empty and not well_observed_data.empty:
            observed_values = []
            simulated_values = []

            for well in well_names:
                observed_value = well_observed_data[well_observed_data['hed_name'] == well]['observed'].values[0]
                observed_values.append(observed_value)
                simulated_values.append(well_simulated_data[well_simulated_data['hed_name'] == well].iloc[:, 1:].values.flatten())

            observed_values = np.array(observed_values)
            simulated_values = np.array(simulated_values)


            lower_bounds = np.percentile(simulated_values, 2.5, axis=1)
            upper_bounds = np.percentile(simulated_values, 97.5, axis=1)

            lower_bounds_all.extend(lower_bounds)
            upper_bounds_all.extend(upper_bounds)


            p_factors_group = (observed_values >= lower_bounds) & (observed_values <= upper_bounds)
            p_factor = np.mean(p_factors_group)

            interval_widths = upper_bounds - lower_bounds
            mean_interval_width = np.mean(interval_widths)
            std_observed = np.std(observed_values)
            r_factor = mean_interval_width / std_observed


            well_name_range = f"{well_names[0]}-{well_names[-1]}"

            p_factors.append({'hed_name': well_name_range, 'p_factor': p_factor, 'r_factor': r_factor})

    ppu_df = pd.DataFrame(p_factors)
    ppu_df.to_csv(output_path, index=False)
    
    return ppu_df, lower_bounds_all, upper_bounds_all

def save_results_to_excel(ppu_df, lower_bounds, upper_bounds, output_path):
    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    ppu_df.to_excel(writer, sheet_name='P_R_Factors', index=False)

    num_wells = len(ppu_df)
    lower_bounds_splits = np.array_split(lower_bounds, num_wells)
    upper_bounds_splits = np.array_split(upper_bounds, num_wells)

    percentile_data = []
    for i, hed_name in enumerate(ppu_df['hed_name']):
        for lb, ub in zip(lower_bounds_splits[i], upper_bounds_splits[i]):
            percentile_data.append({'hed_name': hed_name, '2.5th Percentile': lb, '97.5th Percentile': ub})

    percentile_df = pd.DataFrame(percentile_data)
    percentile_df.to_excel(writer, sheet_name='Percentiles', index=False)

    writer.save()

def main(base_dir, raw_model_dir, config_path):
    clear_run_directory(base_dir)

    sample_count, lpf_ranges, riv_ranges = read_config(config_path)

    results_path = os.path.join(base_dir, "results.csv")
    results = []

    perm_zones = {210: 'zone1', 220: 'zone2', 230: 'zone3', 240: 'zone4', 250: 'zone5', 260: 'zone6'}
    stor_zones = {0.11: 'zone1', 0.12: 'zone2', 0.13: 'zone3', 0.14: 'zone4', 0.15: 'zone5', 0.16: 'zone6'}

    os.makedirs(os.path.join(base_dir, "run"), exist_ok=True)

    run_folders = []

    for sample_index in range(1, int(sample_count) + 1):
        run_folder = os.path.join(base_dir, "run", f'run_{sample_index}')
        os.makedirs(run_folder, exist_ok=True)
        shutil.copytree(raw_model_dir, run_folder, dirs_exist_ok=True)
        run_folders.append(run_folder)

        lpf_path = os.path.join(run_folder, "Taoerhe.lpf")
        riv_path = os.path.join(run_folder, "Taoerhe.riv")

        lpf_samples = modify_lpf_file(lpf_path, sample_index, lpf_ranges, perm_zones, stor_zones)
        riv_samples = modify_riv_file(riv_path, riv_ranges)

        subprocess.run(['mf2k_h5_parallel.exe', 'Taoerhe.mfn'], cwd=run_folder)

        output_path = os.path.join(run_folder, 'Taoerhe.out')
        residual = parse_output(output_path)

        result = {'Sample Index': sample_index}
        for param, zones in lpf_samples.items():
            for zone, value in zones.items():
                result[f'{param}_{zone}'] = value
        result.update({f'riv_sample_{i+1}': val for i, val in enumerate(riv_samples)})
        result['Residual'] = residual
        results.append(result)

    fieldnames = ['Sample Index'] + [f'{p}_zone{z}' for p in ['perm', 'stor'] for z in range(1, 7)] + [f'riv_sample_{i+1}' for i in range(len(riv_samples))] + ['Residual']
    with open(results_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    extract_required_columns(base_dir, run_folders)
    
    simulated_df = pd.read_csv(os.path.join(base_dir, 'first_columns_os.csv'))
    observed_df = pd.read_excel(os.path.join(base_dir, 'observed.xlsx'))
    
    ppu_df, lower_bounds, upper_bounds = calculate_p_r_factors(simulated_df, observed_df, os.path.join(base_dir, 'PPU.csv'))

    save_results_to_excel(ppu_df, lower_bounds, upper_bounds, os.path.join(base_dir, '2.5-97.5th.xlsx'))

if __name__ == "__main__":
    base_dir = "F:\\MODFLOW-github\\modflow-som-pso"
    raw_model_dir = os.path.join(base_dir, "RAWmodel")
    config_path = os.path.join(base_dir, "combined_config.ini")
    main(base_dir, raw_model_dir, config_path)
