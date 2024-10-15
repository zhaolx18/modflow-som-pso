import os
import shutil
import numpy as np
import csv
import configparser
import subprocess
from pyswarm import pso 

def clear_run_directory(run_dir):

    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)

def read_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    if 'LPF' not in config:
        raise ValueError("config should have 'LPF' ")

    lpf_ranges = {
        'perm': {f'zone{zone}': tuple(map(float, config.get('LPF', f'perm_zone{zone}_range').split(','))) for zone in range(1, 7)},
        'stor': {f'zone{zone}': tuple(map(float, config.get('LPF', f'stor_zone{zone}_range').split(','))) for zone in range(1, 7)}
    }

    if 'RIV' not in config:
        raise ValueError("config should have 'RIV' ")

    riv_ranges = []
    for key, value in config['RIV'].items():
        parts = value.split()
        if len(parts) == 3:
            start, end = map(int, parts[0].split('-'))  
            min_val, max_val = map(float, parts[1:])    
            riv_ranges.append((start, end, min_val, max_val))  

    return lpf_ranges, riv_ranges

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
    result = '  '  
    for i, num in enumerate(numbers):
        if i > 0:
            prev = numbers[i - 1]
            if num == 0.0 and prev == 0.0:
                result += '   '  
            elif prev == 0.0 and num != 0.0:
                result += ' ' 
            elif num == 0.0 and prev != 0.0:
                result += '   '  
            else:
                result += ' '  
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
            elif prev == 0.0 and float(num) >= 0.11:
                result += ' '  
            else:
                result += ' ' 
        result += f"{num:.2f}" if num != 0 else "0.0"
    return result

def modify_lpf_file(lpf_path, params, perm_zones, stor_zones):
    with open(lpf_path, 'r') as file:
        lines = file.readlines()

    apply_zone_values(lines, 7, 75, params['perm'], perm_zones, float, permeability_line_format)
    apply_zone_values(lines, 80, 148, params['stor'], stor_zones, float, storage_line_format)

    with open(lpf_path, 'w') as file:
        file.writelines(lines)

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
        if j == 4:  # Ensure 13 spaces between the end of the fourth column and the start of the fifth column
            last_col_end = spaces_format[line_index][j - 1] + len(columns[j - 1])
            current_col_start = len(formatted_line) + len(col)
            spaces_needed = 13 - (current_col_start - last_col_end)
            formatted_line += ' ' * spaces_needed + col
        elif j == 0:
            formatted_line += col
        else:
            formatted_line += ' ' * (spaces_format[line_index][j] - len(formatted_line)) + col
    return formatted_line

def modify_riv_file(riv_path, ranges, sample_values):
    with open(riv_path, 'r') as riv_file:
        lines = riv_file.readlines()

    spaces_format = [get_spaces_format(line) for line in lines]
    
    for i, (start, end, min_val, max_val) in enumerate(ranges):
        sample_value = sample_values[i]
        for j in range(start - 1, end):
            columns = lines[j].split()
            if columns[4].replace('.', '', 1).isdigit():
                original_value = float(columns[4])
                new_value = original_value * sample_value
                formatted_value = f'{new_value:.4f}'
                columns[4] = formatted_value
                lines[j] = format_columns(columns, spaces_format, j) + '\n'

    with open(riv_path, 'w') as riv_file:
        riv_file.writelines(lines)

def parse_output(output_path):
    with open(output_path, 'r') as file:
        for line in file:
            if "SUM OF SQUARED WEIGHTED RESIDUALS (HEADS ONLY)" in line:
                return float(line.strip().split()[-1])
    return None

def objective_function(params, raw_model_dir, config_path, sample_index):
    lpf_ranges, riv_ranges = read_config(config_path)

    perm_samples = {
        'zone1': params[0], 'zone2': params[1], 'zone3': params[2],
        'zone4': params[3], 'zone5': params[4], 'zone6': params[5]
    }
    stor_samples = {
        'zone1': params[6], 'zone2': params[7], 'zone3': params[8],
        'zone4': params[9], 'zone5': params[10], 'zone6': params[11]
    }
    riv_samples = params[12:15]

    run_folder = os.path.join("F:\\MODFLOW-github\\modflow-som-pso", "run", f'run_{sample_index}')
    shutil.copytree(raw_model_dir, run_folder, dirs_exist_ok=True)

    lpf_path = os.path.join(run_folder, "Taoerhe.lpf")
    riv_path = os.path.join(run_folder, "Taoerhe.riv")

    perm_zones = {210: 'zone1', 220: 'zone2', 230: 'zone3', 240: 'zone4', 250: 'zone5', 260: 'zone6'}
    stor_zones = {0.11: 'zone1', 0.12: 'zone2', 0.13: 'zone3', 0.14: 'zone4', 0.15: 'zone5', 0.16: 'zone6'}
    modify_lpf_file(lpf_path, {'perm': perm_samples, 'stor': stor_samples}, perm_zones, stor_zones)

    modify_riv_file(riv_path, riv_ranges, riv_samples)

    subprocess.run(['mf2k_h5_parallel.exe', 'Taoerhe.mfn'], cwd=run_folder)

    output_path = os.path.join(run_folder, 'Taoerhe.out')
    return parse_output(output_path)

np.random.seed(42)  

def pso_optimization(base_dir, raw_model_dir, config_path):
    lpf_ranges, riv_ranges = read_config(config_path)

    lower_bounds = []
    upper_bounds = []

    for zone_ranges in lpf_ranges['perm'].values():
        lower_bounds.append(zone_ranges[0])
        upper_bounds.append(zone_ranges[1])

    for zone_ranges in lpf_ranges['stor'].values():
        lower_bounds.append(zone_ranges[0])
        upper_bounds.append(zone_ranges[1])

    for _, _, min_val, max_val in riv_ranges:
        lower_bounds.append(min_val)
        upper_bounds.append(max_val)

    results = []

    def objective(params):
        sample_index = len(results) + 1
        sswr = objective_function(params, raw_model_dir, config_path, sample_index)
        result = {'sample_index': sample_index}
        for i, val in enumerate(params):
            result[f'param_{i+1}'] = val
        result['SSWR'] = sswr
        results.append(result)
        return sswr

    clear_run_directory(os.path.join(base_dir, 'run'))

    np.random.seed(42)

    best_params, best_value = pso(
        func=objective,
        lb=lower_bounds, 
        ub=upper_bounds,
        swarmsize=5,
        maxiter=10,
        omega=0.7,   
        phip=2,    
        phig=1.2    
    )

    results_path = os.path.join(base_dir, 'pso_results.csv')
    with open(results_path, 'w', newline='') as csvfile:
        fieldnames = ['sample_index'] + [f'param_{i+1}' for i in range(len(lower_bounds))] + ['SSWR']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    return best_params, best_value


if __name__ == "__main__":
    base_dir = "F:\\MODFLOW-github\\modflow-som-pso"
    raw_model_dir = os.path.join(base_dir, "RAWmodel")
    config_path = os.path.join(base_dir, "config.ini")

    best_params, best_value = pso_optimization(base_dir, raw_model_dir, config_path)
    print(f"The best parameter set: {best_params}, Min. SSWR: {best_value}")
