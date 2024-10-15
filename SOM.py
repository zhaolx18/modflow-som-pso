import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# Specify the result directory
result_directory = 'F:\\MODFLOW-github\\modflow-som-pso'
# Load data from Excel file
file_path = os.path.join(result_directory, 'results.xlsx')
data = pd.read_excel(file_path, sheet_name=0, usecols='B:Q', nrows=1000)  

# Normalize the data
data_normalized = (data.iloc[:, :-1] - data.iloc[:, :-1].min()) / (data.iloc[:, :-1].max() - data.iloc[:, :-1].min())

# Define the starting and ending grid sizes for the SOM
start_grid_size = 3
end_grid_size = 6

# Store the results for different grid sizes
results = []
command_line_outputs = []  # Store command line output for saving in the Excel file

# Loop through different rectangular and square grid sizes to calculate the Silhouette score and SSWR average
for grid_x in range(start_grid_size, end_grid_size + 1):
    for grid_y in range(grid_x, end_grid_size + 1): 
        print(f"Calculating {grid_x}x{grid_y} grid...")

        # Initialize and train the Self-Organizing Map (SOM)
        som = MiniSom(x=grid_x, y=grid_y, input_len=data_normalized.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
        som.random_weights_init(data_normalized.values)
        som.train_batch(data_normalized.values, 2000)

        # Get the U-matrix for distance calculations
        u_matrix = som.distance_map()

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(u_matrix.T, cmap='coolwarm', annot=True, fmt='.3f', cbar=True, annot_kws={"size": 16}) 


        ax.set_xlabel('X Grid', fontsize=18)  
        ax.set_ylabel('Y Grid (Distance)', fontsize=18) 

        ax.tick_params(axis='x', labelsize=14)  
        ax.tick_params(axis='y', labelsize=14)  

 
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)  
        cbar.set_label('Distance', fontsize=16) 

        plt.show()

        grid_labels = []
        for i, x in enumerate(data_normalized.values):
            win_position = som.winner(x)
            grid_labels.append(f"grid({win_position[0]},{win_position[1]})")

        data_with_grid = data.assign(Grid=grid_labels)

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(grid_labels)

        silhouette_avg = silhouette_score(data_normalized, encoded_labels)

        min_distance = np.min(u_matrix)
        min_distance_position = np.unravel_index(np.argmin(u_matrix), u_matrix.shape)
        min_distance_grid = f"grid({min_distance_position[0]},{min_distance_position[1]})"

        # Calculate the SSWR average for the closest grid 
        min_distance_samples = data_with_grid[data_with_grid['Grid'] == min_distance_grid]
        q_mean = min_distance_samples['Q'].mean() if not min_distance_samples.empty else np.nan

        # Output grid's Silhouette score and SSWR average for the minimum distance grid
        output_text = (f"Grid Size: {grid_x}x{grid_y}, Silhouette Coefficient: {silhouette_avg}, "
                       f"Min Distance Grid: {min_distance_grid}, Min Distance: {min_distance}, Q Average: {q_mean}")
        print(output_text)

        # Append the output to the list for saving
        command_line_outputs.append(output_text)

        # Store the results for this grid size
        results.append({
            'Grid Size': f'{grid_x}x{grid_y}',
            'Silhouette Coefficient': silhouette_avg,
            'Min Distance': min_distance,
            'Min Distance Grid': min_distance_grid,
            'Min Distance Q Average': q_mean
        })

        # Save results to Excel immediately to avoid overwriting issues
        new_file_path = os.path.join(result_directory, f'SOM_{grid_x}x{grid_y}.xlsx')
        with pd.ExcelWriter(new_file_path, engine='openpyxl') as writer:
            # Save the Q averages for each grid
            averages = data_with_grid.groupby('Grid')['Q'].mean().reset_index()
            averages.to_excel(writer, sheet_name='Grid_Averages', index=False)

            # Save samples in each grid
            for grid in data_with_grid['Grid'].unique():
                grid_data = data_with_grid[data_with_grid['Grid'] == grid]
                grid_data.to_excel(writer, sheet_name=grid, index=False)

            # Save the min distance and Q average to the results sheet
            results_df = pd.DataFrame(results)
            results_df.to_excel(writer, sheet_name='Results_Summary', index=False)

            # Save the command line output to another sheet
            pd.DataFrame(command_line_outputs, columns=['Command_Line_Output']).to_excel(writer, sheet_name='Command_Line_Output', index=False)

# Display the final results
print("\nFinal results saved to the respective Excel files.")
