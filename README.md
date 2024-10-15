## modflow-som-pso
This repository is designed for parameter identification in MODFLOW, using the hypothetical Taoerhe area as a case study. It contains three Python scripts: RUN.py, SOM.py, and PSO.py.

## RUN.py
Main Functionality: Conducts Latin Hypercube Sampling for the parameters to be inverted in the study area. This example includes 15 parameters: K, Sy, and Cond.

Configuration File: combined_config.ini contains the parameter names and their value ranges.

Output Files:

Sampled parameter values and corresponding objective functions (result.csv)
Simulated water levels corresponding to each parameter set (first_columns_os.csv)
Additional output files related to 95 PPU calculations.
## SOM.py
Main Functionality: Performs two-dimensional Self-Organizing Map (SOM) clustering on the parameter sets.

Output Files: Clustering results at different grid sizes and optimized parameter ranges (SOM_n*n.xlsx).

## PSO.py
Main Functionality: Uses Particle Swarm Optimization (PSO) to estimate the optimal parameter sets for the study area.

Configuration File: config.ini contains the parameter names and their value ranges.

Output Files: Records of the PSO optimization results (pso_results.csv).

## For any inquiries, please contact: zlx22@mails.jlu.edu.cn