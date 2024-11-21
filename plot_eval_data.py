import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Specify the column to plot
column_to_plot = 'curriculum_metric'  # Change to 'timing_met', 'heading_err', 'dist_err', or 'curriculum_metric' as needed

# Number of behavior curricula and curricula per behavior curriculum
num_behavior_curricula = 10
num_curricula = 10

# Prepare subplots
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))

# Initialize lists to store the calculated mean and std values for global y-axis scaling
all_means = []
all_stds = []

folder = "experts"

for behavior_curriculum in range(num_behavior_curricula):
    means_none_nan = []
    stds_none_nan = []
    means_timing_nan = []
    stds_timing_nan = []
    means_heading_nan = []
    stds_heading_nan = []
    means_both_nan = []
    stds_both_nan = []
    curricula = list(range(num_curricula))

    for curriculum in curricula:
        file = f"{folder}/data_{behavior_curriculum}_{curriculum}.csv"
        
        try:
            data = pd.read_csv(file)
            
            # Separate the data into the four cases
            data_none_nan = data[data['timing_met'].notna() & data['heading_err'].notna()]
            data_timing_nan = data[data['timing_met'].isna() & data['heading_err'].notna()]
            data_heading_nan = data[data['timing_met'].notna() & data['heading_err'].isna()]
            data_both_nan = data[data['timing_met'].isna() & data['heading_err'].isna()]

            # Calculate mean and std for each subset
            mean_none_nan = data_none_nan[column_to_plot].mean()
            std_none_nan = data_none_nan[column_to_plot].std()
            means_none_nan.append(mean_none_nan)
            stds_none_nan.append(std_none_nan)
            all_means.append(mean_none_nan)
            all_stds.append(std_none_nan)

            mean_timing_nan = data_timing_nan[column_to_plot].mean()
            std_timing_nan = data_timing_nan[column_to_plot].std()
            means_timing_nan.append(mean_timing_nan)
            stds_timing_nan.append(std_timing_nan)
            all_means.append(mean_timing_nan)
            all_stds.append(std_timing_nan)

            mean_heading_nan = data_heading_nan[column_to_plot].mean()
            std_heading_nan = data_heading_nan[column_to_plot].std()
            means_heading_nan.append(mean_heading_nan)
            stds_heading_nan.append(std_heading_nan)
            all_means.append(mean_heading_nan)
            all_stds.append(std_heading_nan)

            mean_both_nan = data_both_nan[column_to_plot].mean()
            std_both_nan = data_both_nan[column_to_plot].std()
            means_both_nan.append(mean_both_nan)
            stds_both_nan.append(std_both_nan)
            all_means.append(mean_both_nan)
            all_stds.append(std_both_nan)

        except FileNotFoundError:
            print(f"File {file} not found. Skipping.")
            continue

    # Plot each subset on the same subplot for the current behavior curriculum
    ax = axes.flatten()[behavior_curriculum]
    ax.errorbar(curricula, means_none_nan, yerr=stds_none_nan, fmt='-o', label='None NaN')
    ax.errorbar(curricula, means_timing_nan, yerr=stds_timing_nan, fmt='-x', label='Timing NaN')
    ax.errorbar(curricula, means_heading_nan, yerr=stds_heading_nan, fmt='-s', label='Heading NaN')
    ax.errorbar(curricula, means_both_nan, yerr=stds_both_nan, fmt='-d', label='Both NaN')

    ax.set_title(f"Behavior Curriculum {behavior_curriculum}")
    ax.set_xlabel("Curriculum")
    ax.set_ylabel(column_to_plot)

# Calculate global y-axis limits based on mean ± std ranges
global_min = min(np.array(all_means) - np.array(all_stds))
global_max = max(np.array(all_means) + np.array(all_stds))

# Set the same y-axis limits for all subplots
for ax in axes.flat:
    global_min = 0
    global_max = 20
    ax.set_ylim(global_min, global_max)

# Add a legend to the first subplot
axes[0, 0].legend(loc='upper right')

# Adjust layout and show plot
plt.tight_layout()
plt.suptitle(f"{column_to_plot} across Curricula for Each Behavior Curriculum", y=1.02)
import os
img_path = f"{folder}/results.png"
plt.savefig(img_path)
plt.show()
