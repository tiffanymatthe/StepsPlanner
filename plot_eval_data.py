import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bottleneck import nanmean, nansum

import warnings
warnings.filterwarnings("ignore")

from matplotlib import rc

# Use Computer Modern Roman as the default font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)

plt.rcParams.update({'font.size': 16})

# Specify the column to plot
column_to_plot = 'curriculum_metric' # 'curriculum_metric'  # Change to 'timing_met', 'heading_err', 'dist_err', or 'curriculum_metric' as needed

# Number of behavior curricula and curricula per behavior curriculum
num_behavior_curricula = 10
num_curricula = 10

# Prepare subplots
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 7))

# Initialize lists to store the calculated mean and std values for global y-axis scaling
all_means = []
all_stds = []

folders = ["reset_all_with_heading"]
# folders = ["all_with_hopping"]
folder_labels=["reset_all"] #, "..."]

for behavior_curriculum in range(num_behavior_curricula):
    all_means_for_task = {}
    for i, folder in enumerate(folders):

        means_none_nan = []
        stds_none_nan = []
        means_timing_nan = []
        stds_timing_nan = []
        means_heading_nan = []
        stds_heading_nan = []
        means_both_nan = []
        stds_both_nan = []
        curricula = list(range(num_curricula))

        curriculum_to_plot = []

        for curriculum in curricula:
            file = f"{folder}/data_{behavior_curriculum}_{curriculum}.csv"
            
            try:
                data = pd.read_csv(file)
                
                # Separate the data into the four cases
                data_none_nan = data[data['timing_met'].notna() & data['heading_err'].notna()][column_to_plot]
                data_timing_nan = data[data['timing_met'].isna() & data['heading_err'].notna()][column_to_plot]
                data_heading_nan = data[data['timing_met'].notna() & data['heading_err'].isna()][column_to_plot]
                data_both_nan = data[data['timing_met'].isna() & data['heading_err'].isna()][column_to_plot]

                if column_to_plot == "curriculum_metric":
                    # data_none_nan = np.exp(np.log(np.array(data_none_nan) / 20) / 20)
                    # data_timing_nan = np.exp(np.log(np.array(data_timing_nan) / 20) / 20)
                    # data_heading_nan = np.exp(np.log(np.array(data_heading_nan) / 20) / 20)
                    # data_both_nan = np.exp(np.log(np.array(data_both_nan) / 20) / 20)
                    data_none_nan = 1/(data_none_nan + 1)
                    data_timing_nan = 1/(data_timing_nan + 1)
                    data_heading_nan = 1/(data_heading_nan + 1)
                    data_both_nan = 1/(data_both_nan + 1)

                # Calculate mean and std for each subset
                mean_none_nan = data_none_nan.mean()
                std_none_nan = data_none_nan.std()
                means_none_nan.append(mean_none_nan)
                stds_none_nan.append(std_none_nan)
                all_means.append(mean_none_nan)
                all_stds.append(std_none_nan)

                mean_timing_nan = data_timing_nan.mean()
                std_timing_nan = data_timing_nan.std()
                means_timing_nan.append(mean_timing_nan)
                stds_timing_nan.append(std_timing_nan)
                all_means.append(mean_timing_nan)
                all_stds.append(std_timing_nan)

                mean_heading_nan = data_heading_nan.mean()
                std_heading_nan = data_heading_nan.std()
                means_heading_nan.append(mean_heading_nan)
                stds_heading_nan.append(std_heading_nan)
                all_means.append(mean_heading_nan)
                all_stds.append(std_heading_nan)

                mean_both_nan = data_both_nan.mean()
                std_both_nan = data_both_nan.std()
                means_both_nan.append(mean_both_nan)
                stds_both_nan.append(std_both_nan)
                all_means.append(mean_both_nan)
                all_stds.append(std_both_nan)

                curriculum_to_plot.append(curriculum)
            except FileNotFoundError:
                print(f"File {file} not found. Skipping.")
                continue

        ax = axes.flatten()[behavior_curriculum]
        
        weights = np.linspace(1,10,len(curriculum_to_plot))
        weights /= sum(weights)

        all_means_for_task[i] = means_none_nan

        ax.errorbar(curriculum_to_plot, means_none_nan, yerr=stds_none_nan, fmt='-o', label=f'(1,1)') # - {folder_labels[i]}')
        ax.errorbar(curriculum_to_plot, means_timing_nan, yerr=stds_timing_nan, fmt='-x', label=f'(0,1)') # - {folder_labels[i]}')
        ax.errorbar(curriculum_to_plot, means_heading_nan, yerr=stds_heading_nan, fmt='-s', label='(1,0)')
        ax.errorbar(curriculum_to_plot, means_both_nan, yerr=stds_both_nan, fmt='-d', label='(0,0)')

    # # 0 = no_plasticity
    # differences = [a - b for a, b in zip(all_means_for_task[1], all_means_for_task[0])]
    # # Compute the average difference
    # average_difference = sum(differences) / len(differences)
    # print(f"{behavior_curriculum}: {average_difference} and {sum([1 for x in differences if x > 0]) / len(differences)}")


    ax.set_title(f"Task {behavior_curriculum}")
    ax.set_xlabel("Curriculum", fontsize=14)
    ax.set_xticks(range(10))
    if behavior_curriculum == 0 or behavior_curriculum == 5:
        ax.set_ylabel("Step Failure Probability", fontsize=14)

# Calculate global y-axis limits based on mean ± std ranges
global_min = min(np.array(all_means) - np.array(all_stds))
global_max = max(np.array(all_means) + np.array(all_stds))

# Set the same y-axis limits for all subplots
# if column_to_plot == "curriculum_metric":
#     for ax in axes.flat:
#         # global_min = 0
#         global_max = 1

for ax in axes.flat:
    ax.set_ylim(global_min, global_max)

# Add a legend to the first subplot
axes[0, 0].legend(loc='lower right')

# Adjust layout and show plot
plt.tight_layout()
# plt.suptitle(f"{column_to_plot} across Curricula for Each Behavior Curriculum", y=1.02)

img_path = f"{folders[0]}/results_{column_to_plot}.png"
plt.savefig(img_path)
plt.show()
