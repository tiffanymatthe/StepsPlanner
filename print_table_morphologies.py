import pandas as pd
import numpy as np
from bottleneck import nanmean
import warnings
warnings.filterwarnings("ignore")

# List of columns to compute metrics for
columns_to_plot = ['curriculum_metric', 'timing_met', 'heading_err', 'dist_err']

# Number of behavior curricula and curricula per behavior curriculum
num_behavior_curricula = 10
num_curricula = 10

folders = ["no_distill_data_all", "mike_final_policy_with_heading"]

output_string = ""

behavior_curriculum_all = list(range(num_behavior_curricula))
# swap for one step plant and transition/combine all
behavior_curriculum_all[9], behavior_curriculum_all[8] = behavior_curriculum_all[8], behavior_curriculum_all[9]

for bi, behavior_curriculum in enumerate(behavior_curriculum_all):
    output_string += f"\n{bi}"

    for column_idx, column_to_plot in enumerate(columns_to_plot):
        means_none_nan = {folder: [] for folder in folders}
        means_timing_nan = {folder: [] for folder in folders}
        means_heading_nan = {folder: [] for folder in folders}
        means_both_nan = {folder: [] for folder in folders}

        for i, folder in enumerate(folders):
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

                    if column_to_plot == "curriculum_metric":
                        data_none_nan = np.exp(np.log(data_none_nan / 20) / 20)
                        data_timing_nan = np.exp(np.log(data_timing_nan / 20) / 20)
                        data_heading_nan = np.exp(np.log(data_heading_nan / 20) / 20)
                        data_both_nan = np.exp(np.log(data_both_nan / 20) / 20)

                    # Calculate mean for each subset
                    means_none_nan[folder].append(data_none_nan[column_to_plot].mean())
                    means_timing_nan[folder].append(data_timing_nan[column_to_plot].mean())
                    means_heading_nan[folder].append(data_heading_nan[column_to_plot].mean())
                    means_both_nan[folder].append(data_both_nan[column_to_plot].mean())

                except FileNotFoundError:
                    print(f"File {file} not found. Skipping.")
                    continue

        # Calculate the weighted averages
        weights = np.linspace(1, 10, num_curricula)
        weights /= sum(weights)

        avg_none_nan = [
            np.average(np.ma.masked_array(means_none_nan[folder], np.isnan(means_none_nan[folder])), weights=weights)
            for folder in folders
        ]
        avg_timing_nan = [
            np.average(np.ma.masked_array(means_timing_nan[folder], np.isnan(means_timing_nan[folder])), weights=weights)
            for folder in folders
        ]
        avg_heading_nan = [
            np.average(np.ma.masked_array(means_heading_nan[folder], np.isnan(means_heading_nan[folder])), weights=weights)
            for folder in folders
        ]
        avg_both_nan = [
            np.average(np.ma.masked_array(means_both_nan[folder], np.isnan(means_both_nan[folder])), weights=weights)
            for folder in folders
        ]
        avgs = []
        for i, folder in enumerate(folders):
            avgs.append(nanmean([avg_none_nan[i], avg_timing_nan[i], avg_heading_nan[i], avg_both_nan[i]]))

        # Append metrics for the current column to the output string
        output_string += "".join([f" & {avgs[i]:.2f}" for i in range(len(folders))])

    # Add LaTeX line break
    output_string += " \\\\"
print(output_string)
