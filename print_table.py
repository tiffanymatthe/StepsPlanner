import pandas as pd
import numpy as np
from bottleneck import nanmean, nansum
import warnings
warnings.filterwarnings("ignore")

# List of columns to compute metrics for
columns_to_plot = ['curriculum_metric'] #, 'timing_met', 'heading_err', 'dist_err']

# Number of behavior curricula and curricula per behavior curriculum
num_behavior_curricula = 12
num_curricula = 10

folders = ["no_plasticity", "reset_actor"] # "reset_all"]

output_string = ""
output_string_second_table = ""

behavior_curriculum_all = list(range(num_behavior_curricula))
# swap for one step plant and transition/combine all
# behavior_curriculum_all[9], behavior_curriculum_all[8] = behavior_curriculum_all[8], behavior_curriculum_all[9]

for bi, behavior_curriculum in enumerate(behavior_curriculum_all):
    output_string += f"\n{bi}"
    output_string_second_table += f"\n{bi}"

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
                    data_none_nan = data[data['timing_met'].notna() & data['heading_err'].notna()][column_to_plot]
                    data_timing_nan = data[data['timing_met'].isna() & data['heading_err'].notna()][column_to_plot]
                    data_heading_nan = data[data['timing_met'].notna() & data['heading_err'].isna()][column_to_plot]
                    data_both_nan = data[data['timing_met'].isna() & data['heading_err'].isna()][column_to_plot]

                    # if column_to_plot == "curriculum_metric":
                    #     # data_none_nan = 1 - nansum(data_none_nan < 19) / nansum(data_none_nan)
                    #     # data_timing_nan = 1 - nansum(data_timing_nan < 19) / nansum(data_timing_nan)
                    #     # data_heading_nan = 1 - nansum(data_heading_nan < 19) / nansum(data_heading_nan)
                    #     # data_both_nan = 1 - nansum(data_both_nan < 19) / nansum(data_both_nan)
                    #     data_none_nan = np.exp(np.log(data_none_nan / 20) / 20)
                    #     data_timing_nan = np.exp(np.log(data_timing_nan / 20) / 20)
                    #     data_heading_nan = np.exp(np.log(data_heading_nan / 20) / 20)
                    #     data_both_nan = np.exp(np.log(data_both_nan / 20) / 20)

                    # Calculate mean for each subset
                    means_none_nan[folder].append(data_none_nan.mean())
                    means_timing_nan[folder].append(data_timing_nan.mean())
                    means_heading_nan[folder].append(data_heading_nan.mean())
                    means_both_nan[folder].append(data_both_nan.mean())

                except FileNotFoundError:
                    means_none_nan[folder].append(np.nan)
                    means_timing_nan[folder].append(np.nan)
                    means_heading_nan[folder].append(np.nan)
                    means_both_nan[folder].append(np.nan)
                    print(f"File {file} not found. Skipping.")
                    continue

        # Calculate the weighted averages
        weights = np.linspace(1, 10, num_curricula)
        weights /= sum(weights)

        if column_to_plot == "curriculum_metric":
            for key in means_timing_nan.keys():
                print(f"{bi}: {[np.round(x, 4) for x in means_timing_nan[key]]} with {key}")
            print(np.sum(np.array(means_timing_nan["reset_actor"]) >= np.array(means_timing_nan["no_plasticity"])) / 10)

        avg_none_nan = np.array([
            np.average(np.ma.masked_array(means_none_nan[folder], np.isnan(means_none_nan[folder])), weights=weights)
            for folder in folders
        ])
        avg_timing_nan = np.array([
            np.average(np.ma.masked_array(means_timing_nan[folder], np.isnan(means_timing_nan[folder])), weights=weights)
            for folder in folders
        ])
        avg_all = np.array([nanmean([avg_none_nan[i],avg_timing_nan[i]]) for i in range(len(folders))])
        # avg_heading_nan = np.average(
        #     np.ma.masked_array(means_heading_nan[folder], np.isnan(means_heading_nan[folder])),
        #     weights=weights
        # )
        # avg_both_nan = np.average(
        #     np.ma.masked_array(means_both_nan[folder], np.isnan(means_both_nan[folder])),
        #     weights=weights
        # )
        num_steps_more = 100 * (avg_all[1]+1)/(avg_all[0]+1)-100
        # Append metrics for the current column to the output string
        output_string += f" & {num_steps_more:.1f}"
        # if column_idx < 2:
        #     output_string += f" & {avg_none_nan[1]:.2f} & {avg_none_nan[0]:.2f} & {avg_timing_nan[1]:.2f} & {avg_timing_nan[0]:.2f}" # & {avg_heading_nan:.2f} & {avg_both_nan:.2f}"
        # else:
        #     output_string_second_table += f" & {avg_none_nan[1]:.2f} & {avg_none_nan[0]:.2f} & {avg_timing_nan[1]:.2f} & {avg_timing_nan[0]:.2f}" # & {avg_heading_nan:.2f} & {avg_both_nan:.2f}"

    # Add LaTeX line break
    output_string += " \\\\"
    output_string_second_table += " \\\\"
print(output_string)
print(output_string_second_table)