import csv
import matplotlib.pyplot as plt
import os

from matplotlib import rc

# Use Computer Modern Roman as the default font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)

plt.rcParams.update({'font.size': 16})

# Function to read the CSV data
def read_csv_data(csv_file):
    data = {
        "behavior_curriculum": [],
        "curriculum": [],
        "dead_actor": [],
        "dead_critic": []
    }
    
    # Ensure the file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} does not exist.")
        return None

    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data["behavior_curriculum"].append(int(row["behavior_curriculum"]))
            data["curriculum"].append(int(row["curriculum"]))
            data["dead_actor"].append(float(row["dead_actor"]))
            data["dead_critic"].append(float(row["dead_critic"]))
    
    return data

# Function to plot the data
def plot_csv_data(data1, data2, labels=["1", "2"], limit=None):
    if data1 is None or data2 is None:
        print("No data to plot.")
        return

    # Create combined x-axis labels for both datasets
    x_labels1 = [f"({b},{c})" for b, c in zip(data1["behavior_curriculum"], data1["curriculum"])]
    x_indices1 = range(len(x_labels1))

    x_labels2 = [f"({b},{c})" for b, c in zip(data2["behavior_curriculum"], data2["curriculum"])]
    x_indices2 = range(len(x_labels2))
    
    if limit:
        x_labels_1_end = x_labels1.index(f"({limit[0]},{limit[1]})") + 1
        x_labels_2_end = x_labels2.index(f"({limit[0]},{limit[1]})") + 1
    else:
        x_labels_1_end = None
        x_labels_2_end = None

    plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots(2, figsize=(12, 10))

    # Plot data1
    ax[0].plot(x_indices1[0:x_labels_1_end], data1["dead_actor"][0:x_labels_1_end], label=labels[0], marker="o", color="blue")
    ax[1].plot(x_indices1[0:x_labels_1_end], data1["dead_critic"][0:x_labels_1_end], label=labels[0], marker="s", color="red")

    # Plot data2
    ax[0].plot(x_indices2[0:x_labels_2_end], data2["dead_actor"][0:x_labels_2_end], label=labels[1], marker="o", linestyle="--", color="cyan")
    ax[1].plot(x_indices2[0:x_labels_2_end], data2["dead_critic"][0:x_labels_2_end], label=labels[1], marker="s", linestyle="--", color="orange")

    # Customize the plot
    ax[0].set_title("Dormant Units in Actor (\%)")
    ax[0].set_xlabel("(Behavior Curriculum, Curriculum)")
    ax[0].set_ylabel("Dormant Units in Actor (\%)")
    ax[0].set_xticks(x_indices1[0:x_labels_1_end], x_labels1[0:x_labels_1_end], rotation=45, ha="right")
    ax[0].legend()

    ax[1].set_title("Dormant Units in Critic (\%)")
    ax[1].set_xlabel("(Behavior Curriculum, Curriculum)")
    ax[1].set_ylabel("Dormant Units in Critic (\%)")
    ax[1].set_xticks(x_indices1[0:x_labels_1_end], x_labels1[0:x_labels_1_end], rotation=45, ha="right")
    ax[1].legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    csv_file1 = "dormant_no_reset_0_01_masking.csv"  # Specify the path to your first CSV file
    csv_file2 = "dormant_proper_reset_0_01.csv"  # Specify the path to your second CSV file

    data1 = read_csv_data(csv_file1)
    data2 = read_csv_data(csv_file2)

    plot_csv_data(data1, data2, labels=["Baseline", "Cont. Backprop."], limit=(1,3))