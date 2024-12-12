import csv
import matplotlib.pyplot as plt
import os

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
def plot_csv_data(data1, data2):
    if data1 is None or data2 is None:
        print("No data to plot.")
        return

    # Create combined x-axis labels for both datasets
    x_labels1 = [f"({b},{c})" for b, c in zip(data1["behavior_curriculum"], data1["curriculum"])]
    x_indices1 = range(len(x_labels1))

    x_labels2 = [f"({b},{c})" for b, c in zip(data2["behavior_curriculum"], data2["curriculum"])]
    x_indices2 = range(len(x_labels2))

    plt.figure(figsize=(12, 6))

    # Plot data1
    plt.plot(x_indices1, data1["dead_actor"], label="Dead Actor (CSV 1)", marker="o", color="blue")
    plt.plot(x_indices1, data1["dead_critic"], label="Dead Critic (CSV 1)", marker="s", color="red")

    # Plot data2
    plt.plot(x_indices2, data2["dead_actor"], label="Dead Actor (CSV 2)", marker="o", linestyle="--", color="cyan")
    plt.plot(x_indices2, data2["dead_critic"], label="Dead Critic (CSV 2)", marker="s", linestyle="--", color="orange")

    # Customize the plot
    plt.title("Comparison of Dead Actor and Dead Critic between Two CSVs")
    plt.xlabel("Behavior Curriculum, Curriculum")
    plt.ylabel("Values")
    plt.xticks(x_indices1, x_labels1, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

# Main execution
if __name__ == "__main__":
    csv_file1 = "dormant_no_reset_0_01_masking.csv"  # Specify the path to your first CSV file
    csv_file2 = "dormant_reset_0_01_masking.csv"  # Specify the path to your second CSV file

    data1 = read_csv_data(csv_file1)
    data2 = read_csv_data(csv_file2)

    plot_csv_data(data1, data2)