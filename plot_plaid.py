import pandas as pd
import matplotlib.pyplot as plt

csvfiles = [
    "runs/dream/plaid_results_1/plaid.csv",
    "runs/dream/plaid_results_2/plaid.csv"
]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for csvfile in csvfiles:
    df = pd.read_csv(csvfile)
    print(df.columns)
    axes[0].plot(df["elapsed_time"], df["value_loss"], label=f"{csvfile} - Value Loss")
    axes[1].plot(df["elapsed_time"], df["action_loss"], label=f"{csvfile} - Action Loss")
    axes[2].plot(df["elapsed_time"], label=f"{csvfile} - Time Elapsed")

axes[0].set_title("Value Loss")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].set_title("Action Loss")
axes[1].set_ylabel("Loss")
axes[1].legend()

axes[2].set_title("Time Elapsed")
axes[2].set_ylabel("Time")
axes[2].legend()

axes[-1].set_xlabel("Iterations")

plt.tight_layout()
plt.show()
