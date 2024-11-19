csvfile = "runs/dream/plaid_results_1/plaid.csv"

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(csvfile)

print(df.columns)

# plt.plot(df["value_loss"])
plt.plot(df["action_loss"])
plt.show()
