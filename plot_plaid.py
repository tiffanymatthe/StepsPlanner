import pandas as pd
import matplotlib.pyplot as plt
import ast

file = "runs/dream/plaid_offset_10.csv"

# Define a function to convert strings to lists
def convert_to_list(val):
    try:
        val = val.replace('nan', 'None')
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        print(val)
        return None  # or any placeholder for malformed entries

df = pd.read_csv(file)
df['curriculum_metric'] = df['curriculum_metric'].apply(convert_to_list)
curriculums = pd.DataFrame(df['curriculum_metric'].to_list(), columns=['curr all','curr no timing'])
plt.plot(df["action_loss"])
# plt.plot(curriculums)
plt.show()