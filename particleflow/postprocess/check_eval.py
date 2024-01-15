import pandas as pd

# Increase the maximum number of rows and columns displayed
pd.set_option('display.max_rows', 1000)  # Replace None with a specific number if needed
pd.set_option('display.max_columns', 1000)  # Replace None with a specific number if needed

# Read the Parquet file
df = pd.read_parquet('particleflow/experiments/cocoa_20231221_162614_121773.wipp-gpu-wn243/evaluation/epoch_50/cocoa_edmjj_clusters_pf/pred_batch1.parquet')

# Print the entire DataFrame
print(df)
