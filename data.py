import pandas as pd
import os
import glob

# Path to the folder containing your CSV files
folder_path = "path/to/your/folder"  # ← Replace with your actual path

# Find all CSV files that match the pattern
csv_files = glob.glob(os.path.join(folder_path, "WorkLog-export-*.csv"))

# Read and combine all CSV files
dataframes = []
for filepath in csv_files:
    filename = os.path.basename(filepath)
    timestamp = filename.split('-')[2]  # Extract the timestamp part
    df = pd.read_csv(filepath)
    df['FileID'] = timestamp  # Add column with timestamp
    dataframes.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Output the combined DataFrame
print("Combined DataFrame:")
print(combined_df.head())
