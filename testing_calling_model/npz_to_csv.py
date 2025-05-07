import numpy as np
import pandas as pd
import sys

# Re-load numpy for compatibility with the current environment
sys.modules['numpy._core'] = np.core

# Load the .npz file
npz_file_path = 'memmap_meta.npz'  # Update with the path to your .npz file
memmap_meta = np.load(npz_file_path, allow_pickle=True)

# Create a DataFrame with the relevant data (start, length, etc.)
df = pd.DataFrame({
    "start": memmap_meta['start'],
    "length": memmap_meta['length']
})

# Save the DataFrame as a CSV
csv_file_path = 'memmap_meta.csv'  # Update with where you want to save the CSV file
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved at: {csv_file_path}")
