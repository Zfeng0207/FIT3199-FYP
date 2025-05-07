import sys
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# Set the paths to your files
NPZ_PATH    = r'memmap_meta.npz'  # Path to memmap_meta.npz
NPY_PATH    = r'memmap.npy'       # Path to memmap.npy
PKL_PATH    = r'df_memmap.pkl'    # Path to df_memmap.pkl
MODEL_PATH  = r'full_model.pkl'  # Path to trained model (.pkl)
OUTPUT_DIR  = r'C:/Monash/FIT3164/FIT3199-FYP/testing_calling_model'         # Directory to save output files
# ────────────────────────────────────────────────────────────────────────────────

# Ensure NumPy pickles load correctly
sys.modules['numpy._core'] = np.core

# 1) Convert .npz → CSV (memmap_meta.npz -> memmap_meta.csv)
meta = np.load(NPZ_PATH, allow_pickle=True)
df_meta = pd.DataFrame({
    "start": meta["start"].astype(int),
    "length": meta["length"].astype(int)
})
meta_csv_path = f"{OUTPUT_DIR}/memmap_meta.csv"
df_meta.to_csv(meta_csv_path, index=False)
print(f"Converted {NPZ_PATH} to {meta_csv_path}")

# 2) Convert .pkl → CSV (df_memmap.pkl -> df_memmap.csv)
df_index = pd.read_pickle(PKL_PATH)
df_index_csv_path = f"{OUTPUT_DIR}/df_memmap.csv"
df_index.to_csv(df_index_csv_path, index=False)
print(f"Converted {PKL_PATH} to {df_index_csv_path}")

# 3) Merge the two CSVs (memmap_meta.csv and df_memmap.csv)
df_meta = pd.read_csv(meta_csv_path)
df_index = pd.read_csv(df_index_csv_path)
df_merged = pd.concat([df_index.reset_index(drop=True), df_meta.reset_index(drop=True)], axis=1)

merged_csv_path = f"{OUTPUT_DIR}/merged.csv"
df_merged.to_csv(merged_csv_path, index=False)
print(f"Merged CSV saved at: {merged_csv_path}")

# 4) Load .npy file and prepare signals
ecg_data = np.memmap(NPY_PATH, dtype=meta["dtype"].item(), mode='r')
if 'shape' in meta:
    ecg_data = ecg_data.reshape(tuple(meta['shape']))

# 5) Extract, normalize, and reshape each ECG
signals = []
for _, row in df_merged.iterrows():
    start, length = int(row['start']), int(row['length'])
    raw = ecg_data[start:start+length*12].reshape(length, 12)
    normed = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-6)
    if np.isnan(normed).any() or np.isinf(normed).any():
        continue
    signals.append(normed)

# 6) Stack into one array and convert to a tensor
signals_np = np.stack(signals, axis=0)  # shape = (N, T, 12)
batch_tensor = torch.from_numpy(signals_np).float()

# 7) Load the trained model using pickle with latin1 encoding
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f, encoding='latin1')
model.eval()

# 8) Run inference
all_probs = []
all_preds = []
loader = DataLoader(TensorDataset(batch_tensor), batch_size=64, shuffle=False)
with torch.no_grad():
    for (x_batch,) in loader:
        logits = model(x_batch)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)

# 9) Build the output DataFrame
out_df = df_merged.iloc[:len(all_preds)].copy().reset_index(drop=True)
out_df['pred'] = all_preds
out_df['prob'] = all_probs

# 10) Display the output DataFrame
print(out_df.head())

# 11) (Optional) Save the predictions DataFrame to a CSV file
predictions_csv_path = f"{OUTPUT_DIR}/predictions.csv"
out_df.to_csv(predictions_csv_path, index=False)
print(f"Predictions saved to {predictions_csv_path}")
