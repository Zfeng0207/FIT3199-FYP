import sys
import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# UPDATE these paths to point at your files
NPZ_PATH    = r'memmap_meta.npz'
NPY_PATH    = r'memmap.npy'
PKL_PATH    = r'C:df_memmap.pkl'
MODEL_PATH  = r'full_model.pkl'
OUT_DIR     = r'C:/Monash/FIT3164/FIT3199-FYP/testing_calling_model'
# ────────────────────────────────────────────────────────────────────────────────


# Ensure NumPy pickles load correctly
sys.modules['numpy._core'] = np.core

# 1) Convert .npz → CSV
meta = np.load(NPZ_PATH, allow_pickle=True)
df_meta = pd.DataFrame({
    "start":  meta["start"].astype(int),
    "length": meta["length"].astype(int)
})
meta_csv = f"{OUT_DIR}/memmap_meta.csv"
df_meta.to_csv(meta_csv, index=False)
print("Wrote:", meta_csv)


# 2) Convert .pkl → CSV
df_map = pd.read_pickle(PKL_PATH)
map_csv = f"{OUT_DIR}/df_memmap.csv"
df_map.to_csv(map_csv, index=False)
print("Wrote:", map_csv)


# 3) Join the two CSVs on their row index
df1 = pd.read_csv(meta_csv)
df2 = pd.read_csv(map_csv)
# they must align 1:1 by row order/index
df_merged = pd.concat([df2.reset_index(drop=True),
                       df1.reset_index(drop=True)], axis=1)
merged_csv = f"{OUT_DIR}/merged.csv"
df_merged.to_csv(merged_csv, index=False)
print("Wrote:", merged_csv)


# 4) Load memmap .npy and reshape if needed
#    The .npz may also include 'shape'; if so you can reshape in one go.
dtype = meta["dtype"].item() if isinstance(meta["dtype"], np.ndarray) else meta["dtype"]
_mm = np.memmap(NPY_PATH, dtype=dtype, mode='r')
# If your npz stored 'shape' as e.g. (total_timesteps,12):
if "shape" in meta:
    ecg_data = _mm.reshape(tuple(meta["shape"]))
    # then you can slice ecg_data[start:start+length, :]
else:
    ecg_data = _mm  # we'll slice flat: raw = ecg_data[start : start+length*12]


# 5) Build signal batches and optional labels
signals = []
labels  = []
for _, row in df_merged.iterrows():
    s, L = int(row["start"]), int(row["length"])
    if ecg_data.ndim == 2:
        raw = ecg_data[s : s+L, :]               # shape=(L,12)
    else:
        raw = ecg_data[s : s + L*12]             # flat
        raw = raw.reshape(L, 12)
    normed = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-6)
    if np.isnan(normed).any() or np.isinf(normed).any():
        continue
    signals.append(normed)
    # if your df_map had a label column, e.g. 'Stroke_YN':
    if "Stroke_YN" in df_merged.columns:
        labels.append(int(row["Stroke_YN"]))

batch_tensor = torch.tensor(signals, dtype=torch.float32)  # (N, T, 12)
loader = DataLoader(TensorDataset(batch_tensor), batch_size=64, shuffle=False)


# 6) Load trained model and run inference
model = joblib.load(MODEL_PATH)
model.eval()

all_probs = []
all_preds = []

with torch.no_grad():
    for (x_batch,) in loader:
        logits = model(x_batch)
        probs  = torch.sigmoid(logits).cpu().numpy().ravel()
        preds  = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)

# 7) Add predictions to DataFrame
out_df = df_merged.copy().reset_index(drop=True)
out_df["pred"] = all_preds
out_df["prob"] = all_probs

# Display the resulting DataFrame
print(out_df.head())

# Optionally save the results
pred_csv = f"{OUT_DIR}/predictions.csv"
out_df.to_csv(pred_csv, index=False)
print("Wrote:", pred_csv)
