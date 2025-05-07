# calling_model.py

import sys
import numpy as np
import pandas as pd
import torch

# ─── IMPORT YOUR MODEL CLASS ──────────────────────────────────────────────────
# Adjust this to point at wherever your RNNAttentionModel is defined.
from cnn_lstm_attention_classifier import RNNAttentionModel
# ────────────────────────────────────────────────────────────────────────────────

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "full_model.pkl"     # your trained model
PKL_PATH   = "df_memmap.pkl"      # per-record metadata + Stroke_YN
NPZ_PATH   = "memmap_meta.npz"    # start/length/shape/dtype
NPY_PATH   = "memmap.npy"         # flat ECG data
# ────────────────────────────────────────────────────────────────────────────────

# 1) Fix numpy pickle compatibility
sys.modules['numpy._core'] = np.core

# 2) Load and merge metadata
df_meta = pd.read_pickle(PKL_PATH)        # must contain at least 'Stroke_YN'
meta    = np.load(NPZ_PATH, allow_pickle=True)
starts  = meta["start"].astype(int)
lengths = meta["length"].astype(int)

df_npz = pd.DataFrame({"start": starts, "length": lengths})
df = pd.concat([df_meta.reset_index(drop=True),
                df_npz .reset_index(drop=True)], axis=1)

# 3) Memory-map and reshape ECG data
dtype    = meta["dtype"].item() if isinstance(meta["dtype"], np.ndarray) else meta["dtype"]
_mm      = np.memmap(NPY_PATH, dtype=dtype, mode="r")
shape_t  = tuple(meta["shape"][0])       # e.g. (total_timesteps, 12)
ecg_data = _mm.reshape(shape_t)

# 4) Load your trained model from full_model.pkl
model = RNNAttentionModel.load_from_checkpoint(MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5) Run inference for each patient
probs = []
preds = []

for idx, row in df.iterrows():
    s, L = int(row["start"]), int(row["length"])
    # slice & normalize
    rec = ecg_data[s : s + L, :]                          # shape = (L,12)
    normed = (rec - rec.mean(axis=0)) / (rec.std(axis=0) + 1e-6)
    # to tensor, add batch dim → [1, L, 12]
    x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)                                 # → [1,1]
        prob   = torch.sigmoid(logits).item()             # float
        pred   = int(prob > 0.5)                          # 0 or 1

    probs.append(prob)
    preds.append(pred)

# 6) Build and display final DataFrame
out_df = df.copy()
out_df["prob"] = probs
out_df["pred"] = preds

print(out_df.head())
