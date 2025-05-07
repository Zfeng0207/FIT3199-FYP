# calling_model.py

import sys
import numpy as np
import pandas as pd
import torch

# ────────────────────────────────────────────────────────────────────────────────
# 1) Monkey-patch memmap_meta.npz so that
#    rnn_attention_model.py’s
#      original_shape = tuple(memmap_meta["shape"][0])
#    line works without change.
# ────────────────────────────────────────────────────────────────────────────────

npz_path = "memmap_meta.npz"
meta     = np.load(npz_path, allow_pickle=True)

# Extract existing fields
starts   = meta["start"]
lengths  = meta["length"]
dtype_md = meta["dtype"]
shape_md = meta["shape"]  # could be flat array or already object-array

# Build a Python tuple for the overall shape:
#  • If `shape_md` is a numeric array, .tolist() → [N,12]
#  • If `shape_md` is object-array of [(N,12)], .tolist() → [(N,12)]
lst = shape_md.tolist()
if isinstance(lst, list) and len(lst) == 1 and isinstance(lst[0], (tuple, list)):
    # already wrapped: e.g. [(N,12)]
    tuple_shape = tuple(lst[0])
else:
    # flat list: [N,12]
    tuple_shape = tuple(lst)

# Now wrap that tuple in an object-array of length 1
shape_obj = np.array([tuple_shape], dtype=object)

# Overwrite the NPZ in place with the same start/length/dtype and new shape
np.savez(
    npz_path,
    start=starts,
    length=lengths,
    dtype=dtype_md,
    shape=shape_obj
)
print(f"🔧 Patched memmap_meta.npz so shape[0] = {tuple_shape}")

# Also patch numpy so downstream imports of np.load still work
sys.modules['numpy._core'] = np.core

# ────────────────────────────────────────────────────────────────────────────────
# 2) Now import your unmodified model file
# ────────────────────────────────────────────────────────────────────────────────
from rnn_attention_model import RNNAttentionModel

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "full_model.pkl"
PKL_PATH   = "df_memmap.pkl"
NPY_PATH   = "memmap.npy"
# ────────────────────────────────────────────────────────────────────────────────

# 3) Load & merge metadata
df_meta = pd.read_pickle(PKL_PATH)         # your per-record metadata + Stroke_YN
meta    = np.load(npz_path, allow_pickle=True)
starts  = meta["start"].astype(int)
lengths = meta["length"].astype(int)
df_idx  = pd.DataFrame({"start": starts, "length": lengths})
df      = pd.concat([df_meta.reset_index(drop=True),
                     df_idx .reset_index(drop=True)], axis=1)

# 4) Memory-map & reshape ECG data
dtype_np = (meta["dtype"].item()
            if isinstance(meta["dtype"], np.ndarray)
            else meta["dtype"])
_mm      = np.memmap(NPY_PATH, dtype=dtype_np, mode="r")
shape_t  = tuple(meta["shape"][0])  # now correct!
ecg_data = _mm.reshape(shape_t)

# 5) Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = RNNAttentionModel.load_from_checkpoint(MODEL_PATH)
model.eval().to(device)

# 6) Inference
probs, preds = [], []
for _, row in df.iterrows():
    s, L = int(row["start"]), int(row["length"])
    rec  = ecg_data[s : s + L, :]                          # (L,12)
    norm = (rec - rec.mean(axis=0)) / (rec.std(axis=0) + 1e-6)
    x    = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x).squeeze(0)
        p     = torch.sigmoid(logit).item()
        yhat  = int(p > 0.5)
    probs.append(p)
    preds.append(yhat)

# 7) Build & display output DataFrame
out_df = df.reset_index(drop=True)
out_df["prob"] = probs
out_df["pred"] = preds

print(out_df.head())
