# calling_model.py

import sys
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# 1) Monkey-fix memmap_meta.npz so that rnn_attention_model.py sees
#    shape[0] as a tuple rather than an int.
# ────────────────────────────────────────────────────────────────────────────────
npz_path = "memmap_meta.npz"
meta = np.load(npz_path, allow_pickle=True)

# pull out existing arrays
starts  = meta["start"]
lengths = meta["length"]
dtype   = meta["dtype"]
shape_v = meta["shape"]           # e.g. array([total_timesteps, 12])

# convert flat shape into a single tuple, then wrap in object array
shape_tuple   = tuple(int(x) for x in shape_v)
shape_obj_arr = np.array([shape_tuple], dtype=object)

# overwrite the NPZ in place
np.savez(
    npz_path,
    start=starts,
    length=lengths,
    dtype=dtype,
    shape=shape_obj_arr
)
print(f"🔧 Fixed memmap_meta.npz: shape[0] → {shape_tuple!r}")

# Also ensure numpy load works inside imported modules
sys.modules['numpy._core'] = np.core



# ────────────────────────────────────────────────────────────────────────────────
# 2) Now safe to import everything from rnn_attention_model
# ────────────────────────────────────────────────────────────────────────────────
from rnn_attention_model import RNNAttentionModel

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "full_model.pkl"
PKL_PATH   = "df_memmap.pkl"
NPZ_PATH   = "memmap_meta.npz"
NPY_PATH   = "memmap.npy"
# ────────────────────────────────────────────────────────────────────────────────

# 3) Load & merge metadata just as before
df_meta = pd.read_pickle(PKL_PATH)
meta    = np.load(NPZ_PATH, allow_pickle=True)
starts  = meta["start"].astype(int)
lengths = meta["length"].astype(int)
df_npz  = pd.DataFrame({ "start": starts, "length": lengths })
df      = pd.concat([df_meta.reset_index(drop=True),
                     df_npz .reset_index(drop=True)], axis=1)

# 4) Memory-map and reshape ECG data
dtype    = (meta["dtype"].item()
            if isinstance(meta["dtype"], np.ndarray)
            else meta["dtype"])
_mm      = np.memmap(NPY_PATH, dtype=dtype, mode="r")
shape_t  = tuple(meta["shape"][0])    # now a tuple thanks to our fix above
ecg_data = _mm.reshape(shape_t)

# 5) Prepare model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = RNNAttentionModel.load_from_checkpoint(MODEL_PATH)
model.eval().to(device)

# 6) Inference loop
probs, preds = [], []
for _, row in df.iterrows():
    s, L = int(row["start"]), int(row["length"])
    rec = ecg_data[s : s+L, :]                            # (L,12)
    norm = (rec - rec.mean(0)) / (rec.std(0) + 1e-6)
    x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x).squeeze(0)
        p      = torch.sigmoid(logit).item()
        ypred  = int(p > 0.5)
    probs.append(p)
    preds.append(ypred)

# 7) Build and show results DataFrame
out_df = df.copy().reset_index(drop=True)
out_df["prob"] = probs
out_df["pred"] = preds

print(out_df.head())
