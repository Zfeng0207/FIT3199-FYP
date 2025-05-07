import sys
import numpy as np
import pandas as pd

# allow numpy pickles
sys.modules['numpy._core'] = np.core

# load
meta      = np.load("memmap_meta.npz", allow_pickle=True)
starts    = meta["start"].astype(int)
lengths   = meta["length"].astype(int)

# pull out the tuple you’re using
shape_vec = tuple(meta["shape"][0])
dtype_str = str(meta["dtype"].item() if isinstance(meta["dtype"], np.ndarray) else meta["dtype"])

# build DataFrame
df = pd.DataFrame({
    "start":  starts,
    "length": lengths,
    "shape":  [str(shape_vec)] * len(starts),
    "dtype":  [dtype_str] * len(starts),
})

# save
df.to_csv("memmap_meta.csv", index=False)
