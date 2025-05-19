# process_and_predict.py

import os
import sys
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib


def _load_mapping_pkl(path: str) -> pd.DataFrame:
    """
    Unpickle df_memmap.pkl in spite of minor NumPy version differences.
    """
    # alias the old numpy._core module name to current numpy.core
    sys.modules['numpy._core'] = __import__('numpy').core
    with open(path, 'rb') as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{path} did not unpickle to a DataFrame.")
    return df


def process_and_predict(
    meta_path:    str,
    npy_path:     str,
    mapping_pkl:  str,
    records_csv:  str,
    model_path:   Optional[str] = None
) -> Dict[str, Any]:
    """
    1) Load 'start','length','shape','dtype' from memmap_meta.npz
    2) np.memmap your .npy (flat float32 vector)
    3) Unpickle df_memmap.pkl for 'study_id','start','length'
    4) Read records_w_diag_icd10.csv and merge on 'study_id'
    5) For each record: slice raw = memmap[start : start + length * num_leads]
       → z-score → reshape(length, num_leads) → flatten
    6) Load full_model.pkl and predict
    """
    # 1) pointers & shape
    meta    = np.load(meta_path, allow_pickle=True)
    starts  = meta['start']    # array of ints
    lengths = meta['length']   # array of ints
    shape_a = meta['shape']    # e.g. [21649000, 12]
    dtype   = meta.get('dtype', np.float32)
    # number of leads is second dim of shape
    num_leads = int(shape_a[1])

    # 2) raw memmap (flat)
    mem = np.memmap(npy_path, dtype=dtype, mode='r')

    # 3) load mapping DataFrame
    df_map = _load_mapping_pkl(mapping_pkl)
    if 'study_id' not in df_map.columns:
        raise ValueError("df_memmap.pkl must contain a 'study_id' column.")
    # keep only the two columns we need
    if not {'start','length'}.issubset(df_map.columns):
        # if your mapping pkl has no start/length, fall back to pointers in the .npz
        # but then df_map must have meta_idx or be in same order as meta arrays:
        df_map = df_map.reset_index().rename(columns={'index':'meta_idx'})
    else:
        # mapping pkl carries its own pointers: great, use those
        df_map = df_map[['study_id','start','length']]

    # 4) read your diagnostic CSV
    df_rec = pd.read_csv(records_csv)
    if 'study_id' not in df_rec.columns:
        raise ValueError("Your CSV must contain a 'study_id' column to match df_memmap.pkl.")

    # 5) merge to get pointers
    df_ptr = df_rec.merge(df_map, on='study_id', how='left')
    if df_ptr[['start','length']].isnull().any().any():
        missing = df_ptr[['start','length']].isnull().sum().sum()
        raise ValueError(f"{missing} rows failed to map to pointers on 'study_id'.")

    # 6) slice, normalize, flatten
    features: List[np.ndarray] = []
    for _, row in df_ptr.iterrows():
        s = int(row['start'])
        L = int(row['length'])
        # exactly as in your notebook:
        raw    = mem[s : s + L * num_leads]                       # flat vector
        normed = (raw - raw.mean()) / (raw.std() + 1e-6)           # z-score
        seg    = normed.reshape(L, num_leads)                     # [length, 12]
        features.append(seg.flatten())

    X = np.vstack(features) if features else np.empty((0, ))

    # 7) load your trained model
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'full_model.pkl')
    model = joblib.load(model_path)

    # 8) inference
    preds = model.predict(X).tolist()
    try:
        probs = model.predict_proba(X).tolist()
    except Exception:
        probs = None

    return {'predictions': preds, 'probabilities': probs}
