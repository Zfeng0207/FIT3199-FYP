# process_and_predict.py

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib


def process_and_predict(
    meta_path:    str,
    npy_path:     str,
    mapping_pkl:  str,
    records_csv:  str,
    model_path:   Optional[str] = None
) -> Dict[str, Any]:
    # 1) load memmap pointers from .npz
    meta = np.load(meta_path, allow_pickle=True)
    starts   = meta['start']
    lengths  = meta['length']
    shape_arr = meta['shape']
    ecg_shape = tuple(shape_arr[0]) if shape_arr.ndim == 2 else tuple(shape_arr)

    # 2) open and reshape memmap
    memmap = np.memmap(npy_path, dtype=np.float32, mode='r').reshape(ecg_shape)

    # 3) load mapping DF for record‐IDs
    df_map = pd.read_pickle(mapping_pkl).reset_index().rename(columns={'index':'meta_idx'})

    # 4) load diagnostic CSV
    df_rec = pd.read_csv(records_csv)

    # 5) choose pointers
    if {'start','length'}.issubset(df_rec.columns):
        df_ptr = df_rec.copy()
        use_raw = True
    else:
        # find shared keys
        shared = list(set(df_map.columns).intersection(df_rec.columns))
        if not shared:
            raise ValueError("No 'start'/'length' and no common key to join on.")
        # if ecg_time is a key, convert both sides to datetime
        if 'ecg_time' in shared:
            df_map['ecg_time'] = pd.to_datetime(df_map['ecg_time'])
            df_rec['ecg_time'] = pd.to_datetime(df_rec['ecg_time'])
        df_ptr = df_rec.merge(df_map, on=shared, how='left')
        if df_ptr['meta_idx'].isnull().any():
            raise ValueError("Some records failed to map after merging on " + ", ".join(shared))
        use_raw = False

    # 6) extract+normalize+flatten
    features: List[np.ndarray] = []
    for _, row in df_ptr.iterrows():
        if use_raw:
            s = int(row['start'])
            L = int(row['length'])
        else:
            idx = int(row['meta_idx'])
            s = int(starts[idx])
            L = int(lengths[idx])
        seg = memmap[s : s + L, :]
        normed = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-6)
        features.append(normed.flatten())

    X = np.vstack(features) if features else np.empty((0,))

    # 7) load model
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'full_model.pkl')
    model = joblib.load(model_path)

    # 8) predict
    preds = model.predict(X).tolist()
    try:
        probs = model.predict_proba(X).tolist()
    except Exception:
        probs = None

    return {'predictions': preds, 'probabilities': probs}
