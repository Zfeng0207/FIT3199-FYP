# process_and_predict.py

"""
Load & preprocess ECG memmap data, then run inference.

Expected inputs:
  • meta_path    – memmap_meta.npz (must contain array 'shape')
  • npy_path     – memmap.npy    (raw ECG memmap, flat .npy)
  • mapping_pkl  – df_memmap.pkl (must contain 'start' & ('length' or 'stop'))
  • records_csv  – records_w_diag_icd10.csv
                   • if it has ['start','length'] itself, those are used
                   • otherwise it must share at least one key (e.g. 'study_id')
                     with df_memmap.pkl for a left‐join

Your trained model: full_model.pkl (sits next to this script).
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib


def process_and_predict(
    meta_path:   str,
    npy_path:    str,
    mapping_pkl: str,
    records_csv: str,
    model_path:  Optional[str] = None
) -> Dict[str, Any]:
    # 1) Load memmap shape
    meta = np.load(meta_path, allow_pickle=True)
    shape_arr = meta['shape']
    ecg_shape = tuple(shape_arr[0]) if shape_arr.ndim == 2 else tuple(shape_arr)

    # 2) Open the raw ECG memmap
    memmap = np.memmap(npy_path, dtype=np.float32, mode='r').reshape(ecg_shape)

    # 3) Load & sanitize your mapping file
    df_map = pd.read_pickle(mapping_pkl)
    if 'start' not in df_map.columns:
        raise ValueError("Mapping file must contain a 'start' column.")
    if 'length' not in df_map.columns:
        if 'stop' in df_map.columns:
            df_map['length'] = df_map['stop'] - df_map['start']
        else:
            raise ValueError("Mapping file must contain either 'length' or 'stop' to compute lengths.")

    # 4) Load your diagnostic CSV
    df_rec = pd.read_csv(records_csv)

    # 5) Determine which pointers to use
    if {'start','length'}.issubset(df_rec.columns):
        # your CSV already has them
        df_ptr = df_rec
    else:
        # find shared key(s)
        shared = list(set(df_map.columns).intersection(df_rec.columns))
        if not shared:
            raise ValueError(
                "records CSV has no 'start'/'length' and no common key to join with mapping."
            )
        df_ptr = df_rec.merge(
            df_map[['start','length'] + shared],
            on=shared, how='left'
        )
        if df_ptr[['start','length']].isnull().any().any():
            raise ValueError("Some rows in your records CSV failed to map to memmap pointers.")

    # 6) Extract, normalize & flatten each ECG segment
    features: List[np.ndarray] = []
    for _, row in df_ptr.iterrows():
        s, L = int(row['start']), int(row['length'])
        seg = memmap[s : s + L, :]  # shape (L, n_leads)
        normed = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-6)
        features.append(normed.flatten())

    X = np.vstack(features) if features else np.empty((0,))

    # 7) Load your trained model
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'full_model.pkl')
    model = joblib.load(model_path)

    # 8) Run inference
    preds = model.predict(X).tolist()
    try:
        probs = model.predict_proba(X).tolist()
    except Exception:
        probs = None

    return {
        'predictions': preds,
        'probabilities': probs
    }
