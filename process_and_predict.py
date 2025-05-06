# process_and_predict.py

"""
Load & preprocess ECG memmap data, then run inference.

Inputs (all required):
  • meta_path    — path to memmap_meta.npz
  • npy_path     — path to memmap.npy
  • mapping_pkl  — path to df_memmap.pkl
  • records_csv  — path to records_w_diag_icd10.csv

Outputs: a dict with
  - 'predictions': List of model outputs
  - 'probabilities': List of prob-vectors (or None)
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
    # 1) load memmap shape
    meta = np.load(meta_path, allow_pickle=True)
    shape_arr = meta['shape']
    # handle nested shape arrays
    ecg_shape = tuple(shape_arr[0]) if shape_arr.ndim == 2 else tuple(shape_arr)

    # 2) open the raw memmap
    memmap = np.memmap(npy_path, dtype=np.float32, mode='r').reshape(ecg_shape)

    # 3) load pointer mapping
    df_map = pd.read_pickle(mapping_pkl)
    if not {'start','length'}.issubset(df_map.columns):
        raise ValueError("Mapping file must contain 'start' & 'length' columns.")

    # 4) load your diagnostic CSV
    df_rec = pd.read_csv(records_csv)

    # 5) figure out pointers
    if {'start','length'}.issubset(df_rec.columns):
        df_ptr = df_rec
    else:
        # find any shared key columns
        shared = list(set(df_map.columns).intersection(df_rec.columns))
        if not shared:
            raise ValueError(
                "records CSV has no 'start'/'length' and no common key to join with mapping."
            )
        # keep only start/length + shared keys
        df_ptr = df_rec.merge(
            df_map[['start','length'] + shared],
            on=shared, how='left'
        )
        if df_ptr[['start','length']].isnull().any().any():
            raise ValueError("Some rows failed to map to memmap pointers.")

    # 6) extract, normalize & flatten
    features: List[np.ndarray] = []
    for _, row in df_ptr.iterrows():
        s, L = int(row['start']), int(row['length'])
        seg = memmap[s:s+L, :]
        normed = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-6)
        features.append(normed.flatten())

    X = np.vstack(features) if features else np.empty((0,))

    # 7) load your model
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'full_model.pkl')
    model = joblib.load(model_path)

    # 8) run inference
    preds = model.predict(X).tolist()
    try:
        probs = model.predict_proba(X).tolist()
    except Exception:
        probs = None

    return {
        'predictions': preds,
        'probabilities': probs
    }
