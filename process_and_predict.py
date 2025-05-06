# process_and_predict.py

"""
Load and preprocess ECG memmap data, then run inference.

Expected files:
  • memmap_meta.npz       — contains arrays 'shape'
  • memmap.npy            — raw ECG memmap (flat .npy)
  • df_memmap.pkl         — DataFrame with at least ['study_id','start','length']
  • records_w_diag_icd10.csv — your diagnostic CSV, must
       – EITHER have ['start','length'] itself
       – OR have a ['study_id'] column to join against df_memmap.pkl

Your trained model: full_model.pkl (next to this file).
"""

import os
from typing import Any, Dict, Optional

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
    """
    Parameters
    ----------
    meta_path : str
        Path to memmap_meta.npz (must contain 'shape' array).
    npy_path : str
        Path to memmap.npy (raw ECG data).
    mapping_pkl : str
        Path to df_memmap.pkl (must contain 'study_id','start','length').
    records_csv : str
        Path to records_w_diag_icd10.csv.
    model_path : Optional[str]
        Path to full_model.pkl (defaults to ./full_model.pkl).

    Returns
    -------
    Dict[str, Any]
        {
          "predictions": [...],
          "probabilities": [[...], ...] or None
        }
    """
    # 1) Load memmap shape
    meta = np.load(meta_path, allow_pickle=True)
    shape_arr = meta['shape']
    ecg_shape = tuple(shape_arr[0]) if shape_arr.ndim == 2 else tuple(shape_arr)

    # 2) Open ECG memmap
    memmap = np.memmap(npy_path, dtype=np.float32, mode='r').reshape(ecg_shape)

    # 3) Load mapping DataFrame (df_memmap.pkl)
    df_map = pd.read_pickle(mapping_pkl)
    if not {'study_id','start','length'}.issubset(df_map.columns):
        raise ValueError("df_memmap.pkl must contain ['study_id','start','length'].")

    # 4) Load your diagnostic CSV
    df_rec = pd.read_csv(records_csv)

    # 5) Determine pointers
    if {'start','length'}.issubset(df_rec.columns):
        df_ptr = df_rec
    else:
        if 'study_id' not in df_rec.columns:
            raise ValueError(
                "records CSV must have ['start','length'] or a 'study_id' column."
            )
        df_ptr = df_rec.merge(
            df_map[['study_id','start','length']],
            on='study_id', how='left'
        )
        if df_ptr[['start','length']].isnull().any().any():
            raise ValueError("Some study_id values did not match df_memmap.pkl.")

    # 6) Extract, normalize & flatten
    feats = []
    for _, row in df_ptr.iterrows():
        s, L = int(row['start']), int(row['length'])
        seg = memmap[s : s + L, :]  # (L, n_leads)
        normed = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-6)
        feats.append(normed.flatten())

    X = np.vstack(feats) if feats else np.empty((0,))

    # 7) Load model
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'full_model.pkl')
    model = joblib.load(model_path)

    # 8) Predict
    preds = model.predict(X).tolist()
    try:
        probs = model.predict_proba(X).tolist()
    except Exception:
        probs = None

    return {"predictions": preds, "probabilities": probs}
