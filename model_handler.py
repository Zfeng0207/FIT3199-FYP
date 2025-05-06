# model_handler.py

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

# 1) Import your model classes exactly as defined in ecg_model.py
from ecg_model import (
    RNNAttentionModel,
    RNN,
    ConvNormPool,
    Swish,
)

# 2) Register them in __main__ so torch.load can unpickle correctly
import __main__
__main__.RNNAttentionModel = RNNAttentionModel
__main__.RNN               = RNN
__main__.ConvNormPool      = ConvNormPool
__main__.Swish             = Swish

# 3) Load the full model object from the pickle
DEVICE    = torch.device('cpu')
MODEL_PKL = 'full_model.pkl'

print("🔄 Loading model from", MODEL_PKL)
model = torch.load(MODEL_PKL, map_location=DEVICE, weights_only=False)
print("✅ Model loaded and ready.")

model.to(DEVICE).eval()

# 4) The single function your Flask route will call
def process_and_predict(meta_path, npy_path, csv_path):
    """
    1) Read the memmap metadata (.npz) and raw ECG (.npy)
    2) Read the CSV to get study_id order
    3) Z-score normalize & tensorize each segment
    4) Pad into a batch and run the model
    5) Return a {study_id: prediction} dict
    """
    # Load memmap metadata & raw ECG
    meta    = np.load(meta_path, allow_pickle=True)
    starts  = meta['start']
    lengths = meta['length']
    shape   = tuple(meta['shape'][0])
    raw     = np.memmap(npy_path, dtype=np.float32, mode='r').reshape(shape)

    # Load study IDs from CSV
    df_ids    = pd.read_csv(csv_path)
    study_ids = df_ids['study_id'].tolist()

    # Build a list of (channels × timesteps) tensors
    signals = []
    for s, L in zip(starts, lengths):
        seg = raw[s : s + L]  
        seg = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-6)
        t   = torch.tensor(seg, dtype=torch.float32).transpose(0,1)
        signals.append(t)

    # Pad into a (batch, channels, max_seq_len) tensor
    batch = pad_sequence(signals, batch_first=True, padding_value=0.0).to(DEVICE)

    # Run inference
    with torch.no_grad():
        logits = model(batch)
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).int().cpu().numpy().tolist()

    # Map back to study_ids
    return dict(zip(study_ids, preds))
