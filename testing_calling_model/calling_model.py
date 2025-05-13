import torch
import numpy as np
import pandas as pd
import rnn_attention_model
import __main__

# Expose classes for safe unpickling
__main__.RNNAttentionModel = rnn_attention_model.RNNAttentionModel
__main__.ConvNormPool      = rnn_attention_model.ConvNormPool
__main__.Swish             = rnn_attention_model.Swish
__main__.RNN               = rnn_attention_model.RNN

def predict_stroke(memmap_data: str = "C:/Monash/FIT3164/FIT3199-FYP/testing_calling_model/memmap_balanced_30_30.npy",
                   label_df: str   = "C:/Monash/FIT3164/FIT3199-FYP/testing_calling_model/label_df.csv",
                   full_model: str = "C:/Monash/FIT3164/FIT3199-FYP/testing_calling_model/full_model.pkl",
                   threshold: float = 0.82,
                   max_records: int = 60) -> pd.DataFrame:
    """
    Preprocess ECG memmap data and run stroke predictions.

    Parameters
    ----------
    memmap_data : str
        Path to the .npy file containing raw ECG floats.
    label_df : str
        Path to the CSV file with metadata (study_id, patient_id, ecg_time, length, ...).
    full_model : str
        Path to the pickled PyTorch model.
    threshold : float
        Probability cutoff for predicting stroke (default 0.82).
    max_records : int
        Number of records to process from the metadata (default first 60).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['study_id', 'patient_id', 'gender', 'ecg_time', 'stroke'].
    """
    # 1. Safe globals for unpickling
    torch.serialization.add_safe_globals([
        rnn_attention_model.RNNAttentionModel,
        rnn_attention_model.ConvNormPool,
        rnn_attention_model.Swish,
        rnn_attention_model.RNN
    ])

    # 2. Load model
    model = torch.load(full_model, map_location="cpu", weights_only=False)
    model.eval()

    # 3. Load metadata
    df = pd.read_csv(label_df)

    # 4. Memory-map the ECG data
    memmap = np.memmap(memmap_data, dtype=np.float32, mode='r')

    results = []
    offset = 0
    total_len = memmap.size

    for idx in range(min(max_records, len(df))):
        row = df.loc[idx]
        study_id  = row.get('study_id')
        patient_id= row.get('patient_id', None)
        ecg_time  = row.get('ecg_time', None)
        gender    = row.get('gender', 'Unknown')
        length    = int(row.get('length', 0))

        # validate slice
        slice_size = length * 12
        if length <= 0 or offset + slice_size > total_len:
            print(f"[SKIP] idx={idx} invalid length or out of bounds")
            offset += slice_size
            continue

        # extract and advance offset
        raw_signal = memmap[offset : offset + slice_size]
        offset += slice_size

        if raw_signal.size != slice_size:
            print(f"[SKIP] idx={idx} size mismatch")
            continue

        # normalize and reshape
        normalized = (raw_signal - raw_signal.mean()) / (raw_signal.std() + 1e-6)
        signal = normalized.reshape(length, 12)

        # check for NaN or Inf
        if np.isnan(signal).any() or np.isinf(signal).any():
            print(f"[SKIP] idx={idx} contains NaN or Inf")
            continue

        # prepare tensor
        input_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        # inference
        with torch.no_grad():
            logits = model(input_tensor)
            prob   = torch.sigmoid(logits).item()
            pred   = 'Y' if prob > threshold else 'N'

        results.append((study_id, patient_id, gender, ecg_time, pred))

    # build DataFrame
    cols = ['study_id', 'patient_id', 'gender', 'ecg_time', 'stroke']
    preds_df = pd.DataFrame(results, columns=cols)
    return preds_df


if __name__ == '__main__':
    df_preds = predict_stroke()
    print(df_preds.head())
    df_preds.to_csv('stroke_predictions.csv', index=False)
