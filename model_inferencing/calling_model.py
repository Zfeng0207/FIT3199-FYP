import torch
import numpy as np
import pandas as pd
import rnn_attention_model
import __main__
import io

# Expose classes for safe unpickling
__main__.RNNAttentionModel = rnn_attention_model.RNNAttentionModel
__main__.ConvNormPool      = rnn_attention_model.ConvNormPool
__main__.Swish             = rnn_attention_model.Swish
__main__.RNN               = rnn_attention_model.RNN

import torch
import numpy as np
import pandas as pd

def predict_stroke(csv_with_ecg: str,
                   full_model: str,
                   threshold: float = 0.82) -> pd.DataFrame:
    """
    Predict stroke risk using a single CSV file that contains ECG signals and metadata.

    Parameters
    ----------
    csv_with_ecg : str
        Path to the CSV file containing metadata and a column with ECG signal (as a list).
    full_model : str
        Path to the pickled PyTorch model.
    threshold : float
        Probability cutoff for predicting stroke (default is 0.82).
    max_records : int
        Number of rows to process from the CSV (default is 60).

    Returns
    -------
    pd.DataFrame
        DataFrame with ['study_id', 'patient_id', 'gender', 'ecg_time', 'stroke'].
    """

    # Allow safe unpickling
    torch.serialization.add_safe_globals([
        rnn_attention_model.RNNAttentionModel,
        rnn_attention_model.ConvNormPool,
        rnn_attention_model.Swish,
        rnn_attention_model.RNN
    ])

    # Load model
    model = torch.load(full_model, map_location='cpu', weights_only=False)
    model.eval()

    # Load CSV with embedded ECG data
    df = pd.read_csv(csv_with_ecg)
    results = []
    for idx, row in df.iterrows():
        subject_id = row['study_id']
        patient_id = row['patient_id']
        ecg_time = row['ecg_time']
        gender = row['gender'] if 'gender' in df.columns else "Unknown"
        length = row['length']
        raw_signal_str = row['ecg_data']

        # Convert the space-separated string to a NumPy array
        try:
            raw_signal = np.loadtxt(io.StringIO(raw_signal_str.strip('[]')))
        except Exception as e:
            print(f"[SKIP] idx={idx} - Error parsing ecg_data: {e}")
            continue

        if len(raw_signal) != length * 12:
            print(f"[SKIP] idx={idx} - Signal length mismatch: got {len(raw_signal)}, expected {length * 12}")
            continue

        normalized = (raw_signal - raw_signal.mean()) / (raw_signal.std() + 1e-6)
        signal = normalized.reshape(length, 12)

        if np.isnan(signal).any() or np.isinf(signal).any():
            print(f"[SKIP] NaN/Inf values at idx={idx}")
            continue

        input_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()
            pred = 'N' if prob > threshold else 'Y'

        print(f"[PREDICT] idx={idx}, prob={prob:.4f}, label={pred}")
        results.append((subject_id, patient_id, gender, ecg_time, pred))

    predictions_df = pd.DataFrame(results, columns=['study_id', 'patient_id', 'gender', 'ecg_time', 'stroke'])
    return predictions_df


if __name__ == '__main__':
    df_preds = predict_stroke()
    print(df_preds.head())
    df_preds.to_csv('stroke_predictions.csv', index=False)
