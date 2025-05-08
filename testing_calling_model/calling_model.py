import sys
import torch
import numpy as np
import pandas as pd
def predict_from_memmap(memmap_path: str) -> pd.DataFrame:
    """
    Run stroke predictions using a pre-loaded model on ECG data stored in a memmap.

    Args:
        memmap_path (str): Path to the .npy memmap file.

    Returns:
        pd.DataFrame: Predictions with columns ['study_id', 'stroke'].
    """
    # Open the memmap for raw ECG values
    memmap = np.memmap(memmap_path, dtype=np.float32, mode="r")
    results = []

    for _, row in df.iterrows():
        start, length = int(row["start"]), int(row["length"])
        if length <= 0:
            continue

        raw = memmap[start: start + length * 12]
        if raw.size != length * 12:
            continue

        # reshape and normalize
        ecg = raw.reshape(length, 12)
        ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-6)

        # tensor of shape (1, seq_len, 12)
        x = torch.from_numpy(ecg).float().unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).item()
            pred = "Y" if prob > 0.5 else "N"

        results.append((row["study_id"], pred))

    return pd.DataFrame(results, columns=["study_id", "stroke"])
