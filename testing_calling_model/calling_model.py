import torch
import numpy as np
import pandas as pd
from rnn_attention_model import RNNAttentionModel, ConvNormPool, Swish, RNN
import rnn_attention_model
import __main__

__main__.RNNAttentionModel = rnn_attention_model.RNNAttentionModel
__main__.ConvNormPool      = rnn_attention_model.ConvNormPool
__main__.Swish             = rnn_attention_model.Swish
__main__.RNN               = rnn_attention_model.RNN

def predict_stroke(memmap_data="C:/Monash/FIT3164/FIT3199-FYP/testing_calling_model/memmap_head.npy",
                   label_df="C:/Monash/FIT3164/FIT3199-FYP/testing_calling_model/label_df.csv",
                   full_model="C:/Monash/FIT3164/FIT3199-FYP/testing_calling_model/full_model.pkl"):
    """
    Load a trained RNN attention model, apply it to ECG data stored
    in a NumPy memmap, and return stroke predictions.

    Parameters
    ----------
    memmap_data : str
        Path to the .npy memmap file containing raw ECG floats.
    label_df : str
        Path to the CSV file with 'start' and 'length' columns (and an ID).
    full_model : str
        Path to the .pkl file of the trained model (full object, not just weights).

    Returns
    -------
    preds_df : pandas.DataFrame
        DataFrame with columns ["study_id", "stroke"], where stroke is "Y" or "N".
    """
    # 1. Register your custom classes so torch.load can unpickle them
    torch.serialization.add_safe_globals([
        RNNAttentionModel,
        ConvNormPool,
        Swish,
        RNN
    ])

    # 2. Load the full model
    model = torch.load(full_model, map_location="cpu", weights_only=False)
    model.eval()

    # 3. Load metadata
    df = pd.read_csv(label_df)

    # 4. Open the memmap of raw ECG values
    memmap = np.memmap(memmap_data, dtype=np.float32, mode="r")

    results = []
    for idx, row in df.iterrows():
        start, length = int(row["start"]), int(row["length"])
        if length <= 0:
            continue

        # extract length×12 floats and reshape into (time_steps, 12 leads)
        raw = memmap[start : start + length * 12]
        if raw.size != length * 12:
            continue

        ecg = raw.reshape(length, 12)
        # normalize to zero mean, unit variance
        ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-6)

        # model expects FloatTensor of shape (batch, seq_len, num_leads)
        x = torch.from_numpy(ecg).float().unsqueeze(0)

        with torch.no_grad():
            logits = model(x)           # shape (1,) or (1,1)
            prob   = torch.sigmoid(logits).item()
            pred   = "Y" if prob > 0.5 else "N"

        results.append((row["study_id"], pred))

    # collect into a DataFrame
    preds_df = pd.DataFrame(results, columns=["study_id", "stroke"])
    return preds_df


if __name__ == "__main__":
    # example usage
    df_preds = predict_stroke(
        memmap_data="memmap_head.npy",
        label_df="label_df.csv",
        full_model="full_model.pkl"
    )
    print(df_preds.head())
    df_preds.to_csv("stroke_predictions.csv", index=False)
