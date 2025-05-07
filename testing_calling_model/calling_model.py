import torch
import numpy as np
import pandas as pd
from rnn_attention_model import RNNAttentionModel,ConvNormPool,Swish,RNN

# 1. Register your classes so torch.load can unpickle them
torch.serialization.add_safe_globals([
    RNNAttentionModel,
    ConvNormPool,
    Swish,
    RNN
])

# 2. Load the full model object

# load full object (not just weights)
model = torch.load(
    "full_model.pkl",
    map_location="cpu",
    weights_only=False           # ← explicitly disable the new default
)
model.eval()

# 3. Load your metadata
df = pd.read_csv("label_df.csv")

# 4. Open the memmap of raw ECG values
memmap = np.memmap("memmap_head.npy", dtype=np.float32, mode="r")

results = []
for idx in range(len(df)):
    row = df.loc[idx]
    start, length = int(row["start"]), int(row["length"])

    if length <= 0:
        continue

    # pull out length×12 floats and reshape into (time_steps, 12 leads)
    raw = memmap[start : start + length * 12]
    if raw.size != length * 12:
        continue

    ecg = raw.reshape(length, 12)
    # normalize to zero mean, unit variance
    ecg = (ecg - ecg.mean()) / (ecg.std() + 1e-6)

    # The model expects a FloatTensor of shape (batch, seq_len, num_leads)
    # Here batch=1, seq_len=length, num_leads=12
    x = torch.from_numpy(ecg).float().unsqueeze(0)  

    with torch.no_grad():
        logits = model(x)             # → shape (1,) or (1,1)
        prob   = torch.sigmoid(logits).item()
        pred   = "Y" if prob > 0.5 else "N"

    results.append((row["study_id"], pred))

# collect into a DataFrame
preds_df = pd.DataFrame(results, columns=["study_id","stroke"])
print(preds_df.head())
preds_df.to_csv("stroke_predictions.csv", index=False)
