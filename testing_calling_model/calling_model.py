# console_predict.py

import sys
import numpy as np
import torch
import rnn_attention_model

# ─── 1) Monkey-patch the model classes into __main__ so unpickling can find them ───
_main = sys.modules['__main__']
for name in ('RNNAttentionModel', 'ConvNormPool', 'Swish', 'RNN', 'CNN'):
    setattr(_main, name, getattr(rnn_attention_model, name))

# ─── 2) Allow those classes for unsafe unpickling ────────────────────────────────
torch.serialization.add_safe_globals([
    rnn_attention_model.RNNAttentionModel,
    rnn_attention_model.ConvNormPool,
    rnn_attention_model.Swish,
    rnn_attention_model.RNN,
    rnn_attention_model.CNN,
])

# ─── 3) Hard-code your memmap shape here ────────────────────────────────────────
ECG_SHAPE = (21649000, 12)  # (number_of_samples, number_of_leads)

def main():
    # ─── 4) Load your full checkpoint with weights_only=False ────────────────────
    model = torch.load("full_model.pkl", map_location="cpu", weights_only=False)
    model.eval()

    # ─── 5) Prompt for the raw memmap file ───────────────────────────────────────
    path = input("Enter path to your raw memmap .npy file: ").strip()

    # ─── 6) Memory-map & reshape ─────────────────────────────────────────────────
    raw = np.memmap(path, dtype=np.float32, mode="r")
    ecg_signal = raw.reshape(ECG_SHAPE)

    # ─── 7) Normalize ────────────────────────────────────────────────────────────
    ecg_signal = (ecg_signal - ecg_signal.mean()) / (ecg_signal.std() + 1e-6)

    # ─── 8) To tensor [1, N, 12] ─────────────────────────────────────────────────
    tensor = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)

    # ─── 9) Inference ────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        prob   = torch.sigmoid(logits).item()
        pred   = "Stroke" if prob > 0.5 else "No Stroke"

    # ─── 10) Print ───────────────────────────────────────────────────────────────
    print(f"\nPrediction : {pred}")
    print(f"Probability: {prob:.4f}")

if __name__ == "__main__":
    main()
