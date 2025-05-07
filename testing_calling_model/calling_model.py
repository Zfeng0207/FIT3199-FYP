# console_predict.py
import numpy as np
import torch
from rnn_attention_model import RNNAttentionModel, ConvNormPool, Swish, RNN, CNN

def main():
    # allow your custom model class for unpickling
    torch.serialization.add_safe_globals({RNNAttentionModel})

    # load the trained model (adjust path if needed)
    model = torch.load(
        "full_model.pkl",
        map_location="cpu",
        weights_only=False
    )
    model.eval()

    # ask user for ECG file
    path = input("Enter path to your ECG .npy file: ").strip()

    # try memory‐mapping first (no pickle)
    try:
        ecg_signal = np.load(path, mmap_mode='r')
    except Exception as e_memmap:
        print(f"  • memmap failed ({e_memmap}), falling back to allow_pickle…")
        try:
            ecg_signal = np.load(path, allow_pickle=True)
        except Exception as e_pickle:
            print(f"ERROR: could not load '{path}': {e_pickle}")
            return

    # normalize
    ecg_signal = (ecg_signal - ecg_signal.mean()) / (ecg_signal.std() + 1e-6)

    # convert to tensor [1, length, 12]
    tensor_input = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)

    # inference
    with torch.no_grad():
        logits = model(tensor_input)
        prob = torch.sigmoid(logits).item()
        pred = "Stroke" if prob > 0.5 else "No Stroke"

    # output
    print("\n=== Result ===")
    print(f"Prediction : {pred}")
    print(f"Probability: {prob:.4f}")

if __name__ == "__main__":
    main()
