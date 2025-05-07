# calling_model.py

import sys
import numpy as np
import torch
import multiprocessing as mp
import rnn_attention_model

# ─── 1) Monkey-patch the classes into this module's globals ────────────────────
for cls_name in ("RNNAttentionModel", "ConvNormPool", "Swish", "RNN", "CNN"):
    globals()[cls_name] = getattr(rnn_attention_model, cls_name)

# ─── 2) Allow these classes for unsafe unpickling with weights_only=False ─────
torch.serialization.add_safe_globals([
    rnn_attention_model.RNNAttentionModel,
    rnn_attention_model.ConvNormPool,
    rnn_attention_model.Swish,
    rnn_attention_model.RNN,
    rnn_attention_model.CNN,
])

# ─── 3) Hard-code your ECG shape + checkpoint path ─────────────────────────────
ECG_SHAPE = (21649000, 12)      # (num_timesteps, num_leads)
MODEL_PATH = "full_model.pkl"

def init_worker():
    """Initializer for each pool worker: load & eval the model once."""
    global model
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()

def infer_chunk(chunk: np.ndarray, mean: float, std: float) -> float:
    """Normalize a chunk, run a forward pass, return sigmoid probability."""
    x = (chunk - mean) / std
    t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(t)
        return torch.sigmoid(logits).item()

def main():
    # 1) Ask for your raw memmap file
    path = input("Enter path to your raw memmap .npy file: ").strip()

    # 2) Memory-map & reshape
    raw = np.memmap(path, dtype=np.float32, mode="r")
    ecg = raw.reshape(ECG_SHAPE)

    # 3) Compute global mean/std once
    mean = float(ecg.mean())
    std  = float(ecg.std() + 1e-6)

    # 4) Split into as many parts as you have CPU cores
    n_workers = mp.cpu_count()
    chunks    = np.array_split(ecg, n_workers, axis=0)

    # 5) Launch pool, each worker runs init_worker() exactly once
    with mp.Pool(processes=n_workers, initializer=init_worker) as pool:
        # 6) Dispatch inference for each chunk in parallel
        args = [(chunk, mean, std) for chunk in chunks]
        probs = pool.starmap(infer_chunk, args)

    # 7) Average the chunk-level probabilities & decide
    avg_prob = sum(probs) / len(probs)
    pred     = "Stroke" if avg_prob > 0.5 else "No Stroke"

    # 8) Print your result
    print("\n=== Result ===")
    print(f"Prediction : {pred}")
    print(f"Probability: {avg_prob:.4f}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
