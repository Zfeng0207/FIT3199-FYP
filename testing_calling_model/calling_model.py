import numpy as np
import torch
import streamlit as st
from rnn_attention_model import RNNAttentionModel, ConvNormPool, Swish, RNN, CNN

# Paths
memmap_meta_path = "memmap_meta.npz"
memmap_path = "memmap.npy"
df_diag_path = "records_w_diag_icd10.csv"
df_memmap_pkl_path = "memmap/df_memmap.pkl"

# Load model (allow unsafe unpickling)
torch.serialization.add_safe_globals({RNNAttentionModel})
model = torch.load("full_model.pkl", map_location='cpu', weights_only=False)
model.eval()

st.title("Stroke Prediction from ECG")
st.write("Upload a .npy ECG file (shape: [length, 12])")

# File uploader
uploaded_file = st.file_uploader("Upload ECG File", type="npy")

if uploaded_file is not None:
    ecg_signal = np.load(uploaded_file)

    # Normalize the signal
    ecg_signal = (ecg_signal - ecg_signal.mean()) / (ecg_signal.std() + 1e-6)

    # Convert to tensor and reshape: [1, length, 12]
    tensor_input = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)

    # Predict
    with torch.no_grad():
        logits = model(tensor_input)
        prob = torch.sigmoid(logits).item()
        pred = int(prob > 0.5)

    # Display results
    st.write(f"**Prediction:** {'Stroke' if pred == 1 else 'No Stroke'}")
    st.write(f"**Probability:** {prob:.4f}")
