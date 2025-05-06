# extract_model.py

import nbformat

NOTEBOOK = '[BINARY]_CNN_LSTM_attention_classifier.ipynb'  # change this to your notebook’s filename
OUTFILE  = 'ecg_model.py'

# 1) Read the notebook
nb = nbformat.read(NOTEBOOK, as_version=4)

# 2) Open the output file and write standard imports
with open(OUTFILE, 'w') as f:
    f.write("""\
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC

""")

    # 3) For every code cell that starts with "class ", dump it verbatim
    for cell in nb.cells:
        if cell.cell_type == 'code':
            src = cell.source.lstrip()
            if src.startswith('class '):
                f.write(src)
                f.write('\n\n')

print(f"✅ Wrote all model classes to {OUTFILE}")
