import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, memmap, X, y):
        self.df = X.reset_index(drop=True)
        self.memmap = memmap
        self.y = y

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
      # Access data directly from the DataFrame
      start = self.df.loc[idx, 'start']
      length = self.df.loc[idx, 'length']
      # file_idx = self.df.loc[idx, 'file_idx'] # You might not need file_idx here anymore

      # Extract the flat signal slice
      signal = self.memmap[start : start + length * 12]  # 12 features per timestep
      signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-6)

      # Reshape to [length, 12]
      signal = signal.reshape(length, 12)

      # Convert signal to PyTorch tensor before checking for NaN/inf
      signal = torch.tensor(signal, dtype=torch.float32)

      if torch.isnan(signal).any() or torch.isinf(signal).any():
        return None

      label = self.y.iloc[idx]  # Access label from DataFrame
      return signal, torch.tensor(label, dtype=torch.long) # signal is already a tensor

from torch.utils.data import DataLoader
import pytorch_lightning as pl

class ECGDataModule(pl.LightningDataModule):
    def __init__(self, memmap, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        super().__init__()
        self.memmap = memmap
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = ECGDataset(self.memmap, self.X_train, self.y_train)
        self.val_dataset = ECGDataset(self.memmap, self.X_val, self.y_val)
        self.test_dataset = ECGDataset(self.memmap, self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=11, collate_fn=safe_collate, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,  num_workers=11, collate_fn=safe_collate, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=11, collate_fn=safe_collate, pin_memory=True)

import pytorch_lightning as pl
import torch

class Swish(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvNormPool(pl.LightningModule):
    """Conv Skip-connection module"""
    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        norm_type='bachnorm'
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.swish_1 = Swish()
        self.swish_2 = Swish()
        self.swish_3 = Swish()
        if norm_type == 'group':
            self.normalization_1 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
            self.normalization_2 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
            self.normalization_3 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
        else:
            self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)

        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1+conv3)
        x = self.swish_3(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))

        x = self.pool(x)
        return x

class CNN(pl.LightningModule):
    def __init__(
        self,
        input_size = 1,
        hid_size = 256,
        kernel_size = 5,
        num_classes = 5,
    ):

        super().__init__()

        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size//2,
            kernel_size=kernel_size,
        )
        self.conv3 = ConvNormPool(
            input_size=hid_size//2,
            hidden_size=hid_size//4,
            kernel_size=kernel_size,
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=hid_size//4, out_features=num_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        # print(x.shape) # num_features * num_channels
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)
        return x

class RNN(pl.LightningModule):
    """RNN module(cell type lstm or gru)"""
    def __init__(
        self,
        input_size,
        hid_size,
        num_rnn_layers=1,
        dropout_p = 0.2,
        bidirectional = False,
        rnn_type = 'lstm',
    ):
        super().__init__()

        if rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=num_rnn_layers,
                dropout=dropout_p if num_rnn_layers>1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )

        else:
            self.rnn_layer = nn.GRU(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=num_rnn_layers,
                dropout=dropout_p if num_rnn_layers>1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
    def forward(self, input):
        outputs, hidden_states = self.rnn_layer(input)
        return outputs, hidden_states

class RNNModel(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hid_size,
        rnn_type,
        bidirectional,
        n_classes=5,
        kernel_size=5,
    ):
        super().__init__()

        self.rnn_layer = RNN(
            input_size=46,#hid_size * 2 if bidirectional else hid_size,
            hid_size=hid_size,
            rnn_type=rnn_type,
            bidirectional=bidirectional
        )
        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=hid_size, out_features=n_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x, _ = self.rnn_layer(x)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.sigmoid(self.fc(x), dim=1)#.squeeze(1)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC

class RNNAttentionModel(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hid_size,
        rnn_type,
        bidirectional,
        kernel_size=5,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = ConvNormPool(
            input_size=input_size,  # input_size = 12 for ECG
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )

        self.rnn_layer = RNN(
            input_size=hid_size,
            hid_size=hid_size,
            rnn_type=rnn_type,
            bidirectional=bidirectional
        )

        self.attn = nn.Linear(hid_size, hid_size, bias=False)
        self.fc = nn.Linear(in_features=hid_size, out_features=1)  # Binary output
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

        # Metrics
        self.train_acc = BinaryAccuracy()
        self.train_f1 = BinaryF1Score()
        self.train_auc = BinaryAUROC()

        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_auc = BinaryAUROC()

    def forward(self, input):
        input = input.permute(0, 2, 1)  # (batch, 12, 1000)
        x = self.conv1(input)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # (batch, time_steps, features)

        x_out, _ = self.rnn_layer(x)  # (batch, time, hid_size)

        attn_weights = torch.softmax(self.attn(x_out), dim=1)  # (batch, time, hid_size)
        x = torch.sum(attn_weights * x_out, dim=1)  # (batch, hid_size)

        logits = self.fc(x)  # (batch, 1)
        return logits

    # def on_train_start(self):
    #     # Log model type as a parameter or tag
    #     # mlflow.pytorch.log_model(self, "model") # Registers the model
    #     mlflow.log_param("model_type", "RNNAttentionModel")  # Log as parameter
    #     mlflow.set_tag("model_type", "RNNAttentionModel")

    def training_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = probs > 0.5

        acc = self.train_acc(probs, y.int())
        f1 = self.train_f1(probs, y.int())
        auc = self.train_auc(probs, y.int())

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        self.log("train_auc", auc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = probs > 0.5

        acc = self.val_acc(probs, y.int())
        f1 = self.val_f1(probs, y.int())
        auc = self.val_auc(probs, y.int())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)

    def on_test_start(self):
        self.test_probs = []
        self.test_preds = []
        self.test_targets = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        self.test_probs.append(probs.detach().cpu())
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())

    def on_test_end(self):
        self.all_probs = torch.cat(self.test_probs)
        self.all_preds = torch.cat(self.test_preds)
        self.all_targets = torch.cat(self.test_targets)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

from torch.nn.utils.rnn import pad_sequence
def safe_collate(batch):
    # Filter out None entries
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # Skip entire batch if empty (optional, or raise Exception)

    signals, labels = zip(*batch)
    signals = pad_sequence(signals, batch_first=True)  # if variable-length ECG
    labels = torch.tensor(labels)
    return signals, labels

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.classification import BinaryF1Score, BinaryAUROC


class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()

        self.fc = nn.Linear(hidden_size * 2, 1)  # bidirectional
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))

    def forward(self, x):
        # x: (B, T, C) → needs to be (B, T, 12)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last timestep
        logits = self.fc(out)
        return logits.squeeze()

    def on_train_start(self):
      # Log model type as a parameter or tag
      mlflow.pytorch.log_model(self, "model") # Registers the model
      mlflow.log_param("model_type", "LSTM")  # Log as parameter
      mlflow.set_tag("model_type", "LSTM")

    def training_step(self, batch, batch_idx):
        self.train()
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        auc = self.train_auc(probs, y.int())
        acc = (preds == y).float().mean()
        f1 = self.train_f1(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        self.log("train_auc", auc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        auc = self.train_auc(probs, y.int())
        acc = (preds == y).float().mean()
        f1 = self.val_f1(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)

        return loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        auc = self.train_auc(probs, y.int())

        acc = (preds == y).float().mean()
        f1 = self.test_f1(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_auc", auc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)  # Reduced lr
