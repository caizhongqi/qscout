"""PyTorch victim models for public image and time-series datasets."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .public_datasets import PublicDataset, to_loader


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageCNN28(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageMLP(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeSeriesCNN(nn.Module):
    def __init__(self, n_classes: int, length: int):
        super().__init__()
        pooled = length // 4
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(64 * pooled, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeSeriesLSTM(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.transpose(1, 2))
        return self.head(out[:, -1])


class TorchBlackBox:
    def __init__(self, name: str, model: nn.Module):
        self.name = name
        self.model = model.to(get_device())

    def predict(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32).to(get_device())
                preds.append(torch.argmax(self.model(xb), dim=1).cpu().numpy())
        return np.concatenate(preds)


def train_victim(
    name: str,
    model: nn.Module,
    data: PublicDataset,
    *,
    epochs: int,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> TorchBlackBox:
    victim = TorchBlackBox(name, model)
    loader = to_loader(data.x_train, data.y_train, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(victim.model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    victim.model.train()
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(get_device())
            yb = yb.to(get_device())
            opt.zero_grad()
            loss = loss_fn(victim.model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            seen += xb.shape[0]
        print(f"{name} epoch {epoch + 1}/{epochs} loss={total / max(seen, 1):.4f}")
    return victim
