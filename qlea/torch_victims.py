"""PyTorch datasets and victim models for larger extraction experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TorchDatasetBundle:
    name: str
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    input_shape: tuple[int, ...]


def make_shape_images(
    *, n_samples: int = 6000, image_size: int = 16, seed: int = 7
) -> TorchDatasetBundle:
    rng = np.random.default_rng(seed)
    images = np.zeros((n_samples, 1, image_size, image_size), dtype=np.float32)
    labels = np.arange(n_samples) % 4
    yy, xx = np.mgrid[:image_size, :image_size]
    for i, cls in enumerate(labels):
        img = rng.normal(0.0, 0.08, size=(image_size, image_size))
        shift = rng.integers(-2, 3)
        if cls == 0:
            img[:, image_size // 2 + shift - 1 : image_size // 2 + shift + 1] += 1.0
        elif cls == 1:
            img[image_size // 2 + shift - 1 : image_size // 2 + shift + 1, :] += 1.0
        elif cls == 2:
            img[np.abs((xx - yy) - shift) <= 1] += 1.0
        else:
            radius = image_size // 4 + shift / 2
            dist = np.sqrt((xx - image_size / 2) ** 2 + (yy - image_size / 2) ** 2)
            img[np.abs(dist - radius) <= 1.3] += 1.0
        images[i, 0] = np.clip(img, 0.0, 1.0)
    order = rng.permutation(n_samples)
    images = images[order]
    labels = labels[order].astype(np.int64)
    split = int(n_samples * 0.75)
    return TorchDatasetBundle(
        "shape_images_16x16",
        images[:split],
        images[split:],
        labels[:split],
        labels[split:],
        (1, image_size, image_size),
    )


def make_long_time_series(
    *, n_samples: int = 8000, length: int = 64, seed: int = 7
) -> TorchDatasetBundle:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length)
    x = np.zeros((n_samples, 1, length), dtype=np.float32)
    y = np.arange(n_samples) % 4
    for i, cls in enumerate(y):
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0.0, 0.10, size=length)
        if cls == 0:
            values = np.sin(2 * np.pi * 2 * t + phase)
        elif cls == 1:
            values = np.sin(2 * np.pi * 5 * t + phase)
        elif cls == 2:
            values = np.sign(np.sin(2 * np.pi * 3 * t + phase))
        else:
            values = 2 * (t - 0.5) + 0.4 * np.sin(2 * np.pi * t + phase)
        x[i, 0] = values + noise
    order = rng.permutation(n_samples)
    x = x[order]
    y = y[order].astype(np.int64)
    split = int(n_samples * 0.75)
    return TorchDatasetBundle(
        "long_time_series",
        x[:split],
        x[split:],
        y[:split],
        y[split:],
        (1, length),
    )


class ImageCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 24, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(48 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceLSTM(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=48, batch_first=True)
        self.head = nn.Linear(48, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1])


class TorchVictim:
    def __init__(self, name: str, model: nn.Module, input_shape: tuple[int, ...]):
        self.name = name
        self.model = model.to(device())
        self.input_shape = input_shape

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device())
                logits = self.model(xb)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds)


def train_torch_victim(
    name: str,
    model: nn.Module,
    bundle: TorchDatasetBundle,
    *,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> TorchVictim:
    victim = TorchVictim(name, model, bundle.input_shape)
    x = torch.tensor(bundle.x_train, dtype=torch.float32)
    y = torch.tensor(bundle.y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(victim.model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    victim.model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device())
            yb = yb.to(device())
            opt.zero_grad()
            loss = loss_fn(victim.model(xb), yb)
            loss.backward()
            opt.step()
    return victim
