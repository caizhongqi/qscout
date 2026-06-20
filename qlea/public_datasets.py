"""Public large datasets for image and time-series extraction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve
import zipfile

import numpy as np
import torch
from torchvision import datasets, transforms


@dataclass
class PublicDataset:
    name: str
    kind: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    input_shape: tuple[int, ...]


def _torchvision_to_numpy(ds) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for img, label in ds:
        x.append(img.numpy())
        y.append(int(label))
    return np.stack(x).astype(np.float32), np.array(y, dtype=np.int64)


def load_image_dataset(name: str, root: Path) -> PublicDataset:
    tfm = transforms.Compose([transforms.ToTensor()])
    if name == "MNIST":
        cls = datasets.MNIST
    elif name == "FashionMNIST":
        cls = datasets.FashionMNIST
    else:
        raise ValueError(f"Unsupported image dataset: {name}")
    train = cls(root=str(root), train=True, download=True, transform=tfm)
    test = cls(root=str(root), train=False, download=True, transform=tfm)
    x_train, y_train = _torchvision_to_numpy(train)
    x_test, y_test = _torchvision_to_numpy(test)
    return PublicDataset(name, "image", x_train, y_train, x_test, y_test, 10, (1, 28, 28))


def _download_ucr_zip(name: str, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / f"{name}.zip"
    if zip_path.exists():
        return zip_path
    urls = [
        f"https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip",
        f"https://timeseriesclassification.com/aeon-toolkit/{name}.zip",
        f"http://www.timeseriesclassification.com/aeon-toolkit/{name}.zip",
    ]
    errors = []
    for url in urls:
        try:
            urlretrieve(url, zip_path)
            return zip_path
        except (OSError, URLError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        f"Could not download UCR dataset {name}. Tried:\n" + "\n".join(errors)
    )


def _parse_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter="\t")
    y = data[:, 0].astype(int)
    x = data[:, 1:].astype(np.float32)
    return x, y


def _parse_ts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    in_data = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("@data"):
                in_data = True
                continue
            if line.startswith("@") or not in_data:
                continue
            parts = line.split(":")
            labels.append(parts[-1].strip())
            dims = []
            for dim in parts[:-1]:
                values = [float(v) for v in dim.replace("?", "nan").split(",") if v]
                dims.extend(values)
            rows.append(dims)
    width = max(len(row) for row in rows)
    x = np.full((len(rows), width), np.nan, dtype=np.float32)
    for idx, row in enumerate(rows):
        x[idx, : len(row)] = row
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_mean, inds[1])
    unique = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    y = np.array([unique[label] for label in labels], dtype=np.int64)
    return x, y


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    return (train_x - mean) / std, (test_x - mean) / std


def load_ucr_dataset(name: str, root: Path) -> PublicDataset:
    zip_path = _download_ucr_zip(name, root)
    extract_dir = root / name
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    candidates = list(extract_dir.rglob("*"))
    train_files = [
        p for p in candidates if p.is_file() and p.stem.lower().endswith("_train")
    ]
    test_files = [
        p for p in candidates if p.is_file() and p.stem.lower().endswith("_test")
    ]
    if not train_files or not test_files:
        raise FileNotFoundError(f"Could not find train/test files in {extract_dir}")
    suffix_rank = {".ts": 0, ".tsv": 1, ".arff": 2, ".txt": 3}
    train_files = sorted(train_files, key=lambda p: suffix_rank.get(p.suffix.lower(), 9))
    test_files = sorted(test_files, key=lambda p: suffix_rank.get(p.suffix.lower(), 9))
    train_path = train_files[0]
    test_path = test_files[0]
    parser = _parse_tsv if train_path.suffix.lower() in {".tsv", ".txt"} else _parse_ts
    x_train, y_train = parser(train_path)
    x_test, y_test = parser(test_path)
    labels = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    remap = {label: idx for idx, label in enumerate(labels)}
    y_train = np.array([remap[int(label)] for label in y_train], dtype=np.int64)
    y_test = np.array([remap[int(label)] for label in y_test], dtype=np.int64)
    x_train, x_test = _standardize(x_train, x_test)
    x_train = x_train[:, None, :].astype(np.float32)
    x_test = x_test[:, None, :].astype(np.float32)
    return PublicDataset(
        name,
        "timeseries",
        x_train,
        y_train,
        x_test,
        y_test,
        len(labels),
        (1, x_train.shape[-1]),
    )


def load_five_public_datasets(root: str = "data_public") -> list[PublicDataset]:
    return load_selected_public_datasets(
        ["MNIST", "FashionMNIST", "FordA", "Wafer", "ElectricDevices"], root=root
    )


def load_selected_public_datasets(
    names: list[str], root: str = "data_public"
) -> list[PublicDataset]:
    root_path = Path(root)
    out = []
    for name in names:
        if name in {"MNIST", "FashionMNIST"}:
            out.append(load_image_dataset(name, root_path / "torchvision"))
        elif name in {"FordA", "Wafer", "ElectricDevices"}:
            out.append(load_ucr_dataset(name, root_path / "ucr"))
        else:
            raise ValueError(f"Unknown public dataset: {name}")
    return out


def to_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    ds = torch.utils.data.TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
