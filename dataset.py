"""
Dataset utilities for NE-Time

Supports:
  ETT family:  ETTh1, ETTh2, ETTm1, ETTm2
  Exchange:    daily FX rates (8 currencies)
  Weather:     10-minute meteorological station data (21 variates)
  ECL:         Electricity Consuming Load — hourly (321 variates)
  Traffic:     road occupancy rates, hourly (862 variates)

Multi-horizon sampling: each sample is annotated with a horizon drawn
from a configured list, forcing the model to generalise across all
horizons in a single training run.

Dataset-aware splits
--------------------
Different datasets use standard benchmark splits to match published
results.  The split logic follows the convention used by PatchTST /
iTransformer papers:

  ETT family : 60 / 20 / 20   (train / val / test)
  Others     : 70 / 10 / 20

This is important for ETTh1 where the previous 70/10/20 split left
only ~1 300 val samples, causing noisy early stopping.
"""

import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────
#  Download helpers
# ─────────────────────────────────────────────────────────

DATASET_URLS = {
    # ETT family (original ETDataset repo — these still work via raw GitHub)
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
    # Weather / Exchange / ECL / Traffic are hosted on HuggingFace.
    # These cannot be downloaded via urllib on clusters with restricted outbound.
    # Run download_datasets.py on a login node first; those CSVs will be placed
    # in ./data/ under these names and download_dataset() will find them via the
    # os.path.exists() check before attempting any HTTP request.
    "Weather":  None,
    "Exchange": None,
    "ECL":      None,
    "Traffic":  None,
}

# Flat filenames used by download_datasets.py (must match local_name there)
DATASET_FILENAMES = {
    "ETTh1":    "ETTh1.csv",
    "ETTh2":    "ETTh2.csv",
    "ETTm1":    "ETTm1.csv",
    "ETTm2":    "ETTm2.csv",
    "Weather":  "Weather.csv",
    "Exchange": "Exchange.csv",
    "ECL":      "ECL.csv",
    "Traffic":  "Traffic.csv",
}

# Standard benchmark split ratios per dataset family
# (train_ratio, val_ratio, test_ratio) — must sum to 1
DATASET_SPLITS = {
    "ETTh1": (0.6, 0.2, 0.2),
    "ETTh2": (0.6, 0.2, 0.2),
    "ETTm1": (0.6, 0.2, 0.2),
    "ETTm2": (0.6, 0.2, 0.2),
    # All other datasets use 70/10/20
}
DEFAULT_SPLIT = (0.7, 0.1, 0.2)


def get_split(dataset_name: str) -> Tuple[float, float, float]:
    return DATASET_SPLITS.get(dataset_name, DEFAULT_SPLIT)


def download_dataset(name: str, root: str = "./data") -> str:
    """
    Locate or download dataset CSV. Returns local path.

    ETT datasets are fetched via raw GitHub if not present.
    Weather / Exchange / ECL / Traffic must be pre-downloaded with
    download_datasets.py (they live on HuggingFace and cannot be fetched
    directly on clusters with restricted outbound HTTP).
    """
    os.makedirs(root, exist_ok=True)

    if name not in DATASET_URLS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Known: {sorted(DATASET_URLS.keys())}"
        )

    filename   = DATASET_FILENAMES.get(name, f"{name}.csv")
    local_path = os.path.join(root, filename)

    if os.path.exists(local_path):
        return local_path

    url = DATASET_URLS[name]
    if url is None:
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {local_path}.\n"
            f"Run `python download_datasets.py --datasets {name}` on a "
            f"login node first to fetch it from HuggingFace."
        )

    print(f"Downloading {name} from {url} ...")
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"Saved to {local_path}")
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise RuntimeError(f"Failed to download {name}: {e}") from e
    return local_path


def load_csv(path: str) -> np.ndarray:
    """
    Load numeric columns from CSV (skip date/timestamp column).
    Handles both comma and tab separators.
    """
    import csv as _csv
    rows = []
    with open(path, newline='') as f:
        # Sniff delimiter
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=',\t')
        except _csv.Error:
            dialect = _csv.excel

        reader = _csv.DictReader(f, dialect=dialect)
        for row in reader:
            values = []
            for k, v in row.items():
                if k is None:
                    continue
                k_lower = k.strip().lower()
                # Skip date/time columns
                if k_lower in ('date', 'datetime', 'timestamp', 'time', ''):
                    continue
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    pass
            if values:
                rows.append(values)

    if not rows:
        raise ValueError(f"No numeric data found in {path}")

    arr = np.array(rows, dtype=np.float32)
    return arr


# ─────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset with multi-horizon sampling.

    Each item:
        x:       [seq_len, C]       input context
        y:       [max_horizon, C]   future values (caller slices to H)
        horizon: int                target horizon for this sample
    """

    def __init__(
        self,
        data:       np.ndarray,
        seq_len:    int,
        horizons:   List[int],
        split:      str   = "train",
        train_ratio: float = 0.7,
        val_ratio:   float = 0.1,
        test_ratio:  float = 0.2,
        stride:     int   = 1,
    ):
        self.seq_len  = seq_len
        self.horizons = horizons
        self.max_h    = max(horizons)

        T = len(data)
        train_end = int(T * train_ratio)
        val_end   = int(T * (train_ratio + val_ratio))

        if split == "train":
            raw = data[:train_end]
        elif split == "val":
            raw = data[train_end:val_end]
        else:
            raw = data[val_end:]

        self.scaler = StandardScaler()
        self.scaler.fit(raw)
        self.data = self.scaler.transform(raw).astype(np.float32)

        max_start = len(self.data) - seq_len - self.max_h
        self.indices = list(range(0, max_start + 1, stride)) if max_start >= 0 else []

    def __len__(self):
        return len(self.indices) * len(self.horizons)

    def __getitem__(self, idx: int):
        win_idx = idx // len(self.horizons)
        hor_idx = idx %  len(self.horizons)
        start   = self.indices[win_idx]
        horizon = self.horizons[hor_idx]

        x = self.data[start : start + self.seq_len]
        y = self.data[start + self.seq_len : start + self.seq_len + self.max_h]

        return {
            "x":       torch.from_numpy(x),
            "y":       torch.from_numpy(y),
            "horizon": horizon,
        }


# ─────────────────────────────────────────────────────────
#  Collate
# ─────────────────────────────────────────────────────────

def collate_fn(batch):
    x       = torch.stack([b["x"] for b in batch])
    y       = torch.stack([b["y"] for b in batch])
    horizon = torch.tensor([b["horizon"] for b in batch], dtype=torch.long)
    return {"x": x, "y": y, "horizon": horizon}


def unique_window_loader(loader: DataLoader, batch_size: Optional[int] = None) -> DataLoader:
    """Dedupe an eval DataLoader down to one row per window.

    TimeSeriesDataset.__len__ is len(indices) * len(horizons): each window
    gets one row per horizon label, but the label only affects how far into
    `y` a caller slices — `x` (and the full-length `y`) is identical across
    a window's len(horizons) copies. Eval code re-slices y[:, :H, :] itself
    and never reads the label, so iterating the raw loader once per H
    forwards every window len(horizons) times *per H* — a len(horizons)**2
    blowup. On wide-channel zero-shot sets (e.g. Traffic, C=862) that extra
    factor was enough to OOM host RAM. This selects the hor_idx==0 copy of
    every window, so each window is forwarded exactly once per H.
    """
    ds = loader.dataset
    n_h = len(ds.horizons)
    sub_indices = list(range(0, len(ds), n_h))
    subset = Subset(ds, sub_indices)
    return DataLoader(
        subset, batch_size=batch_size or loader.batch_size, shuffle=False,
        collate_fn=collate_fn,
    )


# ─────────────────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────────────────

def create_dataloaders(
    dataset_name:  str,
    root_path:     str       = "./data",
    seq_len:       int       = 336,
    horizons:      List[int] = None,
    batch_size:    int       = 32,
    num_workers:   int       = 0,
    train_stride:  int       = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:

    if horizons is None:
        horizons = [96, 192, 336, 720]

    path = download_dataset(dataset_name, root_path)
    data = load_csv(path)
    print(f"Loaded {dataset_name}: shape={data.shape}")

    train_ratio, val_ratio, test_ratio = get_split(dataset_name)
    print(f"Split: train={train_ratio:.0%} / val={val_ratio:.0%} / test={test_ratio:.0%}")

    T = len(data)
    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))

    shared_kw = dict(
        seq_len=seq_len, horizons=horizons,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
    )

    train_ds = TimeSeriesDataset(data, split="train", stride=train_stride, **shared_kw)
    val_ds   = TimeSeriesDataset(data, split="val",   stride=1,            **shared_kw)
    test_ds  = TimeSeriesDataset(data, split="test",  stride=1,            **shared_kw)

    # Share scaler fitted on train
    for ds in [val_ds, test_ds]:
        ds.scaler = train_ds.scaler

    # Re-normalise val/test with train scaler
    val_raw  = data[train_end:val_end].astype(np.float32)
    test_raw = data[val_end:].astype(np.float32)
    val_ds.data  = train_ds.scaler.transform(val_raw).astype(np.float32)
    test_ds.data = train_ds.scaler.transform(test_raw).astype(np.float32)

    # Rebuild indices after data replacement
    for ds in [val_ds, test_ds]:
        max_start = len(ds.data) - seq_len - max(horizons)
        ds.indices = list(range(0, max_start + 1, 1)) if max_start >= 0 else []

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
    )

    print(f"Train: {len(train_ds):,} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds):,} samples ({len(val_loader)} batches)")
    print(f"Test:  {len(test_ds):,} samples ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader, train_ds.scaler


# ─────────────────────────────────────────────────────────
#  Self-tests
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Dataset unit tests (synthetic data)")
    print("=" * 55)

    T, C = 2000, 7
    data = np.random.randn(T, C).astype(np.float32)

    ds = TimeSeriesDataset(
        data, seq_len=336, horizons=[96, 192, 336, 720],
        split="train", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
    )
    print(f"Train dataset: {len(ds)} samples")

    item = ds[0]
    assert item["x"].shape == (336, C)
    assert item["y"].shape == (720, C)
    print(f"[OK] shapes x={item['x'].shape} y={item['y'].shape} H={item['horizon']}")

    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
    batch  = next(iter(loader))
    assert batch["x"].shape == (8, 336, C)
    print(f"[OK] batch x={batch['x'].shape}")

    # Test split ratios for different datasets
    for ds_name, expected in [("ETTh1", (0.6, 0.2, 0.2)), ("Weather", (0.7, 0.1, 0.2))]:
        got = get_split(ds_name)
        assert got == expected, f"{ds_name}: {got} != {expected}"
        print(f"[OK] {ds_name} split = {got}")

    print("\n✓ All dataset tests passed")
