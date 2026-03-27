"""PyTorch training harness for the two-head rank+suit card CNN.

Trains a CNN with a shared backbone and separate heads for rank
(13 classes) and suit (4 classes) classification.  Exports weights
as a numpy ``.npz`` file compatible with
:class:`~src.detection.cnn_detector.CNNDetector`.

Supports training on synthetic data from the synthetic generator
and/or real card images.

Usage::

    python -m src.detection.train_cnn --data-dir data/synthetic
    python -m src.detection.train_cnn --epochs 30 --lr 0.001
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError:
    print("PyTorch is required for training.")
    print("Install with: pip3 install torch --break-system-packages")
    sys.exit(1)

from src.detection.card import Rank, Suit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
MODEL_OUT = PROJECT_ROOT / "models" / "card_detector.npz"

INPUT_SIZE = 64
NUM_RANK_CLASSES = 13
NUM_SUIT_CLASSES = 4

ALL_RANKS: list[Rank] = list(Rank)
ALL_SUITS: list[Suit] = list(Suit)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TwoHeadCardCNN(nn.Module):
    """Two-head CNN for rank and suit classification.

    Architecture: shared backbone (3× Conv+BN+ReLU+MaxPool) with
    separate FC heads for rank (13 classes) and suit (4 classes).
    """

    def __init__(self, input_size: int = INPUT_SIZE) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)

        pool_size = input_size // 8  # 3 rounds of MaxPool2d(2)
        flat_size = 128 * pool_size * pool_size

        self.rank_fc1 = nn.Linear(flat_size, 128)
        self.rank_fc2 = nn.Linear(128, NUM_RANK_CLASSES)

        self.suit_fc1 = nn.Linear(flat_size, 64)
        self.suit_fc2 = nn.Linear(64, NUM_SUIT_CLASSES)

        self.dropout = nn.Dropout(0.3)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Backbone
        x = torch.max_pool2d(torch.relu(self.bn1(self.conv1(x))), 2)
        x = torch.max_pool2d(torch.relu(self.bn2(self.conv2(x))), 2)
        x = torch.max_pool2d(torch.relu(self.bn3(self.conv3(x))), 2)
        x = x.flatten(1)

        # Rank head
        rank = self.dropout(torch.relu(self.rank_fc1(x)))
        rank = self.rank_fc2(rank)

        # Suit head
        suit = self.dropout(torch.relu(self.suit_fc1(x)))
        suit = self.suit_fc2(suit)

        return rank, suit


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SyntheticCardDataset(Dataset):
    """Dataset loading synthetic corner crops from the generator output.

    Expects a directory with ``images/`` subdirectory and ``manifest.csv``
    mapping filenames to rank and suit indices.
    """

    def __init__(self, data_dir: Path, input_size: int = INPUT_SIZE) -> None:
        self._data_dir = data_dir
        self._input_size = input_size
        self._items: list[tuple[Path, int, int]] = []

        manifest_path = data_dir / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Run: python -m src.detection.synthetic_generator"
            )

        img_dir = data_dir / "images"
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                filepath = img_dir / row["filename"]
                if filepath.exists():
                    self._items.append((
                        filepath,
                        int(row["rank_idx"]),
                        int(row["suit_idx"]),
                    ))

        logger.info("Loaded %d samples from %s", len(self._items), data_dir)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        filepath, rank_idx, suit_idx = self._items[idx]
        img = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
        if img is None:
            # Return a blank image on read failure
            img = np.zeros(
                (self._input_size, self._input_size, 3), dtype=np.uint8,
            )

        img = cv2.resize(
            img, (self._input_size, self._input_size),
            interpolation=cv2.INTER_AREA,
        )
        # HWC → CHW, normalise to [0, 1]
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return tensor, rank_idx, suit_idx


# ---------------------------------------------------------------------------
# BatchNorm fusion for export
# ---------------------------------------------------------------------------


def _fuse_bn_conv(
    conv: nn.Conv2d, bn: nn.BatchNorm2d,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse BatchNorm into Conv2d weights for inference.

    Returns fused (weight, bias) as numpy arrays.
    """
    w = conv.weight.data.clone()
    b = conv.bias.data.clone()
    mean = bn.running_mean
    var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)
    w_fused = w * scale.reshape(-1, 1, 1, 1)
    b_fused = (b - mean) * scale + beta
    return w_fused.detach().cpu().numpy(), b_fused.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _export_npz(model: TwoHeadCardCNN, path: Path) -> None:
    """Save model weights as ``.npz`` for numpy-only inference.

    Fuses BatchNorm into conv layers for efficient inference.
    """
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)

    conv1_w, conv1_b = _fuse_bn_conv(model.conv1, model.bn1)
    conv2_w, conv2_b = _fuse_bn_conv(model.conv2, model.bn2)
    conv3_w, conv3_b = _fuse_bn_conv(model.conv3, model.bn3)

    state = model.state_dict()

    np.savez(
        str(path),
        # Backbone (BN fused)
        conv1_w=conv1_w,
        conv1_b=conv1_b,
        conv2_w=conv2_w,
        conv2_b=conv2_b,
        conv3_w=conv3_w,
        conv3_b=conv3_b,
        # Rank head (transposed for x @ W + b)
        rank_fc1_w=state["rank_fc1.weight"].cpu().numpy().T,
        rank_fc1_b=state["rank_fc1.bias"].cpu().numpy(),
        rank_fc2_w=state["rank_fc2.weight"].cpu().numpy().T,
        rank_fc2_b=state["rank_fc2.bias"].cpu().numpy(),
        # Suit head (transposed for x @ W + b)
        suit_fc1_w=state["suit_fc1.weight"].cpu().numpy().T,
        suit_fc1_b=state["suit_fc1.bias"].cpu().numpy(),
        suit_fc2_w=state["suit_fc2.weight"].cpu().numpy().T,
        suit_fc2_b=state["suit_fc2.bias"].cpu().numpy(),
    )
    logger.info("Exported two-head weights to %s", path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_epoch(
    model: TwoHeadCardCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Train one epoch. Returns (loss, rank_acc, suit_acc, exact_acc)."""
    model.train()
    total_loss = 0.0
    rank_correct = 0
    suit_correct = 0
    exact_correct = 0
    total = 0

    for images, rank_labels, suit_labels in loader:
        images = images.to(device)
        rank_labels = rank_labels.to(device)
        suit_labels = suit_labels.to(device)

        optimizer.zero_grad()
        rank_logits, suit_logits = model(images)

        loss_rank = criterion(rank_logits, rank_labels)
        loss_suit = criterion(suit_logits, suit_labels)
        loss = loss_rank + loss_suit

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        rank_pred = rank_logits.argmax(1)
        suit_pred = suit_logits.argmax(1)
        rank_correct += (rank_pred == rank_labels).sum().item()
        suit_correct += (suit_pred == suit_labels).sum().item()
        exact_correct += (
            (rank_pred == rank_labels) & (suit_pred == suit_labels)
        ).sum().item()
        total += images.size(0)

    return (
        total_loss / total,
        rank_correct / total,
        suit_correct / total,
        exact_correct / total,
    )


@torch.no_grad()
def _evaluate(
    model: TwoHeadCardCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Evaluate model. Returns (loss, rank_acc, suit_acc, exact_acc)."""
    model.eval()
    total_loss = 0.0
    rank_correct = 0
    suit_correct = 0
    exact_correct = 0
    total = 0

    for images, rank_labels, suit_labels in loader:
        images = images.to(device)
        rank_labels = rank_labels.to(device)
        suit_labels = suit_labels.to(device)

        rank_logits, suit_logits = model(images)
        loss = criterion(rank_logits, rank_labels) + criterion(
            suit_logits, suit_labels,
        )

        total_loss += loss.item() * images.size(0)
        rank_pred = rank_logits.argmax(1)
        suit_pred = suit_logits.argmax(1)
        rank_correct += (rank_pred == rank_labels).sum().item()
        suit_correct += (suit_pred == suit_labels).sum().item()
        exact_correct += (
            (rank_pred == rank_labels) & (suit_pred == suit_labels)
        ).sum().item()
        total += images.size(0)

    return (
        total_loss / total,
        rank_correct / total,
        suit_correct / total,
        exact_correct / total,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train two-head rank+suit card detection CNN",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=MODEL_OUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--val-split", type=float, default=0.15,
        help="Fraction of data to use for validation (default: 0.15)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.data_dir.is_dir():
        logger.error(
            "Dataset not found at %s. Run: python -m src.detection.synthetic_generator",
            args.data_dir,
        )
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load dataset and split into train/val
    full_dataset = SyntheticCardDataset(args.data_dir)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    logger.info(
        "Dataset: %d train, %d val samples",
        len(train_ds), len(val_ds),
    )

    model = TwoHeadCardCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    best_val_exact = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_rank, train_suit, train_exact = _train_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_rank, val_suit, val_exact = _evaluate(
            model, val_loader, criterion, device,
        )
        scheduler.step(val_loss)

        logger.info(
            "Epoch %2d/%d  loss=%.4f  "
            "rank=%.3f/%.3f  suit=%.3f/%.3f  exact=%.3f/%.3f",
            epoch,
            args.epochs,
            train_loss,
            train_rank,
            val_rank,
            train_suit,
            val_suit,
            train_exact,
            val_exact,
        )

        if val_exact > best_val_exact:
            best_val_exact = val_exact
            _export_npz(model, args.output)
            logger.info("  -> New best model saved (exact=%.3f)", val_exact)

    logger.info("Best validation exact-match accuracy: %.3f", best_val_exact)
    logger.info("Model saved to %s", args.output)


if __name__ == "__main__":
    main()
