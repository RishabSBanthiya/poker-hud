"""PyTorch training harness for the card detection CNN.

Trains a lightweight CNN on the Kaggle cards-classification dataset
and exports weights as a numpy ``.npz`` file compatible with
:class:`~src.detection.cnn_detector.CNNDetector`.

Usage::

    python -m src.detection.train_cnn          # train with defaults
    python -m src.detection.train_cnn --epochs 30 --lr 0.001

The script expects the dataset at ``data/external/kaggle_cards_classification/``
with subdirectories ``train/``, ``valid/``, ``test/`` each containing one folder
per class (e.g. ``ace of spades/``, ``two of hearts/``, etc.).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
except ImportError:
    print("PyTorch and torchvision are required for training.")
    print("Install with: pip3 install torch torchvision --break-system-packages")
    sys.exit(1)

from src.detection.card import Rank, Suit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "external" / "kaggle_cards_classification"
MODEL_OUT = PROJECT_ROOT / "models" / "card_detector.npz"

INPUT_SIZE = 64
NUM_CLASSES = 52  # 52 standard cards, no joker
NUM_FILTERS = 32
KERNEL_SIZE = 5

# Map from Kaggle folder names → (Rank, Suit) for the 52 standard cards.
# Order must match ``cnn_detector._ALL_CARDS``: suit-major, rank-minor.
_ALL_CARDS: list[tuple[Rank, Suit]] = [
    (rank, suit) for suit in Suit for rank in Rank
]

_RANK_FROM_NAME: dict[str, Rank] = {
    "ace": Rank.ACE,
    "two": Rank.TWO,
    "three": Rank.THREE,
    "four": Rank.FOUR,
    "five": Rank.FIVE,
    "six": Rank.SIX,
    "seven": Rank.SEVEN,
    "eight": Rank.EIGHT,
    "nine": Rank.NINE,
    "ten": Rank.TEN,
    "jack": Rank.JACK,
    "queen": Rank.QUEEN,
    "king": Rank.KING,
}

_SUIT_FROM_NAME: dict[str, Suit] = {
    "clubs": Suit.CLUBS,
    "diamonds": Suit.DIAMONDS,
    "hearts": Suit.HEARTS,
    "spades": Suit.SPADES,
}


def _folder_to_index(folder_name: str) -> int | None:
    """Convert a Kaggle folder name like ``'ace of spades'`` to class index.

    Returns ``None`` for the joker class which is excluded.
    """
    parts = folder_name.strip().lower().split(" of ")
    if len(parts) != 2:
        return None  # "joker" or unexpected format
    rank_str, suit_str = parts
    rank = _RANK_FROM_NAME.get(rank_str)
    suit = _SUIT_FROM_NAME.get(suit_str)
    if rank is None or suit is None:
        return None
    try:
        return _ALL_CARDS.index((rank, suit))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CardCNN(nn.Module):
    """Minimal CNN matching the numpy inference in ``CNNDetector``.

    Architecture: Conv2d → ReLU → Flatten → Linear → (softmax at inference).
    """

    def __init__(
        self,
        num_filters: int = NUM_FILTERS,
        kernel_size: int = KERNEL_SIZE,
        input_size: int = INPUT_SIZE,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size, bias=True)
        conv_out_size = input_size - kernel_size + 1
        flat_size = num_filters * conv_out_size * conv_out_size
        self.fc = nn.Linear(flat_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _build_label_remap(class_to_idx: dict[str, int]) -> dict[int, int]:
    """Build a mapping from torchvision class indices to our 52-card indices.

    Entries whose folder name doesn't map (joker) are excluded.
    """
    remap: dict[int, int] = {}
    for folder_name, tv_idx in class_to_idx.items():
        card_idx = _folder_to_index(folder_name)
        if card_idx is not None:
            remap[tv_idx] = card_idx
    return remap


class _RemappedDataset(torch.utils.data.Dataset):
    """Wraps an ``ImageFolder`` dataset to remap labels and skip jokers."""

    def __init__(
        self, base_dataset: datasets.ImageFolder, remap: dict[int, int]
    ) -> None:
        self._items = [
            (path, remap[label])
            for path, label in base_dataset.samples
            if label in remap
        ]
        self._transform = base_dataset.transform
        self._loader = base_dataset.loader

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self._items[idx]
        img = self._loader(path)
        if self._transform is not None:
            img = self._transform(img)
        return img, label


def _get_loaders(
    data_dir: Path, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / validation / test data loaders."""
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_tf)
    valid_ds = datasets.ImageFolder(str(data_dir / "valid"), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(data_dir / "test"), transform=eval_tf)

    remap = _build_label_remap(train_ds.class_to_idx)
    logger.info("Mapped %d / %d classes to 52-card indices", len(remap), len(train_ds.class_to_idx))

    train_loader = DataLoader(
        _RemappedDataset(train_ds, remap),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        _RemappedDataset(valid_ds, remap),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        _RemappedDataset(test_ds, remap),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, valid_loader, test_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_epoch(
    model: CardCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate(
    model: CardCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _export_npz(model: CardCNN, path: Path) -> None:
    """Save model weights as ``.npz`` for numpy-only inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    np.savez(
        str(path),
        conv1_w=state["conv1.weight"].cpu().numpy(),
        conv1_b=state["conv1.bias"].cpu().numpy(),
        fc_w=state["fc.weight"].cpu().numpy().T,  # transpose for x @ W + b
        fc_b=state["fc.bias"].cpu().numpy(),
    )
    logger.info("Exported weights to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train card detection CNN")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=MODEL_OUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.data_dir.is_dir():
        logger.error(
            "Dataset not found at %s. See data/external/README.md for download instructions.",
            args.data_dir,
        )
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, valid_loader, test_loader = _get_loaders(
        args.data_dir, args.batch_size
    )
    logger.info(
        "Dataset: %d train, %d valid, %d test samples",
        len(train_loader.dataset),
        len(valid_loader.dataset),
        len(test_loader.dataset),
    )

    model = CardCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = _evaluate(model, valid_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info(
            "Epoch %2d/%d  train_loss=%.4f train_acc=%.3f  val_loss=%.4f val_acc=%.3f",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _export_npz(model, args.output)
            logger.info("  -> New best model saved (val_acc=%.3f)", val_acc)

    # Final test evaluation
    test_loss, test_acc = _evaluate(model, test_loader, criterion, device)
    logger.info("Test accuracy: %.3f (loss=%.4f)", test_acc, test_loss)
    logger.info("Best validation accuracy: %.3f", best_val_acc)
    logger.info("Model saved to %s", args.output)


if __name__ == "__main__":
    main()
