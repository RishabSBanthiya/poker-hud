"""Dataset management for card detection templates.

Provides functions to load, save, and manage template image sets with
metadata describing each template's card identity, scale, and source.
Designed to support both synthetic and real captured card images.

Usage:
    from src.detection.dataset import TemplateDataset, TemplateMetadata

    dataset = TemplateDataset.from_directory("data/templates")
    templates = dataset.get_templates_for_card(Card(Rank.ACE, Suit.SPADES))
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.detection.card import Card, Rank, Suit

logger = logging.getLogger(__name__)

# Re-use the filename-to-enum maps from template_matcher
_RANK_MAP: dict[str, Rank] = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "10": Rank.TEN,
    "jack": Rank.JACK,
    "queen": Rank.QUEEN,
    "king": Rank.KING,
    "ace": Rank.ACE,
}

_SUIT_MAP: dict[str, Suit] = {
    "hearts": Suit.HEARTS,
    "diamonds": Suit.DIAMONDS,
    "clubs": Suit.CLUBS,
    "spades": Suit.SPADES,
}


@dataclass(frozen=True)
class TemplateMetadata:
    """Metadata describing a single card template image.

    Attributes:
        rank: Card rank identifier (e.g. ``"ace"``).
        suit: Card suit identifier (e.g. ``"spades"``).
        scale: Size label (``"small"``, ``"medium"``, ``"large"``).
        source: Origin of the template (``"synthetic"`` or ``"captured"``).
        width: Image width in pixels.
        height: Image height in pixels.
        filename: Filename relative to the dataset root.
    """

    rank: str
    suit: str
    scale: str
    source: str
    width: int
    height: int
    filename: str


@dataclass
class TemplateEntry:
    """A template image paired with its metadata.

    Attributes:
        metadata: Descriptive metadata for this template.
        image: The template image as a BGR numpy array.
    """

    metadata: TemplateMetadata
    image: np.ndarray


@dataclass
class TemplateDataset:
    """Collection of card template images with metadata.

    Provides lookup by card identity and iteration over all entries.

    Attributes:
        root_dir: Filesystem path where templates are stored.
        entries: All loaded template entries.
    """

    root_dir: Path
    entries: list[TemplateEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        root_dir: str | Path,
        *,
        source: str = "synthetic",
    ) -> TemplateDataset:
        """Load a dataset from a directory tree.

        Supports two layouts:

        * **Flat** -- all ``{rank}_{suit}.png`` files directly in *root_dir*.
        * **Multi-scale** -- subdirectories named by scale containing PNGs.

        Args:
            root_dir: Root directory to scan.
            source: Value to record in metadata ``source`` field.

        Returns:
            Populated ``TemplateDataset``.

        Raises:
            FileNotFoundError: If *root_dir* does not exist.
        """
        root = Path(root_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Template directory not found: {root}")

        dataset = cls(root_dir=root)

        # Try multiscale subdirectories first
        subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
        if subdirs:
            for subdir in subdirs:
                scale_name = subdir.name
                dataset._load_from_dir(subdir, scale=scale_name, source=source)

        # Also load any PNGs directly in root (flat layout / default scale)
        dataset._load_from_dir(root, scale="medium", source=source)

        logger.info(
            "Loaded dataset with %d entries from %s", len(dataset.entries), root,
        )
        return dataset

    def _load_from_dir(
        self,
        directory: Path,
        *,
        scale: str,
        source: str,
    ) -> None:
        """Load PNG templates from a single directory."""
        for path in sorted(directory.glob("*.png")):
            stem = path.stem
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue

            rank_str, suit_str = parts
            if rank_str not in _RANK_MAP or suit_str not in _SUIT_MAP:
                continue

            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Failed to read image: %s", path)
                continue

            h, w = img.shape[:2]
            rel_path = str(path.relative_to(self.root_dir))
            meta = TemplateMetadata(
                rank=rank_str,
                suit=suit_str,
                scale=scale,
                source=source,
                width=w,
                height=h,
                filename=rel_path,
            )
            self.entries.append(TemplateEntry(metadata=meta, image=img))

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_templates_for_card(self, card: Card) -> list[TemplateEntry]:
        """Return all template entries matching *card*.

        Args:
            card: The card to look up.

        Returns:
            List of matching ``TemplateEntry`` objects (may be empty).
        """
        rank_str = card.name.rsplit("_", 1)[0]
        suit_str = card.suit.value
        return [
            e for e in self.entries
            if e.metadata.rank == rank_str and e.metadata.suit == suit_str
        ]

    def get_templates_at_scale(self, scale: str) -> list[TemplateEntry]:
        """Return all entries matching a given scale label.

        Args:
            scale: Scale name (e.g. ``"small"``, ``"medium"``, ``"large"``).

        Returns:
            Filtered list of entries.
        """
        return [e for e in self.entries if e.metadata.scale == scale]

    def card_count(self) -> int:
        """Return number of unique card identities in the dataset."""
        return len({(e.metadata.rank, e.metadata.suit) for e in self.entries})

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[TemplateEntry]:
        return iter(self.entries)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def add_captured_image(
        self,
        card: Card,
        image: np.ndarray,
        *,
        scale: str = "captured",
    ) -> TemplateEntry:
        """Add a real captured card image to the dataset.

        The image is saved to disk under ``{root_dir}/captured/`` and a
        new entry is appended to the dataset.

        Args:
            card: Card identity for the captured image.
            image: BGR numpy array of the captured card.
            scale: Scale label to record (default ``"captured"``).

        Returns:
            The newly created ``TemplateEntry``.
        """
        cap_dir = self.root_dir / "captured"
        cap_dir.mkdir(parents=True, exist_ok=True)

        h, w = image.shape[:2]
        filename = f"{card.name}_{w}x{h}.png"
        filepath = cap_dir / filename
        cv2.imwrite(str(filepath), image)

        rel_path = str(filepath.relative_to(self.root_dir))
        meta = TemplateMetadata(
            rank=card.name.rsplit("_", 1)[0],
            suit=card.suit.value,
            scale=scale,
            source="captured",
            width=w,
            height=h,
            filename=rel_path,
        )
        entry = TemplateEntry(metadata=meta, image=image)
        self.entries.append(entry)
        logger.info("Added captured image for %s (%dx%d)", card.name, w, h)
        return entry

    def save_metadata(self, path: str | Path | None = None) -> Path:
        """Write a JSON manifest of all template metadata.

        Args:
            path: Output file path.  Defaults to
                ``{root_dir}/metadata.json``.

        Returns:
            Path to the written file.
        """
        out = Path(path) if path else self.root_dir / "metadata.json"
        records = [asdict(e.metadata) for e in self.entries]
        out.write_text(json.dumps(records, indent=2))
        logger.info("Saved metadata for %d entries to %s", len(records), out)
        return out

    @classmethod
    def load_metadata(cls, path: str | Path) -> list[TemplateMetadata]:
        """Load a metadata manifest without loading images.

        Args:
            path: Path to a ``metadata.json`` file.

        Returns:
            List of ``TemplateMetadata`` objects.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Metadata file not found: {p}")

        records = json.loads(p.read_text())
        return [TemplateMetadata(**r) for r in records]
