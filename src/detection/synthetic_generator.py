"""Synthetic card corner crop generator for CNN training.

Generates diverse rank+suit corner images (the top-left corner of a
playing card showing rank text and suit symbol) with variation in fonts,
backgrounds, text colours, and augmentations.  Designed to produce
training data that generalises across online poker clients.

Usage::

    python -m src.detection.synthetic_generator                  # defaults
    python -m src.detection.synthetic_generator --samples 100    # more per card
    python -m src.detection.synthetic_generator --output data/synthetic
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detection.card import Rank, Suit

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "synthetic"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Background colours (BGR) — platform-agnostic poker table colours
BG_COLORS: list[tuple[int, int, int]] = [
    # Green felt variants
    (34, 120, 50),
    (20, 100, 40),
    (40, 140, 60),
    (50, 130, 70),
    # Dark backgrounds
    (30, 30, 30),
    (50, 50, 50),
    (40, 40, 45),
    # Navy / blue tables
    (80, 40, 20),
    (100, 60, 30),
    (90, 50, 25),
    # Red felt
    (30, 30, 120),
    (20, 20, 100),
    # Light / white card face
    (230, 230, 230),
    (240, 240, 240),
    (200, 200, 200),
    (255, 255, 255),
]

# Suit text colours (BGR)
RED_SUIT_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 200),
    (0, 0, 180),
    (30, 30, 220),
    (0, 0, 255),
    (20, 20, 190),
]

BLACK_SUIT_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (30, 30, 30),
    (50, 50, 50),
    (20, 20, 20),
]

# White/light text for dark backgrounds
LIGHT_TEXT_COLORS: list[tuple[int, int, int]] = [
    (255, 255, 255),
    (230, 230, 230),
    (210, 210, 210),
]

# OpenCV built-in fonts
CV2_FONTS: list[int] = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
]

# Rank display text (what appears on the card corner)
RANK_TEXT: dict[Rank, str] = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "10",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}

SUIT_SYMBOL: dict[Suit, str] = {
    Suit.HEARTS: "\u2665",
    Suit.DIAMONDS: "\u2666",
    Suit.CLUBS: "\u2663",
    Suit.SPADES: "\u2660",
}


def _draw_suit(
    img: np.ndarray,
    suit: Suit,
    center: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a suit symbol using OpenCV drawing primitives.

    OpenCV's Hershey fonts don't render Unicode suit symbols distinctly,
    so we draw them manually with geometric shapes.
    """
    cx, cy = center
    s = max(size, 4)
    half = s // 2

    if suit == Suit.HEARTS:
        # Heart: two circles on top, triangle below
        r = half // 2
        pts = np.array([
            [cx - half, cy - r // 2],
            [cx, cy + half],
            [cx + half, cy - r // 2],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, color)
        cv2.circle(img, (cx - r, cy - r // 2), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx + r, cy - r // 2), r, color, -1, cv2.LINE_AA)

    elif suit == Suit.DIAMONDS:
        # Diamond: rotated square
        pts = np.array([
            [cx, cy - half],
            [cx + half, cy],
            [cx, cy + half],
            [cx - half, cy],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, color)

    elif suit == Suit.CLUBS:
        # Club: three circles + stem
        r = half // 2 + 1
        cv2.circle(img, (cx, cy - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx - r, cy + r // 2), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx + r, cy + r // 2), r, color, -1, cv2.LINE_AA)
        # Stem
        stem_w = max(2, s // 8)
        cv2.rectangle(
            img,
            (cx - stem_w, cy + r // 2),
            (cx + stem_w, cy + half),
            color,
            -1,
        )

    elif suit == Suit.SPADES:
        # Spade: inverted heart + stem
        r = half // 2
        pts = np.array([
            [cx - half, cy + r // 2],
            [cx, cy - half],
            [cx + half, cy + r // 2],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, color)
        cv2.circle(img, (cx - r, cy + r // 2), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx + r, cy + r // 2), r, color, -1, cv2.LINE_AA)
        # Stem
        stem_w = max(2, s // 8)
        cv2.rectangle(
            img,
            (cx - stem_w, cy + r),
            (cx + stem_w, cy + half),
            color,
            -1,
        )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for synthetic corner crop generation.

    Attributes:
        output_dir: Root directory for generated images.
        crop_size: Output image dimensions (width, height).
        samples_per_card: Number of samples per (rank, suit) pair.
        brightness_range: Random brightness multiplier range.
        contrast_range: Random contrast multiplier range.
        noise_std_range: Range for Gaussian noise standard deviation.
        rotation_range: Max rotation in degrees (±).
        blur_prob: Probability of applying Gaussian blur.
        seed: Random seed for reproducibility (None for random).
    """

    output_dir: Path = DEFAULT_OUTPUT
    crop_size: tuple[int, int] = (64, 64)
    samples_per_card: int = 50
    brightness_range: tuple[float, float] = (0.7, 1.3)
    contrast_range: tuple[float, float] = (0.7, 1.3)
    noise_std_range: tuple[float, float] = (0.0, 15.0)
    rotation_range: float = 5.0
    blur_prob: float = 0.3
    seed: int | None = 42


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pick_text_color(
    suit: Suit,
    bg_color: tuple[int, int, int],
    rng: random.Random,
) -> tuple[int, int, int]:
    """Choose an appropriate text colour for the given suit and background.

    Uses light text on dark backgrounds and standard suit colours on
    light backgrounds.
    """
    # Estimate background brightness
    brightness = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]

    if brightness < 80:
        # Dark background — use light text colours
        return rng.choice(LIGHT_TEXT_COLORS)

    # Light or medium background — use suit-appropriate colours
    if suit in (Suit.HEARTS, Suit.DIAMONDS):
        return rng.choice(RED_SUIT_COLORS)
    return rng.choice(BLACK_SUIT_COLORS)


def render_corner_crop(
    rank: Rank,
    suit: Suit,
    size: tuple[int, int],
    *,
    bg_color: tuple[int, int, int] | None = None,
    text_color: tuple[int, int, int] | None = None,
    font: int | None = None,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Render a single synthetic card corner crop.

    Args:
        rank: Card rank to render.
        suit: Card suit to render.
        size: Output (width, height).
        bg_color: Background colour (BGR). Random if None.
        text_color: Text colour (BGR). Auto-selected if None.
        font: OpenCV font constant. Random if None.
        rng: Random number generator.

    Returns:
        BGR numpy array of shape (height, width, 3).
    """
    if rng is None:
        rng = random.Random()

    w, h = size
    if bg_color is None:
        bg_color = rng.choice(BG_COLORS)
    if text_color is None:
        text_color = _pick_text_color(suit, bg_color, rng)
    if font is None:
        font = rng.choice(CV2_FONTS)

    # Create background
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)

    rank_text = RANK_TEXT[rank]

    # Scale factors — font size relative to crop size
    scale_factor = min(w, h) / 64.0

    # Rank text — top portion
    rank_scale = rng.uniform(0.7, 1.0) * scale_factor
    rank_thickness = max(1, int(round(rng.uniform(1.5, 2.5) * scale_factor)))

    # Position with slight randomness
    rank_x = int(rng.uniform(3, 10) * scale_factor)
    rank_y = int(rng.uniform(18, 26) * scale_factor)

    cv2.putText(
        img,
        rank_text,
        (rank_x, rank_y),
        font,
        rank_scale,
        text_color,
        rank_thickness,
        cv2.LINE_AA,
    )

    # Suit symbol — drawn with geometric primitives below rank
    suit_cx = int(rng.uniform(12, 22) * scale_factor)
    suit_cy = int(rng.uniform(40, 48) * scale_factor)
    suit_size = int(rng.uniform(10, 16) * scale_factor)

    _draw_suit(img, suit, (suit_cx, suit_cy), suit_size, text_color)

    return img


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


def _augment(
    img: np.ndarray,
    config: SyntheticConfig,
    rng: random.Random,
) -> np.ndarray:
    """Apply random augmentations to a rendered corner crop.

    Augmentations: brightness, contrast, Gaussian noise, rotation, blur.
    """
    h, w = img.shape[:2]
    result = img.astype(np.float32)

    # Brightness
    brightness = rng.uniform(*config.brightness_range)
    result = result * brightness

    # Contrast
    contrast = rng.uniform(*config.contrast_range)
    mean = result.mean()
    result = (result - mean) * contrast + mean

    # Gaussian noise
    noise_std = rng.uniform(*config.noise_std_range)
    if noise_std > 0:
        noise = np.random.RandomState(rng.randint(0, 2**31)).normal(
            0, noise_std, result.shape,
        ).astype(np.float32)
        result = result + noise

    result = np.clip(result, 0, 255).astype(np.uint8)

    # Rotation
    if config.rotation_range > 0:
        angle = rng.uniform(-config.rotation_range, config.rotation_range)
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(
            result, matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )

    # Gaussian blur
    if rng.random() < config.blur_prob:
        ksize = rng.choice([3, 5])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    return result


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

# Ordered lists matching enum declaration order (used for index mapping)
ALL_RANKS: list[Rank] = list(Rank)
ALL_SUITS: list[Suit] = list(Suit)


def generate_dataset(config: SyntheticConfig | None = None) -> int:
    """Generate the full synthetic training dataset.

    Creates a directory structure with corner crop images and a CSV
    manifest mapping each image to its rank and suit indices.

    Directory layout::

        {output_dir}/
            images/
                rank{r}_suit{s}_{n}.png
            manifest.csv

    Args:
        config: Generation configuration. Uses defaults if None.

    Returns:
        Total number of images generated.
    """
    if config is None:
        config = SyntheticConfig()

    rng = random.Random(config.seed)
    output = Path(config.output_dir)
    img_dir = output / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int]] = []
    total = 0

    for rank_idx, rank in enumerate(ALL_RANKS):
        for suit_idx, suit in enumerate(ALL_SUITS):
            for sample_idx in range(config.samples_per_card):
                # Render base image
                img = render_corner_crop(
                    rank, suit, config.crop_size, rng=rng,
                )

                # Apply augmentations
                img = _augment(img, config, rng)

                # Save
                filename = f"rank{rank_idx}_suit{suit_idx}_{sample_idx:04d}.png"
                cv2.imwrite(str(img_dir / filename), img)

                manifest_rows.append({
                    "filename": filename,
                    "rank_idx": rank_idx,
                    "suit_idx": suit_idx,
                    "rank": rank.value,
                    "suit": suit.value,
                })
                total += 1

    # Write manifest CSV
    manifest_path = output / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "rank_idx", "suit_idx", "rank", "suit"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(
        "Generated %d synthetic corner crops in %s", total, output,
    )
    return total


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic card corner crops for CNN training",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory (default: data/synthetic)",
    )
    parser.add_argument(
        "--samples", type=int, default=50,
        help="Samples per (rank, suit) pair (default: 50)",
    )
    parser.add_argument(
        "--size", type=int, default=64,
        help="Crop size in pixels (default: 64)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = SyntheticConfig(
        output_dir=args.output,
        crop_size=(args.size, args.size),
        samples_per_card=args.samples,
        seed=args.seed,
    )

    total = generate_dataset(config)
    print(f"Generated {total} synthetic corner crops in {args.output}")


if __name__ == "__main__":
    main()
