"""Generate synthetic card template images for template matching.

Creates recognizable card images for all 52 cards at multiple scales
and saves them to ``data/templates/``. Each template is a white rectangle
with a black border, rank text in the top-left corner, and a colored
suit symbol.  Anti-aliased rendering and slight Gaussian blur improve
match robustness against real-world screenshots.

Usage:
    python -m src.detection.generate_templates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detection.card import Card, Rank, Suit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scale presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemplateScale:
    """Defines a named size preset for card templates.

    Attributes:
        name: Human-readable label (e.g. ``"small"``).
        width: Template width in pixels.
        height: Template height in pixels.
    """

    name: str
    width: int
    height: int


SCALES: tuple[TemplateScale, ...] = (
    TemplateScale("small", 40, 54),
    TemplateScale("medium", 60, 80),
    TemplateScale("large", 90, 120),
)

# Default (medium) dimensions kept for backwards compatibility
TEMPLATE_WIDTH = 60
TEMPLATE_HEIGHT = 80


def _render_card_at_size(card: Card, width: int, height: int) -> np.ndarray:
    """Render a synthetic card image at the given pixel size.

    The card has a white background, thin black border, rank + suit text
    drawn with anti-aliasing, and a light Gaussian blur to smooth edges.

    Args:
        card: Card identity to render.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        BGR numpy array of shape ``(height, width, 3)``.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Black border (thickness scales with size)
    border = max(1, int(round(width / 30)))
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (0, 0, 0), border)

    color = card.suit.color
    rank_text = card.rank.value
    suit_text = card.suit.symbol
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Scale factors relative to medium (60x80)
    sx = width / 60.0
    sy = height / 80.0
    s = min(sx, sy)

    # --- Top-left rank + suit (corner index) ---
    rank_scale = 0.6 * s if len(rank_text) == 1 else 0.5 * s
    rank_thickness = max(1, int(round(2 * s)))
    cv2.putText(
        img, rank_text,
        (int(5 * sx), int(20 * sy)),
        font, rank_scale, color, rank_thickness, cv2.LINE_AA,
    )
    cv2.putText(
        img, suit_text,
        (int(5 * sx), int(40 * sy)),
        font, 0.5 * s, color, max(1, int(round(s))), cv2.LINE_AA,
    )

    # --- Centre rank + suit (for robust matching) ---
    centre_rank_scale = (0.7 if len(rank_text) == 1 else 0.55) * s
    cv2.putText(
        img, rank_text,
        (int(18 * sx), int(55 * sy)),
        font, centre_rank_scale, color, rank_thickness, cv2.LINE_AA,
    )
    cv2.putText(
        img, suit_text,
        (int(20 * sx), int(72 * sy)),
        font, 0.6 * s, color, max(1, int(round(s))), cv2.LINE_AA,
    )

    # Light Gaussian blur for anti-aliasing / edge smoothing
    ksize = 3 if width <= 50 else 5
    img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img


def generate_card_template(card: Card) -> np.ndarray:
    """Generate a synthetic card template image at the default (medium) size.

    Args:
        card: The card to generate a template for.

    Returns:
        BGR numpy array of the card template image (80 x 60).
    """
    return _render_card_at_size(card, TEMPLATE_WIDTH, TEMPLATE_HEIGHT)


def generate_all_templates(output_dir: str | Path) -> int:
    """Generate template images for all 52 playing cards at the default scale.

    Args:
        output_dir: Directory to save template images.

    Returns:
        Number of templates generated.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for suit in Suit:
        for rank in Rank:
            card = Card(rank=rank, suit=suit)
            img = generate_card_template(card)
            filename = f"{card.name}.png"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), img)
            count += 1

    logger.info("Generated %d card templates in %s", count, output_path)
    return count


def generate_multiscale_templates(
    output_dir: str | Path,
    scales: tuple[TemplateScale, ...] | None = None,
) -> int:
    """Generate template images for all 52 cards at every requested scale.

    Templates are written into per-scale subdirectories::

        output_dir/
            small/   (40x54)
            medium/  (60x80)
            large/   (90x120)

    Args:
        output_dir: Root directory for template output.
        scales: Scale presets to generate.  Defaults to ``SCALES``.

    Returns:
        Total number of template images written.
    """
    if scales is None:
        scales = SCALES

    output_path = Path(output_dir)
    total = 0

    for scale in scales:
        scale_dir = output_path / scale.name
        scale_dir.mkdir(parents=True, exist_ok=True)

        for suit in Suit:
            for rank in Rank:
                card = Card(rank=rank, suit=suit)
                img = _render_card_at_size(card, scale.width, scale.height)
                filepath = scale_dir / f"{card.name}.png"
                cv2.imwrite(str(filepath), img)
                total += 1

    logger.info(
        "Generated %d multiscale templates (%d scales) in %s",
        total, len(scales), output_path,
    )
    return total


def main() -> None:
    """Entry point for template generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project_root = Path(__file__).resolve().parent.parent.parent
    template_dir = project_root / "data" / "templates"

    count = generate_all_templates(template_dir)
    print(f"Generated {count} card templates in {template_dir}")

    ms_count = generate_multiscale_templates(template_dir)
    print(f"Generated {ms_count} multiscale card templates in {template_dir}")


if __name__ == "__main__":
    main()
