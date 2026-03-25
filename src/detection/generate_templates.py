"""Generate synthetic card template images for template matching.

Creates simple but recognizable card images for all 52 cards and
saves them to ``data/templates/``. Each template is a white rectangle
with a black border, rank text in the top-left corner, and a colored
suit symbol.

Usage:
    python -m src.detection.generate_templates
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.detection.card import Card, Rank, Suit

logger = logging.getLogger(__name__)

# Template dimensions
TEMPLATE_WIDTH = 60
TEMPLATE_HEIGHT = 80


def generate_card_template(card: Card) -> np.ndarray:
    """Generate a synthetic card template image.

    Creates a white card with black border, rank in top-left,
    and suit symbol centered below the rank.

    Args:
        card: The card to generate a template for.

    Returns:
        BGR numpy array of the card template image.
    """
    img = np.ones((TEMPLATE_HEIGHT, TEMPLATE_WIDTH, 3), dtype=np.uint8) * 255

    # Black border
    cv2.rectangle(img, (0, 0), (TEMPLATE_WIDTH - 1, TEMPLATE_HEIGHT - 1), (0, 0, 0), 2)

    # Get suit color in BGR
    color = card.suit.color

    # Draw rank text in top-left
    rank_text = card.rank.value
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 if len(rank_text) == 1 else 0.5
    cv2.putText(img, rank_text, (5, 20), font, font_scale, color, 2, cv2.LINE_AA)

    # Draw suit symbol centered
    suit_text = card.suit.symbol
    cv2.putText(img, suit_text, (5, 40), font, 0.5, color, 1, cv2.LINE_AA)

    # Draw larger rank + suit in center area for better matching
    cv2.putText(
        img, rank_text, (18, 55), font, 0.7 if len(rank_text) == 1 else 0.55,
        color, 2, cv2.LINE_AA,
    )
    cv2.putText(img, suit_text, (20, 72), font, 0.6, color, 1, cv2.LINE_AA)

    return img


def generate_all_templates(output_dir: str | Path) -> int:
    """Generate template images for all 52 playing cards.

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


def main() -> None:
    """Entry point for template generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project_root = Path(__file__).resolve().parent.parent.parent
    template_dir = project_root / "data" / "templates"

    count = generate_all_templates(template_dir)
    print(f"Generated {count} card templates in {template_dir}")


if __name__ == "__main__":
    main()
