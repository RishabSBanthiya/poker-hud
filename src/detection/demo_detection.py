"""Demo script for template-based card detection.

Generates a synthetic poker table image with cards placed on a green
background, runs template matching, prints results, and saves an
annotated image with bounding boxes.

Usage:
    python -m src.detection.demo_detection
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.detection.card import Card, Rank, Suit
from src.detection.generate_templates import (
    generate_all_templates,
    generate_card_template,
)
from src.detection.template_matcher import TemplateMatcher

logger = logging.getLogger(__name__)

# Table dimensions
TABLE_WIDTH = 800
TABLE_HEIGHT = 500


def create_poker_table(
    cards: list[tuple[Card, int, int]],
) -> np.ndarray:
    """Generate a synthetic poker table image with cards placed on it.

    Args:
        cards: List of (Card, x_position, y_position) tuples specifying
            which cards to place and where.

    Returns:
        BGR numpy array of the synthetic table image.
    """
    # Green felt background
    table = np.zeros((TABLE_HEIGHT, TABLE_WIDTH, 3), dtype=np.uint8)
    table[:] = (34, 120, 50)  # Dark green in BGR

    # Draw table oval
    center = (TABLE_WIDTH // 2, TABLE_HEIGHT // 2)
    cv2.ellipse(table, center, (350, 200), 0, 0, 360, (40, 140, 60), -1)
    cv2.ellipse(table, center, (350, 200), 0, 0, 360, (20, 80, 30), 3)

    # Place cards on the table
    for card, x, y in cards:
        template = generate_card_template(card)
        h, w = template.shape[:2]

        # Ensure card fits within table bounds
        if y + h > TABLE_HEIGHT or x + w > TABLE_WIDTH:
            continue

        table[y : y + h, x : x + w] = template

    return table


def annotate_frame(
    frame: np.ndarray,
    detections: list,
) -> np.ndarray:
    """Draw bounding boxes and labels on detected cards.

    Args:
        frame: The original BGR frame.
        detections: List of DetectedCard objects.

    Returns:
        Annotated copy of the frame.
    """
    annotated = frame.copy()

    for det in detections:
        x, y, w, h = det.bounding_box
        # Green bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Label
        label = f"{det.rank.value}{det.suit.symbol} {det.confidence:.2f}"
        cv2.putText(
            annotated, label, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )

    return annotated


def main() -> None:
    """Run the card detection demo."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project_root = Path(__file__).resolve().parent.parent.parent
    template_dir = project_root / "data" / "templates"
    output_dir = project_root / "data"

    # Ensure templates exist
    if not list(template_dir.glob("*.png")):
        print("Generating card templates...")
        generate_all_templates(template_dir)

    # Define cards to place on the table
    demo_cards = [
        (Card(Rank.ACE, Suit.SPADES), 300, 200),
        (Card(Rank.KING, Suit.HEARTS), 380, 200),
        (Card(Rank.QUEEN, Suit.DIAMONDS), 460, 200),
        (Card(Rank.TEN, Suit.CLUBS), 200, 300),
        (Card(Rank.SEVEN, Suit.HEARTS), 540, 200),
    ]

    print("Creating synthetic poker table...")
    table_image = create_poker_table(demo_cards)

    # Run detection
    print("Running template matching...")
    matcher = TemplateMatcher(template_dir)
    detections = matcher.detect_cards(table_image, threshold=0.8)

    # Print results
    print(f"\nDetected {len(detections)} card(s):")
    for det in detections:
        print(f"  {det}")

    # Save annotated image
    annotated = annotate_frame(table_image, detections)
    output_path = output_dir / "demo_detection_result.png"
    cv2.imwrite(str(output_path), annotated)
    print(f"\nAnnotated image saved to {output_path}")


if __name__ == "__main__":
    main()
