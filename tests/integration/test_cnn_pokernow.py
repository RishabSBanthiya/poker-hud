"""Test CNN card detector against real PokerNow screenshots.

Crops individual card regions from real PokerNow.club screenshots
and tests the trained CNN detector on each card crop.

Usage::

    python -m pytest tests/integration/test_cnn_pokernow.py -v -s
    python -m tests.integration.test_cnn_pokernow   # standalone
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from src.detection.card import DetectedCard, Rank, Suit
from src.detection.cnn_detector import CNNConfig, CNNDetector

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_DIR = PROJECT_ROOT / "tests" / "fixtures" / "pokernow_screenshots"
RESULTS_DIR = PROJECT_ROOT / "tests" / "fixtures" / "pokernow_results"

# ---------------------------------------------------------------------------
# Card crop definitions for each screenshot.
#
# Each entry: (filename, list of (x, y, w, h, expected_rank, expected_suit))
# Coordinates are approximate bounding boxes of card images in the screenshot.
# These were determined by visual inspection of the actual screenshots.
# ---------------------------------------------------------------------------

# potodds_screenshot.png (1280x800):
#   Community cards: 6h, Kc, 10c, Qh
#   Player hole cards (Lorry): Kd, Ad (actual cards: K♦ A♦ based on red suit symbols)
_POTODDS_CARDS = [
    # Community cards (row centered on table)
    (282, 235, 70, 90, Rank.SIX, Suit.HEARTS),
    (375, 235, 70, 90, Rank.KING, Suit.CLUBS),
    (468, 235, 80, 90, Rank.TEN, Suit.CLUBS),
    (560, 235, 70, 90, Rank.QUEEN, Suit.HEARTS),
    # Player hole cards (bottom center)
    (445, 500, 55, 70, Rank.KING, Suit.DIAMONDS),
    (490, 500, 55, 70, Rank.ACE, Suit.DIAMONDS),
]

# pokernow_spectator_1.jpg (1648x931):
#   Community cards: 5d, 3c, Jh
#   Abby's hole cards: 4s, 5s
#   Ellie's hole cards: 8c, Qd  (based on "HIGH CARD" tag)
#   Joel's hole cards: 6s, Qs   (based on "HIGH CARD" tag)
_SPECTATOR1_CARDS = [
    # Community cards (center of red table)
    (535, 210, 80, 100, Rank.FIVE, Suit.DIAMONDS),
    (640, 210, 80, 100, Rank.THREE, Suit.CLUBS),
    (745, 210, 80, 100, Rank.JACK, Suit.HEARTS),
    # Abby's hole cards (left side)
    (60, 260, 50, 65, Rank.FOUR, Suit.SPADES),
    (105, 260, 50, 65, Rank.FIVE, Suit.SPADES),
    # Ellie's hole cards (bottom center-left)
    (250, 370, 40, 55, Rank.EIGHT, Suit.CLUBS),
    (285, 370, 40, 55, Rank.QUEEN, Suit.DIAMONDS),
    # Joel's hole cards (bottom center-right)
    (455, 370, 40, 55, Rank.SIX, Suit.SPADES),
    (490, 370, 40, 55, Rank.QUEEN, Suit.SPADES),
]

# montecarlo_screenshot.png (2838x1547):
#   Community cards: 3s, 8h, 9h, 3c, + (unknown/face-down)
#   Player hole cards (Hifu): 6h, 8d
_MONTECARLO_CARDS = [
    # Community cards (center)
    (820, 195, 115, 150, Rank.THREE, Suit.SPADES),
    (965, 195, 115, 150, Rank.EIGHT, Suit.HEARTS),
    (1110, 195, 115, 150, Rank.NINE, Suit.HEARTS),
    (1255, 195, 115, 150, Rank.THREE, Suit.CLUBS),
    # Player hole cards (bottom - Hifu)
    (780, 630, 65, 90, Rank.SIX, Suit.HEARTS),
    (840, 630, 65, 90, Rank.EIGHT, Suit.DIAMONDS),
]

# pokernow_spectator_4.jpg (1599x897) - red table variant
#   This is spectator mode with red felt. Cards may be visible.

SCREENSHOT_CARD_DEFS: dict[str, list[tuple[int, int, int, int, Rank, Suit]]] = {
    "potodds_screenshot.png": _POTODDS_CARDS,
    "pokernow_spectator_1.jpg": _SPECTATOR1_CARDS,
    "montecarlo_screenshot.png": _MONTECARLO_CARDS,
}


def _crop_and_detect(
    img: np.ndarray,
    x: int, y: int, w: int, h: int,
    detector: CNNDetector,
) -> list[DetectedCard]:
    """Crop a card region from the image and run CNN detection."""
    # Clamp to image bounds
    ih, iw = img.shape[:2]
    x = max(0, min(x, iw - 1))
    y = max(0, min(y, ih - 1))
    x2 = min(x + w, iw)
    y2 = min(y + h, ih)

    card_crop = img[y:y2, x:x2]
    if card_crop.size == 0:
        return []

    return detector.detect(card_crop)


def test_cnn_on_real_pokernow_screenshots() -> dict[str, object]:
    """Test the CNN detector on cropped cards from real PokerNow screenshots.

    Returns:
        Dict with results per screenshot and overall accuracy.
    """
    config = CNNConfig(model_path=str(PROJECT_ROOT / "models" / "card_detector.npz"))
    detector = CNNDetector(config=config)

    if not detector.is_ready:
        logger.warning("CNN model not loaded — skipping PokerNow test")
        return {"total": 0, "correct": 0, "accuracy": 0.0, "results": {}}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_correct = 0
    overall_total = 0
    all_results: dict[str, list[dict]] = {}

    for filename, card_defs in SCREENSHOT_CARD_DEFS.items():
        filepath = SCREENSHOT_DIR / filename
        if not filepath.exists():
            logger.warning("Screenshot not found: %s", filepath)
            continue

        img = cv2.imread(str(filepath))
        if img is None:
            logger.warning("Failed to read: %s", filepath)
            continue

        annotated = img.copy()
        file_results: list[dict] = []

        for x, y, w, h, expected_rank, expected_suit in card_defs:
            detections = _crop_and_detect(img, x, y, w, h, detector)
            overall_total += 1
            expected_str = f"{expected_rank.value}{expected_suit.value}"

            if detections:
                det = detections[0]
                predicted_str = f"{det.rank.value}{det.suit.value}"
                match = det.rank == expected_rank and det.suit == expected_suit
                rank_match = det.rank == expected_rank
                suit_match = det.suit == expected_suit

                if match:
                    overall_correct += 1
                    color = (0, 255, 0)  # green
                else:
                    color = (0, 0, 255)  # red

                file_results.append({
                    "expected": expected_str,
                    "predicted": predicted_str,
                    "confidence": f"{det.confidence:.3f}",
                    "match": match,
                    "rank_match": rank_match,
                    "suit_match": suit_match,
                })

                # Annotate
                label = f"{predicted_str} ({det.confidence:.2f})"
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.putText(annotated, f"exp:{expected_str}", (x, y + h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
            else:
                file_results.append({
                    "expected": expected_str,
                    "predicted": "NONE",
                    "confidence": "N/A",
                    "match": False,
                    "rank_match": False,
                    "suit_match": False,
                })
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(annotated, "NONE", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        all_results[filename] = file_results

        # Save annotated screenshot
        out_path = RESULTS_DIR / f"annotated_{filename}"
        cv2.imwrite(str(out_path), annotated)
        logger.info("Saved annotated: %s", out_path)

    accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    return {
        "total": overall_total,
        "correct": overall_correct,
        "accuracy": accuracy,
        "results": all_results,
    }


# ---------------------------------------------------------------------------
# Pytest test function
# ---------------------------------------------------------------------------


def test_cnn_pokernow_real_screenshots():
    """CNN should attempt detection on real PokerNow screenshots."""
    results = test_cnn_on_real_pokernow_screenshots()
    print(f"\n{'=' * 60}")
    print(f"PokerNow Real Screenshots: {results['correct']}/{results['total']} "
          f"({results['accuracy']:.1%})")
    print(f"{'=' * 60}")

    for filename, cards in results.get("results", {}).items():
        correct = sum(1 for c in cards if c["match"])
        print(f"\n  {filename}: {correct}/{len(cards)}")
        for c in cards:
            status = "OK" if c["match"] else "MISS"
            rank_ok = "R" if c.get("rank_match") else " "
            suit_ok = "S" if c.get("suit_match") else " "
            print(f"    [{status}] {c['expected']:>10} -> {c['predicted']:<10} "
                  f"conf={c['confidence']:>6}  [{rank_ok}{suit_ok}]")

    assert results["total"] > 0, "No screenshots were processed"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("=" * 60)
    print("CNN Card Detector - PokerNow Real Screenshot Test")
    print("=" * 60)

    results = test_cnn_on_real_pokernow_screenshots()

    print(f"\nOverall: {results['correct']}/{results['total']} "
          f"({results['accuracy']:.1%})")

    for filename, cards in results.get("results", {}).items():
        correct = sum(1 for c in cards if c["match"])
        rank_correct = sum(1 for c in cards if c.get("rank_match"))
        suit_correct = sum(1 for c in cards if c.get("suit_match"))
        print(f"\n--- {filename} ---")
        print(f"  Exact match: {correct}/{len(cards)}")
        print(f"  Rank correct: {rank_correct}/{len(cards)}")
        print(f"  Suit correct: {suit_correct}/{len(cards)}")
        for c in cards:
            status = "OK  " if c["match"] else "MISS"
            print(f"    [{status}] expected={c['expected']:>10}  "
                  f"predicted={c['predicted']:<10}  conf={c['confidence']}")

    print(f"\nAnnotated results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
