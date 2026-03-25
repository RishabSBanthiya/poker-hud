"""End-to-end card recognition pipeline.

Orchestrates region extraction, template matching, and optional CNN
fallback to detect all visible cards in a poker table frame.

Usage:
    from src.detection.recognition_pipeline import CardRecognitionPipeline
    from src.detection.table_regions import POKERSTARS_LAYOUT

    pipeline = CardRecognitionPipeline(
        template_dir="data/templates",
        layout=POKERSTARS_LAYOUT,
    )
    results = pipeline.process_frame(frame)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.common.performance import LatencyTracker, PerfTimer
from src.detection.card import DetectedCard
from src.detection.cnn_detector import CNNConfig, CNNDetector
from src.detection.table_regions import (
    ExtractedRegion,
    RegionLocalizer,
    TableLayout,
)
from src.detection.template_matcher import TemplateMatcher

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration knobs for the recognition pipeline.

    Attributes:
        template_threshold: Minimum confidence for template matcher.
        cnn_fallback_threshold: If the best template match is below
            this value, the CNN detector is invoked on the region.
        iou_threshold: IoU threshold for non-maximum suppression.
        use_color: Whether to use colour template matching.
        calibrate: Apply confidence calibration to template scores.
        multiscale: Load multi-scale templates.
        max_latency_ms: Target per-frame budget in milliseconds.
    """

    template_threshold: float = 0.7
    cnn_fallback_threshold: float = 0.6
    iou_threshold: float = 0.3
    use_color: bool = False
    calibrate: bool = False
    multiscale: bool = False
    max_latency_ms: float = 200.0


@dataclass(frozen=True)
class RegionResult:
    """Detection outcome for a single card region.

    Attributes:
        region_name: Name of the source region.
        detection: Best detection found, or ``None`` if nothing was
            detected above threshold.
        source: Which detector produced the result
            (``"template"``, ``"cnn"``, or ``"none"``).
    """

    region_name: str
    detection: DetectedCard | None
    source: str


@dataclass
class PipelineResult:
    """Aggregate result for a full frame.

    Attributes:
        region_results: Per-region detection outcomes.
        detected_cards: All successfully detected cards.
        elapsed_ms: Total processing time in milliseconds.
    """

    region_results: list[RegionResult] = field(default_factory=list)
    detected_cards: list[DetectedCard] = field(default_factory=list)
    elapsed_ms: float = 0.0


class CardRecognitionPipeline:
    """Orchestrates card detection across all table regions.

    Workflow:
      1. Use ``RegionLocalizer`` to crop card regions from the frame.
      2. Run ``TemplateMatcher`` on each region.
      3. If confidence is below *cnn_fallback_threshold*, try the
         ``CNNDetector``.
      4. Collect and return all detected cards with positions and
         confidence scores.

    Args:
        template_dir: Path to card template images.
        layout: Poker table layout defining card regions.
        config: Pipeline configuration overrides.
        cnn_config: CNN detector configuration overrides.
    """

    def __init__(
        self,
        template_dir: str | Path,
        layout: TableLayout,
        config: PipelineConfig | None = None,
        cnn_config: CNNConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.layout = layout

        self._matcher = TemplateMatcher(
            template_dir,
            multiscale=self.config.multiscale,
        )
        self._cnn = CNNDetector(cnn_config)
        self._localizer = RegionLocalizer(layout)
        self._tracker = LatencyTracker("recognition_pipeline")

    @property
    def latency_tracker(self) -> LatencyTracker:
        """Expose the internal latency tracker for external monitoring."""
        return self._tracker

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        seat_indices: list[int] | None = None,
    ) -> PipelineResult:
        """Run the full recognition pipeline on *frame*.

        Args:
            frame: Full poker table image (BGR numpy array).
            seat_indices: Optional subset of seats to scan.  ``None``
                means all seats defined in the layout.

        Returns:
            ``PipelineResult`` with per-region outcomes and aggregate
            detected-card list.
        """
        result = PipelineResult()

        with PerfTimer("pipeline_total", tracker=self._tracker) as timer:
            # 1. Extract regions
            community = self._localizer.extract_community_regions(frame)
            seats = self._localizer.extract_seat_regions(
                frame, seat_indices=seat_indices,
            )
            all_regions = community + seats

            # 2-3. Detect cards in each region
            for extracted in all_regions:
                region_result = self._detect_in_region(extracted)
                result.region_results.append(region_result)
                if region_result.detection is not None:
                    result.detected_cards.append(region_result.detection)

        result.elapsed_ms = timer.elapsed_ms

        if result.elapsed_ms > self.config.max_latency_ms:
            logger.warning(
                "Pipeline exceeded latency target: %.1f ms > %.1f ms",
                result.elapsed_ms,
                self.config.max_latency_ms,
            )

        return result

    def _detect_in_region(self, extracted: ExtractedRegion) -> RegionResult:
        """Attempt detection in a single extracted region.

        Template matching is tried first; the CNN is used as a fallback
        when the best template confidence is below threshold.
        """
        region_name = extracted.name
        px, py, pw, ph = extracted.pixel_box

        # --- Template matching ---
        template_dets = self._matcher.detect_cards(
            extracted.image,
            threshold=self.config.template_threshold,
            iou_threshold=self.config.iou_threshold,
            use_color=self.config.use_color,
            calibrate=self.config.calibrate,
        )

        if template_dets:
            best = template_dets[0]
            # Adjust bounding box to full-frame coords
            bx, by, bw, bh = best.bounding_box
            adjusted = DetectedCard(
                rank=best.rank,
                suit=best.suit,
                confidence=best.confidence,
                bounding_box=(bx + px, by + py, bw, bh),
            )

            if best.confidence >= self.config.cnn_fallback_threshold:
                return RegionResult(
                    region_name=region_name,
                    detection=adjusted,
                    source="template",
                )

        # --- CNN fallback ---
        if self._cnn.is_ready:
            cnn_dets = self._cnn.detect(
                extracted.image,
                bounding_box=extracted.pixel_box,
            )
            if cnn_dets:
                return RegionResult(
                    region_name=region_name,
                    detection=cnn_dets[0],
                    source="cnn",
                )

        # If template had a low-confidence result, still return it
        if template_dets:
            best = template_dets[0]
            bx, by, bw, bh = best.bounding_box
            adjusted = DetectedCard(
                rank=best.rank,
                suit=best.suit,
                confidence=best.confidence,
                bounding_box=(bx + px, by + py, bw, bh),
            )
            return RegionResult(
                region_name=region_name,
                detection=adjusted,
                source="template",
            )

        return RegionResult(
            region_name=region_name,
            detection=None,
            source="none",
        )

    def process_community_only(
        self, frame: np.ndarray,
    ) -> PipelineResult:
        """Convenience: detect only community cards.

        Args:
            frame: Full poker table image.

        Returns:
            Pipeline result scoped to community card regions.
        """
        result = PipelineResult()

        with PerfTimer("pipeline_community") as timer:
            regions = self._localizer.extract_community_regions(frame)
            for extracted in regions:
                rr = self._detect_in_region(extracted)
                result.region_results.append(rr)
                if rr.detection is not None:
                    result.detected_cards.append(rr.detection)

        result.elapsed_ms = timer.elapsed_ms
        return result
