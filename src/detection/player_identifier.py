"""Player name OCR identification and registry for the poker HUD.

Extracts player screen names from captured frames, caches results to avoid
redundant OCR on static names, and maintains a registry of known players
with alias support.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.detection.ocr_engine import Region

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlayerMatch:
    """Result of a player identification attempt.

    Attributes:
        name: Best-guess player name (may be a canonical name if an alias
            mapping exists).
        raw_text: The literal text extracted from the screen.
        confidence: Recognition confidence (0.0-1.0).
        seat_index: Optional seat position the player was detected at.
    """

    name: str
    raw_text: str
    confidence: float
    seat_index: Optional[int] = None


@dataclass
class _CacheEntry:
    """Internal cache entry for a previously recognised name region."""

    name: str
    confidence: float
    region_hash: int
    timestamp: float


class PlayerIdentifier:
    """Extract and identify player names from screen regions.

    Uses contour-based character segmentation and template matching for the
    alphabetic character set commonly found in poker screen names (letters,
    digits, underscores, hyphens).

    Recognised names are cached so that subsequent frames with identical (or
    near-identical) region content skip the OCR step entirely.

    Args:
        cache_ttl: Seconds before a cached name expires and must be re-OCR'd.
        similarity_threshold: Minimum ratio (0.0-1.0) for two strings to be
            considered a partial match.
        template_height: Pixel height for character templates.
        match_threshold: Minimum normalised correlation to accept a character.
    """

    _ALPHA_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"

    def __init__(
        self,
        cache_ttl: float = 30.0,
        similarity_threshold: float = 0.6,
        template_height: int = 20,
        match_threshold: float = 0.50,
    ) -> None:
        self._cache_ttl = cache_ttl
        self._similarity_threshold = similarity_threshold
        self._template_height = template_height
        self._match_threshold = match_threshold
        self._templates = self._generate_alpha_templates(template_height)
        self._cache: dict[int, _CacheEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def identify_player(
        self,
        frame: np.ndarray,
        region: Region,
        seat_index: Optional[int] = None,
    ) -> Optional[PlayerMatch]:
        """Extract a player name from *region* of *frame*.

        If the region content matches a cached entry that has not expired, the
        cached result is returned immediately.

        Args:
            frame: Source image as a BGR numpy array.
            region: Rectangular area containing the player name.
            seat_index: Optional seat number for context.

        Returns:
            A :class:`PlayerMatch` if a name was successfully extracted,
            otherwise ``None``.
        """
        region_img = self._crop_region(frame, region)
        if region_img.size == 0:
            return None

        r_hash = self._region_hash(region_img)

        # Check cache.
        cached = self._cache.get(r_hash)
        not_expired = (
            cached is not None
            and (time.monotonic() - cached.timestamp) < self._cache_ttl
        )
        if not_expired and cached is not None:
            return PlayerMatch(
                name=cached.name,
                raw_text=cached.name,
                confidence=cached.confidence,
                seat_index=seat_index,
            )

        binary = self._preprocess(region_img)
        raw_text, confidence = self._recognize_text(binary)

        if not raw_text:
            return None

        # Store in cache.
        self._cache[r_hash] = _CacheEntry(
            name=raw_text,
            confidence=confidence,
            region_hash=r_hash,
            timestamp=time.monotonic(),
        )

        return PlayerMatch(
            name=raw_text,
            raw_text=raw_text,
            confidence=confidence,
            seat_index=seat_index,
        )

    def clear_cache(self) -> None:
        """Remove all cached name recognitions."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_region(frame: np.ndarray, region: Region) -> np.ndarray:
        """Safely crop *region* from *frame*, returning an empty array on OOB."""
        y1, y2 = region.y, region.y + region.h
        x1, x2 = region.x, region.x + region.w
        h, w = frame.shape[:2]
        if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
            return np.empty(0, dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """Convert a cropped BGR region to a denoised binary image."""
        if image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        return binary

    # ------------------------------------------------------------------
    # Template generation & matching
    # ------------------------------------------------------------------

    @classmethod
    def _generate_alpha_templates(cls, height: int) -> dict[str, np.ndarray]:
        """Generate binary templates for alphanumeric characters."""
        templates: dict[str, np.ndarray] = {}
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        font_scale = cv2.getFontScaleFromHeight(font_face, height, thickness)

        for ch in cls._ALPHA_CHARS:
            (tw, th), baseline = cv2.getTextSize(
                ch, font_face, font_scale, thickness
            )
            canvas_h = th + baseline + 4
            canvas_w = tw + 4
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            cv2.putText(
                canvas, ch, (2, th + 2), font_face, font_scale, 255, thickness
            )
            scale = height / canvas_h
            new_w = max(int(canvas_w * scale), 1)
            resized = cv2.resize(canvas, (new_w, height), interpolation=cv2.INTER_AREA)
            _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
            templates[ch] = binary

        return templates

    def _recognize_text(
        self, binary: np.ndarray
    ) -> tuple[str, float]:
        """Run template matching across *binary* for alphanumeric characters.

        Returns:
            Tuple of (recognised_text, mean_confidence).
        """
        if binary.shape[0] < 2 or binary.shape[1] < 2:
            return "", 0.0

        scale = self._template_height / binary.shape[0]
        new_w = max(int(binary.shape[1] * scale), 1)
        resized = cv2.resize(
            binary, (new_w, self._template_height), interpolation=cv2.INTER_AREA
        )
        _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        img_h, img_w = resized.shape[:2]
        chars: list[str] = []
        confidences: list[float] = []
        x_pos = 0

        while x_pos < img_w - 2:
            best_score = -1.0
            best_char = ""
            best_width = 1

            for ch, tmpl in self._templates.items():
                t_h, t_w = tmpl.shape[:2]
                if t_w > img_w - x_pos or t_h > img_h:
                    continue
                roi = resized[:t_h, x_pos : x_pos + t_w]
                if roi.shape != tmpl.shape:
                    continue
                result = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
                score = float(result[0, 0]) if result.size > 0 else -1.0
                if score > best_score:
                    best_score = score
                    best_char = ch
                    best_width = t_w

            if best_score >= self._match_threshold and best_char:
                chars.append(best_char)
                confidences.append(best_score)
                x_pos += best_width
            else:
                x_pos += max(best_width // 2, 1)

        text = "".join(chars)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return text, mean_conf

    # ------------------------------------------------------------------
    # Hashing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _region_hash(image: np.ndarray) -> int:
        """Compute a fast perceptual hash of a small image region.

        Resizes to 8x8 grayscale, computes the mean, and packs the
        above/below-mean bitmap into an integer.  This is intentionally coarse
        so that minor anti-aliasing or compression differences still hit the
        cache.
        """
        if image.size == 0:
            return 0
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        mean_val = resized.mean()
        bits = (resized > mean_val).flatten()
        hash_val = 0
        for bit in bits:
            hash_val = (hash_val << 1) | int(bit)
        return hash_val


@dataclass
class _AliasEntry:
    """Internal alias mapping for a player."""

    canonical_name: str
    aliases: set[str] = field(default_factory=set)


class PlayerRegistry:
    """Track known players across sessions with alias support.

    Maintains a mapping of canonical player names and their aliases.  When a
    new name is encountered it is checked against existing entries for partial
    matches (e.g. truncated display names) and can be auto-linked.

    Args:
        similarity_threshold: Minimum ratio (0.0-1.0) for two strings to be
            considered a partial match for auto-aliasing.
    """

    def __init__(self, similarity_threshold: float = 0.6) -> None:
        self._similarity_threshold = similarity_threshold
        # canonical_name -> _AliasEntry
        self._players: dict[str, _AliasEntry] = {}
        # alias -> canonical_name (reverse index)
        self._alias_index: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, name: str) -> str:
        """Register a player name, returning its canonical form.

        If *name* matches an existing alias, the canonical name is returned
        without creating a duplicate.  If it partially matches an existing
        player (above the similarity threshold), it is added as an alias.

        Args:
            name: Screen name to register.

        Returns:
            Canonical player name.
        """
        if not name:
            return name

        normalised = name.strip()

        # Exact match in alias index?
        if normalised in self._alias_index:
            return self._alias_index[normalised]

        # Exact match as canonical?
        if normalised in self._players:
            return normalised

        # Check partial matches against known names.
        best_match: Optional[str] = None
        best_ratio = 0.0
        for canonical in self._players:
            ratio = self._similarity(normalised, canonical)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = canonical

        if best_match and best_ratio >= self._similarity_threshold:
            self.add_alias(best_match, normalised)
            return best_match

        # New player.
        entry = _AliasEntry(canonical_name=normalised)
        self._players[normalised] = entry
        self._alias_index[normalised] = normalised
        return normalised

    def add_alias(self, canonical_name: str, alias: str) -> None:
        """Explicitly add an alias for an existing canonical player name.

        Args:
            canonical_name: The primary name the player is known by.
            alias: Alternative display name to map to *canonical_name*.

        Raises:
            KeyError: If *canonical_name* is not a registered player.
        """
        if canonical_name not in self._players:
            raise KeyError(
                f"Player '{canonical_name}' is not registered. "
                "Register the canonical name first."
            )
        self._players[canonical_name].aliases.add(alias)
        self._alias_index[alias] = canonical_name

    def resolve(self, name: str) -> Optional[str]:
        """Look up the canonical name for *name*.

        Args:
            name: A player name or alias to resolve.

        Returns:
            Canonical name if found, otherwise ``None``.
        """
        normalised = name.strip()
        return self._alias_index.get(normalised)

    def get_aliases(self, canonical_name: str) -> set[str]:
        """Return all known aliases for a canonical player name.

        Args:
            canonical_name: The primary player name.

        Returns:
            Set of alias strings (may be empty). Includes the canonical name
            itself.

        Raises:
            KeyError: If *canonical_name* is not registered.
        """
        entry = self._players.get(canonical_name)
        if entry is None:
            raise KeyError(f"Player '{canonical_name}' is not registered.")
        return entry.aliases | {canonical_name}

    @property
    def known_players(self) -> list[str]:
        """Return a sorted list of all canonical player names."""
        return sorted(self._players.keys())

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Compute a simple character-level similarity ratio.

        Uses the ratio of matching characters (longest common subsequence
        length) to the max string length.  This is cheaper than full
        Levenshtein and sufficient for screen-name matching.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Similarity ratio between 0.0 and 1.0.
        """
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        # Longest common subsequence via DP.
        m, n = len(a), len(b)
        # Use a rolling two-row approach for memory efficiency.
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)

        lcs_len = prev[n]
        return lcs_len / max(m, n)
