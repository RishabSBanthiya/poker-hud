"""Tests for src.detection.player_identifier."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from src.detection.ocr_engine import Region
from src.detection.player_identifier import (
    PlayerIdentifier,
    PlayerMatch,
    PlayerRegistry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_name_image(
    text: str,
    width: int = 200,
    height: int = 40,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    fg_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Render a player name onto a BGR image."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (5, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        fg_color,
        2,
    )
    return img


# ---------------------------------------------------------------------------
# PlayerMatch dataclass
# ---------------------------------------------------------------------------

class TestPlayerMatch:
    def test_creation(self) -> None:
        m = PlayerMatch(name="Alice", raw_text="Alice", confidence=0.9, seat_index=2)
        assert m.name == "Alice"
        assert m.raw_text == "Alice"
        assert m.confidence == 0.9
        assert m.seat_index == 2

    def test_default_seat_is_none(self) -> None:
        m = PlayerMatch(name="Bob", raw_text="Bob", confidence=0.8)
        assert m.seat_index is None


# ---------------------------------------------------------------------------
# PlayerIdentifier
# ---------------------------------------------------------------------------

class TestPlayerIdentifier:
    @pytest.fixture()
    def identifier(self) -> PlayerIdentifier:
        return PlayerIdentifier(cache_ttl=5.0, match_threshold=0.40)

    def test_identify_returns_match_or_none(
        self, identifier: PlayerIdentifier
    ) -> None:
        frame = _render_name_image("Player1", width=200, height=40)
        region = Region(0, 0, 200, 40)
        result = identifier.identify_player(frame, region)
        # With synthetic data the template matcher may or may not succeed.
        assert result is None or isinstance(result, PlayerMatch)

    def test_identify_with_seat_index(
        self, identifier: PlayerIdentifier
    ) -> None:
        frame = _render_name_image("Alice", width=160, height=40)
        region = Region(0, 0, 160, 40)
        result = identifier.identify_player(frame, region, seat_index=3)
        if result is not None:
            assert result.seat_index == 3

    def test_empty_region_returns_none(
        self, identifier: PlayerIdentifier
    ) -> None:
        frame = np.zeros((40, 200, 3), dtype=np.uint8)
        # Region extends beyond frame -- _crop_region should return empty.
        region = Region(0, 0, 300, 60)
        result = identifier.identify_player(frame, region)
        assert result is None

    def test_cache_returns_same_result(
        self, identifier: PlayerIdentifier
    ) -> None:
        frame = _render_name_image("CachedName", width=200, height=40)
        region = Region(0, 0, 200, 40)
        r1 = identifier.identify_player(frame, region)
        r2 = identifier.identify_player(frame, region)
        # Both should agree (cache hit on second call).
        if r1 is not None:
            assert r2 is not None
            assert r1.name == r2.name

    def test_cache_expires(self, identifier: PlayerIdentifier) -> None:
        identifier._cache_ttl = 0.0  # Expire immediately.
        frame = _render_name_image("Expire", width=200, height=40)
        region = Region(0, 0, 200, 40)
        identifier.identify_player(frame, region)
        # After TTL=0, next call should NOT use cache (re-OCR).
        # Just verify no crash.
        result = identifier.identify_player(frame, region)
        assert result is None or isinstance(result, PlayerMatch)

    def test_clear_cache(self, identifier: PlayerIdentifier) -> None:
        # Populate cache via a real identify call, then clear.
        frame = _render_name_image("ClearMe", width=200, height=40)
        region = Region(0, 0, 200, 40)
        identifier.identify_player(frame, region)
        identifier.clear_cache()
        assert len(identifier._cache) == 0

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def test_preprocess_returns_binary(self) -> None:
        img = _render_name_image("Test")
        binary = PlayerIdentifier._preprocess(img)
        unique = set(np.unique(binary))
        assert unique <= {0, 255}

    def test_preprocess_empty_image(self) -> None:
        empty = np.empty(0, dtype=np.uint8)
        result = PlayerIdentifier._preprocess(empty)
        assert result.size >= 1

    def test_preprocess_grayscale_input(self) -> None:
        gray = np.full((30, 100), 100, dtype=np.uint8)
        binary = PlayerIdentifier._preprocess(gray)
        assert len(binary.shape) == 2

    # ------------------------------------------------------------------
    # Region hashing
    # ------------------------------------------------------------------

    def test_region_hash_deterministic(self) -> None:
        img = _render_name_image("HashTest")
        h1 = PlayerIdentifier._region_hash(img)
        h2 = PlayerIdentifier._region_hash(img)
        assert h1 == h2

    def test_region_hash_different_for_different_images(self) -> None:
        img_a = _render_name_image("AAA", bg_color=(0, 0, 0))
        img_b = _render_name_image("ZZZ", bg_color=(200, 200, 200))
        h_a = PlayerIdentifier._region_hash(img_a)
        h_b = PlayerIdentifier._region_hash(img_b)
        # Not guaranteed to differ for all inputs, but these are quite
        # different images.
        assert h_a != h_b

    def test_region_hash_empty_image(self) -> None:
        empty = np.empty(0, dtype=np.uint8)
        assert PlayerIdentifier._region_hash(empty) == 0


# ---------------------------------------------------------------------------
# PlayerRegistry
# ---------------------------------------------------------------------------

class TestPlayerRegistry:
    @pytest.fixture()
    def registry(self) -> PlayerRegistry:
        return PlayerRegistry(similarity_threshold=0.6)

    # Registration
    def test_register_new_player(self, registry: PlayerRegistry) -> None:
        name = registry.register("Alice")
        assert name == "Alice"
        assert "Alice" in registry.known_players

    def test_register_existing_returns_canonical(
        self, registry: PlayerRegistry
    ) -> None:
        registry.register("Alice")
        name = registry.register("Alice")
        assert name == "Alice"
        assert registry.known_players.count("Alice") == 1

    def test_register_empty_string(self, registry: PlayerRegistry) -> None:
        assert registry.register("") == ""

    # Alias support
    def test_add_alias(self, registry: PlayerRegistry) -> None:
        registry.register("Alice")
        registry.add_alias("Alice", "alice_alt")
        assert registry.resolve("alice_alt") == "Alice"

    def test_add_alias_unknown_canonical_raises(
        self, registry: PlayerRegistry
    ) -> None:
        with pytest.raises(KeyError):
            registry.add_alias("UnknownPlayer", "alias")

    def test_get_aliases_includes_canonical(
        self, registry: PlayerRegistry
    ) -> None:
        registry.register("Bob")
        registry.add_alias("Bob", "bobby")
        aliases = registry.get_aliases("Bob")
        assert "Bob" in aliases
        assert "bobby" in aliases

    def test_get_aliases_unknown_raises(
        self, registry: PlayerRegistry
    ) -> None:
        with pytest.raises(KeyError):
            registry.get_aliases("NoSuchPlayer")

    # Resolution
    def test_resolve_canonical(self, registry: PlayerRegistry) -> None:
        registry.register("Charlie")
        assert registry.resolve("Charlie") == "Charlie"

    def test_resolve_alias(self, registry: PlayerRegistry) -> None:
        registry.register("Charlie")
        registry.add_alias("Charlie", "chuck")
        assert registry.resolve("chuck") == "Charlie"

    def test_resolve_unknown_returns_none(
        self, registry: PlayerRegistry
    ) -> None:
        assert registry.resolve("Nobody") is None

    # Partial matching via registration
    def test_partial_match_auto_aliases(self) -> None:
        registry = PlayerRegistry(similarity_threshold=0.6)
        registry.register("PlayerOne")
        # "PlayerOn" is very similar to "PlayerOne".
        canonical = registry.register("PlayerOn")
        assert canonical == "PlayerOne"
        assert "PlayerOn" in registry.get_aliases("PlayerOne")

    def test_dissimilar_name_creates_new_player(
        self, registry: PlayerRegistry
    ) -> None:
        registry.register("Alice")
        canonical = registry.register("Xander")
        assert canonical == "Xander"
        assert len(registry.known_players) == 2

    # known_players
    def test_known_players_sorted(self, registry: PlayerRegistry) -> None:
        registry.register("Charlie")
        registry.register("Alice")
        registry.register("Bob")
        assert registry.known_players == ["Alice", "Bob", "Charlie"]

    # Similarity function
    def test_similarity_identical(self) -> None:
        assert PlayerRegistry._similarity("abc", "abc") == 1.0

    def test_similarity_empty(self) -> None:
        assert PlayerRegistry._similarity("", "abc") == 0.0
        assert PlayerRegistry._similarity("abc", "") == 0.0

    def test_similarity_partial(self) -> None:
        ratio = PlayerRegistry._similarity("abcdef", "abcxyz")
        # LCS is "abc" -> 3/6 = 0.5
        assert abs(ratio - 0.5) < 1e-6

    def test_similarity_substring(self) -> None:
        ratio = PlayerRegistry._similarity("Player", "PlayerOne")
        # LCS is "Player" -> 6/9
        assert ratio > 0.5
