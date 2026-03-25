"""Player statistics calculations and data models.

Provides the PlayerStats dataclass that holds per-player HUD statistics
computed from hand history data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlayerStats:
    """Aggregated statistics for a single player.

    All percentage values are stored as floats in [0, 100] range.
    Aggression factor is a ratio (typically 0.0 to 5.0+).

    Attributes:
        player_id: Unique identifier for the player.
        player_name: Display name of the player.
        hands_played: Total number of hands observed.
        vpip: Voluntarily Put money In Pot percentage.
        pfr: Pre-Flop Raise percentage.
        three_bet_pct: 3-Bet percentage.
        fold_to_three_bet_pct: Fold to 3-Bet percentage.
        cbet_pct: Continuation bet percentage.
        aggression_factor: Aggression factor (bets+raises / calls).
        wtsd_pct: Went To ShowDown percentage.
    """

    player_id: str = ""
    player_name: str = ""
    hands_played: int = 0
    vpip: float = 0.0
    pfr: float = 0.0
    three_bet_pct: float = 0.0
    fold_to_three_bet_pct: float = 0.0
    cbet_pct: float = 0.0
    aggression_factor: float = 0.0
    wtsd_pct: float = 0.0
