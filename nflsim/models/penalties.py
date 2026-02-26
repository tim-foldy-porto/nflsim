"""Penalty model — occurrence rates and yard distributions from PBP data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from nflsim.data.features import add_features

logger = logging.getLogger(__name__)


@dataclass
class PenaltyModel:
    """Penalty occurrence and outcome model."""

    # Rate of penalty per play (on either team)
    penalty_rate: float = 0.14  # ~14% of plays have a penalty
    # Of penalties, fraction on offense vs defense
    offense_penalty_share: float = 0.45
    # Common penalty yard values and their probabilities
    yard_distribution: dict[int, float] = field(default_factory=lambda: {
        5: 0.55,
        10: 0.25,
        15: 0.15,
        0: 0.03,   # offsetting/declined
        30: 0.02,
    })
    # Auto first down rate (for defensive penalties)
    auto_first_down_rate: float = 0.35
    # Declined/offsetting rate (penalty occurred but no effect)
    declined_rate: float = 0.25

    def sample_penalty(self, rng: np.random.Generator) -> dict | None:
        """Sample whether a penalty occurs and its properties.

        Returns None if no penalty, or a dict with penalty details.
        """
        if rng.random() >= self.penalty_rate:
            return None

        # Declined/offsetting
        if rng.random() < self.declined_rate:
            return None

        on_offense = rng.random() < self.offense_penalty_share

        # Sample yards
        yard_values = list(self.yard_distribution.keys())
        yard_probs = np.array(list(self.yard_distribution.values()))
        yard_probs = yard_probs / yard_probs.sum()
        yards = rng.choice(yard_values, p=yard_probs)

        auto_first = not on_offense and rng.random() < self.auto_first_down_rate

        return {
            "on_offense": on_offense,
            "yards": int(yards),
            "auto_first_down": auto_first,
        }


def build_penalty_model(pbp: pl.DataFrame, current_season: int) -> PenaltyModel:
    """Build penalty model from PBP data."""
    logger.info("Building penalty model...")

    df = add_features(pbp, current_season=current_season)

    # Filter to real plays
    plays = df.filter(
        pl.col("play_type").is_in(["pass", "run", "punt", "field_goal", "kickoff"])
        & pl.col("down").is_not_null()
    )

    model = PenaltyModel()

    total_plays = plays.shape[0]
    penalty_plays = plays.filter(pl.col("penalty") == 1).shape[0]
    model.penalty_rate = penalty_plays / total_plays if total_plays > 0 else 0.14

    # Penalty yards distribution
    pen_df = plays.filter(
        (pl.col("penalty") == 1) & pl.col("penalty_yards").is_not_null()
    )
    if pen_df.shape[0] > 0:
        yard_counts: dict[int, int] = {}
        for y in pen_df["penalty_yards"].to_list():
            y_int = int(abs(y)) if y is not None else 0
            # Bucket to standard values
            if y_int <= 2:
                y_int = 0
            elif y_int <= 7:
                y_int = 5
            elif y_int <= 12:
                y_int = 10
            elif y_int <= 20:
                y_int = 15
            else:
                y_int = 30
            yard_counts[y_int] = yard_counts.get(y_int, 0) + 1

        total = sum(yard_counts.values())
        model.yard_distribution = {k: v / total for k, v in yard_counts.items()}

    logger.info("Penalty model: rate=%.3f", model.penalty_rate)
    return model
