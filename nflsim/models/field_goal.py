"""Field goal and extra point model from PBP data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class FieldGoalModel:
    """Field goal success probability by distance bucket."""

    # Distance bucket -> success rate
    success_rates: dict[int, float] = field(default_factory=lambda: {
        20: 0.97, 25: 0.95, 30: 0.93, 35: 0.90, 40: 0.87,
        45: 0.80, 50: 0.72, 55: 0.58, 60: 0.40, 65: 0.15,
    })
    extra_point_rate: float = 0.94
    two_point_rate: float = 0.48

    def fg_probability(self, distance: int) -> float:
        """Get success probability for a field goal at given distance."""
        # Find nearest bucket
        buckets = sorted(self.success_rates.keys())
        if distance <= buckets[0]:
            return self.success_rates[buckets[0]]
        if distance >= buckets[-1]:
            return max(0.05, self.success_rates[buckets[-1]] - 0.05 * (distance - buckets[-1]))

        # Linear interpolation between buckets
        for i in range(len(buckets) - 1):
            if buckets[i] <= distance <= buckets[i + 1]:
                lo, hi = buckets[i], buckets[i + 1]
                frac = (distance - lo) / (hi - lo)
                return self.success_rates[lo] * (1 - frac) + self.success_rates[hi] * frac

        return 0.50  # shouldn't reach here

    def sample_fg(self, distance: int, rng: np.random.Generator) -> bool:
        return rng.random() < self.fg_probability(distance)

    def sample_xp(self, rng: np.random.Generator) -> bool:
        return rng.random() < self.extra_point_rate

    def sample_two_point(self, rng: np.random.Generator) -> bool:
        return rng.random() < self.two_point_rate


def build_field_goal_model(pbp: pl.DataFrame) -> FieldGoalModel:
    """Build FG model from PBP data."""
    logger.info("Building field goal model...")

    model = FieldGoalModel()

    # Field goals
    fgs = pbp.filter(
        (pl.col("play_type") == "field_goal")
        & pl.col("kick_distance").is_not_null()
        & pl.col("field_goal_result").is_not_null()
    )

    if fgs.shape[0] > 50:
        # Build success rates by 5-yard buckets
        rates: dict[int, float] = {}
        bucketed = fgs.with_columns(
            ((pl.col("kick_distance") / 5).round(0) * 5).cast(pl.Int32).alias("dist_bucket")
        )
        grouped = bucketed.group_by("dist_bucket").agg([
            (pl.col("field_goal_result") == "made").sum().alias("made"),
            pl.len().alias("total"),
        ])
        for row in grouped.iter_rows(named=True):
            bucket = row["dist_bucket"]
            if row["total"] >= 10:
                rates[bucket] = row["made"] / row["total"]

        if rates:
            model.success_rates = rates

    # Extra points
    xps = pbp.filter(
        (pl.col("play_type") == "extra_point")
        & pl.col("extra_point_result").is_not_null()
    )
    if xps.shape[0] > 50:
        made = xps.filter(pl.col("extra_point_result") == "good").shape[0]
        model.extra_point_rate = made / xps.shape[0]

    # Two-point conversions
    twos = pbp.filter(
        (pl.col("play_type") == "two_point_attempt")
        & pl.col("two_point_conv_result").is_not_null()
    )
    if twos.shape[0] > 20:
        success = twos.filter(pl.col("two_point_conv_result") == "success").shape[0]
        model.two_point_rate = success / twos.shape[0]

    logger.info(
        "FG model: %d distance buckets, XP rate=%.3f, 2pt rate=%.3f",
        len(model.success_rates), model.extra_point_rate, model.two_point_rate,
    )
    return model
