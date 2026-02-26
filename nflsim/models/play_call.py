"""Play call prediction — conditional probability tables from PBP data.

Given a game context (down, distance, field zone, score), what play type
does the offense call? Built from historical PBP data with recency weighting.

The lookup key is (down, distance_bucket, field_zone, score_bucket).
Values are probability distributions over {pass, run, punt, field_goal, kneel, spike}.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from nflsim.config import PROCESSED_DIR
from nflsim.data.features import CONTEXT_COLS, add_features

logger = logging.getLogger(__name__)

# Play types we model
PLAY_TYPES = ["pass", "run"]
# Fourth-down-only types
FOURTH_DOWN_TYPES = ["pass", "run", "punt", "field_goal"]

# Minimum weighted sample size before blending with broader context
MIN_SAMPLES = 30
BLEND_THRESHOLD = 100


@dataclass
class PlayCallModel:
    """Conditional probability tables for play calling.

    Attributes:
        tables: Mapping from context tuple -> dict of play_type -> probability
        global_dist: League-wide play type distribution (fallback)
        fourth_down_tables: Separate tables for 4th down decisions
        team_factors: Per-team pass rate multiplier vs league average
    """
    tables: dict[tuple, dict[str, float]] = field(default_factory=dict)
    global_dist: dict[str, float] = field(default_factory=lambda: {"pass": 0.58, "run": 0.42})
    fourth_down_tables: dict[tuple, dict[str, float]] = field(default_factory=dict)
    fourth_down_global: dict[str, float] = field(
        default_factory=lambda: {"punt": 0.55, "field_goal": 0.25, "pass": 0.12, "run": 0.08}
    )
    team_factors: dict[str, float] = field(default_factory=dict)  # team -> pass_rate / league_avg

    def sample_play(
        self,
        down: int,
        distance_bucket: str,
        field_zone: str,
        score_bucket: str,
        team: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> str:
        """Sample a play type from the conditional distribution."""
        if rng is None:
            rng = np.random.default_rng()

        key = (down, distance_bucket, field_zone, score_bucket)

        if down == 4:
            dist = self._get_blended_dist(key, self.fourth_down_tables, self.fourth_down_global)
        else:
            dist = self._get_blended_dist(key, self.tables, self.global_dist)

        # Apply team factor (adjust pass rate)
        if team and team in self.team_factors and down != 4:
            dist = self._apply_team_factor(dist, self.team_factors[team])

        # Sample
        types = list(dist.keys())
        probs = np.array([dist[t] for t in types])
        probs = probs / probs.sum()  # renormalize
        return rng.choice(types, p=probs)

    def _get_blended_dist(
        self,
        key: tuple,
        tables: dict[tuple, dict[str, float]],
        global_dist: dict[str, float],
    ) -> dict[str, float]:
        """Look up distribution, blending with global if sample size is small."""
        if key in tables:
            return tables[key]

        # Try progressively broader keys (drop score_bucket, then field_zone)
        if len(key) >= 4:
            broader = key[:3]  # (down, distance_bucket, field_zone)
            if broader in tables:
                return tables[broader]
        if len(key) >= 3:
            broader = key[:2]  # (down, distance_bucket)
            if broader in tables:
                return tables[broader]

        return global_dist

    def _apply_team_factor(self, dist: dict[str, float], factor: float) -> dict[str, float]:
        """Adjust pass/run split by team tendency factor."""
        if "pass" not in dist or "run" not in dist:
            return dist

        result = dict(dist)
        pass_rate = result["pass"] * factor
        run_rate = result["run"] * (2.0 - factor)  # inverse adjustment
        total = pass_rate + run_rate
        result["pass"] = pass_rate / total * (result["pass"] + result["run"])
        result["run"] = run_rate / total * (result["pass"] + result["run"])
        # Re-derive to keep other play types (if any) and sum to ~1
        return result


def build_play_call_model(pbp: pl.DataFrame, current_season: int) -> PlayCallModel:
    """Build play call model from PBP data.

    Args:
        pbp: Raw nflverse play-by-play data (multiple seasons)
        current_season: Season for recency weighting

    Returns:
        PlayCallModel with conditional probability tables
    """
    logger.info("Building play call model...")

    # Add features
    df = add_features(pbp, current_season=current_season)

    # Filter to real plays (not timeouts, end-of-quarter, etc.)
    df = df.filter(
        pl.col("play_category").is_in(["pass", "run", "punt", "field_goal", "kneel", "spike"])
        & pl.col("down").is_not_null()
        & pl.col("ydstogo").is_not_null()
        & pl.col("yardline_100").is_not_null()
    )

    model = PlayCallModel()

    # Build tables for downs 1-3 (pass/run only)
    df_123 = df.filter(pl.col("down").is_in([1, 2, 3]) & pl.col("play_category").is_in(["pass", "run"]))
    model.tables = _build_conditional_tables(df_123, CONTEXT_COLS, ["pass", "run"])
    model.global_dist = _compute_global_dist(df_123, ["pass", "run"])

    # Build tables for 4th down (pass/run/punt/field_goal)
    df_4 = df.filter(
        pl.col("down") == 4,
        pl.col("play_category").is_in(["pass", "run", "punt", "field_goal"]),
    )
    model.fourth_down_tables = _build_conditional_tables(
        df_4, ["distance_bucket", "field_zone", "score_bucket"], ["pass", "run", "punt", "field_goal"]
    )
    model.fourth_down_global = _compute_global_dist(df_4, ["pass", "run", "punt", "field_goal"])

    # Team pass rate factors
    model.team_factors = _compute_team_factors(df_123)

    logger.info(
        "Play call model: %d context keys, %d 4th-down keys, %d teams",
        len(model.tables), len(model.fourth_down_tables), len(model.team_factors),
    )

    return model


def _build_conditional_tables(
    df: pl.DataFrame,
    context_cols: list[str],
    play_types: list[str],
) -> dict[tuple, dict[str, float]]:
    """Build conditional probability tables grouped by context columns."""
    tables: dict[tuple, dict[str, float]] = {}

    # Compute at full context granularity
    valid_cols = [c for c in context_cols if c in df.columns]
    if not valid_cols:
        return tables

    grouped = (
        df.group_by(valid_cols + ["play_category"])
        .agg(pl.col("recency_weight").sum().alias("weighted_count"))
    )

    # Pivot to get play type columns
    for context_vals, group_df in grouped.group_by(valid_cols):
        if isinstance(context_vals, (list, tuple)):
            key = tuple(context_vals)
        else:
            key = (context_vals,)

        total_weight = group_df["weighted_count"].sum()
        if total_weight < MIN_SAMPLES:
            continue

        dist = {}
        for row in group_df.iter_rows(named=True):
            ptype = row["play_category"]
            if ptype in play_types:
                dist[ptype] = row["weighted_count"] / total_weight

        # Ensure all play types present
        for pt in play_types:
            if pt not in dist:
                dist[pt] = 0.0

        # Normalize
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
            tables[key] = dist

    # Also build broader context tables (fewer columns) for fallback
    for level in range(len(valid_cols) - 1, 0, -1):
        sub_cols = valid_cols[:level]
        sub_grouped = (
            df.group_by(sub_cols + ["play_category"])
            .agg(pl.col("recency_weight").sum().alias("weighted_count"))
        )
        for context_vals, group_df in sub_grouped.group_by(sub_cols):
            if isinstance(context_vals, (list, tuple)):
                key = tuple(context_vals)
            else:
                key = (context_vals,)

            if key in tables:
                continue  # don't overwrite more specific table

            total_weight = group_df["weighted_count"].sum()
            if total_weight < MIN_SAMPLES:
                continue

            dist = {}
            for row in group_df.iter_rows(named=True):
                ptype = row["play_category"]
                if ptype in play_types:
                    dist[ptype] = row["weighted_count"] / total_weight

            for pt in play_types:
                if pt not in dist:
                    dist[pt] = 0.0

            total = sum(dist.values())
            if total > 0:
                dist = {k: v / total for k, v in dist.items()}
                tables[key] = dist

    return tables


def _compute_global_dist(df: pl.DataFrame, play_types: list[str]) -> dict[str, float]:
    """Compute league-wide play type distribution."""
    counts = (
        df.filter(pl.col("play_category").is_in(play_types))
        .group_by("play_category")
        .agg(pl.col("recency_weight").sum().alias("weighted_count"))
    )

    total = counts["weighted_count"].sum()
    dist = {}
    for row in counts.iter_rows(named=True):
        dist[row["play_category"]] = row["weighted_count"] / total

    for pt in play_types:
        if pt not in dist:
            dist[pt] = 0.0

    return dist


def _compute_team_factors(df: pl.DataFrame) -> dict[str, float]:
    """Compute per-team pass rate as a multiplier vs league average."""
    league_pass_rate = (
        df.filter(pl.col("play_category") == "pass")["recency_weight"].sum()
        / df["recency_weight"].sum()
    )

    team_rates = (
        df.group_by("posteam")
        .agg([
            pl.col("recency_weight").filter(pl.col("play_category") == "pass").sum().alias("pass_weight"),
            pl.col("recency_weight").sum().alias("total_weight"),
        ])
    )

    factors = {}
    for row in team_rates.iter_rows(named=True):
        team = row["posteam"]
        if team and row["total_weight"] > 0:
            team_rate = row["pass_weight"] / row["total_weight"]
            factors[team] = team_rate / league_pass_rate if league_pass_rate > 0 else 1.0

    return factors
