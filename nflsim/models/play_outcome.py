"""Play outcome model — empirical distributions for yards gained, turnovers, sacks.

Instead of parametric distributions (normal, exponential), we sample directly from
historical play outcomes binned by context. This captures the true shape of NFL
yard distributions: bimodal passes, spike-at-zero rushes, fat tails.

Key rates modeled:
- Sack rate (per dropback, by context)
- Completion rate (by down/distance/field)
- Yards gained distribution (empirical histogram sampling)
- Interception rate (per incomplete)
- Fumble rate (per play)
- Out-of-bounds rate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from nflsim.data.features import CONTEXT_COLS, add_features

logger = logging.getLogger(__name__)

MIN_SAMPLES = 20
BROAD_CONTEXT = ["down", "distance_bucket"]


@dataclass
class PlayOutcomeModel:
    """Empirical outcome distributions keyed by context."""

    # Pass outcomes
    sack_rates: dict[tuple, float] = field(default_factory=dict)
    completion_rates: dict[tuple, float] = field(default_factory=dict)
    pass_yard_bins: dict[tuple, np.ndarray] = field(default_factory=dict)  # completed pass yards
    air_yard_bins: dict[tuple, np.ndarray] = field(default_factory=dict)
    yac_bins: dict[tuple, np.ndarray] = field(default_factory=dict)
    int_rate: float = 0.025  # league-wide INT rate per pass attempt
    sack_fumble_rate: float = 0.12  # fumble rate per sack
    catch_fumble_rate: float = 0.01  # fumble rate per completion

    # Rush outcomes
    rush_yard_bins: dict[tuple, np.ndarray] = field(default_factory=dict)
    rush_fumble_rate: float = 0.015

    # Out of bounds rates
    pass_oob_rate: float = 0.15
    rush_oob_rate: float = 0.08

    # Global fallbacks
    global_sack_rate: float = 0.065
    global_comp_rate: float = 0.65
    global_pass_yards: np.ndarray | None = None
    global_rush_yards: np.ndarray | None = None

    def sample_sack(self, key: tuple, rng: np.random.Generator) -> bool:
        rate = self._lookup_rate(key, self.sack_rates, self.global_sack_rate)
        return rng.random() < rate

    def sample_completion(self, key: tuple, rng: np.random.Generator) -> bool:
        rate = self._lookup_rate(key, self.completion_rates, self.global_comp_rate)
        return rng.random() < rate

    def get_comp_rate(self, key: tuple) -> float | None:
        """Get the completion rate for a context key (for matchup adjustments)."""
        return self._lookup_rate(key, self.completion_rates, self.global_comp_rate)

    def sample_pass_yards(self, key: tuple, rng: np.random.Generator) -> int:
        arr = self._lookup_array(key, self.pass_yard_bins, self.global_pass_yards)
        if arr is not None and len(arr) > 0:
            return int(rng.choice(arr))
        return int(rng.exponential(7))  # ultimate fallback

    def sample_air_yards(self, key: tuple, rng: np.random.Generator) -> int:
        arr = self._lookup_array(key, self.air_yard_bins, None)
        if arr is not None and len(arr) > 0:
            return int(rng.choice(arr))
        return int(rng.exponential(7))

    def sample_yac(self, key: tuple, rng: np.random.Generator) -> int:
        arr = self._lookup_array(key, self.yac_bins, None)
        if arr is not None and len(arr) > 0:
            return max(0, int(rng.choice(arr)))
        return max(0, int(rng.exponential(5)))

    def sample_rush_yards(self, key: tuple, rng: np.random.Generator) -> int:
        arr = self._lookup_array(key, self.rush_yard_bins, self.global_rush_yards)
        if arr is not None and len(arr) > 0:
            return int(rng.choice(arr))
        return int(rng.normal(4.2, 3.5))

    def sample_sack_yards(self, rng: np.random.Generator) -> int:
        return -int(rng.uniform(3, 12))

    def _lookup_rate(
        self, key: tuple, table: dict[tuple, float], fallback: float
    ) -> float:
        if key in table:
            return table[key]
        # Try broader keys
        for i in range(len(key) - 1, 0, -1):
            broader = key[:i]
            if broader in table:
                return table[broader]
        return fallback

    def _lookup_array(
        self,
        key: tuple,
        table: dict[tuple, np.ndarray],
        fallback: np.ndarray | None,
    ) -> np.ndarray | None:
        if key in table:
            return table[key]
        for i in range(len(key) - 1, 0, -1):
            broader = key[:i]
            if broader in table:
                return table[broader]
        return fallback


def build_play_outcome_model(pbp: pl.DataFrame, current_season: int) -> PlayOutcomeModel:
    """Build outcome model from PBP data."""
    logger.info("Building play outcome model...")

    df = add_features(pbp, current_season=current_season)

    model = PlayOutcomeModel()

    # ---- Pass plays ----
    passes = df.filter(
        (pl.col("play_type") == "pass")
        & pl.col("down").is_not_null()
        & pl.col("yardline_100").is_not_null()
    )

    # Sack rates by context
    model.sack_rates, model.global_sack_rate = _build_rate_table(
        passes, CONTEXT_COLS, "sack", fallback=0.065
    )

    # Completion rates (among non-sacks)
    non_sacks = passes.filter(pl.col("sack") == 0)
    model.completion_rates, model.global_comp_rate = _build_rate_table(
        non_sacks, CONTEXT_COLS, "complete_pass", fallback=0.65
    )

    # Pass yard distributions (completed passes)
    completions = non_sacks.filter(pl.col("complete_pass") == 1)
    model.pass_yard_bins = _build_yard_bins(completions, CONTEXT_COLS, "yards_gained")
    model.global_pass_yards = _global_yard_array(completions, "yards_gained")

    # Air yards and YAC distributions
    valid_air = completions.filter(pl.col("air_yards").is_not_null())
    model.air_yard_bins = _build_yard_bins(valid_air, BROAD_CONTEXT, "air_yards")

    valid_yac = completions.filter(pl.col("yards_after_catch").is_not_null())
    model.yac_bins = _build_yard_bins(valid_yac, BROAD_CONTEXT, "yards_after_catch")

    # INT rate
    total_attempts = non_sacks.shape[0]
    total_ints = non_sacks.filter(pl.col("interception") == 1).shape[0]
    model.int_rate = total_ints / total_attempts if total_attempts > 0 else 0.025

    # Fumble rates
    sacks = passes.filter(pl.col("sack") == 1)
    sack_fumbles = sacks.filter(pl.col("fumble") == 1).shape[0]
    model.sack_fumble_rate = sack_fumbles / sacks.shape[0] if sacks.shape[0] > 0 else 0.12

    catch_fumbles = completions.filter(pl.col("fumble") == 1).shape[0]
    model.catch_fumble_rate = catch_fumbles / completions.shape[0] if completions.shape[0] > 0 else 0.01

    # ---- Rush plays ----
    rushes = df.filter(
        (pl.col("play_type") == "run")
        & pl.col("down").is_not_null()
        & pl.col("yardline_100").is_not_null()
    )

    model.rush_yard_bins = _build_yard_bins(rushes, CONTEXT_COLS, "yards_gained")
    model.global_rush_yards = _global_yard_array(rushes, "yards_gained")

    rush_fumbles = rushes.filter(pl.col("fumble_lost") == 1).shape[0]
    model.rush_fumble_rate = rush_fumbles / rushes.shape[0] if rushes.shape[0] > 0 else 0.015

    # OOB rates (approximate from data if available)
    # nflverse doesn't have a direct OOB column, so we keep defaults

    logger.info(
        "Outcome model: %d sack keys, %d comp keys, %d pass yard keys, %d rush yard keys",
        len(model.sack_rates), len(model.completion_rates),
        len(model.pass_yard_bins), len(model.rush_yard_bins),
    )

    return model


def _build_rate_table(
    df: pl.DataFrame,
    context_cols: list[str],
    target_col: str,
    fallback: float,
) -> tuple[dict[tuple, float], float]:
    """Build a rate table: P(target_col == 1 | context)."""
    tables: dict[tuple, float] = {}
    valid_cols = [c for c in context_cols if c in df.columns]

    if not valid_cols or target_col not in df.columns:
        return tables, fallback

    # Global rate
    total = df.shape[0]
    positives = df.filter(pl.col(target_col) == 1).shape[0]
    global_rate = positives / total if total > 0 else fallback

    # By context
    grouped = (
        df.group_by(valid_cols)
        .agg([
            pl.col(target_col).sum().alias("positives"),
            pl.len().alias("total"),
        ])
    )

    for row in grouped.iter_rows(named=True):
        key = tuple(row[c] for c in valid_cols)
        n = row["total"]
        if n < MIN_SAMPLES:
            continue
        raw_rate = row["positives"] / n
        # Blend with global
        alpha = min(1.0, n / 100)
        blended = alpha * raw_rate + (1 - alpha) * global_rate
        tables[key] = blended

    # Broader context fallbacks
    for level in range(len(valid_cols) - 1, 0, -1):
        sub_cols = valid_cols[:level]
        sub_grouped = (
            df.group_by(sub_cols)
            .agg([
                pl.col(target_col).sum().alias("positives"),
                pl.len().alias("total"),
            ])
        )
        for row in sub_grouped.iter_rows(named=True):
            key = tuple(row[c] for c in sub_cols)
            if key in tables:
                continue
            n = row["total"]
            if n < MIN_SAMPLES:
                continue
            raw_rate = row["positives"] / n
            alpha = min(1.0, n / 100)
            tables[key] = alpha * raw_rate + (1 - alpha) * global_rate

    return tables, global_rate


def _build_yard_bins(
    df: pl.DataFrame,
    context_cols: list[str],
    yard_col: str,
) -> dict[tuple, np.ndarray]:
    """Build empirical yard arrays by context for sampling."""
    bins: dict[tuple, np.ndarray] = {}
    valid_cols = [c for c in context_cols if c in df.columns]

    if not valid_cols or yard_col not in df.columns:
        return bins

    df_clean = df.filter(pl.col(yard_col).is_not_null())

    for context_vals, group_df in df_clean.group_by(valid_cols):
        if isinstance(context_vals, (list, tuple)):
            key = tuple(context_vals)
        else:
            key = (context_vals,)

        if group_df.shape[0] < MIN_SAMPLES:
            continue

        yards = group_df[yard_col].to_numpy().astype(np.int32)
        bins[key] = yards

    # Broader fallbacks
    for level in range(len(valid_cols) - 1, 0, -1):
        sub_cols = valid_cols[:level]
        for context_vals, group_df in df_clean.group_by(sub_cols):
            if isinstance(context_vals, (list, tuple)):
                key = tuple(context_vals)
            else:
                key = (context_vals,)
            if key in bins:
                continue
            if group_df.shape[0] < MIN_SAMPLES:
                continue
            yards = group_df[yard_col].to_numpy().astype(np.int32)
            bins[key] = yards

    return bins


def _global_yard_array(df: pl.DataFrame, yard_col: str) -> np.ndarray | None:
    """Get global yard distribution as a numpy array."""
    clean = df.filter(pl.col(yard_col).is_not_null())
    if clean.shape[0] == 0:
        return None
    return clean[yard_col].to_numpy().astype(np.int32)
