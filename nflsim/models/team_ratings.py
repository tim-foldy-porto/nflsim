"""Team power ratings — per-team offensive and defensive multipliers from PBP data.

Each multiplier is relative to league average (1.0). Values > 1.0 mean
above-average for that metric (good offense = more yards; bad defense = more
yards allowed).

Matchup adjustment = off_mult * def_mult.  Both at 1.0 → no change.
Good offense (1.1) vs bad defense (1.1) → 1.21 (boosted).
Bad offense (0.9) vs good defense (0.9) → 0.81 (suppressed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from nflsim.data.features import add_recency_weight
from nflsim.season.schedule import normalize_team

logger = logging.getLogger(__name__)


@dataclass
class TeamRating:
    """Offensive and defensive multipliers for a single team."""
    team: str

    # Offensive multipliers (1.0 = league average)
    pass_yards_off: float = 1.0   # yards per completed pass
    rush_yards_off: float = 1.0   # yards per rush
    comp_rate_off: float = 1.0    # completion rate
    sack_rate_off: float = 1.0    # sack rate (>1 = MORE sacks = bad OL)
    int_rate_off: float = 1.0     # interception rate (>1 = MORE ints = bad QB)
    fumble_rate_off: float = 1.0  # fumble rate

    # Defensive multipliers (1.0 = league average, >1 = MORE allowed = bad defense)
    pass_yards_def: float = 1.0
    rush_yards_def: float = 1.0
    comp_rate_def: float = 1.0
    sack_rate_def: float = 1.0    # >1 = MORE sacks generated = good defense
    int_rate_def: float = 1.0     # >1 = MORE ints forced = good defense
    fumble_rate_def: float = 1.0  # >1 = MORE fumbles forced = good defense

    @property
    def off_rating(self) -> float:
        """Composite offensive rating (higher = better)."""
        # Weight pass yards most, then rush, then completion, penalize turnovers
        return (
            0.35 * self.pass_yards_off
            + 0.25 * self.rush_yards_off
            + 0.20 * self.comp_rate_off
            + 0.10 * (2.0 - self.sack_rate_off)    # invert: low sacks = good
            + 0.10 * (2.0 - self.int_rate_off)      # invert: low INTs = good
        )

    @property
    def def_rating(self) -> float:
        """Composite defensive rating (higher = better)."""
        return (
            0.35 * (2.0 - self.pass_yards_def)  # invert: low yards allowed = good
            + 0.25 * (2.0 - self.rush_yards_def)
            + 0.20 * (2.0 - self.comp_rate_def)
            + 0.10 * self.sack_rate_def          # high sacks = good
            + 0.10 * self.int_rate_def            # high INTs forced = good
        )

    @property
    def overall(self) -> float:
        """Overall power rating (higher = better)."""
        return (self.off_rating + self.def_rating) / 2


@dataclass
class MatchupAdjustment:
    """Pre-computed adjustments for offense-vs-defense matchup."""
    pass_yard_mult: float = 1.0
    rush_yard_mult: float = 1.0
    comp_rate_mult: float = 1.0
    sack_rate_mult: float = 1.0
    int_rate_mult: float = 1.0
    fumble_rate_mult: float = 1.0

    @staticmethod
    def from_matchup(off: TeamRating, def_: TeamRating) -> MatchupAdjustment:
        """Compute matchup adjustment from offensive and defensive ratings."""
        return MatchupAdjustment(
            pass_yard_mult=off.pass_yards_off * def_.pass_yards_def,
            rush_yard_mult=off.rush_yards_off * def_.rush_yards_def,
            comp_rate_mult=off.comp_rate_off * def_.comp_rate_def,
            sack_rate_mult=off.sack_rate_off * def_.sack_rate_def,
            int_rate_mult=off.int_rate_off * def_.int_rate_def,
            fumble_rate_mult=off.fumble_rate_off * def_.fumble_rate_def,
        )


def build_team_ratings(
    pbp: pl.DataFrame, current_season: int
) -> dict[str, TeamRating]:
    """Build per-team ratings from PBP data with recency weighting.

    Returns dict mapping team abbreviation to TeamRating.
    """
    # Filter to meaningful plays in regular season
    df = pbp.filter(
        pl.col("play_type").is_in(["pass", "run"]),
        pl.col("season_type") == "REG",
        pl.col("posteam").is_not_null(),
        pl.col("defteam").is_not_null(),
    )

    # Add recency weights
    df = add_recency_weight(df, current_season)

    # Normalize team names
    df = df.with_columns([
        pl.col("posteam").map_elements(normalize_team, return_dtype=pl.Utf8).alias("posteam"),
        pl.col("defteam").map_elements(normalize_team, return_dtype=pl.Utf8).alias("defteam"),
    ])

    # --- Compute league averages ---
    pass_plays = df.filter(pl.col("play_type") == "pass")
    run_plays = df.filter(pl.col("play_type") == "run")

    league_avg = _compute_league_averages(pass_plays, run_plays)

    # --- Compute per-team offensive stats ---
    off_stats = _compute_team_stats(pass_plays, run_plays, group_col="posteam")

    # --- Compute per-team defensive stats ---
    def_stats = _compute_team_stats(pass_plays, run_plays, group_col="defteam")

    # --- Build ratings ---
    ratings = {}
    all_teams = set(off_stats.keys()) | set(def_stats.keys())

    for team in all_teams:
        off = off_stats.get(team, league_avg)
        def_ = def_stats.get(team, league_avg)

        rating = TeamRating(
            team=team,
            # Offensive: team's stat / league average
            pass_yards_off=_safe_ratio(off["pass_ypa"], league_avg["pass_ypa"]),
            rush_yards_off=_safe_ratio(off["rush_ypc"], league_avg["rush_ypc"]),
            comp_rate_off=_safe_ratio(off["comp_rate"], league_avg["comp_rate"]),
            sack_rate_off=_safe_ratio(off["sack_rate"], league_avg["sack_rate"]),
            int_rate_off=_safe_ratio(off["int_rate"], league_avg["int_rate"]),
            fumble_rate_off=_safe_ratio(off["fumble_rate"], league_avg["fumble_rate"]),
            # Defensive: opponent's stats against this team / league average
            pass_yards_def=_safe_ratio(def_["pass_ypa"], league_avg["pass_ypa"]),
            rush_yards_def=_safe_ratio(def_["rush_ypc"], league_avg["rush_ypc"]),
            comp_rate_def=_safe_ratio(def_["comp_rate"], league_avg["comp_rate"]),
            sack_rate_def=_safe_ratio(def_["sack_rate"], league_avg["sack_rate"]),
            int_rate_def=_safe_ratio(def_["int_rate"], league_avg["int_rate"]),
            fumble_rate_def=_safe_ratio(def_["fumble_rate"], league_avg["fumble_rate"]),
        )

        # Dampen extreme values toward 1.0 to avoid unrealistic outcomes
        rating = _dampen_rating(rating, strength=0.3)
        ratings[team] = rating

    logger.info(
        "Built team ratings for %d teams (range: %.2f to %.2f overall)",
        len(ratings),
        min(r.overall for r in ratings.values()),
        max(r.overall for r in ratings.values()),
    )

    return ratings


def _compute_league_averages(
    pass_plays: pl.DataFrame, run_plays: pl.DataFrame
) -> dict[str, float]:
    """Compute weighted league averages."""
    w = "recency_weight"

    completions = pass_plays.filter(pl.col("complete_pass") == 1)

    # Weighted pass yards per attempt (completed passes only)
    pass_ypa = _weighted_mean(completions, "yards_gained", w)
    # Weighted rush yards per carry
    rush_ypc = _weighted_mean(run_plays, "yards_gained", w)
    # Weighted completion rate
    comp_rate = _weighted_mean(pass_plays, "complete_pass", w)
    # Weighted sack rate
    sack_rate = _weighted_mean(pass_plays, "sack", w)
    # Weighted INT rate (among incomplete passes)
    int_rate = _weighted_mean(pass_plays, "interception", w)
    # Fumble rate
    all_plays = pl.concat([pass_plays, run_plays])
    fumble_rate = _weighted_mean(all_plays, "fumble_lost", w)

    return {
        "pass_ypa": pass_ypa,
        "rush_ypc": rush_ypc,
        "comp_rate": comp_rate,
        "sack_rate": sack_rate,
        "int_rate": int_rate,
        "fumble_rate": fumble_rate,
    }


def _compute_team_stats(
    pass_plays: pl.DataFrame,
    run_plays: pl.DataFrame,
    group_col: str,
) -> dict[str, dict[str, float]]:
    """Compute weighted stats per team (grouped by posteam or defteam)."""
    w = "recency_weight"
    result = {}

    all_teams = set(
        pass_plays.select(group_col).unique().to_series().to_list()
        + run_plays.select(group_col).unique().to_series().to_list()
    )

    for team in all_teams:
        tp = pass_plays.filter(pl.col(group_col) == team)
        tr = run_plays.filter(pl.col(group_col) == team)

        completions = tp.filter(pl.col("complete_pass") == 1)

        pass_ypa = _weighted_mean(completions, "yards_gained", w) if len(completions) > 0 else 0.0
        rush_ypc = _weighted_mean(tr, "yards_gained", w) if len(tr) > 0 else 0.0
        comp_rate = _weighted_mean(tp, "complete_pass", w) if len(tp) > 0 else 0.0
        sack_rate = _weighted_mean(tp, "sack", w) if len(tp) > 0 else 0.0
        int_rate = _weighted_mean(tp, "interception", w) if len(tp) > 0 else 0.0

        all_team = pl.concat([tp, tr]) if len(tp) > 0 and len(tr) > 0 else (tp if len(tp) > 0 else tr)
        fumble_rate = _weighted_mean(all_team, "fumble_lost", w) if len(all_team) > 0 else 0.0

        result[team] = {
            "pass_ypa": pass_ypa,
            "rush_ypc": rush_ypc,
            "comp_rate": comp_rate,
            "sack_rate": sack_rate,
            "int_rate": int_rate,
            "fumble_rate": fumble_rate,
        }

    return result


def _weighted_mean(df: pl.DataFrame, col: str, weight_col: str) -> float:
    """Compute weighted mean of a column."""
    if len(df) == 0:
        return 0.0
    result = df.select(
        (pl.col(col).cast(pl.Float64) * pl.col(weight_col)).sum()
        / pl.col(weight_col).sum()
    ).item()
    return float(result) if result is not None else 0.0


def _safe_ratio(team_val: float, league_val: float) -> float:
    """Compute ratio, avoiding division by zero."""
    if league_val == 0 or team_val == 0:
        return 1.0
    return team_val / league_val


def _dampen_rating(rating: TeamRating, strength: float = 0.6) -> TeamRating:
    """Dampen multipliers toward 1.0 to prevent extreme outcomes.

    strength: 0.0 = no dampening (raw ratios), 1.0 = fully dampened (all 1.0).
    Recommended: 0.5–0.7 for realistic spread without chaos.
    """
    def damp(v: float) -> float:
        return 1.0 + (v - 1.0) * (1.0 - strength)

    return TeamRating(
        team=rating.team,
        pass_yards_off=damp(rating.pass_yards_off),
        rush_yards_off=damp(rating.rush_yards_off),
        comp_rate_off=damp(rating.comp_rate_off),
        sack_rate_off=damp(rating.sack_rate_off),
        int_rate_off=damp(rating.int_rate_off),
        fumble_rate_off=damp(rating.fumble_rate_off),
        pass_yards_def=damp(rating.pass_yards_def),
        rush_yards_def=damp(rating.rush_yards_def),
        comp_rate_def=damp(rating.comp_rate_def),
        sack_rate_def=damp(rating.sack_rate_def),
        int_rate_def=damp(rating.int_rate_def),
        fumble_rate_def=damp(rating.fumble_rate_def),
    )
