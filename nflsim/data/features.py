"""Feature engineering — bin continuous PBP columns into discrete keys for lookup tables.

These bins define the "context" for conditional distributions. A play's context is:
    (down, distance_bucket, field_zone, score_bucket, situation)

The goal is bins wide enough to have sufficient sample sizes (30+ weighted plays)
but narrow enough to capture meaningful behavioral differences.
"""

from __future__ import annotations

import polars as pl


def add_features(df: pl.DataFrame, current_season: int | None = None) -> pl.DataFrame:
    """Add all derived feature columns to a PBP dataframe.

    Args:
        df: Raw nflverse PBP data
        current_season: If provided, compute recency weights relative to this season

    Returns:
        DataFrame with added feature columns
    """
    df = add_distance_bucket(df)
    df = add_field_zone(df)
    df = add_score_bucket(df)
    df = add_situation(df)
    df = add_play_category(df)
    if current_season is not None:
        df = add_recency_weight(df, current_season)
    return df


def add_distance_bucket(df: pl.DataFrame) -> pl.DataFrame:
    """Bucket yards-to-go into discrete categories."""
    return df.with_columns(
        pl.when(pl.col("ydstogo") <= 1).then(pl.lit("1"))
        .when(pl.col("ydstogo") <= 3).then(pl.lit("2-3"))
        .when(pl.col("ydstogo") <= 6).then(pl.lit("4-6"))
        .when(pl.col("ydstogo") <= 10).then(pl.lit("7-10"))
        .when(pl.col("ydstogo") <= 15).then(pl.lit("11-15"))
        .otherwise(pl.lit("16+"))
        .alias("distance_bucket")
    )


def add_field_zone(df: pl.DataFrame) -> pl.DataFrame:
    """Bucket field position into zones.

    yardline_100 = yards from opponent's end zone.
    So yardline_100=20 means "in the red zone", yardline_100=95 means "backed up".
    """
    return df.with_columns(
        pl.when(pl.col("yardline_100") <= 3).then(pl.lit("goal_line"))    # inside 3
        .when(pl.col("yardline_100") <= 10).then(pl.lit("green_zone"))    # inside 10
        .when(pl.col("yardline_100") <= 20).then(pl.lit("red_zone"))      # inside 20
        .when(pl.col("yardline_100") <= 40).then(pl.lit("plus_territory"))  # opp 21-40
        .when(pl.col("yardline_100") <= 60).then(pl.lit("midfield"))      # 41-60
        .when(pl.col("yardline_100") <= 80).then(pl.lit("own_territory")) # own 21-40
        .when(pl.col("yardline_100") <= 90).then(pl.lit("own_deep"))      # own 11-20
        .otherwise(pl.lit("backed_up"))                                     # own 1-10
        .alias("field_zone")
    )


def add_score_bucket(df: pl.DataFrame) -> pl.DataFrame:
    """Bucket score differential into categories (from possession team's perspective)."""
    return df.with_columns(
        pl.when(pl.col("score_differential") <= -17).then(pl.lit("down_17+"))
        .when(pl.col("score_differential") <= -9).then(pl.lit("down_9_16"))
        .when(pl.col("score_differential") <= -4).then(pl.lit("down_4_8"))
        .when(pl.col("score_differential") <= -1).then(pl.lit("down_1_3"))
        .when(pl.col("score_differential") == 0).then(pl.lit("tied"))
        .when(pl.col("score_differential") <= 3).then(pl.lit("up_1_3"))
        .when(pl.col("score_differential") <= 8).then(pl.lit("up_4_8"))
        .when(pl.col("score_differential") <= 16).then(pl.lit("up_9_16"))
        .otherwise(pl.lit("up_17+"))
        .alias("score_bucket")
    )


def add_situation(df: pl.DataFrame) -> pl.DataFrame:
    """Classify game situation from PBP data.

    Mirrors GameState.situation but computed from raw data columns.
    """
    return df.with_columns(
        pl.when(
            (pl.col("half_seconds_remaining") <= 120)
            & (pl.col("game_half").is_in(["Half1", "Half2"]))
        ).then(pl.lit("two_minute"))
        .when(
            (pl.col("quarter_seconds_remaining") <= 120)
            & (pl.col("down").is_not_null())
            & (pl.col("game_half") == "Half2")
            & (pl.col("score_differential").abs() > 16)
        ).then(pl.lit("garbage_time"))
        .when(pl.col("yardline_100") <= 3).then(pl.lit("goal_line"))
        .when(pl.col("yardline_100") <= 20).then(pl.lit("red_zone"))
        .when(pl.col("yardline_100") >= 90).then(pl.lit("backed_up"))
        .otherwise(pl.lit("normal"))
        .alias("situation")
    )


def add_play_category(df: pl.DataFrame) -> pl.DataFrame:
    """Simplify play_type into categories we care about for modeling."""
    return df.with_columns(
        pl.when(pl.col("play_type") == "pass").then(pl.lit("pass"))
        .when(pl.col("play_type") == "run").then(pl.lit("run"))
        .when(pl.col("play_type") == "punt").then(pl.lit("punt"))
        .when(pl.col("play_type") == "field_goal").then(pl.lit("field_goal"))
        .when(pl.col("play_type") == "kickoff").then(pl.lit("kickoff"))
        .when(pl.col("play_type") == "qb_kneel").then(pl.lit("kneel"))
        .when(pl.col("play_type") == "qb_spike").then(pl.lit("spike"))
        .when(pl.col("play_type") == "no_play").then(pl.lit("no_play"))
        .otherwise(pl.lit("other"))
        .alias("play_category")
    )


def add_recency_weight(df: pl.DataFrame, current_season: int) -> pl.DataFrame:
    """Add exponential recency weight with half-life of 1.5 seasons.

    w = 2^(-age / 1.5) where age = current_season - play_season.
    """
    return df.with_columns(
        (2.0 ** (-(current_season - pl.col("season")).cast(pl.Float64) / 1.5))
        .alias("recency_weight")
    )


# Context key columns used for lookup tables
CONTEXT_COLS = ["down", "distance_bucket", "field_zone", "score_bucket"]

# Extended context that includes situation
FULL_CONTEXT_COLS = ["down", "distance_bucket", "field_zone", "score_bucket", "situation"]
