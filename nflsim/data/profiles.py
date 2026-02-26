"""Player and team profile building from PBP data.

Builds per-team rosters with usage shares:
- QB: primary passer, completion rate, sack rate, INT rate
- Receivers: target share per player
- Rushers: carry share per player (excluding QB scrambles)

Profiles are recency-weighted so recent seasons matter more.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from nflsim.data.features import add_recency_weight

logger = logging.getLogger(__name__)

# Minimum share to keep a player (avoid long tail of 1-touch players)
MIN_SHARE = 0.02


@dataclass
class PlayerProfile:
    """A single player's profile."""
    player_id: str
    name: str
    position: str = ""  # QB, WR, RB, TE, etc.
    team: str = ""


@dataclass
class QBProfile(PlayerProfile):
    """Quarterback profile."""
    snap_share: float = 1.0  # fraction of team's pass plays


@dataclass
class ReceiverProfile(PlayerProfile):
    """Receiver/target profile."""
    target_share: float = 0.0  # fraction of team's targets


@dataclass
class RusherProfile(PlayerProfile):
    """Rusher profile (non-QB carries)."""
    carry_share: float = 0.0  # fraction of team's designed runs


@dataclass
class TeamProfile:
    """Complete offensive profile for a team."""
    team: str
    qbs: list[QBProfile] = field(default_factory=list)
    receivers: list[ReceiverProfile] = field(default_factory=list)
    rushers: list[RusherProfile] = field(default_factory=list)

    def primary_qb(self) -> QBProfile | None:
        return self.qbs[0] if self.qbs else None

    def sample_qb(self, rng: np.random.Generator) -> QBProfile | None:
        if not self.qbs:
            return None
        if len(self.qbs) == 1:
            return self.qbs[0]
        shares = np.array([q.snap_share for q in self.qbs])
        shares = shares / shares.sum()
        return self.qbs[rng.choice(len(self.qbs), p=shares)]

    def sample_receiver(self, rng: np.random.Generator) -> ReceiverProfile | None:
        if not self.receivers:
            return None
        shares = np.array([r.target_share for r in self.receivers])
        shares = shares / shares.sum()
        return self.receivers[rng.choice(len(self.receivers), p=shares)]

    def sample_rusher(self, rng: np.random.Generator) -> RusherProfile | None:
        if not self.rushers:
            return None
        shares = np.array([r.carry_share for r in self.rushers])
        shares = shares / shares.sum()
        return self.rushers[rng.choice(len(self.rushers), p=shares)]


def build_team_profiles(
    pbp: pl.DataFrame,
    current_season: int,
) -> dict[str, TeamProfile]:
    """Build profiles for all teams from PBP data.

    Args:
        pbp: Raw nflverse PBP data (multiple seasons)
        current_season: For recency weighting

    Returns:
        Dict mapping team abbreviation -> TeamProfile
    """
    logger.info("Building team profiles...")

    df = add_recency_weight(pbp, current_season)

    # Filter to real plays with valid team
    df = df.filter(
        pl.col("posteam").is_not_null()
        & (pl.col("play_type").is_in(["pass", "run"]))
    )

    # Build set of (player_id, team) pairs from the current season
    # so we only include players currently on the roster
    current_df = df.filter(pl.col("season") == current_season)
    _current_passers = set()
    _current_receivers = set()
    _current_rushers = set()
    for col, s in [
        ("passer_player_id", _current_passers),
        ("receiver_player_id", _current_receivers),
        ("rusher_player_id", _current_rushers),
    ]:
        if col in current_df.columns:
            pairs = (
                current_df.filter(pl.col(col).is_not_null())
                .select([col, "posteam"])
                .unique()
            )
            for row in pairs.iter_rows():
                s.add((row[0], row[1]))

    teams = df["posteam"].unique().to_list()
    profiles: dict[str, TeamProfile] = {}

    for team in teams:
        if team is None:
            continue
        team_df = df.filter(pl.col("posteam") == team)
        profile = TeamProfile(team=team)

        # QBs — only those on team in current season
        profile.qbs = _build_qb_profiles(team_df, team, _current_passers)

        # Receivers
        profile.receivers = _build_receiver_profiles(team_df, team, _current_receivers)

        # Rushers
        profile.rushers = _build_rusher_profiles(team_df, team, profile.qbs, _current_rushers)

        profiles[team] = profile

    logger.info("Built profiles for %d teams", len(profiles))
    return profiles


def _build_qb_profiles(
    df: pl.DataFrame, team: str, current_set: set[tuple[str, str]]
) -> list[QBProfile]:
    """Build QB profiles from pass play data."""
    passes = df.filter(
        (pl.col("play_type") == "pass")
        & pl.col("passer_player_id").is_not_null()
    )

    if passes.shape[0] == 0:
        return []

    qb_stats = (
        passes.group_by(["passer_player_id", "passer_player_name"])
        .agg(pl.col("recency_weight").sum().alias("weighted_attempts"))
        .sort("weighted_attempts", descending=True)
    )

    total_weight = qb_stats["weighted_attempts"].sum()
    qbs = []
    for row in qb_stats.iter_rows(named=True):
        pid = row["passer_player_id"]
        if (pid, team) not in current_set:
            continue
        share = row["weighted_attempts"] / total_weight
        if share < MIN_SHARE:
            continue
        qbs.append(QBProfile(
            player_id=pid,
            name=row["passer_player_name"],
            position="QB",
            team=team,
            snap_share=share,
        ))

    # Renormalize after filtering
    total_share = sum(q.snap_share for q in qbs)
    if total_share > 0:
        for q in qbs:
            q.snap_share /= total_share

    return qbs


def _build_receiver_profiles(
    df: pl.DataFrame, team: str, current_set: set[tuple[str, str]]
) -> list[ReceiverProfile]:
    """Build receiver profiles from target data."""
    targets = df.filter(
        (pl.col("play_type") == "pass")
        & pl.col("receiver_player_id").is_not_null()
        & (pl.col("sack") == 0)
    )

    if targets.shape[0] == 0:
        return []

    rec_stats = (
        targets.group_by(["receiver_player_id", "receiver_player_name"])
        .agg(pl.col("recency_weight").sum().alias("weighted_targets"))
        .sort("weighted_targets", descending=True)
    )

    total_weight = rec_stats["weighted_targets"].sum()
    receivers = []
    for row in rec_stats.iter_rows(named=True):
        pid = row["receiver_player_id"]
        if (pid, team) not in current_set:
            continue
        share = row["weighted_targets"] / total_weight
        if share < MIN_SHARE:
            continue
        receivers.append(ReceiverProfile(
            player_id=pid,
            name=row["receiver_player_name"],
            position="WR",
            team=team,
            target_share=share,
        ))

    # Normalize shares to sum to 1.0
    total_share = sum(r.target_share for r in receivers)
    if total_share > 0:
        for r in receivers:
            r.target_share /= total_share

    return receivers


def _build_rusher_profiles(
    df: pl.DataFrame, team: str, qbs: list[QBProfile],
    current_set: set[tuple[str, str]]
) -> list[RusherProfile]:
    """Build rusher profiles from carry data, separating QB scrambles."""
    rushes = df.filter(
        (pl.col("play_type") == "run")
        & pl.col("rusher_player_id").is_not_null()
    )

    if rushes.shape[0] == 0:
        return []

    qb_ids = {q.player_id for q in qbs}

    rush_stats = (
        rushes.group_by(["rusher_player_id", "rusher_player_name"])
        .agg(pl.col("recency_weight").sum().alias("weighted_carries"))
        .sort("weighted_carries", descending=True)
    )

    total_weight = rush_stats["weighted_carries"].sum()
    rushers = []
    for row in rush_stats.iter_rows(named=True):
        pid = row["rusher_player_id"]
        if (pid, team) not in current_set:
            continue
        share = row["weighted_carries"] / total_weight
        if share < MIN_SHARE:
            continue

        is_qb = pid in qb_ids
        rushers.append(RusherProfile(
            player_id=pid,
            name=row["rusher_player_name"],
            position="QB" if is_qb else "RB",
            team=team,
            carry_share=share,
        ))

    # Normalize
    total_share = sum(r.carry_share for r in rushers)
    if total_share > 0:
        for r in rushers:
            r.carry_share /= total_share

    return rushers
