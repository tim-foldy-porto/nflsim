"""Season orchestrator — simulate a full NFL season with standings and playoffs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from typing import TYPE_CHECKING

from nflsim.engine.game import simulate_game
from nflsim.season.playoffs import PlayoffResult, simulate_playoffs
from nflsim.season.schedule import (
    ScheduledGame,
    get_regular_season_games,
    get_teams_in_schedule,
    load_season_schedule,
)
from nflsim.season.standings import GameOutcome, Standings

if TYPE_CHECKING:
    from nflsim.models.store import ModelBundle

logger = logging.getLogger(__name__)


@dataclass
class SeasonResult:
    """Result of a full season simulation."""
    season: int
    standings: Standings
    playoff_result: PlayoffResult
    champion: str | None = None

    # Per-team season records for easy access
    team_records: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.team_records and self.standings:
            for team, rec in self.standings.records.items():
                self.team_records[team] = rec.record_str()
        if self.champion is None and self.playoff_result:
            self.champion = self.playoff_result.champion


def simulate_season(
    season: int,
    *,
    seed: int | None = None,
    models: ModelBundle | None = None,
    schedule: list[ScheduledGame] | None = None,
) -> SeasonResult:
    """Simulate a full NFL season: 18-week regular season + playoffs.

    Args:
        season: Year to simulate (uses that year's schedule)
        seed: Random seed for reproducibility
        models: Statistical models for game simulation
        schedule: Pre-loaded schedule (avoids re-loading for bulk sims)

    Returns:
        SeasonResult with standings, playoff bracket, and champion
    """
    rng = np.random.default_rng(seed)

    # Load schedule (or use pre-loaded)
    if schedule is None:
        schedule = load_season_schedule(season)
    reg_games = get_regular_season_games(schedule)
    teams = get_teams_in_schedule(schedule)

    logger.info("Simulating %d season: %d regular season games", season, len(reg_games))

    # Initialize standings
    standings = Standings(teams)

    # Simulate regular season
    for game in reg_games:
        game_seed = int(rng.integers(0, 2**31))
        result = simulate_game(
            game.home_team, game.away_team,
            seed=game_seed, models=models,
        )
        standings.record_game(GameOutcome(
            home_team=game.home_team,
            away_team=game.away_team,
            home_score=result.home_score,
            away_score=result.away_score,
        ))

    # Determine playoff seeds
    afc_seeds = standings.playoff_seeds("AFC")
    nfc_seeds = standings.playoff_seeds("NFC")

    logger.info(
        "AFC #1: %s (%s), NFC #1: %s (%s)",
        afc_seeds[0].team, afc_seeds[0].record_str(),
        nfc_seeds[0].team, nfc_seeds[0].record_str(),
    )

    # Simulate playoffs
    playoff_result = simulate_playoffs(afc_seeds, nfc_seeds, rng, models)

    return SeasonResult(
        season=season,
        standings=standings,
        playoff_result=playoff_result,
    )
