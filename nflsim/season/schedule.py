"""Schedule loading and NFL division/conference structure."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nflsim.data.loader import load_schedules

# NFL division structure (2024)
DIVISIONS: dict[str, dict[str, list[str]]] = {
    "AFC": {
        "East": ["BUF", "MIA", "NE", "NYJ"],
        "North": ["BAL", "CIN", "CLE", "PIT"],
        "South": ["HOU", "IND", "JAX", "TEN"],
        "West": ["DEN", "KC", "LAC", "LV"],
    },
    "NFC": {
        "East": ["DAL", "NYG", "PHI", "WAS"],
        "North": ["CHI", "DET", "GB", "MIN"],
        "South": ["ATL", "CAR", "NO", "TB"],
        "West": ["ARI", "LA", "SEA", "SF"],
    },
}

# Flat lookups
TEAM_TO_CONFERENCE: dict[str, str] = {}
TEAM_TO_DIVISION: dict[str, str] = {}
for conf, divs in DIVISIONS.items():
    for div, teams in divs.items():
        for team in teams:
            TEAM_TO_CONFERENCE[team] = conf
            TEAM_TO_DIVISION[team] = f"{conf} {div}"

# Alias mapping for team abbreviation discrepancies
TEAM_ALIASES = {
    "LAR": "LA",  # nflverse schedules use "LA" for Rams
}


def normalize_team(abbr: str) -> str:
    """Normalize team abbreviation to our canonical form."""
    return TEAM_ALIASES.get(abbr, abbr)


@dataclass
class ScheduledGame:
    """A single scheduled game."""
    week: int
    away_team: str
    home_team: str
    game_type: str = "REG"  # REG, WC, DIV, CON, SB


def load_season_schedule(season: int) -> list[ScheduledGame]:
    """Load the schedule for a season from cached data.

    Returns list of ScheduledGame sorted by week.
    """
    df = load_schedules([season])
    df = df.filter(pl.col("season") == season)

    games = []
    for row in df.sort("week").iter_rows(named=True):
        games.append(ScheduledGame(
            week=row["week"],
            away_team=normalize_team(row["away_team"]),
            home_team=normalize_team(row["home_team"]),
            game_type=row.get("game_type", "REG"),
        ))

    return games


def get_regular_season_games(schedule: list[ScheduledGame]) -> list[ScheduledGame]:
    """Filter to regular season games only."""
    return [g for g in schedule if g.game_type == "REG"]


def get_teams_in_schedule(schedule: list[ScheduledGame]) -> set[str]:
    """Get all teams that appear in the schedule."""
    teams = set()
    for g in schedule:
        teams.add(g.away_team)
        teams.add(g.home_team)
    return teams
