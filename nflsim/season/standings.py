"""Standings computation with NFL tiebreaker rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

from nflsim.season.schedule import TEAM_TO_CONFERENCE, TEAM_TO_DIVISION


@dataclass
class TeamRecord:
    """Win-loss-tie record for a single team."""
    team: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: int = 0
    points_against: int = 0
    div_wins: int = 0
    div_losses: int = 0
    div_ties: int = 0
    conf_wins: int = 0
    conf_losses: int = 0
    conf_ties: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_pct(self) -> float:
        g = self.games
        return (self.wins + 0.5 * self.ties) / g if g > 0 else 0.0

    @property
    def div_win_pct(self) -> float:
        g = self.div_wins + self.div_losses + self.div_ties
        return (self.div_wins + 0.5 * self.div_ties) / g if g > 0 else 0.0

    @property
    def conf_win_pct(self) -> float:
        g = self.conf_wins + self.conf_losses + self.conf_ties
        return (self.conf_wins + 0.5 * self.conf_ties) / g if g > 0 else 0.0

    @property
    def point_diff(self) -> int:
        return self.points_for - self.points_against

    def record_str(self) -> str:
        if self.ties > 0:
            return f"{self.wins}-{self.losses}-{self.ties}"
        return f"{self.wins}-{self.losses}"

    def __str__(self) -> str:
        return f"{self.team:>4s} {self.record_str():>7s}  PF:{self.points_for:4d}  PA:{self.points_against:4d}  Diff:{self.point_diff:+4d}"


@dataclass
class GameOutcome:
    """Result of a single game for standings purposes."""
    home_team: str
    away_team: str
    home_score: int
    away_score: int


class Standings:
    """NFL standings tracker with tiebreaker logic."""

    def __init__(self, teams: set[str] | None = None):
        self.records: dict[str, TeamRecord] = {}
        if teams:
            for t in teams:
                self.records[t] = TeamRecord(team=t)

    def record_game(self, outcome: GameOutcome) -> None:
        """Record a game result into standings."""
        home = outcome.home_team
        away = outcome.away_team

        # Ensure records exist
        if home not in self.records:
            self.records[home] = TeamRecord(team=home)
        if away not in self.records:
            self.records[away] = TeamRecord(team=away)

        hr = self.records[home]
        ar = self.records[away]

        hr.points_for += outcome.home_score
        hr.points_against += outcome.away_score
        ar.points_for += outcome.away_score
        ar.points_against += outcome.home_score

        is_div = TEAM_TO_DIVISION.get(home) == TEAM_TO_DIVISION.get(away)
        is_conf = TEAM_TO_CONFERENCE.get(home) == TEAM_TO_CONFERENCE.get(away)

        if outcome.home_score > outcome.away_score:
            hr.wins += 1
            ar.losses += 1
            if is_div:
                hr.div_wins += 1
                ar.div_losses += 1
            if is_conf:
                hr.conf_wins += 1
                ar.conf_losses += 1
        elif outcome.away_score > outcome.home_score:
            ar.wins += 1
            hr.losses += 1
            if is_div:
                ar.div_wins += 1
                hr.div_losses += 1
            if is_conf:
                ar.conf_wins += 1
                hr.conf_losses += 1
        else:
            hr.ties += 1
            ar.ties += 1
            if is_div:
                hr.div_ties += 1
                ar.div_ties += 1
            if is_conf:
                hr.conf_ties += 1
                ar.conf_ties += 1

    def division_standings(self, division: str) -> list[TeamRecord]:
        """Get sorted standings for a division.

        Tiebreakers (simplified): win%, div win%, conf win%, point diff.
        """
        teams = [
            r for r in self.records.values()
            if TEAM_TO_DIVISION.get(r.team) == division
        ]
        return sorted(teams, key=lambda r: (
            r.win_pct, r.div_win_pct, r.conf_win_pct, r.point_diff
        ), reverse=True)

    def conference_standings(self, conference: str) -> list[TeamRecord]:
        """Get sorted standings for a conference."""
        teams = [
            r for r in self.records.values()
            if TEAM_TO_CONFERENCE.get(r.team) == conference
        ]
        return sorted(teams, key=lambda r: (
            r.win_pct, r.conf_win_pct, r.point_diff
        ), reverse=True)

    def playoff_seeds(self, conference: str) -> list[TeamRecord]:
        """Determine the 7 playoff seeds for a conference.

        Seeds 1-4: Division winners (sorted by record).
        Seeds 5-7: Best remaining wild cards.
        """
        divisions_in_conf = {
            div for div, teams_list in _conf_divisions(conference)
        }

        # Division winners
        div_winners = []
        remaining = []
        for div in sorted(divisions_in_conf):
            standings = self.division_standings(div)
            if standings:
                div_winners.append(standings[0])
                remaining.extend(standings[1:])

        # Sort division winners by record
        div_winners.sort(key=lambda r: (
            r.win_pct, r.conf_win_pct, r.point_diff
        ), reverse=True)

        # Wild cards: best non-division-winners
        remaining.sort(key=lambda r: (
            r.win_pct, r.conf_win_pct, r.point_diff
        ), reverse=True)

        seeds = div_winners[:4] + remaining[:3]
        return seeds


def _conf_divisions(conference: str):
    """Yield (division_name, teams) for a conference."""
    from nflsim.season.schedule import DIVISIONS
    if conference in DIVISIONS:
        for div_name, teams in DIVISIONS[conference].items():
            yield f"{conference} {div_name}", teams
