"""Playoff bracket simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from nflsim.season.standings import TeamRecord

logger = logging.getLogger(__name__)

# Type alias for the game simulator function
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nflsim.models.store import ModelBundle


@dataclass
class PlayoffResult:
    """Results of a full playoff bracket."""
    afc_seeds: list[str]    # 7 team abbreviations in seed order
    nfc_seeds: list[str]
    # Round results: list of (winner, loser, score) tuples
    wildcard: list[tuple[str, str, str]] = None
    divisional: list[tuple[str, str, str]] = None
    conference: list[tuple[str, str, str]] = None
    super_bowl: tuple[str, str, str] | None = None

    @property
    def champion(self) -> str | None:
        return self.super_bowl[0] if self.super_bowl else None

    def __post_init__(self):
        if self.wildcard is None:
            self.wildcard = []
        if self.divisional is None:
            self.divisional = []
        if self.conference is None:
            self.conference = []


def simulate_playoffs(
    afc_seeds: list[TeamRecord],
    nfc_seeds: list[TeamRecord],
    rng: np.random.Generator,
    models: ModelBundle | None = None,
) -> PlayoffResult:
    """Simulate the full NFL playoff bracket.

    Bracket structure (per conference):
      Wild Card: #2 vs #7, #3 vs #6, #4 vs #5  (higher seed hosts)
      Divisional: #1 vs lowest surviving, other two
      Conference Championship: survivors
      Super Bowl: AFC champ vs NFC champ (neutral site)
    """
    from nflsim.engine.game import simulate_game

    afc = [r.team for r in afc_seeds[:7]]
    nfc = [r.team for r in nfc_seeds[:7]]

    result = PlayoffResult(afc_seeds=afc, nfc_seeds=nfc)

    def play_game(home: str, away: str) -> tuple[str, str, str]:
        """Simulate a playoff game, return (winner, loser, score_str)."""
        seed = int(rng.integers(0, 2**31))
        gr = simulate_game(home, away, seed=seed, models=models)
        # No ties in playoffs — if tied, home team wins (simplified OT)
        if gr.home_score > gr.away_score:
            winner, loser = home, away
        elif gr.away_score > gr.home_score:
            winner, loser = away, home
        else:
            # Tie in our sim = go to OT which is already handled,
            # but if still tied, give to higher seed (home)
            winner, loser = home, away
        score = f"{gr.away_team} {gr.away_score} @ {gr.home_team} {gr.home_score}"
        return winner, loser, score

    # ---- Wild Card Round ----
    afc_wc_winners = []
    nfc_wc_winners = []

    for conf, seeds, winners in [
        ("AFC", afc, afc_wc_winners),
        ("NFC", nfc, nfc_wc_winners),
    ]:
        # #2 vs #7
        w, l, s = play_game(seeds[1], seeds[6])
        result.wildcard.append((w, l, s))
        winners.append(w)

        # #3 vs #6
        w, l, s = play_game(seeds[2], seeds[5])
        result.wildcard.append((w, l, s))
        winners.append(w)

        # #4 vs #5
        w, l, s = play_game(seeds[3], seeds[4])
        result.wildcard.append((w, l, s))
        winners.append(w)

    # ---- Divisional Round ----
    afc_div_winners = []
    nfc_div_winners = []

    for conf, seeds, wc_winners, div_winners in [
        ("AFC", afc, afc_wc_winners, afc_div_winners),
        ("NFC", nfc, nfc_wc_winners, nfc_div_winners),
    ]:
        # #1 seed vs lowest remaining seed
        # Sort WC winners by their original seed
        wc_by_seed = sorted(wc_winners, key=lambda t: seeds.index(t))
        lowest = wc_by_seed[-1]  # worst seed
        others = [t for t in wc_by_seed if t != lowest]

        # #1 vs lowest
        w, l, s = play_game(seeds[0], lowest)
        result.divisional.append((w, l, s))
        div_winners.append(w)

        # Remaining two play each other — better seed hosts
        if seeds.index(others[0]) < seeds.index(others[1]):
            home, away = others[0], others[1]
        else:
            home, away = others[1], others[0]
        w, l, s = play_game(home, away)
        result.divisional.append((w, l, s))
        div_winners.append(w)

    # ---- Conference Championships ----
    afc_champ = None
    nfc_champ = None

    for conf, seeds, div_winners in [
        ("AFC", afc, afc_div_winners),
        ("NFC", nfc, nfc_div_winners),
    ]:
        # Better seed hosts
        if seeds.index(div_winners[0]) < seeds.index(div_winners[1]):
            home, away = div_winners[0], div_winners[1]
        else:
            home, away = div_winners[1], div_winners[0]
        w, l, s = play_game(home, away)
        result.conference.append((w, l, s))
        if conf == "AFC":
            afc_champ = w
        else:
            nfc_champ = w

    # ---- Super Bowl (neutral site — pick one as "home") ----
    w, l, s = play_game(nfc_champ, afc_champ)
    result.super_bowl = (w, l, s)

    return result
