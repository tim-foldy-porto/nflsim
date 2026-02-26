"""Play resolution — decides play call and simulates outcomes.

When a ModelBundle is provided, uses data-driven conditional distributions.
Falls back to Phase 1 heuristics when models are None.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nflsim.engine.game_state import GameState, PlayOutcome, PlayResult, Situation
from nflsim.models.team_ratings import MatchupAdjustment

if TYPE_CHECKING:
    from nflsim.models.store import ModelBundle


# ── Binning helpers (mirror data/features.py for live game state) ──

def _distance_bucket(ytg: int) -> str:
    if ytg <= 1: return "1"
    if ytg <= 3: return "2-3"
    if ytg <= 6: return "4-6"
    if ytg <= 10: return "7-10"
    if ytg <= 15: return "11-15"
    return "16+"


def _field_zone(opp_yl: int) -> str:
    """opp_yl = yards from opponent's goal line (= yardline_100 in nflverse)."""
    if opp_yl <= 3: return "goal_line"
    if opp_yl <= 10: return "green_zone"
    if opp_yl <= 20: return "red_zone"
    if opp_yl <= 40: return "plus_territory"
    if opp_yl <= 60: return "midfield"
    if opp_yl <= 80: return "own_territory"
    if opp_yl <= 90: return "own_deep"
    return "backed_up"


def _score_bucket(diff: int) -> str:
    if diff <= -17: return "down_17+"
    if diff <= -9: return "down_9_16"
    if diff <= -4: return "down_4_8"
    if diff <= -1: return "down_1_3"
    if diff == 0: return "tied"
    if diff <= 3: return "up_1_3"
    if diff <= 8: return "up_4_8"
    if diff <= 16: return "up_9_16"
    return "up_17+"


def _context_key(state: GameState) -> tuple:
    """Build the context lookup key from live game state."""
    return (
        state.down,
        _distance_bucket(state.yards_to_go),
        _field_zone(state.opponent_yard_line),
        _score_bucket(state.score_diff),
    )


# ── Play calling ──

def choose_play(
    state: GameState,
    rng: np.random.Generator,
    models: ModelBundle | None = None,
) -> str:
    """Decide what play to call given the game state."""
    down = state.down
    score_diff = state.score_diff
    qtr = state.quarter
    secs = state.quarter_seconds_remaining

    # Kneel to run out the clock
    def_timeouts = (
        state.away_timeouts if state.possession == state.home_team else state.home_timeouts
    )
    if score_diff > 0 and qtr == 4 and secs <= 120 and def_timeouts == 0 and down <= 3:
        return "kneel"

    # Spike to stop the clock
    if score_diff < 0 and secs <= 60 and qtr in (2, 4) and down <= 3:
        if rng.random() < 0.15:
            return "spike"

    # Data-driven play calling
    if models is not None:
        key = _context_key(state)
        return models.play_call.sample_play(
            down=key[0],
            distance_bucket=key[1],
            field_zone=key[2],
            score_bucket=key[3],
            team=state.possession,
            rng=rng,
        )

    # ── Phase 1 fallback ──
    if down == 4:
        fg_dist = state.field_goal_distance
        opp_yl = state.opponent_yard_line
        if fg_dist <= 55 and opp_yl <= 38:
            return "field_goal"
        go_prob = _fourth_down_go_prob(state.yards_to_go, opp_yl, score_diff, qtr, secs)
        if rng.random() < go_prob:
            return "pass" if rng.random() < 0.55 else "run"
        return "punt"

    pass_prob = _pass_probability(down, state.yards_to_go, state.situation, score_diff, qtr, secs)
    return "pass" if rng.random() < pass_prob else "run"


# ── Play execution ──

def resolve_pass(
    state: GameState,
    rng: np.random.Generator,
    models: ModelBundle | None = None,
    matchup: MatchupAdjustment | None = None,
) -> PlayOutcome:
    """Simulate a pass play."""
    key = _context_key(state)
    om = models.play_outcome if models else None
    adj = matchup or MatchupAdjustment()

    # Sack check (adjusted by matchup)
    sack_rate = om.global_sack_rate if om else 0.065
    if om:
        # Apply matchup: >1 sack_rate_mult means more sacks
        base_sacked = om.sample_sack(key, rng)
        if not base_sacked and adj.sack_rate_mult > 1.0:
            # Extra chance of sack for bad OL vs good pass rush
            extra = sack_rate * (adj.sack_rate_mult - 1.0)
            sacked = rng.random() < extra
        elif base_sacked and adj.sack_rate_mult < 1.0:
            # Chance to avoid sack for good OL vs weak pass rush
            avoid = 1.0 - adj.sack_rate_mult
            sacked = rng.random() > avoid
        else:
            sacked = base_sacked
    else:
        sacked = rng.random() < (sack_rate * adj.sack_rate_mult)

    if sacked:
        sack_yards = om.sample_sack_yards(rng) if om else -int(rng.uniform(3, 12))
        fumble_rate = om.sack_fumble_rate if om else 0.12
        if rng.random() < fumble_rate:
            return PlayOutcome(
                play_type="pass", result=PlayResult.FUMBLE_LOST,
                yards_gained=sack_yards, sack=True, turnover=True, fumble_lost=True,
                clock_running=True,
                description=f"Sacked for {sack_yards} yards, FUMBLE! Recovered by defense.",
            )
        return PlayOutcome(
            play_type="pass", result=PlayResult.SACK,
            yards_gained=sack_yards, sack=True, clock_running=True,
            description=f"Sacked for {sack_yards} yards.",
        )

    # Completion check (adjusted by matchup)
    if om:
        base_completed = om.sample_completion(key, rng)
        # Adjust: good passing offense + bad pass defense → more completions
        if not base_completed and adj.comp_rate_mult > 1.0:
            extra = (om.get_comp_rate(key) or om.global_comp_rate) * (adj.comp_rate_mult - 1.0)
            completed = rng.random() < extra
        elif base_completed and adj.comp_rate_mult < 1.0:
            drop = 1.0 - adj.comp_rate_mult
            completed = rng.random() > drop
        else:
            completed = base_completed
    else:
        completed = rng.random() < (0.65 * adj.comp_rate_mult)

    if not completed:
        # Interception check (adjusted by matchup)
        int_rate = om.int_rate if om else 0.025
        int_rate *= adj.int_rate_mult
        if rng.random() < int_rate:
            int_return = max(0, int(rng.normal(15, 10)))
            if rng.random() < 0.15:  # pick-six
                return PlayOutcome(
                    play_type="pass", result=PlayResult.INTERCEPTION,
                    yards_gained=0, turnover=True, interception=True,
                    turnover_return_yards=int_return,
                    touchdown=True, scoring_team=state.defense, points=6,
                    clock_running=False,
                    description="INTERCEPTED! Returned for a TOUCHDOWN!",
                )
            return PlayOutcome(
                play_type="pass", result=PlayResult.INTERCEPTION,
                yards_gained=0, turnover=True, interception=True,
                turnover_return_yards=int_return, clock_running=False,
                description=f"INTERCEPTED! Returned {int_return} yards.",
            )
        return PlayOutcome(
            play_type="pass", result=PlayResult.INCOMPLETE,
            yards_gained=0, is_complete=False, clock_running=False,
            description="Pass incomplete.",
        )

    # Complete — sample yards (adjusted by matchup)
    if om:
        total_yards = om.sample_pass_yards(key, rng)
        air_yards = om.sample_air_yards(key, rng)
    else:
        air_yards = int(rng.exponential(7))
        yac = max(0, int(rng.exponential(5)))
        total_yards = air_yards + yac

    # Apply pass yard matchup multiplier
    if adj.pass_yard_mult != 1.0:
        total_yards = int(round(total_yards * adj.pass_yard_mult))
        air_yards = int(round(air_yards * adj.pass_yard_mult))

    # Clamp to endzone
    opp_yl = state.opponent_yard_line
    if total_yards >= opp_yl:
        total_yards = opp_yl
        return PlayOutcome(
            play_type="pass", result=PlayResult.TOUCHDOWN,
            yards_gained=total_yards, air_yards=air_yards, is_complete=True,
            touchdown=True, scoring_team=state.possession, points=6,
            clock_running=False,
            description=f"Pass complete for {total_yards} yards, TOUCHDOWN!",
        )

    # Clamp negative (screen passes tackled behind LOS)
    total_yards = max(total_yards, -(state.yard_line - 1))

    # Fumble after catch (adjusted by matchup)
    fumble_rate = om.catch_fumble_rate if om else 0.01
    fumble_rate *= adj.fumble_rate_mult
    if rng.random() < fumble_rate:
        return PlayOutcome(
            play_type="pass", result=PlayResult.FUMBLE_LOST,
            yards_gained=total_yards, air_yards=air_yards, is_complete=True,
            turnover=True, fumble_lost=True, clock_running=True,
            description=f"Pass complete for {total_yards} yards, FUMBLE! Recovered by defense.",
        )

    oob_rate = om.pass_oob_rate if om else 0.15
    oob = rng.random() < oob_rate

    return PlayOutcome(
        play_type="pass",
        result=PlayResult.FIRST_DOWN if total_yards >= state.yards_to_go else PlayResult.SHORT_OF_FIRST,
        yards_gained=total_yards, air_yards=air_yards, is_complete=True,
        out_of_bounds=oob, clock_running=not oob,
        description=f"Pass complete for {total_yards} yards.{' (out of bounds)' if oob else ''}",
    )


def resolve_run(
    state: GameState,
    rng: np.random.Generator,
    models: ModelBundle | None = None,
    matchup: MatchupAdjustment | None = None,
) -> PlayOutcome:
    """Simulate a run play."""
    key = _context_key(state)
    om = models.play_outcome if models else None
    adj = matchup or MatchupAdjustment()

    # Sample yards (adjusted by matchup)
    if om:
        yards = om.sample_rush_yards(key, rng)
    else:
        if rng.random() < 0.05:
            yards = int(rng.uniform(15, 50))
        else:
            yards = int(rng.normal(4.2, 3.5))
            yards = max(-5, yards)

    # Apply rush yard matchup multiplier
    if adj.rush_yard_mult != 1.0:
        yards = int(round(yards * adj.rush_yard_mult))

    # Clamp to endzone
    opp_yl = state.opponent_yard_line
    if yards >= opp_yl:
        yards = opp_yl
        return PlayOutcome(
            play_type="run", result=PlayResult.TOUCHDOWN,
            yards_gained=yards, touchdown=True, scoring_team=state.possession, points=6,
            clock_running=False,
            description=f"Rush for {yards} yards, TOUCHDOWN!",
        )

    # Safety
    if state.yard_line + yards <= 0:
        return PlayOutcome(
            play_type="run", result=PlayResult.SAFETY,
            yards_gained=yards, points=2, scoring_team=state.defense,
            clock_running=False,
            description="Tackled in the end zone, SAFETY!",
        )

    # Fumble (adjusted by matchup)
    fumble_rate = om.rush_fumble_rate if om else 0.015
    fumble_rate *= adj.fumble_rate_mult
    if rng.random() < fumble_rate:
        return PlayOutcome(
            play_type="run", result=PlayResult.FUMBLE_LOST,
            yards_gained=yards, turnover=True, fumble_lost=True, clock_running=True,
            description=f"Rush for {yards} yards, FUMBLE! Recovered by defense.",
        )

    oob_rate = om.rush_oob_rate if om else 0.08
    oob = rng.random() < oob_rate

    return PlayOutcome(
        play_type="run",
        result=PlayResult.FIRST_DOWN if yards >= state.yards_to_go else PlayResult.SHORT_OF_FIRST,
        yards_gained=yards, out_of_bounds=oob, clock_running=not oob,
        description=f"Rush for {yards} yards.{' (out of bounds)' if oob else ''}",
    )


def resolve_kneel(state: GameState) -> PlayOutcome:
    return PlayOutcome(
        play_type="kneel", result=PlayResult.KNEEL,
        yards_gained=-1, clock_running=True, description="QB kneel.",
    )


def resolve_spike(state: GameState) -> PlayOutcome:
    return PlayOutcome(
        play_type="spike", result=PlayResult.SPIKE,
        yards_gained=0, clock_running=False, description="QB spikes the ball.",
    )


# ── Phase 1 heuristic fallbacks ──

def _pass_probability(
    down: int, ytg: int, situation: Situation, score_diff: int, qtr: int, secs: int,
) -> float:
    base = 0.58
    if down == 1: base = 0.52
    elif down == 2: base = 0.65 if ytg >= 7 else 0.50
    elif down == 3:
        if ytg >= 5: base = 0.82
        elif ytg >= 3: base = 0.65
        else: base = 0.48
    if situation == Situation.TWO_MINUTE: base = min(0.90, base + 0.20)
    elif situation == Situation.GOAL_LINE: base = max(0.30, base - 0.15)
    elif situation == Situation.GARBAGE_TIME:
        base = max(0.30, base - 0.20) if score_diff > 0 else min(0.85, base + 0.15)
    if qtr == 4 and score_diff < -8: base = min(0.85, base + 0.15)
    elif qtr == 4 and score_diff > 8: base = max(0.35, base - 0.15)
    return base


def _fourth_down_go_prob(ytg: int, opp_yl: int, score_diff: int, qtr: int, secs: int) -> float:
    if ytg <= 1: prob = 0.40
    elif ytg <= 2: prob = 0.25
    elif ytg <= 3: prob = 0.15
    else: prob = 0.05
    if opp_yl <= 45 and opp_yl > 38: prob += 0.15
    if qtr == 4 and score_diff < 0:
        prob += 0.25
        if secs <= 300: prob += 0.20
    if opp_yl > 60: prob *= 0.3
    return min(0.95, prob)
