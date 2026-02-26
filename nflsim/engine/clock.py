"""Clock management — time runoff, quarter transitions, two-minute warning."""

from __future__ import annotations

from nflsim.config import (
    CLOCK_RUNOFF_COMPLETE_INBOUNDS,
    CLOCK_RUNOFF_FIELD_GOAL,
    CLOCK_RUNOFF_HURRY_UP_RUNNING,
    CLOCK_RUNOFF_HURRY_UP_STOPPED,
    CLOCK_RUNOFF_INCOMPLETE,
    CLOCK_RUNOFF_KICKOFF,
    CLOCK_RUNOFF_OUT_OF_BOUNDS,
    CLOCK_RUNOFF_PENALTY,
    CLOCK_RUNOFF_PUNT,
    CLOCK_RUNOFF_RUN,
    CLOCK_RUNOFF_SACK,
    CLOCK_RUNOFF_SCORING,
    QUARTER_SECONDS,
    TIMEOUTS_PER_HALF,
)
from nflsim.engine.game_state import GameState, Phase, PlayOutcome


def _is_hurry_up(state: GameState) -> bool:
    """Check if we're in hurry-up mode (last 2 min of a half)."""
    return (
        state.quarter in (2, 4)
        and state.quarter_seconds_remaining <= 120
        and state.score_diff < 0  # only trailing team hurries
    )


def compute_runoff(state: GameState, outcome: PlayOutcome) -> int:
    """Determine how many seconds run off the clock for this play.

    Calibrated from 2024 NFL PBP data (median inter-play time).
    """
    if state.phase == Phase.GAME_OVER:
        return 0

    hurry = _is_hurry_up(state)

    # ---- Special plays with fixed durations ----

    if outcome.play_type == "kickoff":
        return CLOCK_RUNOFF_KICKOFF

    if outcome.play_type == "spike":
        return 1

    if outcome.play_type == "kneel":
        return 40

    # ---- Scoring plays — clock stops ----
    if outcome.touchdown or outcome.points > 0:
        return CLOCK_RUNOFF_SCORING

    # ---- Punts and field goals ----
    if outcome.play_type == "punt":
        return CLOCK_RUNOFF_PUNT

    if outcome.play_type in ("field_goal", "extra_point", "two_point"):
        return CLOCK_RUNOFF_FIELD_GOAL

    # ---- Penalties ----
    if outcome.penalty:
        return CLOCK_RUNOFF_PENALTY

    # ---- Incomplete passes — clock stops ----
    if outcome.play_type == "pass" and not outcome.is_complete and not outcome.sack:
        return CLOCK_RUNOFF_HURRY_UP_STOPPED if hurry else CLOCK_RUNOFF_INCOMPLETE

    # ---- Out of bounds — clock stops briefly, but teams still huddle ----
    if outcome.out_of_bounds:
        return CLOCK_RUNOFF_HURRY_UP_STOPPED if hurry else CLOCK_RUNOFF_OUT_OF_BOUNDS

    # ---- Clock-running plays ----

    if outcome.sack:
        base = CLOCK_RUNOFF_SACK
    elif outcome.play_type == "run":
        base = CLOCK_RUNOFF_RUN
    elif outcome.play_type == "pass" and outcome.is_complete:
        base = CLOCK_RUNOFF_COMPLETE_INBOUNDS
    else:
        base = CLOCK_RUNOFF_RUN  # default

    return CLOCK_RUNOFF_HURRY_UP_RUNNING if hurry else base


def advance_clock(state: GameState, seconds: int) -> None:
    """Subtract time from the clock, handling quarter/half/game transitions."""
    state.quarter_seconds_remaining -= seconds
    state.game_seconds_remaining -= seconds

    # Clamp to zero
    if state.quarter_seconds_remaining < 0:
        state.game_seconds_remaining -= state.quarter_seconds_remaining  # add back overshoot
        state.quarter_seconds_remaining = 0
    if state.game_seconds_remaining < 0:
        state.game_seconds_remaining = 0


def check_two_minute_warning(state: GameState) -> bool:
    """Check if two-minute warning should trigger. Returns True if it does."""
    q = state.quarter
    if q not in (2, 4):
        return False
    if state.two_min_warning_given.get(q, False):
        return False
    if state.quarter_seconds_remaining <= 120:
        state.two_min_warning_given[q] = True
        return True
    return False


def check_quarter_end(state: GameState) -> bool:
    """Check if the quarter has ended. Returns True if transition needed."""
    return state.quarter_seconds_remaining <= 0


def transition_quarter(state: GameState) -> None:
    """Handle quarter/half/game transitions."""
    if state.quarter == 2:
        _halftime(state)
    elif state.quarter == 4:
        if state.home_score == state.away_score:
            _start_overtime(state)
        else:
            state.phase = Phase.GAME_OVER
    elif state.quarter == 5:
        state.phase = Phase.GAME_OVER
    else:
        state.quarter += 1
        state.quarter_seconds_remaining = QUARTER_SECONDS


def _halftime(state: GameState) -> None:
    """Handle halftime: flip possession for second half kickoff, reset timeouts."""
    state.quarter = 3
    state.quarter_seconds_remaining = QUARTER_SECONDS
    state.home_timeouts = TIMEOUTS_PER_HALF
    state.away_timeouts = TIMEOUTS_PER_HALF

    if state.home_receives_2h:
        state.possession = state.away_team
        state.scoring_team_last = state.away_team
    else:
        state.possession = state.home_team
        state.scoring_team_last = state.home_team

    state.phase = Phase.KICKOFF
    state.kickoff_reason = "start_of_half"


def _start_overtime(state: GameState) -> None:
    """Initialize overtime period."""
    state.quarter = 5
    state.quarter_seconds_remaining = 600
    state.game_seconds_remaining = 600
    state.phase = Phase.OVERTIME
