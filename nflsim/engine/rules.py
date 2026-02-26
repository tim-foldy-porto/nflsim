"""NFL rules engine — applies play results to game state.

This is the heart of the state machine. Every state transition (down changes,
possession changes, scoring, turnovers) flows through here.
"""

from __future__ import annotations

from nflsim.config import (
    KICKOFF_TOUCHBACK_YARD_LINE,
    PUNT_TOUCHBACK_YARD_LINE,
)
from nflsim.engine.game_state import GameState, Phase, PlayOutcome, PlayResult


def apply_play_result(state: GameState, outcome: PlayOutcome) -> None:
    """Apply a play outcome to the game state.

    Handles: yard advancement, down/distance, first downs, turnovers,
    scoring, possession changes, and phase transitions.
    """
    state.play_number += 1
    if state.possession == state.home_team:
        state.home_plays += 1
    else:
        state.away_plays += 1

    state.log_play(outcome.description)

    # ---- Scoring plays ----
    if outcome.touchdown and not outcome.turnover:
        _apply_touchdown(state, state.possession)
        return

    if outcome.touchdown and outcome.turnover:
        # Defensive/return touchdown
        _apply_touchdown(state, outcome.scoring_team or state.defense)
        return

    if outcome.result == PlayResult.SAFETY:
        _apply_safety(state, outcome)
        return

    if outcome.result == PlayResult.FIELD_GOAL_MADE:
        _apply_field_goal(state, outcome)
        return

    if outcome.result == PlayResult.FIELD_GOAL_MISSED:
        _apply_missed_fg(state)
        return

    # ---- PAT results ----
    if state.phase == Phase.PAT:
        _apply_pat_result(state, outcome)
        return

    # ---- Kickoff results ----
    if outcome.play_type == "kickoff":
        _apply_kickoff_result(state, outcome)
        return

    # ---- Punt results ----
    if outcome.play_type == "punt":
        _apply_punt_result(state, outcome)
        return

    # ---- Turnovers ----
    if outcome.turnover:
        _apply_turnover(state, outcome)
        return

    # ---- Normal plays (pass, run, kneel, spike) ----
    _apply_normal_play(state, outcome)


def _apply_touchdown(state: GameState, scoring_team: str) -> None:
    """Score a touchdown and set up PAT."""
    if scoring_team == state.home_team:
        state.home_score += 6
    else:
        state.away_score += 6

    state.scoring_team_last = scoring_team
    state.possession = scoring_team
    state.yard_line = 98  # PAT from 2-yard line (opponent's side)
    state.down = 1
    state.yards_to_go = 2
    state.phase = Phase.PAT


def _apply_pat_result(state: GameState, outcome: PlayOutcome) -> None:
    """Handle extra point or two-point conversion result, then kickoff."""
    if outcome.points > 0:
        if outcome.scoring_team == state.home_team:
            state.home_score += outcome.points
        elif outcome.scoring_team == state.away_team:
            state.away_score += outcome.points

    # After PAT, the scoring team kicks off
    _setup_kickoff(state, kicking_team=state.scoring_team_last)


def _apply_safety(state: GameState, outcome: PlayOutcome) -> None:
    """Safety: 2 points to defense, defense gets a free kick."""
    defense = state.defense
    if defense == state.home_team:
        state.home_score += 2
    else:
        state.away_score += 2

    # After safety, team that was scored ON kicks (free kick from own 20)
    _setup_kickoff(state, kicking_team=state.possession)


def _apply_field_goal(state: GameState, outcome: PlayOutcome) -> None:
    """Field goal made: 3 points, then kickoff."""
    if outcome.scoring_team == state.home_team:
        state.home_score += 3
    else:
        state.away_score += 3

    state.scoring_team_last = outcome.scoring_team or state.possession
    _setup_kickoff(state, kicking_team=state.scoring_team_last)


def _apply_missed_fg(state: GameState) -> None:
    """Missed field goal: defense gets ball at spot of kick or own 20."""
    kicker_yard_line = state.yard_line  # where the kicking team was
    # Defense gets ball at the spot of the kick (LOS), or own 20 if deeper
    new_yard_line = max(20, 100 - kicker_yard_line)
    _change_possession(state, new_yard_line)


def _apply_kickoff_result(state: GameState, outcome: PlayOutcome) -> None:
    """Handle kickoff return/touchback result."""
    # Receiving team gets the ball
    receiving_team = state.defense  # on kickoffs, "defense" = receiving team
    state.possession = receiving_team

    if outcome.result == PlayResult.TOUCHBACK:
        state.yard_line = KICKOFF_TOUCHBACK_YARD_LINE
    elif outcome.result == PlayResult.FUMBLE_LOST:
        # Kicking team recovers — they get ball at return spot
        kicking_team = state.home_team if receiving_team == state.away_team else state.away_team
        state.possession = kicking_team
        state.yard_line = max(1, min(99, 100 - (KICKOFF_TOUCHBACK_YARD_LINE + outcome.yards_gained)))
    else:
        # Normal return: ball starts at ~own 25, returned N yards from there
        start = KICKOFF_TOUCHBACK_YARD_LINE
        state.yard_line = max(1, min(99, start + outcome.yards_gained))

    state.down = 1
    state.yards_to_go = 10
    state.first_down_marker = min(state.yard_line + 10, 100)
    state.phase = Phase.NORMAL


def _apply_punt_result(state: GameState, outcome: PlayOutcome) -> None:
    """Handle punt result — possession changes to receiving team."""
    if outcome.result == PlayResult.FUMBLE_LOST and not outcome.turnover:
        # Muffed punt, punting team recovers — keep possession
        state.yard_line = max(1, min(99, state.yard_line + outcome.yards_gained))
        state.down = 1
        state.yards_to_go = 10
        state.first_down_marker = min(state.yard_line + 10, 100)
        return

    if outcome.result == PlayResult.FUMBLE_LOST and outcome.turnover:
        # Blocked punt recovered by defense
        _change_possession(state, max(1, min(99, state.yard_line - outcome.yards_gained)))
        return

    if outcome.result == PlayResult.TOUCHBACK:
        _change_possession(state, PUNT_TOUCHBACK_YARD_LINE)
        return

    # Fair catch or normal return
    net_punt = outcome.yards_gained
    receiving_yard_line = 100 - (state.yard_line + net_punt)
    receiving_yard_line = max(1, min(99, receiving_yard_line))
    _change_possession(state, receiving_yard_line)


def _apply_turnover(state: GameState, outcome: PlayOutcome) -> None:
    """Handle turnovers (interceptions, fumbles)."""
    if outcome.interception:
        # INT: defense gets ball. Return yards from where INT happened.
        # Approximate: INT at the spot where pass was targeted
        int_spot = state.yard_line + max(0, outcome.air_yards)
        int_spot = min(99, int_spot)
        # Return yards move toward the intercepting team's goal
        return_spot = 100 - int_spot + outcome.turnover_return_yards
        return_spot = max(1, min(99, return_spot))
        _change_possession(state, return_spot)
    elif outcome.fumble_lost:
        # Fumble: defense gets ball at the spot of the fumble
        fumble_spot = state.yard_line + outcome.yards_gained
        fumble_spot = max(1, min(99, fumble_spot))
        new_spot = 100 - fumble_spot  # flip for new possessing team
        new_spot = max(1, min(99, new_spot))
        _change_possession(state, new_spot)


def _apply_normal_play(state: GameState, outcome: PlayOutcome) -> None:
    """Handle normal offensive plays: advance yards, check first down, etc."""
    yards = outcome.yards_gained

    # Advance the ball
    state.yard_line += yards
    state.yard_line = max(1, min(99, state.yard_line))

    # Check first down
    if outcome.penalty and outcome.penalty_auto_first:
        _award_first_down(state)
    elif outcome.penalty and outcome.penalty_on_offense:
        # Offensive penalty: replay down with penalty yards
        state.yards_to_go = min(state.yards_to_go + outcome.penalty_yards, 99)
    elif yards >= state.yards_to_go or outcome.result == PlayResult.FIRST_DOWN:
        _award_first_down(state)
    else:
        # Short of first down
        state.yards_to_go -= yards
        state.yards_to_go = max(1, state.yards_to_go)
        state.down += 1

        if state.down > 4:
            # Turnover on downs
            _change_possession(state, 100 - state.yard_line)


def _award_first_down(state: GameState) -> None:
    """Reset downs for a new first down."""
    state.down = 1
    remaining_to_endzone = 100 - state.yard_line
    state.yards_to_go = min(10, remaining_to_endzone)
    state.first_down_marker = min(state.yard_line + state.yards_to_go, 100)


def _change_possession(state: GameState, new_yard_line: int) -> None:
    """Flip possession to the other team at the given yard line."""
    state.possession = state.defense
    state.yard_line = max(1, min(99, new_yard_line))
    state.down = 1
    remaining_to_endzone = 100 - state.yard_line
    state.yards_to_go = min(10, remaining_to_endzone)
    state.first_down_marker = min(state.yard_line + state.yards_to_go, 100)


def _setup_kickoff(state: GameState, kicking_team: str) -> None:
    """Set up state for a kickoff."""
    state.possession = kicking_team
    state.yard_line = 35  # kickoff from own 35
    state.down = 1
    state.yards_to_go = 10
    state.phase = Phase.KICKOFF
    state.kickoff_reason = "after_score"
