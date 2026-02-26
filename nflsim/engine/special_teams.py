"""Special teams: kickoffs, punts, field goals, extra points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nflsim.engine.game_state import GameState, PlayOutcome, PlayResult

if TYPE_CHECKING:
    from nflsim.models.field_goal import FieldGoalModel


def resolve_kickoff(state: GameState, rng: np.random.Generator) -> PlayOutcome:
    """Simulate a kickoff.

    Outcomes: touchback (~55%), return (variable yards), onside kick (rare).
    """
    # Phase 1: simple distributions
    touchback_prob = 0.55
    roll = rng.random()

    if roll < touchback_prob:
        return PlayOutcome(
            play_type="kickoff",
            result=PlayResult.TOUCHBACK,
            yards_gained=0,
            clock_running=False,
            description="Kickoff, touchback.",
        )

    # Return
    return_yards = max(0, int(rng.normal(24, 8)))
    return_yards = min(return_yards, 100)  # can't return past own endzone backwards

    # Rare: return TD (~1%)
    if rng.random() < 0.01:
        return PlayOutcome(
            play_type="kickoff",
            result=PlayResult.TOUCHDOWN,
            yards_gained=return_yards,
            touchdown=True,
            scoring_team=state.defense,  # receiving team
            points=6,
            clock_running=False,
            description=f"Kickoff returned for a TOUCHDOWN! ({return_yards} yards)",
        )

    # Fumble on return (~1%)
    if rng.random() < 0.01:
        return PlayOutcome(
            play_type="kickoff",
            result=PlayResult.FUMBLE_LOST,
            yards_gained=return_yards,
            turnover=True,
            fumble_lost=True,
            clock_running=False,
            description=f"Kickoff return, FUMBLE! Recovered by kicking team at the {return_yards} yard line.",
        )

    return PlayOutcome(
        play_type="kickoff",
        result=PlayResult.SHORT_OF_FIRST,
        yards_gained=return_yards,
        clock_running=False,
        description=f"Kickoff returned {return_yards} yards.",
    )


def resolve_punt(state: GameState, rng: np.random.Generator) -> PlayOutcome:
    """Simulate a punt.

    Punt distance depends on field position. Can be touchback, fair catch,
    return, blocked, or muffed.
    """
    max_punt = state.opponent_yard_line  # can't punt past the endzone meaningfully

    # Average punt ~45 yards, but limited by field position
    punt_distance = max(20, min(int(rng.normal(45, 8)), max_punt + 10))

    # Blocked punt (~1.5%)
    if rng.random() < 0.015:
        block_recovery_yards = int(rng.uniform(0, 15))
        return PlayOutcome(
            play_type="punt",
            result=PlayResult.FUMBLE_LOST,
            yards_gained=-block_recovery_yards,
            turnover=True,
            fumble_lost=True,
            clock_running=False,
            description=f"Punt BLOCKED! Recovered by defense.",
        )

    net_yards = punt_distance
    land_yard = state.yard_line + punt_distance  # where ball lands (from own goal)

    # Touchback if punt reaches endzone
    if land_yard >= 100:
        return PlayOutcome(
            play_type="punt",
            result=PlayResult.TOUCHBACK,
            yards_gained=punt_distance,
            clock_running=False,
            description=f"Punt {punt_distance} yards, touchback.",
        )

    # Fair catch (~40%)
    if rng.random() < 0.40:
        return PlayOutcome(
            play_type="punt",
            result=PlayResult.FAIR_CATCH,
            yards_gained=punt_distance,
            clock_running=False,
            description=f"Punt {punt_distance} yards, fair catch.",
        )

    # Return
    return_yards = max(0, int(rng.normal(9, 5)))

    # Muffed punt (~2%)
    if rng.random() < 0.02:
        return PlayOutcome(
            play_type="punt",
            result=PlayResult.FUMBLE_LOST,
            yards_gained=punt_distance,
            turnover=False,  # punting team recovers muff = keeps ball
            fumble_lost=True,
            clock_running=False,
            description=f"Punt {punt_distance} yards, MUFFED! Punting team recovers.",
        )

    # Rare punt return TD (~1%)
    if rng.random() < 0.01:
        return PlayOutcome(
            play_type="punt",
            result=PlayResult.TOUCHDOWN,
            yards_gained=punt_distance,
            touchdown=True,
            scoring_team=state.defense,  # return team
            points=6,
            clock_running=False,
            description=f"Punt returned for a TOUCHDOWN!",
        )

    return PlayOutcome(
        play_type="punt",
        result=PlayResult.PUNT,
        yards_gained=punt_distance - return_yards,  # net punt yards
        clock_running=False,
        description=f"Punt {punt_distance} yards, returned {return_yards} yards.",
    )


def resolve_field_goal(
    state: GameState,
    rng: np.random.Generator,
    fg_model: FieldGoalModel | None = None,
) -> PlayOutcome:
    """Simulate a field goal attempt."""
    distance = state.field_goal_distance

    if fg_model:
        prob = fg_model.fg_probability(distance)
    else:
        if distance <= 30: prob = 0.93
        elif distance <= 40: prob = 0.87
        elif distance <= 50: prob = 0.75
        elif distance <= 55: prob = 0.60
        elif distance <= 60: prob = 0.40
        else: prob = 0.15

    if rng.random() < prob:
        return PlayOutcome(
            play_type="field_goal",
            result=PlayResult.FIELD_GOAL_MADE,
            points=3,
            scoring_team=state.possession,
            clock_running=False,
            description=f"{distance}-yard field goal is GOOD!",
        )
    else:
        # Miss — ball goes to defense at spot of kick (or 20, whichever is farther)
        return PlayOutcome(
            play_type="field_goal",
            result=PlayResult.FIELD_GOAL_MISSED,
            clock_running=False,
            description=f"{distance}-yard field goal is NO GOOD.",
        )


def resolve_extra_point(
    state: GameState,
    rng: np.random.Generator,
    fg_model: FieldGoalModel | None = None,
) -> PlayOutcome:
    """Simulate an extra point attempt (33 yards, ~94% success)."""
    rate = fg_model.extra_point_rate if fg_model else 0.94
    if rng.random() < rate:
        return PlayOutcome(
            play_type="extra_point",
            result=PlayResult.FIELD_GOAL_MADE,
            points=1,
            scoring_team=state.possession,
            clock_running=False,
            description="Extra point is GOOD.",
        )
    return PlayOutcome(
        play_type="extra_point",
        result=PlayResult.FIELD_GOAL_MISSED,
        clock_running=False,
        description="Extra point is NO GOOD.",
    )


def resolve_two_point(
    state: GameState,
    rng: np.random.Generator,
    fg_model: FieldGoalModel | None = None,
) -> PlayOutcome:
    """Simulate a two-point conversion attempt (~48% success)."""
    rate = fg_model.two_point_rate if fg_model else 0.48
    if rng.random() < rate:
        return PlayOutcome(
            play_type="two_point",
            result=PlayResult.TOUCHDOWN,
            points=2,
            scoring_team=state.possession,
            clock_running=False,
            description="Two-point conversion is GOOD!",
        )
    return PlayOutcome(
        play_type="two_point",
        result=PlayResult.SHORT_OF_FIRST,
        clock_running=False,
        description="Two-point conversion FAILED.",
    )
