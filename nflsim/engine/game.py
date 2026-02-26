"""Game orchestrator — runs a complete game from kickoff to final whistle."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from nflsim.engine.clock import (
    advance_clock,
    check_quarter_end,
    check_two_minute_warning,
    compute_runoff,
    transition_quarter,
)
from typing import TYPE_CHECKING

from nflsim.engine.game_state import GameState, Phase, PlayOutcome, PlayResult
from nflsim.models.personnel import PlayPersonnel, assign_personnel
from nflsim.models.team_ratings import MatchupAdjustment
from nflsim.output.box_score import BoxScore
from nflsim.output.player_stats import PlayerStatsTracker
from nflsim.engine.play_resolver import (
    choose_play,
    resolve_kneel,
    resolve_pass,
    resolve_run,
    resolve_spike,
)
from nflsim.engine.rules import apply_play_result
from nflsim.engine.special_teams import (
    resolve_extra_point,
    resolve_field_goal,
    resolve_kickoff,
    resolve_punt,
    resolve_two_point,
)

if TYPE_CHECKING:
    from nflsim.models.store import ModelBundle

logger = logging.getLogger(__name__)

MAX_PLAYS = 300  # safety limit to prevent infinite loops


@dataclass
class GameResult:
    """Summary of a completed game."""
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    total_plays: int
    home_plays: int
    away_plays: int
    play_log: list[str]
    validation_errors: list[str]
    box_score: BoxScore | None = None
    player_stats: PlayerStatsTracker | None = None
    play_events: list[dict] | None = None  # structured play data for web UI

    @property
    def winner(self) -> str | None:
        if self.home_score > self.away_score:
            return self.home_team
        elif self.away_score > self.home_score:
            return self.away_team
        return None  # tie

    def __str__(self) -> str:
        return (
            f"{self.away_team} {self.away_score} @ {self.home_team} {self.home_score} "
            f"({self.total_plays} plays)"
        )


def simulate_game(
    home_team: str,
    away_team: str,
    *,
    seed: int | None = None,
    verbose: bool = False,
    models: ModelBundle | None = None,
    capture_events: bool = False,
) -> GameResult:
    """Simulate a complete NFL game.

    Args:
        home_team: Home team abbreviation (e.g., "KC")
        away_team: Away team abbreviation (e.g., "SF")
        seed: Random seed for reproducibility
        verbose: Print play-by-play during simulation

    Returns:
        GameResult with final score and play log
    """
    rng = np.random.default_rng(seed)
    state = GameState(home_team=home_team, away_team=away_team)
    box = BoxScore(home_team=home_team, away_team=away_team)
    pstats = PlayerStatsTracker()
    all_errors: list[str] = []
    play_events: list[dict] = [] if capture_events else None

    # Team profiles for player assignment
    team_profiles = models.team_profiles if models else None

    # Pre-compute matchup adjustments from team ratings
    home_on_off = None  # home team offense vs away defense
    away_on_off = None  # away team offense vs home defense
    if models and models.team_ratings:
        ratings = models.team_ratings
        home_r = ratings.get(home_team)
        away_r = ratings.get(away_team)
        if home_r and away_r:
            home_on_off = MatchupAdjustment.from_matchup(home_r, away_r)
            away_on_off = MatchupAdjustment.from_matchup(away_r, home_r)

    # Coin toss
    _coin_toss(state, rng)

    # Main game loop
    while state.phase != Phase.GAME_OVER and state.play_number < MAX_PLAYS:
        # Validate state
        errors = state.validate()
        if errors:
            all_errors.extend(errors)
            logger.error("Validation errors: %s", errors)
            break

        # Snapshot pre-play state for box score and events
        pre_possession = state.possession
        pre_down = state.down
        pre_ytg = state.yards_to_go
        pre_yard_line = state.yard_line
        pre_quarter = state.quarter
        pre_clock = state.quarter_seconds_remaining
        pre_home_score = state.home_score
        pre_away_score = state.away_score
        pre_phase = state.phase

        # Resolve one play (with matchup adjustment for current possession)
        if state.possession == home_team:
            matchup = home_on_off
        else:
            matchup = away_on_off
        outcome = _resolve_one_play(state, rng, models, matchup)

        # Apply result to state
        apply_play_result(state, outcome)

        # Clock management (skip for PAT)
        runoff = 0
        if outcome.play_type not in ("extra_point", "two_point"):
            runoff = compute_runoff(state, outcome)
            advance_clock(state, runoff)

            # Two-minute warning
            if check_two_minute_warning(state):
                state.log_play("** Two-Minute Warning **")

            # Quarter end
            if check_quarter_end(state) and state.phase not in (Phase.PAT, Phase.GAME_OVER):
                transition_quarter(state)

        # Record play in box score
        is_first_down = (
            outcome.result == PlayResult.FIRST_DOWN
            or (outcome.yards_gained >= pre_ytg and outcome.play_type in ("pass", "run")
                and not outcome.turnover)
        )
        box.record_play(
            possession=pre_possession,
            play_type=outcome.play_type,
            yards_gained=outcome.yards_gained,
            is_complete=outcome.is_complete,
            sack=outcome.sack,
            touchdown=outcome.touchdown,
            interception=outcome.interception,
            fumble_lost=outcome.fumble_lost,
            first_down=is_first_down,
            down=pre_down,
            clock_elapsed=runoff,
            penalty=outcome.penalty,
            penalty_on_offense=outcome.penalty_on_offense,
            penalty_yards=outcome.penalty_yards,
        )

        # Assign players and record player stats
        if outcome.play_type in ("pass", "run", "kneel", "spike"):
            off_profile = (
                team_profiles.get(pre_possession) if team_profiles else None
            )
            personnel = assign_personnel(
                off_profile, outcome.play_type,
                outcome.is_complete, outcome.sack, rng,
            )
            if outcome.play_type == "pass":
                pstats.record_pass(
                    team=pre_possession,
                    passer_id=personnel.passer_id,
                    passer_name=personnel.passer_name,
                    receiver_id=personnel.receiver_id,
                    receiver_name=personnel.receiver_name,
                    complete=outcome.is_complete,
                    yards=outcome.yards_gained,
                    touchdown=outcome.touchdown and not outcome.turnover,
                    interception=outcome.interception,
                    sack=outcome.sack,
                    sack_yards=outcome.yards_gained if outcome.sack else 0,
                )
            elif outcome.play_type == "run":
                pstats.record_rush(
                    team=pre_possession,
                    rusher_id=personnel.rusher_id,
                    rusher_name=personnel.rusher_name,
                    yards=outcome.yards_gained,
                    touchdown=outcome.touchdown and not outcome.turnover,
                )

        # Capture structured play event for web UI
        if play_events is not None:
            play_events.append({
                "play_num": state.play_number,
                "quarter": pre_quarter,
                "clock_seconds": pre_clock,
                "phase": pre_phase.name.lower() if hasattr(pre_phase, 'name') else str(pre_phase),
                "possession": pre_possession,
                "down": pre_down,
                "yards_to_go": pre_ytg,
                "yard_line": pre_yard_line,
                "play_type": outcome.play_type,
                "result": outcome.result.name.lower() if hasattr(outcome.result, 'name') else str(outcome.result),
                "yards_gained": outcome.yards_gained,
                "air_yards": outcome.air_yards,
                "description": outcome.description or "",
                "is_complete": outcome.is_complete,
                "sack": outcome.sack,
                "turnover": outcome.turnover,
                "interception": outcome.interception,
                "fumble_lost": outcome.fumble_lost,
                "touchdown": outcome.touchdown,
                "scoring_team": outcome.scoring_team,
                "points": outcome.points,
                "penalty": outcome.penalty,
                "out_of_bounds": outcome.out_of_bounds,
                "home_score": state.home_score,
                "away_score": state.away_score,
                "new_possession": state.possession,
                "new_yard_line": state.yard_line,
                "new_down": state.down,
                "new_yards_to_go": state.yards_to_go,
                "new_quarter": state.quarter,
                "new_clock_seconds": state.quarter_seconds_remaining,
            })

        # Overtime scoring rules (regular season)
        if state.quarter == 5 and state.phase != Phase.PAT:
            if state.home_score != state.away_score:
                state.phase = Phase.GAME_OVER

        if verbose and outcome.description:
            print(state.play_log[-1] if state.play_log else outcome.description)

    # Game over — finalize box score
    if state.play_number >= MAX_PLAYS:
        logger.warning("Game hit max plays limit (%d)", MAX_PLAYS)

    box.home_score = state.home_score
    box.away_score = state.away_score

    return GameResult(
        home_team=home_team,
        away_team=away_team,
        home_score=state.home_score,
        away_score=state.away_score,
        total_plays=state.play_number,
        home_plays=state.home_plays,
        away_plays=state.away_plays,
        play_log=state.play_log,
        validation_errors=all_errors,
        box_score=box,
        player_stats=pstats,
        play_events=play_events,
    )


def _coin_toss(state: GameState, rng: np.random.Generator) -> None:
    """Simulate coin toss — winner usually defers to second half."""
    winner = state.home_team if rng.random() < 0.5 else state.away_team

    # ~60% of coin toss winners defer
    if rng.random() < 0.60:
        # Winner defers — they receive in 2nd half
        if winner == state.home_team:
            state.home_receives_2h = True
            state.possession = state.home_team  # home kicks first
            state.scoring_team_last = state.home_team
        else:
            state.home_receives_2h = False
            state.possession = state.away_team  # away kicks first
            state.scoring_team_last = state.away_team
    else:
        # Winner receives first
        if winner == state.home_team:
            state.home_receives_2h = False
            state.possession = state.away_team  # away kicks to home
            state.scoring_team_last = state.away_team
        else:
            state.home_receives_2h = True
            state.possession = state.home_team  # home kicks to away
            state.scoring_team_last = state.home_team

    state.phase = Phase.KICKOFF
    state.kickoff_reason = "start_of_game"
    state.log_play(f"Coin toss: {winner} wins. {'Defers.' if state.home_receives_2h == (winner == state.home_team) else 'Receives.'}")


def _resolve_one_play(
    state: GameState,
    rng: np.random.Generator,
    models: ModelBundle | None = None,
    matchup: MatchupAdjustment | None = None,
) -> PlayOutcome:
    """Resolve a single play based on current game phase and state."""
    if state.phase == Phase.KICKOFF:
        return resolve_kickoff(state, rng)

    if state.phase == Phase.PAT:
        return _resolve_pat(state, rng, models)

    if state.phase in (Phase.NORMAL, Phase.OVERTIME):
        play_call = choose_play(state, rng, models)
        return _execute_play(state, play_call, rng, models, matchup)

    # Shouldn't reach here
    logger.error("Unexpected phase: %s", state.phase)
    return PlayOutcome(play_type="kneel", result=PlayResult.KNEEL, description="ERROR: unexpected phase")


def _resolve_pat(
    state: GameState, rng: np.random.Generator, models: ModelBundle | None = None,
) -> PlayOutcome:
    """Decide and resolve PAT attempt.

    Phase 1: ~95% kick XP, ~5% go for 2.
    Actual decision depends on score and time, but we keep it simple for now.
    """
    score_diff = state.score_diff - 6  # before the TD points but after PAT setup

    # Go for 2 in specific scenarios
    go_for_2 = False
    if state.quarter >= 4:
        # Down by 2 after TD = go for 2 to tie
        if score_diff in (-2, -8, -9):
            go_for_2 = True
    if rng.random() < 0.05:
        go_for_2 = True

    fg_model = models.field_goal if models else None
    if go_for_2:
        return resolve_two_point(state, rng, fg_model)
    return resolve_extra_point(state, rng, fg_model)


def _execute_play(
    state: GameState,
    play_call: str,
    rng: np.random.Generator,
    models: ModelBundle | None = None,
    matchup: MatchupAdjustment | None = None,
) -> PlayOutcome:
    """Execute a called play."""
    fg_model = models.field_goal if models else None
    if play_call == "pass":
        return resolve_pass(state, rng, models, matchup)
    elif play_call == "run":
        return resolve_run(state, rng, models, matchup)
    elif play_call == "punt":
        return resolve_punt(state, rng)
    elif play_call == "field_goal":
        return resolve_field_goal(state, rng, fg_model)
    elif play_call == "kneel":
        return resolve_kneel(state)
    elif play_call == "spike":
        return resolve_spike(state)
    else:
        logger.error("Unknown play call: %s", play_call)
        return resolve_pass(state, rng, models)
