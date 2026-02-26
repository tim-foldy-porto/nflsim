"""Game state representation — the single source of truth during simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class Phase(Enum):
    """Current phase of the game."""
    COIN_TOSS = auto()
    KICKOFF = auto()
    NORMAL = auto()       # regular offensive play
    PAT = auto()          # point-after-touchdown attempt
    OVERTIME = auto()
    GAME_OVER = auto()


class Situation(Enum):
    """Contextual game situation for play-calling."""
    NORMAL = auto()
    TWO_MINUTE = auto()       # under 2:00 in half
    GOAL_LINE = auto()        # inside 3 yard line
    RED_ZONE = auto()         # inside 20
    BACKED_UP = auto()        # inside own 10
    GARBAGE_TIME = auto()     # 3+ score lead, 4th quarter
    LATE_AND_CLOSE = auto()   # within 8 points, 4th quarter


class PlayResult(Enum):
    """Outcome category of a play."""
    FIRST_DOWN = auto()
    SHORT_OF_FIRST = auto()
    TOUCHDOWN = auto()
    TURNOVER_DOWNS = auto()
    INTERCEPTION = auto()
    FUMBLE_LOST = auto()
    INCOMPLETE = auto()
    SACK = auto()
    PENALTY_OFFENSE = auto()
    PENALTY_DEFENSE = auto()
    SAFETY = auto()
    FIELD_GOAL_MADE = auto()
    FIELD_GOAL_MISSED = auto()
    PUNT = auto()
    TOUCHBACK = auto()
    FAIR_CATCH = auto()
    KNEEL = auto()
    SPIKE = auto()
    TWO_MINUTE_WARNING = auto()


@dataclass
class PlayOutcome:
    """The resolved outcome of a single play."""
    play_type: str              # "pass", "run", "punt", "field_goal", "kickoff", "kneel", "spike", "extra_point", "two_point"
    result: PlayResult
    yards_gained: int = 0
    turnover: bool = False
    turnover_return_yards: int = 0
    touchdown: bool = False
    scoring_team: str | None = None
    points: int = 0
    penalty: bool = False
    penalty_on_offense: bool = False
    penalty_yards: int = 0
    penalty_auto_first: bool = False
    clock_running: bool = True  # does clock continue after play
    out_of_bounds: bool = False
    fumble_lost: bool = False
    interception: bool = False
    sack: bool = False
    air_yards: int = 0
    is_complete: bool = False
    description: str = ""


@dataclass
class GameState:
    """Complete state of a game at any point."""
    # Teams
    home_team: str = ""
    away_team: str = ""

    # Score
    home_score: int = 0
    away_score: int = 0

    # Possession
    possession: str = ""     # team abbreviation with the ball
    home_receives_2h: bool = False  # who deferred

    # Field position
    yard_line: int = 25      # yards from possessing team's own goal (1-99)
    down: int = 1
    yards_to_go: int = 10
    first_down_marker: int = 35  # absolute yard line of first down

    # Clock
    quarter: int = 1
    game_seconds_remaining: int = 3600  # full game
    quarter_seconds_remaining: int = 900
    play_clock: int = 40

    # Timeouts
    home_timeouts: int = 3
    away_timeouts: int = 3

    # Phase
    phase: Phase = Phase.COIN_TOSS

    # Kickoff tracking
    kickoff_reason: str = ""  # "start_of_game", "start_of_half", "after_score"
    scoring_team_last: str = ""  # who scored last (for kickoff direction)

    # Stats tracking
    play_number: int = 0
    home_plays: int = 0
    away_plays: int = 0
    play_log: list[str] = field(default_factory=list)

    # Two-minute warning
    two_min_warning_given: dict[int, bool] = field(
        default_factory=lambda: {1: False, 2: False, 3: False, 4: False, 5: False}
    )

    @property
    def defense(self) -> str:
        """Team on defense."""
        return self.away_team if self.possession == self.home_team else self.home_team

    @property
    def score_diff(self) -> int:
        """Score differential from possessing team's perspective."""
        if self.possession == self.home_team:
            return self.home_score - self.away_score
        return self.away_score - self.home_score

    @property
    def half(self) -> int:
        """Current half (1 or 2)."""
        return 1 if self.quarter <= 2 else 2

    @property
    def situation(self) -> Situation:
        """Classify the current game situation."""
        q = self.quarter
        secs = self.quarter_seconds_remaining
        diff = self.score_diff

        # Two-minute situations
        if secs <= 120 and q in (2, 4):
            return Situation.TWO_MINUTE

        # Fourth quarter situations
        if q == 4:
            if abs(diff) > 16:
                return Situation.GARBAGE_TIME
            if abs(diff) <= 8:
                return Situation.LATE_AND_CLOSE

        # Field position situations
        if self.yard_line >= 90:  # inside opponent's 10
            return Situation.GOAL_LINE
        if self.yard_line >= 80:  # inside opponent's 20
            return Situation.RED_ZONE
        if self.yard_line <= 10:  # backed up inside own 10
            return Situation.BACKED_UP

        return Situation.NORMAL

    @property
    def opponent_yard_line(self) -> int:
        """Yards from opponent's goal line (i.e., yards to endzone)."""
        return 100 - self.yard_line

    @property
    def field_goal_distance(self) -> int:
        """Distance of a field goal attempt from current position (add 17 for snap+hold)."""
        return self.opponent_yard_line + 17

    @property
    def possession_score(self) -> int:
        return self.home_score if self.possession == self.home_team else self.away_score

    @property
    def defense_score(self) -> int:
        return self.away_score if self.possession == self.home_team else self.home_score

    def validate(self) -> list[str]:
        """Check invariants. Returns list of violations (empty = valid)."""
        errors = []
        if self.phase == Phase.GAME_OVER:
            return errors

        if self.yard_line < 1 or self.yard_line > 99:
            errors.append(f"yard_line out of bounds: {self.yard_line}")
        if self.down < 1 or self.down > 4:
            errors.append(f"invalid down: {self.down}")
        if self.yards_to_go < 1:
            errors.append(f"invalid yards_to_go: {self.yards_to_go}")
        if self.quarter < 1 or self.quarter > 5:
            errors.append(f"invalid quarter: {self.quarter}")
        if self.quarter_seconds_remaining < 0:
            errors.append(f"negative quarter_seconds: {self.quarter_seconds_remaining}")
        if self.game_seconds_remaining < 0:
            errors.append(f"negative game_seconds: {self.game_seconds_remaining}")
        if self.home_timeouts < 0 or self.home_timeouts > 3:
            errors.append(f"invalid home_timeouts: {self.home_timeouts}")
        if self.away_timeouts < 0 or self.away_timeouts > 3:
            errors.append(f"invalid away_timeouts: {self.away_timeouts}")
        if self.possession not in (self.home_team, self.away_team) and self.phase != Phase.COIN_TOSS:
            errors.append(f"invalid possession: {self.possession}")

        return errors

    def log_play(self, desc: str) -> None:
        """Append a play description to the log."""
        self.play_log.append(
            f"Q{self.quarter} {self._clock_str()} | {self.possession} "
            f"{'%d' % self.down}&{'%d' % self.yards_to_go} at {self._field_pos_str()} | {desc}"
        )

    def _clock_str(self) -> str:
        mins = self.quarter_seconds_remaining // 60
        secs = self.quarter_seconds_remaining % 60
        return f"{mins}:{secs:02d}"

    def _field_pos_str(self) -> str:
        if self.yard_line == 50:
            return "50"
        if self.yard_line < 50:
            return f"{self.possession} {self.yard_line}"
        opp = self.defense
        return f"{opp} {100 - self.yard_line}"
