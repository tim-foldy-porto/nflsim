"""Box score aggregation from play-by-play outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BoxScore:
    """Aggregated game statistics, updated play-by-play."""
    home_team: str
    away_team: str
    home_score: int = 0
    away_score: int = 0

    # Passing
    home_pass_attempts: int = 0
    home_completions: int = 0
    home_pass_yards: int = 0
    home_pass_tds: int = 0
    home_interceptions: int = 0
    home_sacks: int = 0
    home_sack_yards: int = 0

    away_pass_attempts: int = 0
    away_completions: int = 0
    away_pass_yards: int = 0
    away_pass_tds: int = 0
    away_interceptions: int = 0
    away_sacks: int = 0
    away_sack_yards: int = 0

    # Rushing
    home_rush_attempts: int = 0
    home_rush_yards: int = 0
    home_rush_tds: int = 0

    away_rush_attempts: int = 0
    away_rush_yards: int = 0
    away_rush_tds: int = 0

    # Turnovers
    home_fumbles_lost: int = 0
    away_fumbles_lost: int = 0

    # Totals
    home_first_downs: int = 0
    away_first_downs: int = 0
    home_third_down_att: int = 0
    home_third_down_conv: int = 0
    away_third_down_att: int = 0
    away_third_down_conv: int = 0
    home_penalties: int = 0
    away_penalties: int = 0
    home_penalty_yards: int = 0
    away_penalty_yards: int = 0

    # Time of possession (seconds)
    home_possession_secs: int = 0
    away_possession_secs: int = 0

    # Punts
    home_punts: int = 0
    away_punts: int = 0

    def record_play(
        self,
        possession: str,
        play_type: str,
        yards_gained: int,
        is_complete: bool,
        sack: bool,
        touchdown: bool,
        interception: bool,
        fumble_lost: bool,
        first_down: bool,
        down: int,
        clock_elapsed: int,
        penalty: bool = False,
        penalty_on_offense: bool = False,
        penalty_yards: int = 0,
    ) -> None:
        """Record a single play's stats into the box score."""
        side = "home" if possession == self.home_team else "away"

        # Time of possession
        _add(self, side, "possession_secs", clock_elapsed)

        # Skip non-scrimmage plays for stat tracking
        if play_type in ("kickoff", "extra_point", "two_point"):
            return

        if play_type == "punt":
            _add(self, side, "punts", 1)
            return

        if play_type == "field_goal":
            return

        # Penalties
        if penalty:
            pen_side = side if penalty_on_offense else _other(side)
            _add(self, pen_side, "penalties", 1)
            _add(self, pen_side, "penalty_yards", penalty_yards)

        # Third down tracking
        if down == 3:
            _add(self, side, "third_down_att", 1)
            if first_down or touchdown:
                _add(self, side, "third_down_conv", 1)

        # First downs
        if first_down and not touchdown:
            _add(self, side, "first_downs", 1)

        # Pass plays
        if play_type == "pass":
            if sack:
                _add(self, side, "sacks", 1)
                _add(self, side, "sack_yards", yards_gained)  # negative
            else:
                _add(self, side, "pass_attempts", 1)
                if is_complete:
                    _add(self, side, "completions", 1)
                    _add(self, side, "pass_yards", yards_gained)
                    if touchdown:
                        _add(self, side, "pass_tds", 1)
                if interception:
                    _add(self, side, "interceptions", 1)
            if fumble_lost and not sack:
                _add(self, side, "fumbles_lost", 1)
            if sack and fumble_lost:
                _add(self, side, "fumbles_lost", 1)

        # Run plays
        elif play_type == "run":
            _add(self, side, "rush_attempts", 1)
            _add(self, side, "rush_yards", yards_gained)
            if touchdown:
                _add(self, side, "rush_tds", 1)
            if fumble_lost:
                _add(self, side, "fumbles_lost", 1)

    @property
    def home_total_yards(self) -> int:
        return self.home_pass_yards + self.home_rush_yards + self.home_sack_yards

    @property
    def away_total_yards(self) -> int:
        return self.away_pass_yards + self.away_rush_yards + self.away_sack_yards

    @property
    def home_turnovers(self) -> int:
        return self.home_interceptions + self.home_fumbles_lost

    @property
    def away_turnovers(self) -> int:
        return self.away_interceptions + self.away_fumbles_lost

    def summary(self) -> str:
        """Format a box score summary."""
        lines = [
            "",
            f"{'':22s} {self.away_team:>8s} {self.home_team:>8s}",
            f"{'─' * 40}",
            f"{'Final':22s} {self.away_score:8d} {self.home_score:8d}",
            f"{'─' * 40}",
            f"{'First Downs':22s} {self.away_first_downs:8d} {self.home_first_downs:8d}",
            f"{'3rd Down Eff':22s} {self._third_down('away'):>8s} {self._third_down('home'):>8s}",
            f"{'Total Yards':22s} {self.away_total_yards:8d} {self.home_total_yards:8d}",
            f"{'Passing':22s} {self._pass_line('away'):>8s} {self._pass_line('home'):>8s}",
            f"{'Pass Yards':22s} {self.away_pass_yards:8d} {self.home_pass_yards:8d}",
            f"{'Rush Att-Yds':22s} {self._rush_line('away'):>8s} {self._rush_line('home'):>8s}",
            f"{'Sacks-Yards':22s} {self._sack_line('away'):>8s} {self._sack_line('home'):>8s}",
            f"{'Turnovers':22s} {self.away_turnovers:8d} {self.home_turnovers:8d}",
            f"{'  INT':22s} {self.away_interceptions:8d} {self.home_interceptions:8d}",
            f"{'  Fumbles Lost':22s} {self.away_fumbles_lost:8d} {self.home_fumbles_lost:8d}",
            f"{'Penalties':22s} {self._pen_line('away'):>8s} {self._pen_line('home'):>8s}",
            f"{'Punts':22s} {self.away_punts:8d} {self.home_punts:8d}",
            f"{'Time of Possession':22s} {self._top('away'):>8s} {self._top('home'):>8s}",
        ]
        return "\n".join(lines)

    def _pass_line(self, side: str) -> str:
        c = getattr(self, f"{side}_completions")
        a = getattr(self, f"{side}_pass_attempts")
        return f"{c}/{a}"

    def _rush_line(self, side: str) -> str:
        a = getattr(self, f"{side}_rush_attempts")
        y = getattr(self, f"{side}_rush_yards")
        return f"{a}-{y}"

    def _sack_line(self, side: str) -> str:
        n = getattr(self, f"{side}_sacks")
        y = abs(getattr(self, f"{side}_sack_yards"))
        return f"{n}-{y}"

    def _pen_line(self, side: str) -> str:
        n = getattr(self, f"{side}_penalties")
        y = getattr(self, f"{side}_penalty_yards")
        return f"{n}-{y}"

    def _third_down(self, side: str) -> str:
        conv = getattr(self, f"{side}_third_down_conv")
        att = getattr(self, f"{side}_third_down_att")
        return f"{conv}/{att}"

    def _top(self, side: str) -> str:
        secs = getattr(self, f"{side}_possession_secs")
        mins = secs // 60
        s = secs % 60
        return f"{mins}:{s:02d}"


def _add(obj: BoxScore, side: str, attr: str, value: int) -> None:
    """Helper to increment a side-prefixed attribute."""
    full = f"{side}_{attr}"
    setattr(obj, full, getattr(obj, full) + value)


def _other(side: str) -> str:
    return "away" if side == "home" else "home"
