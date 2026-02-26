"""Player-level stat tracking and ESPN-style box score rendering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class PassingStats:
    name: str
    completions: int = 0
    attempts: int = 0
    yards: int = 0
    touchdowns: int = 0
    interceptions: int = 0
    sacks: int = 0
    sack_yards: int = 0

    @property
    def passer_rating(self) -> float:
        """NFL passer rating formula."""
        if self.attempts == 0:
            return 0.0
        a = max(0, min(((self.completions / self.attempts) - 0.3) * 5, 2.375))
        b = max(0, min(((self.yards / self.attempts) - 3) * 0.25, 2.375))
        c = max(0, min((self.touchdowns / self.attempts) * 20, 2.375))
        d = max(0, min(2.375 - ((self.interceptions / self.attempts) * 25), 2.375))
        return ((a + b + c + d) / 6) * 100

    def line(self) -> str:
        return (
            f"  {self.name:<22s} {self.completions}/{self.attempts}  "
            f"{self.yards:>4d} yds  {self.touchdowns} TD  {self.interceptions} INT  "
            f"{self.passer_rating:.1f} RTG"
        )


@dataclass
class ReceivingStats:
    name: str
    targets: int = 0
    receptions: int = 0
    yards: int = 0
    touchdowns: int = 0

    @property
    def avg(self) -> float:
        return self.yards / self.receptions if self.receptions > 0 else 0.0

    def line(self) -> str:
        return (
            f"  {self.name:<22s} {self.receptions:>2d} rec  "
            f"{self.yards:>4d} yds  {self.avg:>4.1f} avg  {self.touchdowns} TD"
        )


@dataclass
class RushingStats:
    name: str
    carries: int = 0
    yards: int = 0
    touchdowns: int = 0

    @property
    def avg(self) -> float:
        return self.yards / self.carries if self.carries > 0 else 0.0

    def line(self) -> str:
        return (
            f"  {self.name:<22s} {self.carries:>2d} car  "
            f"{self.yards:>4d} yds  {self.avg:>4.1f} avg  {self.touchdowns} TD"
        )


@dataclass
class PlayerStatsTracker:
    """Tracks per-player stats during a game, keyed by (team, player_id)."""
    passing: dict[str, dict[str, PassingStats]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    receiving: dict[str, dict[str, ReceivingStats]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    rushing: dict[str, dict[str, RushingStats]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def record_pass(
        self,
        team: str,
        passer_id: str | None,
        passer_name: str | None,
        receiver_id: str | None,
        receiver_name: str | None,
        complete: bool,
        yards: int,
        touchdown: bool,
        interception: bool,
        sack: bool,
        sack_yards: int = 0,
    ) -> None:
        if not passer_id or not passer_name:
            return

        # Passer stats
        if passer_id not in self.passing[team]:
            self.passing[team][passer_id] = PassingStats(name=passer_name)
        ps = self.passing[team][passer_id]

        if sack:
            ps.sacks += 1
            ps.sack_yards += sack_yards
        else:
            ps.attempts += 1
            if complete:
                ps.completions += 1
                ps.yards += yards
                if touchdown:
                    ps.touchdowns += 1
            if interception:
                ps.interceptions += 1

        # Receiver stats (only on non-sacks)
        if not sack and receiver_id and receiver_name:
            if receiver_id not in self.receiving[team]:
                self.receiving[team][receiver_id] = ReceivingStats(name=receiver_name)
            rs = self.receiving[team][receiver_id]
            rs.targets += 1
            if complete:
                rs.receptions += 1
                rs.yards += yards
                if touchdown:
                    rs.touchdowns += 1

    def record_rush(
        self,
        team: str,
        rusher_id: str | None,
        rusher_name: str | None,
        yards: int,
        touchdown: bool,
    ) -> None:
        if not rusher_id or not rusher_name:
            return
        if rusher_id not in self.rushing[team]:
            self.rushing[team][rusher_id] = RushingStats(name=rusher_name)
        rs = self.rushing[team][rusher_id]
        rs.carries += 1
        rs.yards += yards
        if touchdown:
            rs.touchdowns += 1

    def render(self, away_team: str, home_team: str) -> str:
        """Render ESPN-style player box score."""
        lines: list[str] = []

        for team in (away_team, home_team):
            lines.append("")
            lines.append(f"{'─' * 60}")
            lines.append(f"  {team}")
            lines.append(f"{'─' * 60}")

            # Passing
            if team in self.passing and self.passing[team]:
                lines.append("  PASSING")
                for ps in sorted(
                    self.passing[team].values(),
                    key=lambda x: x.attempts, reverse=True,
                ):
                    if ps.attempts > 0 or ps.sacks > 0:
                        lines.append(ps.line())

            # Rushing
            if team in self.rushing and self.rushing[team]:
                lines.append("  RUSHING")
                for rs in sorted(
                    self.rushing[team].values(),
                    key=lambda x: x.carries, reverse=True,
                ):
                    if rs.carries > 0:
                        lines.append(rs.line())

            # Receiving
            if team in self.receiving and self.receiving[team]:
                lines.append("  RECEIVING")
                for rs in sorted(
                    self.receiving[team].values(),
                    key=lambda x: x.targets, reverse=True,
                ):
                    if rs.receptions > 0:
                        lines.append(rs.line())

        return "\n".join(lines)
