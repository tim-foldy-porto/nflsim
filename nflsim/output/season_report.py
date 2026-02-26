"""Season report rendering — standings, playoff bracket, and multi-sim aggregates."""

from __future__ import annotations

from collections import Counter, defaultdict

from nflsim.season.schedule import DIVISIONS, TEAM_TO_CONFERENCE
from nflsim.season.season import SeasonResult
from nflsim.season.standings import Standings


def render_standings(standings: Standings) -> str:
    """Render full NFL standings by division."""
    lines = []

    for conf in ("AFC", "NFC"):
        lines.append(f"\n{'═' * 50}")
        lines.append(f"  {conf}")
        lines.append(f"{'═' * 50}")

        for div_name, teams in sorted(DIVISIONS[conf].items()):
            full_div = f"{conf} {div_name}"
            div_standings = standings.division_standings(full_div)
            lines.append(f"\n  {div_name}")
            lines.append(f"  {'Team':>4s}  {'Record':>7s}  {'PF':>4s}  {'PA':>4s}  {'Diff':>5s}  {'Div':>5s}  {'Conf':>5s}")
            lines.append(f"  {'─' * 46}")
            for r in div_standings:
                div_rec = f"{r.div_wins}-{r.div_losses}"
                conf_rec = f"{r.conf_wins}-{r.conf_losses}"
                lines.append(
                    f"  {r.team:>4s}  {r.record_str():>7s}  {r.points_for:4d}  "
                    f"{r.points_against:4d}  {r.point_diff:+5d}  {div_rec:>5s}  {conf_rec:>5s}"
                )

    return "\n".join(lines)


def render_playoff_bracket(result: SeasonResult) -> str:
    """Render the playoff bracket."""
    pr = result.playoff_result
    lines = []

    lines.append(f"\n{'═' * 50}")
    lines.append("  PLAYOFF BRACKET")
    lines.append(f"{'═' * 50}")

    # Seeds
    for conf, seeds in [("AFC", pr.afc_seeds), ("NFC", pr.nfc_seeds)]:
        lines.append(f"\n  {conf} Seeds:")
        for i, team in enumerate(seeds, 1):
            rec = result.standings.records.get(team)
            rec_str = rec.record_str() if rec else "?"
            lines.append(f"    {i}. {team} ({rec_str})")

    # Wild Card
    lines.append(f"\n  Wild Card Round")
    lines.append(f"  {'─' * 40}")
    for w, l, score in pr.wildcard:
        lines.append(f"    {score}  →  {w} advances")

    # Divisional
    lines.append(f"\n  Divisional Round")
    lines.append(f"  {'─' * 40}")
    for w, l, score in pr.divisional:
        lines.append(f"    {score}  →  {w} advances")

    # Conference Championships
    lines.append(f"\n  Conference Championships")
    lines.append(f"  {'─' * 40}")
    for w, l, score in pr.conference:
        conf = TEAM_TO_CONFERENCE.get(w, "?")
        lines.append(f"    {score}  →  {w} wins {conf}")

    # Super Bowl
    if pr.super_bowl:
        w, l, score = pr.super_bowl
        lines.append(f"\n  {'═' * 40}")
        lines.append(f"  SUPER BOWL")
        lines.append(f"    {score}")
        lines.append(f"    CHAMPION: {w}")
        lines.append(f"  {'═' * 40}")

    return "\n".join(lines)


def render_multi_season_summary(results: list[SeasonResult], n: int) -> str:
    """Render aggregate stats from N season simulations."""
    lines = []
    champion_counts: Counter[str] = Counter()
    sb_appearance: Counter[str] = Counter()
    conf_champ: Counter[str] = Counter()
    playoff_counts: Counter[str] = Counter()
    division_winner_counts: Counter[str] = Counter()
    total_wins: defaultdict[str, int] = defaultdict(int)

    for result in results:
        # Champion
        if result.champion:
            champion_counts[result.champion] += 1

        # Super Bowl appearances
        pr = result.playoff_result
        if pr.super_bowl:
            sb_appearance[pr.super_bowl[0]] += 1
            sb_appearance[pr.super_bowl[1]] += 1

        # Conference champions
        for w, l, s in pr.conference:
            conf_champ[w] += 1

        # Playoff appearances
        for team in pr.afc_seeds + pr.nfc_seeds:
            playoff_counts[team] += 1

        # Division winners (seeds 1-4 in each conference)
        for team in pr.afc_seeds[:4] + pr.nfc_seeds[:4]:
            division_winner_counts[team] += 1

        # Win totals
        for team, rec in result.standings.records.items():
            total_wins[team] += rec.wins

    lines.append(f"\n{'═' * 60}")
    lines.append(f"  {n:,} SEASON SIMULATIONS")
    lines.append(f"{'═' * 60}")

    # Super Bowl odds
    lines.append(f"\n  Super Bowl Champions")
    lines.append(f"  {'Team':>4s}  {'Wins':>6s}  {'Odds':>7s}")
    lines.append(f"  {'─' * 22}")
    for team, count in champion_counts.most_common(16):
        lines.append(f"  {team:>4s}  {count:6d}  {count/n:7.1%}")

    # Playoff odds (top 16)
    lines.append(f"\n  Playoff Probability (top 16)")
    lines.append(f"  {'Team':>4s}  {'Playoffs':>8s}  {'Div Win':>8s}  {'Conf':>6s}  {'SB App':>7s}  {'Champ':>6s}  {'Avg W':>5s}")
    lines.append(f"  {'─' * 55}")
    for team, _ in playoff_counts.most_common(16):
        avg_w = total_wins[team] / n
        lines.append(
            f"  {team:>4s}  {playoff_counts[team]/n:8.1%}  "
            f"{division_winner_counts[team]/n:8.1%}  "
            f"{conf_champ.get(team, 0)/n:6.1%}  "
            f"{sb_appearance.get(team, 0)/n:7.1%}  "
            f"{champion_counts.get(team, 0)/n:6.1%}  "
            f"{avg_w:5.1f}"
        )

    return "\n".join(lines)
