"""CLI interface for NFL simulator."""

from __future__ import annotations

import logging
import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="NFL Game Simulator")
console = Console()


def _load_models():
    """Try to load saved models. Returns None if not available."""
    from nflsim.models.store import load_models
    return load_models()


@app.command()
def simulate_game(
    away_team: str = typer.Argument(help="Away team abbreviation (e.g., SF)"),
    home_team: str = typer.Argument(help="Home team abbreviation (e.g., KC)"),
    n: int = typer.Option(1, "--n", "-n", help="Number of simulations"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show play-by-play"),
    no_models: bool = typer.Option(False, "--no-models", help="Use Phase 1 heuristics only"),
):
    """Simulate NFL game(s) between two teams."""
    from nflsim.data.schemas import NFL_TEAMS
    from nflsim.engine.game import simulate_game as run_game

    # Validate teams
    away_team = away_team.upper()
    home_team = home_team.upper()
    for team in (away_team, home_team):
        if team not in NFL_TEAMS:
            console.print(f"[red]Unknown team: {team}[/red]")
            console.print(f"Valid teams: {', '.join(sorted(NFL_TEAMS))}")
            raise typer.Exit(1)

    # Load models
    models = None
    if not no_models:
        models = _load_models()
        if models:
            console.print("[dim]Using data-driven models[/dim]")
        else:
            console.print("[dim]No models found — using heuristics (run build-profiles first)[/dim]")

    if n == 1:
        console.print(f"\n[bold]{away_team} @ {home_team}[/bold]\n")
        result = run_game(home_team, away_team, seed=seed, verbose=verbose, models=models)

        if verbose:
            console.print()

        console.print(f"[bold]Final: {result}[/bold]")

        if result.validation_errors:
            console.print(f"\n[red]Validation errors: {len(result.validation_errors)}[/red]")
            for err in result.validation_errors:
                console.print(f"  [red]• {err}[/red]")

        if result.box_score:
            console.print(result.box_score.summary())

        if result.player_stats:
            console.print(result.player_stats.render(away_team, home_team))

    else:
        console.print(f"\n[bold]Simulating {away_team} @ {home_team} ({n:,} games)...[/bold]\n")

        home_wins = 0
        away_wins = 0
        ties = 0
        total_points = 0
        total_plays = 0
        validation_error_count = 0

        t0 = time.perf_counter()
        for i in range(n):
            game_seed = seed + i if seed is not None else None
            result = run_game(home_team, away_team, seed=game_seed, models=models)

            if result.home_score > result.away_score:
                home_wins += 1
            elif result.away_score > result.home_score:
                away_wins += 1
            else:
                ties += 1

            total_points += result.home_score + result.away_score
            total_plays += result.total_plays
            validation_error_count += len(result.validation_errors)

        elapsed = time.perf_counter() - t0

        table = Table(title=f"{away_team} @ {home_team} — {n:,} Simulations")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row(f"{home_team} Wins", f"{home_wins:,} ({home_wins/n:.1%})")
        table.add_row(f"{away_team} Wins", f"{away_wins:,} ({away_wins/n:.1%})")
        table.add_row("Ties", f"{ties:,} ({ties/n:.1%})")
        table.add_row("Avg Total Points", f"{total_points/n:.1f}")
        table.add_row("Avg Plays/Game", f"{total_plays/n:.1f}")
        table.add_row("Validation Errors", f"{validation_error_count:,}")
        table.add_row("Time", f"{elapsed:.2f}s ({n/elapsed:.0f} games/sec)")

        console.print(table)


@app.command()
def sync(
    seasons: list[int] = typer.Argument(help="Seasons to download (e.g., 2020 2021 2022)"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-download even if cached"),
):
    """Download and cache nflverse data."""
    from nflsim.data.loader import load_pbp, load_rosters, load_schedules

    logging.basicConfig(level=logging.INFO)

    console.print(f"[bold]Syncing data for seasons: {seasons}[/bold]\n")

    with console.status("Downloading play-by-play..."):
        pbp = load_pbp(seasons, force=force)
    console.print(f"  PBP: {pbp.shape[0]:,} plays, {pbp.shape[1]} columns")

    with console.status("Downloading rosters..."):
        rosters = load_rosters(seasons, force=force)
    console.print(f"  Rosters: {rosters.shape[0]:,} players")

    with console.status("Downloading schedules..."):
        schedules = load_schedules(seasons, force=force)
    console.print(f"  Schedules: {schedules.shape[0]:,} games")

    console.print("\n[green]Sync complete![/green]")


@app.command()
def build_profiles(
    seasons: list[int] = typer.Argument(help="Seasons to build from (e.g., 2020 2021 2022 2023 2024)"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-download data even if cached"),
):
    """Build statistical models from historical PBP data."""
    from nflsim.data.loader import load_pbp
    from nflsim.models.store import build_all_models, save_models

    logging.basicConfig(level=logging.INFO)

    console.print(f"[bold]Building models from {len(seasons)} seasons: {seasons}[/bold]\n")

    with console.status("Loading play-by-play data..."):
        pbp = load_pbp(seasons, force=force)
    console.print(f"  Loaded {pbp.shape[0]:,} plays")

    current_season = max(seasons)
    with console.status("Building models..."):
        bundle = build_all_models(pbp, current_season)

    path = save_models(bundle)
    console.print(f"\n[green]Models saved to {path}[/green]")

    # Print summary
    console.print(f"\n  Play call contexts: {len(bundle.play_call.tables):,}")
    console.print(f"  4th down contexts: {len(bundle.play_call.fourth_down_tables):,}")
    console.print(f"  Team factors: {len(bundle.play_call.team_factors)}")
    console.print(f"  Pass yard contexts: {len(bundle.play_outcome.pass_yard_bins):,}")
    console.print(f"  Rush yard contexts: {len(bundle.play_outcome.rush_yard_bins):,}")
    console.print(f"  Global sack rate: {bundle.play_outcome.global_sack_rate:.3f}")
    console.print(f"  Global comp rate: {bundle.play_outcome.global_comp_rate:.3f}")
    console.print(f"  Global INT rate: {bundle.play_outcome.int_rate:.3f}")
    console.print(f"  Penalty rate: {bundle.penalties.penalty_rate:.3f}")
    console.print(f"  FG buckets: {len(bundle.field_goal.success_rates)}")
    console.print(f"  XP rate: {bundle.field_goal.extra_point_rate:.3f}")
    if bundle.team_profiles:
        console.print(f"  Team profiles: {len(bundle.team_profiles)}")
        total_players = sum(
            len(p.qbs) + len(p.receivers) + len(p.rushers)
            for p in bundle.team_profiles.values()
        )
        console.print(f"  Total player profiles: {total_players}")
    if bundle.team_ratings:
        ratings = sorted(bundle.team_ratings.values(), key=lambda r: r.overall, reverse=True)
        console.print(f"  Team ratings: {len(ratings)}")
        console.print(f"    Best: {ratings[0].team} ({ratings[0].overall:.3f})")
        console.print(f"    Worst: {ratings[-1].team} ({ratings[-1].overall:.3f})")


@app.command()
def simulate_season(
    season: int = typer.Argument(help="Season year (e.g., 2024)"),
    n: int = typer.Option(1, "--n", "-n", help="Number of season simulations"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    no_models: bool = typer.Option(False, "--no-models", help="Use Phase 1 heuristics only"),
):
    """Simulate a full NFL season (or multiple) with standings and playoffs."""
    from nflsim.season.season import simulate_season as run_season
    from nflsim.season.schedule import load_season_schedule
    from nflsim.output.season_report import (
        render_standings,
        render_playoff_bracket,
        render_multi_season_summary,
    )

    logging.basicConfig(level=logging.INFO)

    models = None
    if not no_models:
        models = _load_models()
        if models:
            console.print("[dim]Using data-driven models[/dim]")
        else:
            console.print("[dim]No models found — using heuristics[/dim]")

    # Pre-load schedule once
    schedule = load_season_schedule(season)

    if n == 1:
        console.print(f"\n[bold]Simulating {season} NFL season...[/bold]\n")
        t0 = time.perf_counter()
        result = run_season(season, seed=seed, models=models, schedule=schedule)
        elapsed = time.perf_counter() - t0

        console.print(render_standings(result.standings))
        console.print(render_playoff_bracket(result))
        console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")
    else:
        console.print(f"\n[bold]Simulating {season} NFL season x{n:,}...[/bold]\n")

        # Suppress per-season logging for bulk runs
        logging.getLogger("nflsim").setLevel(logging.WARNING)

        t0 = time.perf_counter()
        results = []
        for i in range(n):
            sim_seed = seed + i if seed is not None else None
            result = run_season(season, seed=sim_seed, models=models, schedule=schedule)
            results.append(result)
            if (i + 1) % 100 == 0 or (i + 1) == n:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                console.print(
                    f"  [dim]{i+1:,}/{n:,} seasons ({rate:.1f} seasons/sec)[/dim]",
                    end="\r",
                )
        console.print()

        elapsed = time.perf_counter() - t0
        console.print(render_multi_season_summary(results, n))
        console.print(f"\n[dim]Completed in {elapsed:.1f}s ({n/elapsed:.1f} seasons/sec)[/dim]")


@app.command()
def validate(
    season: int = typer.Argument(help="Season to validate against"),
    n: int = typer.Option(100, "--n", "-n", help="Simulations per game"),
):
    """Validate simulator against actual season results. (Phase 2+)"""
    console.print("[yellow]Validation not yet implemented (requires Phase 3+)[/yellow]")


if __name__ == "__main__":
    app()
