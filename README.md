# NFL Simulator

A probabilistic NFL game simulator that models football play-by-play using empirical distributions derived from real NFL data. Simulate individual games, full seasons, or run Monte Carlo projections for playoff odds and Super Bowl probabilities.

## Features

### Game Engine
- Play-by-play simulation with a state machine (kickoff, normal play, PAT, overtime, game over)
- Realistic clock management calibrated from 2024 NFL play-by-play medians
- Full rules: downs, turnovers, penalties, safeties, two-minute warning, overtime

### Data-Driven Models
- Built from 5 seasons of nflverse play-by-play data (~246K plays, 2020-2024)
- Conditional play-calling tables keyed by down, distance, field zone, and score differential
- Empirical yard distributions (histogram sampling, not parametric)
- Field goal success rates by distance, calibrated XP and 2pt conversion rates
- Recency weighting with 1.5-season half-life so recent data matters more

### Team Power Ratings
- Per-team offensive and defensive multipliers computed from PBP data
- Six dimensions: pass yards, rush yards, completion rate, sack rate, interception rate, fumble rate
- Matchup adjustments applied during simulation (good offense vs bad defense = more yards)

### Player Modeling
- Per-team roster profiles with real player names and usage shares
- Target share, carry share, and snap share from current-season data
- ESPN-style player box scores: passing, rushing, receiving stats with passer rating

### Season Simulation
- Full 18-week regular season using real NFL schedules
- Division standings with simplified tiebreakers (win%, division record, conference record, point differential)
- 7-team playoff bracket per conference (Wild Card, Divisional, Conference Championship, Super Bowl)
- Multi-season Monte Carlo: run 100-1000+ simulations for playoff probability tables and Super Bowl odds

### Web Viewer
- 2D football field rendered on HTML canvas with team-colored end zones
- Animated play-by-play: line of scrimmage, first-down marker, ball position, gain/loss highlights
- Score bug with clock, quarter, possession indicator, and down & distance
- Playback controls: play/pause, step forward/back, speed slider, keyboard shortcuts
- Scrollable play log with click-to-jump

## Quick Start

```bash
# Install
pip install -e .
pip install pyarrow fastapi uvicorn

# Download data and build models
nflsim sync 2020 2021 2022 2023 2024
nflsim build-profiles 2020 2021 2022 2023 2024

# Simulate a single game
nflsim simulate-game KC BAL --seed 42

# Simulate a full season
nflsim simulate-season 2024 --seed 42

# Run 100 season Monte Carlo
nflsim simulate-season 2024 -n 100 --seed 42

# Launch the web viewer
python -m nflsim.web.server
# Open http://localhost:8080
```

## Project Structure

```
nflsim/
  data/        # Data loading, feature engineering, player profiles
  engine/      # Game state machine, play resolution, clock, rules
  models/      # Play calling, outcomes, penalties, field goals, team ratings
  output/      # Box scores, player stats, season reports
  season/      # Schedule, standings, playoffs, season orchestration
  web/         # FastAPI server and HTML/JS game viewer
```
