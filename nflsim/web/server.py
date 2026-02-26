"""Web server for NFL simulator — serves the game viewer UI."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nflsim.engine.game import simulate_game
from nflsim.models.store import load_models
from nflsim.season.schedule import DIVISIONS, TEAM_TO_CONFERENCE

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="NFL Simulator")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load models once at startup
_models = None


def _get_models():
    global _models
    if _models is None:
        _models = load_models()
    return _models


class SimRequest(BaseModel):
    home_team: str
    away_team: str
    seed: int | None = None


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/teams")
async def get_teams():
    """Return teams with division/conference info and ratings."""
    models = _get_models()
    ratings = models.team_ratings if models else {}

    teams = []
    for conf, divs in DIVISIONS.items():
        for div, team_list in divs.items():
            for team in team_list:
                r = ratings.get(team)
                teams.append({
                    "abbr": team,
                    "conference": conf,
                    "division": f"{conf} {div}",
                    "overall": round(r.overall, 3) if r else 1.0,
                    "off_rating": round(r.off_rating, 3) if r else 1.0,
                    "def_rating": round(r.def_rating, 3) if r else 1.0,
                })

    teams.sort(key=lambda t: t["overall"], reverse=True)
    return JSONResponse(teams)


@app.post("/api/simulate")
async def simulate(req: SimRequest):
    """Simulate a game and return structured play events."""
    models = _get_models()
    result = simulate_game(
        req.home_team, req.away_team,
        seed=req.seed, models=models, capture_events=True,
    )

    return JSONResponse({
        "home_team": result.home_team,
        "away_team": result.away_team,
        "home_score": result.home_score,
        "away_score": result.away_score,
        "total_plays": result.total_plays,
        "plays": result.play_events or [],
    })


def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting NFL Simulator web server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
