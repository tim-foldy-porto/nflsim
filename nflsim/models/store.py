"""Model persistence — build, save, and load all models as a single bundle."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nflsim.config import PROCESSED_DIR
from nflsim.data.profiles import TeamProfile, build_team_profiles
from nflsim.models.field_goal import FieldGoalModel, build_field_goal_model
from nflsim.models.penalties import PenaltyModel, build_penalty_model
from nflsim.models.play_call import PlayCallModel, build_play_call_model
from nflsim.models.play_outcome import PlayOutcomeModel, build_play_outcome_model
from nflsim.models.team_ratings import TeamRating, build_team_ratings

logger = logging.getLogger(__name__)

MODEL_FILE = "models.pkl"


@dataclass
class ModelBundle:
    """All models needed for simulation."""
    play_call: PlayCallModel
    play_outcome: PlayOutcomeModel
    penalties: PenaltyModel
    field_goal: FieldGoalModel
    team_profiles: dict[str, TeamProfile] | None = None
    team_ratings: dict[str, TeamRating] | None = None


def build_all_models(pbp: pl.DataFrame, current_season: int) -> ModelBundle:
    """Build all models from PBP data."""
    return ModelBundle(
        play_call=build_play_call_model(pbp, current_season),
        play_outcome=build_play_outcome_model(pbp, current_season),
        penalties=build_penalty_model(pbp, current_season),
        field_goal=build_field_goal_model(pbp),
        team_profiles=build_team_profiles(pbp, current_season),
        team_ratings=build_team_ratings(pbp, current_season),
    )


def save_models(bundle: ModelBundle, path: Path | None = None) -> Path:
    """Save model bundle to disk."""
    if path is None:
        path = PROCESSED_DIR / MODEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Models saved to %s", path)
    return path


def load_models(path: Path | None = None) -> ModelBundle | None:
    """Load model bundle from disk. Returns None if not found."""
    if path is None:
        path = PROCESSED_DIR / MODEL_FILE
    if not path.exists():
        logger.info("No saved models found at %s", path)
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    logger.info("Models loaded from %s", path)
    return bundle
