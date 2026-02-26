"""Data loading from nflverse via nfl_data_py, with local parquet caching."""

import logging
from pathlib import Path

import polars as pl

from nflsim.config import RAW_DIR

logger = logging.getLogger(__name__)


def _cache_path(name: str, seasons: list[int]) -> Path:
    """Build a cache file path for a dataset + season range."""
    tag = f"{min(seasons)}-{max(seasons)}" if len(seasons) > 1 else str(seasons[0])
    return RAW_DIR / f"{name}_{tag}.parquet"


def load_pbp(seasons: list[int], *, force: bool = False) -> pl.DataFrame:
    """Load play-by-play data for given seasons, caching to parquet."""
    import nfl_data_py as nfl

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for season in seasons:
        cache = RAW_DIR / f"pbp_{season}.parquet"
        if cache.exists() and not force:
            logger.info("Loading cached PBP for %d", season)
            frames.append(pl.read_parquet(cache))
        else:
            logger.info("Downloading PBP for %d...", season)
            pdf = nfl.import_pbp_data([season])
            df = pl.from_pandas(pdf)
            df.write_parquet(cache)
            frames.append(df)

    if not frames:
        return pl.DataFrame()
    # Seasons may have different columns — align to common set, coerce types
    common_cols = set(frames[0].columns)
    for f in frames[1:]:
        common_cols &= set(f.columns)
    aligned = [f.select(sorted(common_cols)) for f in frames]
    return pl.concat(aligned, how="vertical_relaxed")


def load_rosters(seasons: list[int], *, force: bool = False) -> pl.DataFrame:
    """Load roster data for given seasons."""
    import nfl_data_py as nfl

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path("rosters", seasons)

    if cache.exists() and not force:
        logger.info("Loading cached rosters")
        return pl.read_parquet(cache)

    logger.info("Downloading rosters...")
    pdf = nfl.import_seasonal_rosters(seasons)
    df = pl.from_pandas(pdf)
    df.write_parquet(cache)
    return df


def load_schedules(seasons: list[int], *, force: bool = False) -> pl.DataFrame:
    """Load schedule data for given seasons."""
    import nfl_data_py as nfl

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path("schedules", seasons)

    if cache.exists() and not force:
        logger.info("Loading cached schedules")
        return pl.read_parquet(cache)

    logger.info("Downloading schedules...")
    pdf = nfl.import_schedules(seasons)
    df = pl.from_pandas(pdf)
    df.write_parquet(cache)
    return df
