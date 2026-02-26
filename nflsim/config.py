"""Paths, constants, and configuration."""

from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# Game constants
QUARTERS = 4
QUARTER_SECONDS = 900  # 15 minutes
OVERTIME_SECONDS = 600  # 10 minutes (regular season)
HALF_SECONDS = QUARTER_SECONDS * 2

YARDS_FOR_FIRST_DOWN = 10
FIELD_LENGTH = 100
ENDZONE_DEPTH = 10

# Scoring
TOUCHDOWN = 6
FIELD_GOAL = 3
EXTRA_POINT = 1
TWO_POINT_CONVERSION = 2
SAFETY = 2

# Play clock
PLAY_CLOCK_SECONDS = 40
AVG_PLAY_DURATION = 6  # seconds of actual play

# Kickoff
KICKOFF_YARD_LINE = 35
TOUCHBACK_YARD_LINE = 25  # where ball is placed after touchback (from own goal)
KICKOFF_TOUCHBACK_YARD_LINE = 30  # 2024 rule: touchback to 30

# Punt
PUNT_TOUCHBACK_YARD_LINE = 20

# Clock runoff — calibrated from 2024 PBP data (median values)
# "Clock-running" = play + huddle + play clock countdown
CLOCK_RUNOFF_RUN = 38              # run inbounds: median 39s
CLOCK_RUNOFF_COMPLETE_INBOUNDS = 38  # complete pass inbounds: median 38s
CLOCK_RUNOFF_SACK = 40             # sack: median 40s
CLOCK_RUNOFF_INCOMPLETE = 5        # clock stops, just play duration
CLOCK_RUNOFF_OUT_OF_BOUNDS = 34    # clock stops briefly but teams still huddle: median 34s
CLOCK_RUNOFF_SCORING = 5           # clock stops on scoring plays
CLOCK_RUNOFF_PENALTY = 14          # penalties: median ~8s but some have runoff
CLOCK_RUNOFF_PUNT = 9              # punt: median 9s
CLOCK_RUNOFF_FIELD_GOAL = 4        # FG attempt: median 4s
CLOCK_RUNOFF_KICKOFF = 6           # kickoff: median 6s
CLOCK_RUNOFF_TIMEOUT = 3
CLOCK_RUNOFF_TWO_MINUTE_WARNING = 0

# Hurry-up mode (last 2 min of half): clock-running plays drop to ~18s
CLOCK_RUNOFF_HURRY_UP_RUNNING = 18
CLOCK_RUNOFF_HURRY_UP_STOPPED = 5

# Timeouts
TIMEOUTS_PER_HALF = 3
