"""Column constants and validation for nflverse play-by-play data."""

# Key columns we use from nflverse PBP data
PBP_COLUMNS = [
    "play_id",
    "game_id",
    "season",
    "week",
    "game_date",
    "posteam",  # team with possession
    "defteam",
    "posteam_type",  # home/away
    "side_of_field",
    "yardline_100",  # yards from opponent's end zone
    "game_seconds_remaining",
    "half_seconds_remaining",
    "quarter_seconds_remaining",
    "game_half",
    "quarter_end",
    "down",
    "ydstogo",  # yards to go for first down
    "goal_to_go",
    "play_type",  # pass, run, punt, field_goal, kickoff, etc.
    "yards_gained",
    "air_yards",
    "yards_after_catch",
    "pass_length",  # short/deep
    "pass_location",  # left/middle/right
    "run_location",  # left/middle/right
    "run_gap",  # end/tackle/guard
    "shotgun",
    "no_huddle",
    "qb_dropback",
    "qb_scramble",
    "rush_attempt",
    "pass_attempt",
    "sack",
    "touchdown",
    "fumble",
    "interception",
    "penalty",
    "penalty_yards",
    "first_down",
    "third_down_converted",
    "third_down_failed",
    "fourth_down_converted",
    "fourth_down_failed",
    "incomplete_pass",
    "complete_pass",
    "passer_player_name",
    "rusher_player_name",
    "receiver_player_name",
    "score_differential",
    "posteam_score",
    "defteam_score",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "ep",  # expected points
    "epa",  # expected points added
    "wp",  # win probability
    "wpa",  # win probability added
    "field_goal_result",
    "kick_distance",
    "extra_point_result",
    "two_point_conv_result",
    "punt_blocked",
    "return_yards",
    "tackled_for_loss",
    "fumble_lost",
    "own_kickoff_recovery",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "result",  # home score - away score
    "total",  # total points scored
]

# Play types from nflverse
PLAY_TYPES = {
    "pass",
    "run",
    "punt",
    "field_goal",
    "kickoff",
    "extra_point",
    "two_point_attempt",
    "qb_kneel",
    "qb_spike",
    "no_play",  # penalties that negate the play
}

# Teams (current 32 NFL teams, standard abbreviations)
NFL_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}
