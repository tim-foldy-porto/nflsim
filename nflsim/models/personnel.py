"""Personnel selection — assigns players to each play.

On a pass play: select QB (passer), and if complete, select receiver.
On a run play: select rusher.
On sacks: select QB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nflsim.data.profiles import TeamProfile


@dataclass
class PlayPersonnel:
    """Players involved in a single play."""
    passer_id: str | None = None
    passer_name: str | None = None
    receiver_id: str | None = None
    receiver_name: str | None = None
    rusher_id: str | None = None
    rusher_name: str | None = None


def assign_personnel(
    team_profile: TeamProfile | None,
    play_type: str,
    is_complete: bool,
    sack: bool,
    rng: np.random.Generator,
) -> PlayPersonnel:
    """Assign players to a play based on team profile and play type.

    Args:
        team_profile: The offensive team's profile (None = no player assignment)
        play_type: "pass", "run", "kneel", "spike", etc.
        is_complete: Whether a pass was completed
        sack: Whether the play was a sack
        rng: Random number generator

    Returns:
        PlayPersonnel with assigned player IDs and names
    """
    if team_profile is None:
        return PlayPersonnel()

    personnel = PlayPersonnel()

    if play_type == "pass":
        qb = team_profile.sample_qb(rng)
        if qb:
            personnel.passer_id = qb.player_id
            personnel.passer_name = qb.name

        # Assign receiver on completions and targeted incompletions
        if not sack:
            rec = team_profile.sample_receiver(rng)
            if rec:
                personnel.receiver_id = rec.player_id
                personnel.receiver_name = rec.name

    elif play_type == "run":
        rusher = team_profile.sample_rusher(rng)
        if rusher:
            personnel.rusher_id = rusher.player_id
            personnel.rusher_name = rusher.name

    elif play_type in ("kneel", "spike"):
        qb = team_profile.sample_qb(rng)
        if qb:
            personnel.passer_id = qb.player_id
            personnel.passer_name = qb.name

    return personnel
