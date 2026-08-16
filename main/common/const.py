import retro
import os
import numpy as np


SF_BONUS_LEVEL = [4, 8, 12]

SF_DEFAULT_STATE = "Champion.Level1.RyuVsGuile"

retro_directory = os.path.dirname(retro.__file__)
sf_game_dir = "data/stable/StreetFighterIISpecialChampionEdition-Genesis"
SF_STATE_DIR = os.path.join(retro_directory, sf_game_dir)

sf_game = "StreetFighterIISpecialChampionEdition-Genesis"

START_STATUS = 0

END_STATUS = 1

BUTTONS = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

SF_COMBOS_BUTTONS = [
    [['DOWN'], ['DOWN', 'RIGHT'], ['RIGHT'], ['X']], # 'Hadouken-R'
    [['RIGHT'], ['DOWN'], ['DOWN', 'RIGHT'], ['X']], # 'Shoryuken-R'
    [['DOWN'], ['DOWN', 'LEFT'], ['LEFT'], ['A']], # 'Tatsumaki-R'
    [['DOWN'], ['DOWN', 'LEFT'], ['LEFT'], ['X']], # 'Hadouken-L'
    [['LEFT'], ['DOWN'], ['DOWN', 'LEFT'], ['X']], #'Shoryuken-L'
    [['DOWN'], ['DOWN', 'RIGHT'], ['RIGHT'], ['A']], # 'Tatsumaki-L'
]
def build_sf_combos(num_step_frames=8):
    """Motion inputs stretched to fill exactly num_step_frames emulator frames.

    SFWrapper asserts num_step_frames == len(combo), so the historical hardcoded
    `range(2)` pinned the frame skip at 8 (4 inputs x 2 frames). Deriving the
    repeat count from the frame budget is what makes the skip adjustable, which
    is the point: at 8 frames a whole exchange resolves inside ONE decision, so
    the joint dependence is settled before the agent chooses again.

    Fails loudly rather than silently truncating a motion input -- a Hadouken
    missing its final frame is not a Hadouken, and the agent would just see an
    action that never works.
    """
    n_inputs = len(SF_COMBOS_BUTTONS[0])
    if any(len(c) != n_inputs for c in SF_COMBOS_BUTTONS):
        raise ValueError("SF_COMBOS_BUTTONS entries differ in length")
    if num_step_frames % n_inputs != 0:
        raise ValueError(
            f"num_step_frames={num_step_frames} is not divisible by the {n_inputs} "
            f"inputs in a motion command; a combo cannot be stretched to fit it "
            f"without dropping or duplicating an input unevenly.")
    reps = num_step_frames // n_inputs
    out = []
    for combo_buttons in SF_COMBOS_BUTTONS:
        action_seq = []
        for combo_button in combo_buttons:
            button = [int(b in combo_button) for b in BUTTONS]
            for _ in range(reps):
                action_seq.append(np.array(button))
        out.append(action_seq)
    return out


SF_COMBOS = build_sf_combos(8)

DIRECTIONS_BUTTONS = [
    [], ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], 
    ['UP', 'LEFT'], ['UP', 'RIGHT'], ['DOWN', 'LEFT'], ['DOWN', 'RIGHT'], 
]

ATTACKS_BUTTONS = [
    [], ['B'], ['A'], ['C'], ['Y'], ['X'], ['Z'],
]

SELECT_CHARACTER_MOVEMENTS = {
    'NO_OP': [],
    'START': ['START'],
    'LEFT': ['LEFT'],
    'RIGHT': ['RIGHT'],
    'UP': ['UP'],
    'DOWN': ['DOWN'],
}
SELECT_CHARACTER_BUTTONS = {}
for k, v in SELECT_CHARACTER_MOVEMENTS.items():
    SELECT_CHARACTER_BUTTONS[k] = np.array([int(b in v) for b in BUTTONS])
SELECT_CHARACTER_SEQUENCES = {
    'Ryu': [],
    'Honda': ['RIGHT'],
    'Blanka': ['RIGHT'] * 2,
    'Guile': ['RIGHT'] * 3,
    'Balrog': ['RIGHT'] * 4,
    'Vega': ['RIGHT'] * 5,
    'Ken': ['DOWN'],
    'Chunli': ['DOWN'] + ['RIGHT'],
    'Zangief': ['DOWN'] + ['RIGHT'] * 2,
    'Dhalsim': ['DOWN'] + ['RIGHT'] * 3,
    'Sagat': ['DOWN'] + ['RIGHT'] * 4,
    'Bison': ['DOWN'] + ['RIGHT'] * 5,
}