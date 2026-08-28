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
def _stretch(seq, nsf):
    """Stretch a k-input motion command to fill nsf frames (each input held nsf//k frames).
    Fails loudly rather than truncating -- a Hadouken missing its last frame is not a Hadouken."""
    if nsf % len(seq) != 0:
        raise ValueError(f"num_step_frames={nsf} not divisible by motion length {len(seq)}")
    reps = nsf // len(seq)
    out = []
    for grp in seq:
        b = np.array([int(x in grp) for x in BUTTONS])
        out += [b for _ in range(reps)]
    return out

def _frames(frame_list, nsf):
    """Build an exact nsf-frame program from a list of button-name groups."""
    if len(frame_list) != nsf:
        raise ValueError(f"macro program length {len(frame_list)} != num_step_frames {nsf}")
    return [np.array([int(x in f) for x in BUTTONS]) for f in frame_list]

# --- tuned macro programs (each verified to fire via the RAM special-flag; see harness). ---
# Char-agnostic: the input PROGRAM; the character decides which special that input produces.
def _motion(seq):            # QCF/DP/QCB motion, stretched to nsf
    return lambda nsf: _stretch(seq, nsf)
def _rel_fwd(fwd, btn):      # charge release: FORWARD 1-frame head-start, then forward+button (10/10)
    return lambda nsf: _frames([[fwd]] + [[fwd, btn]] * (nsf - 1), nsf)
def _rel_up(btn):            # down-charge release: up head-start, then up+button
    return lambda nsf: _frames([['UP']] + [['UP', btn]] * (nsf - 1), nsf)
def _mash(btn):              # tap every other frame; select repeatedly to sustain the mash
    return lambda nsf: _frames([[btn] if i % 2 == 0 else [] for i in range(nsf)], nsf)
def _hold(btns):             # multi-button held (Lariat)
    return lambda nsf: _frames([list(btns)] * nsf, nsf)
def _spd(btn):               # 360 grab: half-circle into up-forward+button (needs nsf==8)
    return lambda nsf: _frames([['LEFT'], ['DOWN', 'LEFT'], ['DOWN'], ['DOWN', 'RIGHT'], ['RIGHT'],
                                ['UP', 'RIGHT'], ['UP', 'RIGHT'], ['UP', 'RIGHT', btn]][:nsf], nsf)

MACRO_DEFS = {
    'hadouken_r':  _motion(SF_COMBOS_BUTTONS[0]), 'shoryuken_r': _motion(SF_COMBOS_BUTTONS[1]),
    'tatsumaki_r': _motion(SF_COMBOS_BUTTONS[2]), 'hadouken_l':  _motion(SF_COMBOS_BUTTONS[3]),
    'shoryuken_l': _motion(SF_COMBOS_BUTTONS[4]), 'tatsumaki_l': _motion(SF_COMBOS_BUTTONS[5]),
    'chg_fwd_p_r': _rel_fwd('RIGHT', 'X'), 'chg_fwd_p_l': _rel_fwd('LEFT', 'X'),
    'chg_fwd_k_r': _rel_fwd('RIGHT', 'A'), 'chg_fwd_k_l': _rel_fwd('LEFT', 'A'),
    'chg_dwn_k': _rel_up('A'), 'chg_dwn_p': _rel_up('X'),
    'mash_p': _mash('X'), 'mash_k': _mash('A'),
    'spd_p': _spd('X'), 'spd_k': _spd('A'),
    'lariat_p': _hold(['Y', 'X']), 'lariat_k': _hold(['B', 'A']),
}

# per-character moveset (SF2 Champion Edition): which macros that character actually has.
_SHOTO = ['hadouken_r', 'hadouken_l', 'shoryuken_r', 'shoryuken_l', 'tatsumaki_r', 'tatsumaki_l']
MOVESETS = {
    'Ryu': _SHOTO, 'Ken': _SHOTO,
    'Sagat': ['hadouken_r', 'hadouken_l', 'shoryuken_r', 'shoryuken_l'],   # Tiger Shot(QCF), Tiger Uppercut(DP)
    'Dhalsim': ['hadouken_r', 'hadouken_l'],                               # Yoga Fire(QCF)
    'Guile': ['chg_fwd_p_r', 'chg_fwd_p_l', 'chg_dwn_k'],                  # Sonic Boom, Flash Kick
    'ChunLi': ['chg_dwn_k', 'mash_k'],                                     # Spinning Bird Kick, Lightning Kick
    'Blanka': ['chg_fwd_p_r', 'chg_fwd_p_l', 'mash_p'],                    # Rolling Attack, Electricity
    'EHonda': ['chg_fwd_p_r', 'chg_fwd_p_l', 'mash_p'],                    # Sumo Headbutt, Hundred Hand Slap
    'Balrog': ['chg_fwd_p_r', 'chg_fwd_p_l', 'chg_fwd_k_r', 'chg_fwd_k_l'],# Dash Straight / Dash Low
    'Vega': ['chg_fwd_p_r', 'chg_fwd_p_l'],                                # Rolling Crystal Flash
    'MBison': ['chg_fwd_p_r', 'chg_fwd_p_l', 'chg_fwd_k_r', 'chg_fwd_k_l'],# Psycho Crusher, Scissor Kick
    'Zangief': ['spd_p', 'spd_k', 'lariat_p', 'lariat_k'],                 # SPD, Lariat
}
_MOVESETS_LC = {k.lower(): v for k, v in MOVESETS.items()}

def build_sf_combos(num_step_frames=8, ego_char=None):
    """Return the ego character's tuned macro programs (each nsf frames). ego_char=None keeps the
    legacy behaviour (the 6 shoto motions), so callers that don't pass an ego are unchanged."""
    names = _SHOTO if ego_char is None else _MOVESETS_LC.get(str(ego_char).lower(), _SHOTO)
    return [MACRO_DEFS[n](num_step_frames) for n in names]


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