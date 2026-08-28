"""Reusable special-move detection harness.

Turns "did move X fire?" into a measured firing-RATE, robustly and char-agnostically.
Fixes the detection traps that caused so much thrash:
  - char-specific status codes            -> no hardcoded codes; "special = status the char's NORMALS never produce"
  - jump/normal confounds the new-status  -> baseline includes standing, crouching AND jumping normals
  - anti-airs/grabs don't connect at range -> positioning (ranged vs close); antiair reported by fire, not connect
  - single-shot flakiness / frame parity  -> N frame-alignments, report a rate, never a bool

Usage:
    python main/special_detection_harness.py <Char>                 # measure all of <Char>'s specials
    python main/special_detection_harness.py <Char> --render <Move> # save a montage of one firing trial
"""
import sys, argparse
import numpy as np
from common.const import sf_game, DIRECTIONS_BUTTONS, ATTACKS_BUTTONS
from train_ma import make_env, _build_states_from_roster

N_ATT = len(ATTACKS_BUTTONS)
def A(d, a=0):                      # (direction_idx, attack_idx) -> flat factored action index
    return d * N_ATT + a
NEUT = A(0, 0)
RIGHT, LEFT, DOWN, UP = A(4), A(3), A(2), A(1)
DB = A(7, 0)                        # DOWN-LEFT = down-back (ego faces right in <Char>VsRyu)
FWDP, FWDK = A(4, 5), A(4, 2)       # forward + punch(X) / kick(A)
UPK, UPP = A(1, 2), A(1, 5)         # up + kick / punch  (charge-down releases)
PUNCH, KICK = A(0, 5), A(0, 2)      # X-only / A-only
COMBO0 = len(DIRECTIONS_BUTTONS) * N_ATT + 0    # combo idx 0 = QCF+X (Hadouken/TigerShot/YogaFire/...)
ROT = [A(4), A(8), A(2), A(7), A(3), A(5), A(1), A(6)]  # 360 clockwise from forward

def prog_charge_fwd(): return [DB]*16 + [FWDP]
def prog_charge_down(): return [DB]*16 + [UPK]
def prog_motion():      return [COMBO0]
def prog_mashP():       return [PUNCH, NEUT]*12
def prog_mashK():       return [KICK, NEUT]*12
def prog_spd():         return ROT + [A(6, 5)]

# (move_name, program, move_type)   move_type -> positioning + how firing is judged
#   ranged : projectile/lunge, connects standing dummy at range (fire=new-status, connect=hp-drop)
#   antiair: anti-air special; won't connect a grounded dummy -> judged by FIRE only
#   close  : mash/grab; walk into range first, connect=hp-drop
MOVES = {
    'Ryu':    [('Hadouken', prog_motion(), 'ranged')],
    'Ken':    [('Hadouken', prog_motion(), 'ranged')],
    'Sagat':  [('TigerShot', prog_motion(), 'ranged')],
    'Dhalsim':[('YogaFire', prog_motion(), 'ranged')],
    'Guile':  [('SonicBoom', prog_charge_fwd(), 'ranged'), ('FlashKick', prog_charge_down(), 'antiair')],
    'ChunLi': [('SpinBirdKick', prog_charge_down(), 'antiair'), ('LightningKick', prog_mashK(), 'close')],
    'Blanka': [('RollingAttack', prog_charge_fwd(), 'ranged'), ('Electricity', prog_mashP(), 'close')],
    'EHonda': [('SumoHeadbutt', prog_charge_fwd(), 'ranged'), ('HundredHand', prog_mashP(), 'close')],
    'Balrog': [('DashStraight', prog_charge_fwd(), 'ranged')],
    'Vega':   [('RollCrystal', prog_charge_fwd(), 'ranged')],
    'MBison': [('PsychoCrusher', prog_charge_fwd(), 'ranged')],
    'Zangief':[('SPD', prog_spd(), 'close')],
}


class Harness:
    SPECIAL_FLAG_ADDR = 32773   # RAM idx (Genesis 0xFF8005): 0 = no special, non-zero = special active.
                                # char-agnostic + connect-independent (component-7 RAM-diff find).
    SETTLE = 90                 # >=80 needed for slow-intro chars (Zangief) to be controllable.

    def __init__(self, ego, opponent="Ryu"):
        self.ego = ego
        key = f"{ego.lower()}_{opponent.lower()}"
        state = _build_states_from_roster([ego], [opponent], "both")[key]
        # ego drives P1 (left seat of <ego>Vs<opponent>); build the per-ego / per-seat macro tables.
        self.env = make_env(sf_game, state_name=state, side="both", reset_type="round", rendering=False,
                            enable_combo=True, null_combo=False, transform_action=True, seed=0, reward_scale=1.0,
                            ego_char=ego, left_char=ego, right_char=opponent)()
        self.raw = self.env                       # underlying retro env, for RAM reads
        while hasattr(self.raw, "env"):
            self.raw = self.raw.env
        self.base = None

    def _flag(self):
        return int(self.raw.get_ram()[self.SPECIAL_FLAG_ADDR])

    def _step(self, ego_a, opp_a=NEUT, frames=None):
        o, r, ro, d, info = self.env.step(np.array([ego_a, opp_a]))
        return o, info

    def _settle(self, n=None):
        if n is None:
            n = self.SETTLE
        self.env.reset()
        info = None
        for _ in range(n):
            _, info = self._step(NEUT)
        return info

    # --- component 1: comprehensive per-char baseline (stand + crouch + JUMP normals) ---
    def build_baseline(self):
        base = set()
        for d in range(len(DIRECTIONS_BUTTONS)):          # neutral + all 8 directions
            self._settle(6)
            for _ in range(5): base.add(self._step(A(d, 0))[1].get('agent_status'))
        for att in range(1, N_ATT):                       # standing & crouching normals
            self._settle()
            for _ in range(4): base.add(self._step(A(0, att))[1].get('agent_status'))
            for _ in range(4): base.add(self._step(A(2, att))[1].get('agent_status'))
        for att in range(1, N_ATT):                       # JUMPING normals (fixes the up+kick confound)
            self._settle()
            self._step(UP)
            for _ in range(4): base.add(self._step(A(0, att))[1].get('agent_status'))
        base.discard(None)
        self.base = base
        return base

    # --- components 3+4: one trial -> (fired, connected, new_statuses) ---
    def trial(self, prog, move_type, offset):
        self._settle(self.SETTLE + offset)                 # vary frame alignment (slow-intro safe)
        if move_type == 'close':
            for _ in range(20): self._step(RIGHT)          # approach for grabs/mash
        hp0 = self._step(NEUT)[1].get('enemy_hp'); emin = hp0; seen = set(); flag = 0
        for a in prog:
            info = self._step(a)[1]; seen.add(info.get('agent_status')); emin = min(emin, info.get('enemy_hp')); flag = max(flag, self._flag())
        for _ in range(30):
            info = self._step(NEUT)[1]; seen.add(info.get('agent_status')); emin = min(emin, info.get('enemy_hp')); flag = max(flag, self._flag())
        fired = flag != 0                                  # PRIMARY: RAM special flag (bulletproof, char-agnostic)
        return fired, (hp0 - emin) > 0, sorted(x for x in (seen - self.base) if x is not None)[:3]

    # --- component 3: stats over alignments -> rate ---
    def measure(self, prog, move_type, n_align=10):
        fired = conn = 0; ex = []
        for off in range(n_align):
            f, c, nw = self.trial(prog, move_type, off)
            fired += f; conn += c
            if nw: ex = nw
        return fired, conn, n_align, ex

    # --- component 6: on-demand render of one firing trial ---
    def render(self, prog, move_type, path):
        from PIL import Image
        self._settle(17)
        if move_type == 'close':
            for _ in range(8): self._step(RIGHT)
        frames = []
        rel = None
        for a in prog:
            o, _ = self._step(a); frames.append(o)
        rel = len(frames)
        for _ in range(30):
            o, _ = self._step(NEUT); frames.append(o)
        sel = [frames[int(k)] for k in np.linspace(max(0, rel-2), len(frames)-1, 8)]
        w, h = sel[0].shape[1], sel[0].shape[0]
        grid = Image.new('RGB', (w*4, h*2), 'black')
        for i, f in enumerate(sel):
            grid.paste(Image.fromarray(f), ((i % 4)*w, (i//4)*h))
        grid.resize((grid.width*2, grid.height*2), Image.NEAREST).save(path)

    def close(self):
        self.env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("char")
    ap.add_argument("--render", default=None, help="render one firing trial of this move name")
    args = ap.parse_args()

    h = Harness(args.char)
    h.build_baseline()
    if args.render:
        spec = next(m for m in MOVES[args.char] if m[0] == args.render)
        out = f"/tmp/{args.char}_{args.render}.png"
        h.render(spec[1], spec[2], out); print(f"rendered -> {out}")
    else:
        print(f"[{args.char}] baseline={len(h.base)} normal-statuses")
        for name, prog, mtype in MOVES[args.char]:
            fired, conn, n, ex = h.measure(prog, mtype)
            verdict = "FIRES" if fired >= max(1, n//5) else ("marginal" if fired else "NONE")
            conninfo = "" if mtype == 'antiair' else f" | connect {conn}/{n}"
            print(f"  {name:14} [{mtype:7}] fire {fired}/{n}{conninfo} -> {verdict}   new_status~{ex}")
    h.close()


if __name__ == "__main__":
    main()
