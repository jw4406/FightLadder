""" Adapter for Retro: https://github.com/Farama-Foundation/stable-retro. Game dynamics implementation is inspired from https://github.com/linyiLYi/street-fighter-ai. """
import os
import copy
import math
import time
import gzip
import gym
import torch
import hashlib
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Union
from gym.wrappers import LazyFrames, FrameStack
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Dict

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

from .const import *
from common.utils import linear_schedule


# Local Best Response (LBR) support. Cap the per-worker snapshot registry: each
# entry is ~1 MB of emulator state plus the frame stack, and a leaked key would
# grow without bound across a long evaluation.
LBR_MAX_SNAPSHOTS = 4

# SFWrapper attributes that em.set_state() does NOT restore but that determine
# reward and termination. prev_agent_hp/prev_enemy_hp drive the dense reward in
# step(); the remainder are the edge-triggered match/round state machine that
# update_status() advances and asserts on.
LBR_SF_ATTRS = (
    "prev_agent_hp", "prev_enemy_hp", "level", "save_state",
    "match_status", "round_status", "during_transation", "round_num",
    "extra_round", "total_timesteps", "aggresive_coeff", "dense_coeff",
)


class SFWrapper(gym.Wrapper):

    def __init__(self, env, side, reset_type="round", init_level=1, rendering=False, num_stack=12, num_step_frames=8, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False, counterhit_kappa=0.0, trade_kappa=0.0, pressure_beta=0.0, pressure_range=0.0, attack_statuses=(), reset_close_range=0.0, close_max_steps=40, reward_scale=0.001, aggresive_coeff=1.0, decision_timing="off", actionable_statuses=(), max_skip_frames=90, dwell_frames=1, sticky_prob=0.0, ego_char=None, left_char=None, right_char=None, charge_obs=False, charge_preserving_skip=False):
        super(SFWrapper, self).__init__(env)
        self.env = FrameStack(env, num_stack=num_stack)

        assert side in ['left', 'right', 'both'], "side should be 'left', 'right' or 'both'"
        self.side = side

        self.num_stack = num_stack
        self.num_step_frames = num_step_frames
        # Motion inputs are rebuilt to fill exactly num_step_frames, so the
        # frame skip is adjustable without breaking the combo assert.
        # Per-SEAT macro tables: P1 is decoded with the LEFT char's specials and P2 with the RIGHT
        # char's, so an opponent's combos aren't mis-decoded with the learner's table. self.sf_combos
        # (the ego/learner's table) sizes the action space. None -> legacy shoto motions.
        self.sf_combos_p1 = build_sf_combos(num_step_frames, left_char if left_char is not None else ego_char)
        self.sf_combos_p2 = build_sf_combos(num_step_frames, right_char if right_char is not None else ego_char)
        self.sf_combos = build_sf_combos(num_step_frames, ego_char)

        # aggresive_coeff weights damage DEALT vs damage TAKEN in both the dense
        # and terminal rewards. 1.0 = the historical zero-sum game (default). The
        # FightLadder paper uses 3 to incentivise combat, which makes the game
        # GENERAL-SUM (r_ego + r_adv = (a-1)(D_e+D_a) != 0), so minimax-Q does not
        # apply to a != 1 -- an a=3 arm is for measuring STATE VISITATION only.
        self.aggresive_coeff = float(aggresive_coeff)
        self.dense_coeff = 1.0

        # CONTACT-DENSITY REWARD VARIANTS. Both default to 0.0, which leaves the
        # in-fight reward bitwise unchanged.
        #
        # The problem they address: the dense reward is D_e - D_a, which is
        # IDENTICALLY ZERO whenever no damage lands -- for all 484 joint actions
        # at once. On such a state the payoff is constant in the actions, so the
        # ANOVA interaction term gamma is exactly zero and a joint-action critic
        # has nothing to represent. Damage is the only channel by which the
        # payoff depends on actions at all, which is why gamma is zero on ~94%
        # of states.
        #
        # Both terms below are ANTISYMMETRIC -- r_inverse = -r exactly -- so the
        # game stays ZERO-SUM, which minimax-Q requires. Note that the existing
        # `aggresive_coeff` is NOT: r + r_inv = (a-1)(D_e + D_a), zero only at
        # a == 1. It is the obvious dial for "more aggression" and it would
        # silently invalidate the operator under evaluation.
        #
        #   counterhit_kappa  weights damage landed while the RECIPIENT is in an
        #                     attack state. A counter-hit is the purest joint
        #                     event in the game -- neither action alone predicts
        #                     it -- so this raises gamma SPECIFICALLY rather
        #                     than just raising contact.
        #   pressure_beta     antisymmetric bonus for being the one in range and
        #                     attacking. Raises contact directly. This one
        #                     CHANGES THE GAME (it is not potential-based), so
        #                     the equilibrium moves and prior numbers do not
        #                     transfer.
        #   trade_kappa       scales the whole exchange by whether BOTH sides
        #                     are attacking. This is the only one of the three
        #                     that is JOINT BY CONSTRUCTION: the multiplier is a
        #                     PRODUCT of the two players' indicators, so it lands
        #                     in the ANOVA interaction term rather than in the
        #                     main effects. pressure_beta, by contrast, is a
        #                     DIFFERENCE of per-player indicators -- additive by
        #                     construction, so it can only dilute gamma's share.
        # START-STATE INTERVENTION. Measured scaling law (contact_density.py,
        # 1600 policy-free roots): as root separation |agent_x - enemy_x| falls
        # from 185-210px to 0-64px, contact goes 3.4% -> 18.8% and the
        # interaction magnitude |gamma| goes 0.0039 -> 0.0316, an 8x swing. This
        # walks the fighters together at the start of each round so training
        # BEGINS in the regime where the joint payoff actually has structure.
        #
        # It touches no reward, so the game stays EXACTLY zero-sum and the
        # payoff is untouched -- unlike the three reward variants, all of which
        # were measured to DILUTE gamma's share rather than raise it.
        #
        # CAVEAT worth stating: this changes where each round STARTS, not where
        # the equilibrium lives. Agents remain free to retreat, so whether the
        # contact rate survives training is the open question, not a given.
        self.reset_close_range = float(reset_close_range)
        self.close_max_steps = int(close_max_steps)
        self.counterhit_kappa = float(counterhit_kappa)
        self.trade_kappa = float(trade_kappa)
        self.pressure_beta = float(pressure_beta)
        self.pressure_range = float(pressure_range)
        self.attack_statuses = frozenset(int(x) for x in attack_statuses)
        # DECISION TIMING. 'off' = fixed num_step_frames clock (default, inert).
        # 'ego'/'joint' = after applying the action, keep stepping NEUTRAL until
        # the ego (or either player) is actionable again, so decisions land on
        # frames where the action actually changes the render. Measured: 81% of a
        # SPAR ego's decision points are non-actionable (agent_status in a locked
        # animation) and collapse all 21 actions to 1 identical frame; on
        # ACTIONABLE frames the current pixel pipeline already resolves ~11/21.
        # actionable_statuses is the empirical set (obs_attribution --by_status);
        # it is a STRONG but IMPERFECT proxy (the byte spans startup/free
        # sub-phases), so validate the distinctness jump before trusting it.
        assert decision_timing in ("off", "ego", "joint"), decision_timing
        self.decision_timing = decision_timing
        self.actionable_statuses = frozenset(int(x) for x in actionable_statuses)
        self.max_skip_frames = int(max_skip_frames)
        # DWELL: require the ego to be actionable for this many CONSECUTIVE frames
        # before returning, so control comes back PAST the recovery-settle rather
        # than at the first frame the status byte flips actionable (the byte spans
        # startup/settle/free sub-phases -- the imperfect-proxy tail). 1 = return
        # on the first actionable frame (identical to no dwell).
        self.dwell_frames = max(1, int(dwell_frames))
        self._last_skip = 0
        # COUNTERFACTUAL (opt-in, default off -> bit-identical to before): during the
        # decision-timing skip, hold each seat's last-commanded DIRECTION buttons (attacks
        # masked out) instead of neutral, so a held charge survives the skip. Lets charge
        # chars actually fire Flash Kick / Sonic Boom under decision timing. See memory
        # decision-timing-disables-charge-specials.
        self.charge_preserving_skip = bool(charge_preserving_skip)
        if self.decision_timing != "off" and not self.actionable_statuses:
            raise ValueError(
                "decision_timing needs actionable_statuses; derive them with "
                "obs_attribution.py --by_status (statuses whose everything-distinct "
                "is high) rather than guessing.")
        if (self.counterhit_kappa or self.trade_kappa or self.pressure_beta) and not self.attack_statuses:
            raise ValueError(
                "counterhit_kappa/pressure_beta need attack_statuses; derive them "
                "with contact_density.py --mode analyze rather than guessing.")
        if self.pressure_beta and self.pressure_range <= 0:
            raise ValueError("pressure_beta needs a positive pressure_range")

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_agent_hp = self.full_hp
        self.prev_enemy_hp = self.full_hp

        # self.observation_space = Box(low=0, high=255, shape=(100, 128, 3 * self.num_stack), dtype=np.uint8)
        self.observation_space = Box(low=0, high=255, shape=(100, 128, len(range(0, self.num_stack, self.num_step_frames // 2))), dtype=np.uint8)
        self.action_dim = 12 + 3 if (enable_combo or null_combo) else 12 # 3 bits for combos
        if transform_action:
            # self.action_space = MultiDiscrete([len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS) + len(COMBOS) for _ in range(self.players)])
            # Factored (flat-encoded) action space: one scalar index over all (direction, attack)
            # PAIRS so the agent can press a direction AND a button on the same frame -- required for
            # charge-move releases (forward+punch), command normals, etc. -- plus n_combo scripted
            # shoto-motion shortcuts appended after the pairs. A single scalar per player is preserved
            # so the 2-player hstack assembly in algorithms.py stays unchanged.
            n_dir, n_att, n_combo = len(DIRECTIONS_BUTTONS), len(ATTACKS_BUTTONS), len(self.sf_combos)
            self.action_space = MultiDiscrete([n_dir * n_att + n_combo]) if (enable_combo or null_combo) else MultiDiscrete([n_dir * n_att])
            def action_transformer(action):
                players_action = []
                n_dir, n_att = len(DIRECTIONS_BUTTONS), len(ATTACKS_BUTTONS)
                for player_action in action:
                    if player_action >= n_dir * n_att:                 # scripted shoto-motion combo
                        button_bits = [0 for _ in range(12)]
                        combo_bits = [int(i) for i in np.binary_repr(player_action - n_dir * n_att).zfill(3)]
                    else:                                              # direction + attack, SIMULTANEOUS
                        direction_buttons = DIRECTIONS_BUTTONS[player_action // n_att]
                        attack_buttons = ATTACKS_BUTTONS[player_action % n_att]
                        button_bits = [int(b in direction_buttons + attack_buttons) for b in BUTTONS]
                        combo_bits = [1 for _ in range(3)]             # 7 >= len(sf_combos) => "no combo"
                    players_action.append(np.array(button_bits + combo_bits))
                return np.hstack(players_action)
            self.action_transformer = action_transformer
        else:
            # self.action_space = MultiBinary(self.players * self.action_dim)
            self.action_space = MultiBinary(self.action_dim)
            self.action_transformer = None
        
        # Per-emulator-frame hook. An agent step advances num_step_frames (8)
        # emulator frames, so anything sampled once per agent STEP is sampled at
        # stride 8 and ALIASES OUT events shorter than that -- a special move's
        # active frames last ~2-4. RamObsWrapper installs a callback here when it
        # needs sub-step resolution. None => zero cost, the historical path.
        self.ram_tap = None

        # RENDER CAPTURE (duel/eval video ONLY; off by default => zero training
        # cost, no effect on obs/reward/done). When on, step() records every
        # emulator frame's rgb so decision-timing skips render smoothly instead
        # of teleporting between decision points. Frames are pulled out-of-band
        # via pop_render_frames() (SubprocVecEnv.env_method), not through obs.
        self._render_capture = False
        self._render_frames = []

        self.reset_type = reset_type
        self.rendering = rendering

        self.init_level = init_level
        self.state_dir = state_dir
        self.verbose = verbose
        self.enable_combo = enable_combo
        self.null_combo = null_combo
        self.reward_scale = float(reward_scale)
        # Sticky-action exploration: per-player, repeat the previous executed action w.p. sticky_prob.
        # 0.0 = off (default). Payoff-eval envs are forced to 0 (see league.construct_agent).
        self.sticky_prob = float(sticky_prob)
        self._prev_raw_action = None
        # Curriculum special-move reward bonus: added to the firing seat's reward on the RISING EDGE of a
        # special (status high byte 0x0C). 0.0 = off (default); the training callback anneals it to 0.
        self.special_bonus_coef = 0.0
        self._prev_p1_special = False
        self._prev_p2_special = False
        # Demonstration-guided injection: with prob inject_prob, force the EGO seat through a scripted
        # charge->release so a charge policy EXPERIENCES a fire; the executed action is reported via
        # info['injected_action'] so the algorithm can store it in the buffer (imitation). Annealed to 0.
        self.inject_prob = 0.0
        self._inject_pos = -1                # -1 = not injecting; else index into the program
        self._injected_action = None
        _ego_is_right = (ego_char is not None and right_char is not None
                         and str(ego_char).lower() == str(right_char).lower())
        self._ego_seat_idx = 1 if _ego_is_right else 0
        self._inject_program = injection_program(
            ego_char, "right" if _ego_is_right else "left",
            len(DIRECTIONS_BUTTONS), len(ATTACKS_BUTTONS)) if ego_char is not None else None
        # Charge-hold reward (4th lever): credit the ego for HOLDING its charge direction (building a
        # charge), capped at the charge threshold so it rewards a FULL charge, not holding forever. The
        # charge direction + length come from the injection program's charge steps. Annealed to 0.
        self.charge_bonus_coef = 0.0
        self._charge_count = 0
        self._charge_this_step = 0
        # Charge-in-observation (the OBSERVABILITY fix): paint a charge-progress bar into the image so the
        # near-memoryless vision policy can SEE how long it has charged (obs window ~1.5 steps << 16-step
        # charge). Opt-in; the charge counter is then tracked every step (not just when the reward is on).
        self.charge_obs = bool(charge_obs)
        if self._inject_program is not None:
            self._charge_dir = int(self._inject_program[0]) // len(ATTACKS_BUTTONS)   # e.g. down-back / down
            self._charge_threshold = len(self._inject_program) - 1                    # # of charge steps
        else:
            self._charge_dir = None
            self._charge_threshold = 16

    def save_state_to_file(self, name="test.state"):
        content = self.env.em.get_state()
        print(f"Save state to {os.path.join(self.state_dir, name)}")
        with gzip.open(os.path.join(self.state_dir, name), 'wb') as f:
            f.write(content)

    # --- Local Best Response (LBR) snapshot/restore ---------------------------
    # Save and rewind a live env mid-episode so a one-step lookahead can try
    # every action from the same state. em.set_state() restores the emulator and
    # none of the Python state above it, so we capture both here.
    #
    # These stay public on purpose: SubprocVecEnv2P.env_method reaches them via
    # gym.Wrapper.__getattr__, which raises on underscore-prefixed names. Every
    # return value crosses a multiprocessing pipe, so none of them may return the
    # snapshot itself (~3 MB pickled) -- it stays in the worker.

    def _lbr_store(self):
        # Reach into __dict__ directly: gym.Wrapper.__getattr__ forwards unknown
        # attributes down the wrapper chain, so a plain read of a not-yet-created
        # attribute resolves against FrameStack/RetroEnv and raises instead of
        # creating the registry.
        return self.__dict__.setdefault("lbr_snapshots", {})

    def lbr_snapshot(self, key="lbr_root"):
        store = self._lbr_store()
        if key not in store and len(store) >= LBR_MAX_SNAPSHOTS:
            # Deliberately fatal. _worker2p does not guard the env_method call, so
            # this kills the worker and the parent sees an opaque EOFError. That is
            # the right trade for a leak guard: silently evicting an entry could
            # drop the root snapshot mid-branch-loop and corrupt results with no
            # signal. Callers must lbr_drop() in a finally block, never catch this.
            raise RuntimeError(
                f"LBR snapshot registry full ({LBR_MAX_SNAPSHOTS} keys: {sorted(store)}). "
                f"Call lbr_drop() before taking another snapshot."
            )
        retro_env = self.env.env
        # Read the SFWrapper attributes out of __dict__ rather than with getattr:
        # getattr would silently forward down the chain if one were unset, and we
        # want a missing field to fail loudly here rather than corrupt a reward later.
        store[key] = {
            "em": retro_env.em.get_state(),
            "img": None if retro_env.img is None else retro_env.img.copy(),
            "sf": {k: self.__dict__[k] for k in LBR_SF_ATTRS},
            "frames": copy.deepcopy(self.env.frames),
        }
        return None

    def lbr_restore(self, key="lbr_root"):
        snap = self._lbr_store()[key]
        retro_env = self.env.env
        # Order matters. update_ram() must follow set_state(), or info is read off
        # the pre-restore RAM. Do NOT call data.reset() here: that clears the
        # scenario delta trackers, which is reset semantics, not continuation.
        # Those trackers going stale is harmless only because SFWrapper.step
        # discards retro's own _reward/_done -- do not "fix" that discard.
        retro_env.em.set_state(snap["em"])
        retro_env.data.update_ram()
        retro_env.img = None if snap["img"] is None else snap["img"].copy()
        self.__dict__.update(snap["sf"])
        self.env.frames = copy.deepcopy(snap["frames"])
        return None

    def lbr_drop(self, key=None):
        store = self._lbr_store()
        if key is None:
            dropped = len(store)
            store.clear()
            return dropped
        return int(store.pop(key, None) is not None)

    def lbr_has(self, key="lbr_root"):
        return key in self._lbr_store()

    def lbr_fingerprint(self):
        """Digest of the restored machine state. A pipe-cheap equality check for
        the restore-fidelity test.

        Hashes RAM rather than em.get_state(): the savestate blob is NOT
        idempotent through set_state(). Measured on this core, get_state() ->
        set_state() -> get_state() differs in 76% of bytes past offset 144228
        with no stepping in between, while RAM, the frame stack, and every
        wrapper attribute stay byte-equal and replay stays bit-identical. That
        trailing region is emulator scratch, so hashing the blob would report
        spurious mismatches.
        """
        retro_env = self.env.env
        h = hashlib.md5()
        h.update(np.ascontiguousarray(retro_env.get_ram()).tobytes())
        for frame in self.env.frames:
            h.update(np.ascontiguousarray(frame).tobytes())
        h.update(repr([self.__dict__[k] for k in LBR_SF_ATTRS]).encode())
        return h.hexdigest()

    def lbr_obs_variants(self):
        """Digests of the CURRENT frame stack under alternative subsamplings.

        Measured: RAM distinguishes 441 = 21^2 successors at a median decision
        point (every genuinely distinct joint action; 0 and 9 are byte-identical
        no-ops), while the agent's observation distinguishes ONE. This attributes
        that 441 -> 1 collapse to the three independent downsamplings in
        `_get_obs`:

            temporal  frames[::4] -> stack indices 0, 4, 8 of 12. The NEWEST
                      THREE (9,10,11) are never shown, so the freshest content
                      is already 3 emulator frames stale -- and an action's
                      effect appears LAST, not first.
            spatial   o[::2, ::2] -> 100x128 from 200x256
            channel   `i % 3`: frame 0 contributes only RED, frame 4 only GREEN,
                      frame 8 only BLUE. Time and colour are entangled.

        `recent` is the interesting one: frames[3::4] -> indices 3, 7, 11. The
        SAME tensor shape and the same compute as today, only sampled to include
        the newest frame. If that alone recovers the distinctions, the fix costs
        nothing.

        Digests are computed HERE, in the worker, and only hex strings cross the
        pipe -- a full stack is ~1.8 MB and blobs must never be piped.
        """
        frames = [np.ascontiguousarray(f) for f in self.env.frames]
        step = max(1, self.num_step_frames // 2)
        sub = list(range(0, len(frames), step))            # 0, 4, 8
        rec = list(range(step - 1, len(frames), step))     # 3, 7, 11
        allf = list(range(len(frames)))

        def dig(idx, half=True, one_channel=True):
            h = hashlib.md5()
            for n, i in enumerate(idx):
                o = frames[i]
                o = o[::2, ::2] if half else o
                o = o[:, :, n % 3] if one_channel else o
                h.update(np.ascontiguousarray(o).tobytes())
            return h.hexdigest()

        return {
            "current":      dig(sub),                       # exactly _get_obs
            "recent":       dig(rec),                       # free: same shape
            "all_frames":   dig(allf),                      # temporal fixed
            "full_spatial": dig(sub, half=False),           # spatial fixed
            "all_channels": dig(sub, one_channel=False),    # channel fixed
            "everything":   dig(allf, half=False, one_channel=False),
        }

    def lbr_probe_sample(self, ds=4):
        """One probe row: the CURRENT (color-cycled) and ALL_CHANNELS (full RGB)
        observations of this frame stack, plus RAM ground-truth targets.

        `current` reproduces _get_obs exactly (frames[::step], o[::2,::2], channel
        n%3 -> K=3). `all_channels` keeps ALL 3 RGB per sampled frame (-> 3*K), so
        a probe on the two isolates what the colour reduction costs. Both are
        further decimated by `ds` HERE so only a few KB cross the pipe (a full
        stack is ~1.8 MB and blobs must never be piped). Targets are the raw retro
        info vars the pixels are regressed onto (seat = which sprite is the agent;
        enemy_character = character ID)."""
        frames = [np.ascontiguousarray(f) for f in self.env.frames]
        step = max(1, self.num_step_frames // 2)
        sub = list(range(0, len(frames), step))
        cur, allc = [], []
        for n, i in enumerate(sub):
            o = frames[i][::2, ::2][::ds, ::ds]     # _get_obs spatial, then probe ds
            cur.append(o[:, :, n % 3])              # single cycled channel
            allc.append(o)                          # full RGB
        info = self.env.env.data.lookup_all()
        keys = ("agent_x", "agent_y", "enemy_x", "enemy_y", "agent_status",
                "enemy_status", "enemy_character", "agent_hp", "enemy_hp")
        return {"current": np.stack(cur, axis=-1).astype(np.uint8),
                "all_channels": np.concatenate(allc, axis=-1).astype(np.uint8),
                "targets": {k: float(info[k]) for k in keys}}

    def lbr_ram(self):
        """Raw RAM as a uint8 array, for build_ram_mask.py.

        65,536 bytes DOES cross the pipe here, unlike the other lbr_* methods --
        acceptable because this runs offline in a one-shot mask build, never in
        a training or LBR loop.
        """
        return np.ascontiguousarray(self.env.env.get_ram()).astype(np.uint8)

    def lbr_state_variants(self):
        """Digest of PURE RAM plus the raw info variables, for choosing an
        observation type.

        Pixels resolve 1 of 21 action-distinct successors at a median decision
        point, at ANY resolution or frame coverage, so the discriminating state
        is not rendered. This exposes the two candidate replacements so their
        discriminating power can be measured before either is built:

            ram    md5 of get_ram() ALONE. `lbr_fingerprint` mixes RAM with the
                   frame stack and wrapper attrs, so it cannot attribute the 21
                   distinct successors to RAM by itself. This can.
            vals   the raw retro info variables, per key, so per-variable
                   discriminating power is visible -- if only agent_status moves,
                   a 14-dim vector suffices and 64KB of RAM is unnecessary.

        Cheap over the pipe: one hex string and ~14 scalars.
        """
        retro_env = self.env.env
        ram = np.ascontiguousarray(retro_env.get_ram())
        info = retro_env.data.lookup_all()
        keys = sorted(info.keys())
        return {"ram": hashlib.md5(ram.tobytes()).hexdigest(),
                "ram_bytes": int(ram.size),
                "keys": keys,
                "vals": [float(info[k]) for k in keys]}

    def lbr_config(self):
        """Scalar facts the LBR driver preflights on before stepping anything."""
        space = self.action_space
        return {
            "transform_action": self.action_transformer is not None,
            "action_space": type(space).__name__,
            "n_actions": int(space.nvec[0]) if hasattr(space, "nvec") else None,
            "reset_type": self.reset_type,
            "side": self.side,
            "init_level": self.init_level,
            "num_stack": self.num_stack,
            "num_step_frames": self.num_step_frames,
            "enable_combo": self.enable_combo,
        }

    def _paint_charge(self, img):
        """Paint a charge-progress bar into the image so the near-memoryless vision policy can SEE its
        charge level. WIDE + high-contrast (bottom 10 rows, full-width fill L->R = charge; whole strip
        white when fully charged) so the CNN can actually read it. Charge egos only, opt-in via charge_obs.
        Off -> untouched."""
        if not self.charge_obs or self._charge_dir is None:
            return img
        prog = min(1.0, self._charge_count / max(1, self._charge_threshold))
        H, W = img.shape[0], img.shape[1]
        img[H - 10:H, :, :] = 0                             # clear the bottom 10-row strip (black)
        w = int(round(prog * W))
        if w > 0:
            img[H - 10:H, 0:w, :] = 255                     # white fill proportional to charge
        return img

    def _get_obs(self, obs):
        # return np.concatenate([o[::2, ::2, :] for o in obs], axis=-1)
        if isinstance(obs, dict):
            # print(np.stack([o[::2, ::2, i % 3] for (i, o) in enumerate(obs['observations'][::(self.num_step_frames // 2)])], axis=-1).shape, flush=True)
            # print(obs['actions'], flush=True)
            # print(obs['actions'][::(self.num_step_frames // 2)], flush=True)
            # print(np.squeeze(obs['actions'][::(self.num_step_frames // 2)], axis=-1), flush=True)
            return {
                'observations': np.stack([o[::2, ::2, i % 3] for (i, o) in enumerate(obs['observations'][::(self.num_step_frames // 2)])], axis=-1),
                'actions': np.squeeze(obs['actions'][::(self.num_step_frames // 2)], axis=-1),
            }
        else:
            img = np.stack([o[::2, ::2, i % 3] for (i, o) in enumerate(obs[::(self.num_step_frames // 2)])], axis=-1)
            return self._paint_charge(img)

    def reset(self):
        obs = self.env.reset()
        self._prev_raw_action = None   # don't carry a sticky hold across episodes
        self._prev_p1_special = False  # reset special rising-edge tracking across episodes
        self._prev_p2_special = False
        self._inject_pos = -1          # don't carry a half-finished injection across episodes
        self._injected_action = None
        self._charge_count = 0         # reset charge-hold counter across episodes
        self._charge_this_step = 0

        self.prev_agent_hp = self.full_hp
        self.prev_enemy_hp = self.full_hp
        
        self.level = self.init_level # NOTE: only valid for some states
        self.save_state = False
        self.match_status = START_STATUS # NOTE: only valid for some states
        self.round_status = END_STATUS # NOTE: only valid for some states
        self.during_transation = True
        self.round_num = 0
        self.extra_round = False

        self.total_timesteps = 0
    
        out = self._get_obs(obs)
        if self.reset_close_range > 0:
            out = self._walk_close(out)
        return out

    def _walk_close(self, out):
        """Walk both fighters toward each other until they are within range.

        Goes through self.step() rather than the raw emulator so the round/match
        state machine, the frame stack and the RAM tap all advance exactly as
        they do in play. At round start both are at full hp with a full timer, so
        no damage or termination is possible during the walk.
        """
        if self.action_transformer is None:
            raise ValueError("reset_close_range needs --transform_action True "
                             "(it moves via DIRECTIONS_BUTTONS indices)")
        LEFT, RIGHT = 3 * len(ATTACKS_BUTTONS), 4 * len(ATTACKS_BUTTONS)  # flat-encoded dir-only (attack=neutral); DIRECTIONS_BUTTONS 3=LEFT, 4=RIGHT
        for _ in range(self.close_max_steps):
            d = self.env.env.data.lookup_all()
            ax, ex = d['agent_x'], d['enemy_x']
            if abs(ex - ax) <= self.reset_close_range:
                break
            act = np.array([RIGHT, LEFT]) if ex > ax else np.array([LEFT, RIGHT])
            res = self.step(act)
            out = res[0]
            if res[-2]:             # done -- should be unreachable at round start
                break
        # The walk consumed steps and hp is untouched, but prev_*_hp must match
        # the CURRENT hp or the first real step bills their difference as damage.
        d = self.env.env.data.lookup_all()
        self.prev_agent_hp = d['agent_hp']
        self.prev_enemy_hp = d['enemy_hp']
        return out

    def update_status(self, info, bonus=False):
        # info['level'] = self.level
        # info['match'] = 'start' if self.match_status == START_STATUS else 'end'
        # info['round'] = 'start' if self.round_status == START_STATUS else 'end'
        # print(info, flush=True)
        max_round = 1 if bonus else 3

        agent_hp = info['agent_hp']
        enemy_hp = info['enemy_hp']
        agent_victories = info['agent_victories']
        enemy_victories = info['enemy_victories']
        round_countdown = info['round_countdown']
        timesup = (round_countdown <= 0)

        if self.match_status == END_STATUS and (agent_victories == 0 and enemy_victories == 0):
            self.match_status = START_STATUS
            self.save_state = True
            self.level += 1
            self.round_num = 0
            self.extra_round = False
        elif self.match_status == START_STATUS and (self.round_num == max_round or agent_victories == 2 or enemy_victories == 2):
            self.match_status = END_STATUS
            if self.verbose:
                if agent_victories < enemy_victories:
                    print(f"Level {self.level} is over and player loses")
                elif agent_victories > enemy_victories:
                    print(f"Level {self.level} is over and player wins")
                else:
                    print(f"Draw level {self.level}")
                    if (not bonus) and (not self.extra_round):
                        self.match_status = START_STATUS
                        self.round_num -= 1
                        self.extra_round = True
                        print(f"One more round for draw level {self.level}")
        if self.round_status == END_STATUS and (agent_hp == self.full_hp and enemy_hp == self.full_hp and round_countdown > 0):
            self.round_status = START_STATUS
            self.prev_agent_hp = self.full_hp
            self.prev_enemy_hp = self.full_hp
        elif self.round_status == START_STATUS and ((agent_hp < 0 or enemy_hp < 0) or timesup):
            self.round_status = END_STATUS
            self.round_num += 1
            if self.verbose:
                if agent_hp < enemy_hp:
                    print(f"The round is over and player loses")
                elif agent_hp > enemy_hp:
                    print(f"The round is over and player wins")
                else:
                    print(f"Draw round")
        if self.match_status == END_STATUS:
            info['match'] = 'start' if self.match_status == START_STATUS else 'end'
            info['round'] = 'start' if self.round_status == START_STATUS else 'end'
            assert self.round_status == END_STATUS, info
        if self.round_status == START_STATUS:
            info['match'] = 'start' if self.match_status == START_STATUS else 'end'
            info['round'] = 'start' if self.round_status == START_STATUS else 'end'
            assert self.match_status == START_STATUS, info

    def set_sticky_prob(self, sticky_prob):
        """Runtime setter (kept for flexibility; eval gating is done at construction)."""
        self.sticky_prob = float(sticky_prob)

    def set_special_bonus_coef(self, coef):
        """Runtime setter for the annealed special-move reward bonus (0 = off). Driven by
        AnnealSpecialBonusCallback from the agent's global num_timesteps."""
        self.special_bonus_coef = float(coef)

    def set_inject_prob(self, prob):
        """Runtime setter for the annealed charge->release injection probability (0 = off)."""
        self.inject_prob = float(prob)

    def set_charge_bonus_coef(self, coef):
        """Runtime setter for the annealed charge-hold reward (0 = off)."""
        self.charge_bonus_coef = float(coef)

    def set_render_capture(self, on=True):
        """Enable/disable per-emulator-frame rgb capture for smooth video."""
        self._render_capture = bool(on)
        self._render_frames = []

    def pop_render_frames(self):
        """Return and clear the rgb frames captured since the last call."""
        frames = self._render_frames
        self._render_frames = []
        return frames

    def step(self, action):
        # Sticky-action exploration: with prob sticky_prob, per player, repeat the previously
        # EXECUTED raw (pre-transform) action. The rollout buffer stores the SAMPLED action, so PPO
        # sees this purely as environment stochasticity; correlated executed-action runs are what let
        # charge moves (long down-back holds) be discovered.
        if self.sticky_prob > 0.0 and self.action_transformer is not None:
            if self._prev_raw_action is not None:
                a = np.array(action, copy=True)
                for i in range(len(a)):
                    if np.random.random() < self.sticky_prob:
                        a[i] = self._prev_raw_action[i]
                action = a
            self._prev_raw_action = np.array(action, copy=True)
        # Charge-hold tracking: is the ego holding its charge direction this step? (credited in the reward
        # below, capped at a full charge). Uses the effective (post-sticky) action = the policy's choice.
        self._charge_this_step = 0
        if self._charge_dir is not None and (self.charge_bonus_coef > 0.0 or self.charge_obs):
            _a1 = np.atleast_1d(action)
            if len(_a1) > self._ego_seat_idx:
                _ego_a = int(_a1[self._ego_seat_idx])
                _holding = (_ego_a < len(DIRECTIONS_BUTTONS) * len(ATTACKS_BUTTONS)
                            and _ego_a // len(ATTACKS_BUTTONS) == self._charge_dir)
                self._charge_count = self._charge_count + 1 if _holding else 0
                self._charge_this_step = self._charge_count
        # Demonstration injection (after sticky, before transform): override the EGO seat's raw factored
        # action with the current step of the scripted charge->release program. Reported via
        # info['injected_action'] so the algorithm stores the injected action (not the sampled one).
        self._injected_action = None
        if (self._inject_program is not None and self.inject_prob > 0.0
                and self.action_transformer is not None and len(np.atleast_1d(action)) > self._ego_seat_idx):
            if self._inject_pos < 0 and np.random.random() < self.inject_prob:
                self._inject_pos = 0                     # start a fresh charge->release
            if self._inject_pos >= 0:
                action = np.array(action, copy=True)
                action[self._ego_seat_idx] = self._inject_program[self._inject_pos]
                self._injected_action = int(self._inject_program[self._inject_pos])
                self._inject_pos += 1
                if self._inject_pos >= len(self._inject_program):
                    self._inject_pos = -1                # program done
        if self.action_transformer is not None:
            action = self.action_transformer(action)
        if self.side == 'both':
            assert action.shape[-1] == 2 * self.action_dim, f"action.shape[-1]={action.shape[-1]}, 2 * self.action_dim={2 * self.action_dim}"
        else:
            assert action.shape[-1] == self.action_dim, f"action.shape[-1]={action.shape[-1]}, self.action_dim={self.action_dim}"

        if self.level in SF_BONUS_LEVEL: # skip bonus level
            skip_level = self.level
            no_op = np.zeros_like(action[:24])
            while self.level == skip_level:
                obs, _reward, _done, info = self.env.step(no_op)
                self.update_status(info, bonus=True)
            if self.verbose:
                print(f"Skip bonus level {skip_level}")
        
        custom_done = False

        if self.side == 'left':
            action[3] = 0 # Filter out the "START/PAUSE" button
            if self.enable_combo:
                combo_id = int(4 * action[-3] + 2 * action[-2] + action[-1])
            else:
                combo_id = len(self.sf_combos)
            if combo_id >= len(self.sf_combos):
                action_seq = [np.hstack([action[:12], np.zeros_like(action[:12])]) for _ in range(self.num_step_frames)]
            else:
                combo = self.sf_combos[combo_id]
                assert self.num_step_frames == len(combo)
                action_seq = combo
                action_seq = [np.hstack([combo[t], np.zeros_like(combo[t])]) for t in range(self.num_step_frames)]
        elif self.side == 'right':
            action[3] = 0 # Filter out the "START/PAUSE" button
            if self.enable_combo:
                combo_id = int(4 * action[-3] + 2 * action[-2] + action[-1])
            else:
                combo_id = len(self.sf_combos)
            if combo_id >= len(self.sf_combos):
                action_seq = [np.hstack([np.zeros_like(action[:12]), action[:12]]) for _ in range(self.num_step_frames)]
            else:
                combo = self.sf_combos[combo_id]
                assert self.num_step_frames == len(combo)
                action_seq = combo
                action_seq = [np.hstack([np.zeros_like(combo[t]), combo[t]]) for t in range(self.num_step_frames)]
        else:
            action[3] = 0 # Filter out the "START/PAUSE" button
            action[self.action_dim + 3] = 0
            tables = [self.sf_combos_p1, self.sf_combos_p2]   # decode each seat with ITS char's macros
            if self.enable_combo:
                combo_ids = [int(4 * action[self.action_dim - 3] + 2 * action[self.action_dim - 2] + action[self.action_dim - 1]), int(4 * action[-3] + 2 * action[-2] + action[-1])]
            else:
                combo_ids = [len(tables[0]), len(tables[1])]
            action_seqs = []
            for player_id, combo_id in enumerate(combo_ids):
                table = tables[player_id]
                if combo_id >= len(table):
                    action_seq = [action[player_id * self.action_dim : player_id * self.action_dim + 12] for _ in range(self.num_step_frames)]
                else:
                    combo = table[combo_id]
                    assert self.num_step_frames == len(combo)
                    action_seq = combo
                action_seqs.append(action_seq)
            action_seq = [np.hstack([action_1, action_2]) for action_1, action_2 in zip(action_seqs[0], action_seqs[1])]
        
        for i in range(self.num_step_frames):
            # Keep the button pressed for (num_step_frames - 1) frames.
            obs, _reward, _done, info = self.env.step(action_seq[i])
            if self.ram_tap is not None:
                self.ram_tap()          # AFTER the frame, so it sees post-frame RAM
            self.update_status(info)
            if self._render_capture:
                self._render_frames.append(self.env.render(mode='rgb_array'))
            if self.rendering:
                self.env.render()
                time.sleep(0.01)

        # DECISION TIMING: hold NEUTRAL past the locked (non-actionable) frames
        # so the observation the agent next sees is one where its action can
        # actually change the render. Reward accrues over these frames (the
        # opponent can punish during recovery), which is captured because the hp
        # deltas below read the POST-skip hp. Round-ending frames stop the skip.
        self._last_skip = 0
        if self.decision_timing != "off":
            neutral = np.zeros_like(action_seq[0])
            if self.charge_preserving_skip:
                # hold each seat's last-commanded DIRECTION buttons (UP/DOWN/LEFT/RIGHT =
                # per-player indices 4..7), attacks masked to 0, so a charge survives the skip.
                _hold = np.array(action_seq[-1]).copy()
                _keep = np.zeros(len(_hold), dtype=bool)
                _nper = len(_hold) // 2
                for _b in (0, _nper):
                    _keep[_b + 4:_b + 8] = True
                neutral = (np.array(action_seq[-1]) * _keep).astype(neutral.dtype)
            consec = 0                       # consecutive actionable frames so far
            while True:
                a_act = int(info['agent_status']) in self.actionable_statuses
                e_act = int(info['enemy_status']) in self.actionable_statuses
                ready = a_act if self.decision_timing == "ego" else (a_act or e_act)
                consec = consec + 1 if ready else 0
                if consec >= self.dwell_frames or info['agent_hp'] < 0 \
                        or info['enemy_hp'] < 0 or info['round_countdown'] <= 0 \
                        or self._last_skip >= self.max_skip_frames:
                    break
                obs, _reward, _done, info = self.env.step(neutral)
                if self.ram_tap is not None:
                    self.ram_tap()
                self.update_status(info)
                if self._render_capture:
                    self._render_frames.append(self.env.render(mode='rgb_array'))
                if self.rendering:
                    self.env.render()
                    time.sleep(0.01)
                self._last_skip += 1

        agent_hp = info['agent_hp']
        enemy_hp = info['enemy_hp']
        agent_victories = info['agent_victories']
        enemy_victories = info['enemy_victories']
        round_countdown = info['round_countdown']
        timesup = (round_countdown <= 0)

        self.total_timesteps += self.num_step_frames + self._last_skip
        
        if self.during_transation and (self.match_status == END_STATUS or self.round_status == END_STATUS):
            # During transation between episodes, do nothing
            custom_done = False
            custom_reward = 0
            custom_reward_inverse = 0
            if (enemy_victories == 2) or ((self.match_status == END_STATUS) and (enemy_victories >= agent_victories)): # also need to handle 2nd condition during transation
                # Player loses the game
                custom_done = not self.reset_type == "never"
            if (agent_victories == 2) or ((self.match_status == END_STATUS) and (agent_victories > enemy_victories)): # also need to handle 2nd condition during transation
                # Player wins the match
                custom_done = self.reset_type == "match"
                if self.level == 15:
                    print(f"Player wins the game")
                    custom_done = True
                    self.save_state = True
                    self.level += 1
        else:
            self.during_transation = False
            # if self.save_state and self.state_dir is not None:
            #     self.save_state = False
            #     self.save_state_to_file(f"Level{self.level}.{self.total_timesteps}.state")

            if (agent_hp < 0 and enemy_hp < 0) or (timesup and agent_hp == enemy_hp):
                custom_reward = -1
                custom_reward_inverse = 1
                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    custom_done = False
                    self.during_transation = True
            elif agent_hp < 0 or (timesup and agent_hp < enemy_hp):
                custom_reward = -math.pow(self.full_hp, (enemy_hp + 1) / (self.full_hp + 1))     
                custom_reward_inverse = math.pow(self.full_hp, (enemy_hp + 1) / (self.full_hp + 1)) * self.aggresive_coeff
                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    self.during_transation = True
                    if (enemy_victories >= 2) or ((self.match_status == END_STATUS) and (enemy_victories >= agent_victories)): # also need to handle 2nd condition during transation
                        # Player loses the game
                        # if self.verbose:
                        #     print("Player loses the game")
                        custom_done = not self.reset_type == "never"
            elif enemy_hp < 0 or (timesup and agent_hp > enemy_hp):
                custom_reward = math.pow(self.full_hp, (agent_hp + 1) / (self.full_hp + 1)) * self.aggresive_coeff
                custom_reward_inverse = -math.pow(self.full_hp, (agent_hp + 1) / (self.full_hp + 1))

                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    self.during_transation = True
                    if (agent_victories >= 2) or ((self.match_status == END_STATUS) and (agent_victories > enemy_victories)): # also need to handle 2nd condition during transation
                        # Player wins the match
                        # if self.verbose:
                        #     print("Player wins the match")
                        custom_done = self.reset_type == "match"
                        if self.level == 15:
                            print(f"Player wins the game")
                            custom_done = True
                            self.save_state = True
                            self.level += 1
            # While the fighting is still going on
            else:
                D_e = self.prev_enemy_hp - enemy_hp
                D_a = self.prev_agent_hp - agent_hp
                # counter-hit: scale each side's damage by whether the side that
                # RECEIVED it was mid-attack. w_e weights damage dealt TO the
                # enemy, so it reads the ENEMY's status.
                w_e = w_a = 1.0
                if self.counterhit_kappa:
                    w_e += self.counterhit_kappa * (info['enemy_status'] in self.attack_statuses)
                    w_a += self.counterhit_kappa * (info['agent_status'] in self.attack_statuses)
                # trade: a PRODUCT of both indicators, so it scales the whole
                # antisymmetric exchange and stays zero-sum while contributing
                # to the interaction term rather than the main effects.
                trade = 1.0
                if self.trade_kappa:
                    trade += self.trade_kappa * (
                        (info['agent_status'] in self.attack_statuses) and
                        (info['enemy_status'] in self.attack_statuses))
                custom_reward = trade * self.dense_coeff * (self.aggresive_coeff * D_e * w_e - D_a * w_a)
                custom_reward_inverse = trade * self.dense_coeff * (self.aggresive_coeff * D_a * w_a - D_e * w_e)
                if self.pressure_beta:
                    in_rng = abs(info['agent_x'] - info['enemy_x']) <= self.pressure_range
                    p_a = float(in_rng and (info['agent_status'] in self.attack_statuses))
                    p_e = float(in_rng and (info['enemy_status'] in self.attack_statuses))
                    custom_reward += self.pressure_beta * (p_a - p_e)
                    custom_reward_inverse += self.pressure_beta * (p_e - p_a)
                self.prev_agent_hp = agent_hp
                self.prev_enemy_hp = enemy_hp
                custom_done = False

        # if custom_reward != 0:
        #     print("reward:{}".format(custom_reward))

        info['level'] = self.level
        info['match'] = 'start' if self.match_status == START_STATUS else 'end'
        info['round'] = 'start' if self.round_status == START_STATUS else 'end'
        # -1 = no injection this step; else the FACTORED action the ego seat was forced to execute
        # (the algorithm stores this in the buffer so the policy imitates the charge->release).
        info['injected_action'] = -1 if self._injected_action is None else self._injected_action
        # frames the decision-timing loop held NEUTRAL past the action (lockdown duration proxy)
        info['last_skip'] = int(self._last_skip)
        if custom_done:
            info['outcome'] = 'win' if (agent_hp > enemy_hp) else ('lose' if (agent_hp < enemy_hp) else 'draw')

        # reward_scale defaults to 0.001 -> bitwise the historical behaviour.
        # It sets the magnitude the VALUE optimizer sees: Adam is scale-invariant
        # except through eps, and at 0.001 the value grads' second moment sqrt(v)
        # falls to ~1e-9 (below eps=1e-8), knocking the value head out of Adam's
        # adaptive regime. Unscaling (1.0) restores it. Measured, not assumed.
        # Curriculum special-move bonus: reward the RISING EDGE of a special once (not every animation
        # frame), added pre-scale so it rides the same reward_scale as damage. Non-zero-sum by design;
        # the training callback anneals special_bonus_coef to 0, restoring the true zero-sum game.
        # Detection uses the VALIDATED raw-RAM special-active flag (see special_detection memory):
        # get_ram()[32770]==12 (P1/agent), get_ram()[33410]==12 (P2/enemy). Sustained across the move,
        # so a once-per-step read at reward time catches it. NOTE: info['agent_status'] is byte-swapped
        # vs get_ram (the special byte is the LOW byte there) -- read get_ram to stay consistent.
        if self.special_bonus_coef > 0.0:
            _ram = self.unwrapped.get_ram()
            p1_fire_step = int(_ram[32770]) == 12
            p2_fire_step = int(_ram[33410]) == 12
            if p1_fire_step and not self._prev_p1_special:
                custom_reward += self.special_bonus_coef
            if p2_fire_step and not self._prev_p2_special:
                custom_reward_inverse += self.special_bonus_coef
            self._prev_p1_special = p1_fire_step
            self._prev_p2_special = p2_fire_step
        else:
            self._prev_p1_special = False
            self._prev_p2_special = False

        # Charge-hold reward: credit the ego for building a charge, CAPPED at a full charge (so it is not
        # rewarded for holding forever). The larger fire bonus makes charge->fire beat charge-farming.
        if self.charge_bonus_coef > 0.0 and 0 < self._charge_this_step <= self._charge_threshold:
            if self._ego_seat_idx == 0:
                custom_reward += self.charge_bonus_coef
            else:
                custom_reward_inverse += self.charge_bonus_coef

        rs = getattr(self, 'reward_scale', 0.001)
        if self.side == 'left':
            return self._get_obs(obs), rs * custom_reward, custom_done, info
        elif self.side == 'right':
            return self._get_obs(obs), rs * custom_reward_inverse, custom_done, info
        else:
            return self._get_obs(obs), rs * custom_reward, rs * custom_reward_inverse, custom_done, info


class Monitor2P(Monitor):
    
    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords, override_existing)
        
        self.rewards_other = None
        self.episode_returns_other = []
    
    def reset(self, **kwargs) -> GymObs:
        self.rewards_other = []
        return super().reset(**kwargs)
    
    def lbr_pause_monitor(self, paused: bool = True) -> bool:
        """Suspend episode bookkeeping so the env can be BRANCHED and rewound.

        Monitor2P corrupts a branching search in three separate ways, and all
        three are silent-to-fatal rather than obvious:

          1. `needs_reset` is set the moment any branch ends an episode, so the
             NEXT restore-and-step raises RuntimeError and kills the worker.
          2. every branch reward is appended to self.rewards, so the episode
             return reported for the real trajectory includes counterfactual
             rewards that were rewound away.
          3. a done branch writes a full episode record, producing phantom
             episodes in the monitor log and in rollout/ep_rew_mean.

        Pausing is preferable to snapshotting the monitor: during enumeration we
        want NO bookkeeping at all, so there is nothing to restore and no way for
        the two copies to drift. The counterfactual steps are counted separately
        by the algorithm as train/enum_env_steps.

        Reaches through env_method like the other lbr_* methods, so it must be
        public and must not start with an underscore.
        """
        self.__dict__["lbr_paused"] = bool(paused)
        return bool(paused)

    def lbr_monitor_paused(self) -> bool:
        return bool(self.__dict__.get("lbr_paused", False))

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.__dict__.get("lbr_paused", False):
            # No guard, no accumulation, no episode record -- see
            # lbr_pause_monitor. needs_reset is deliberately NOT set: the caller
            # is about to rewind this env to a pre-branch snapshot.
            return self.env.step(action)
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, reward_other, done, info = self.env.step(action)
        self.rewards.append(reward)
        self.rewards_other.append(reward_other)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_rew_other = sum(self.rewards_other)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "ro": round(ep_rew_other, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_returns_other.append(ep_rew_other)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, reward_other, done, info


class InfoObsWrapper(gym.Wrapper):
    """Replaces image observations with a normalized ego-centric feature vector from the info dict.
    Must wrap SFWrapper (which provides 2P step returns with info containing game state).
    Features 0-4 are always ego, 5-9 are always opponent, regardless of controller assignment.
    """

    _AGENT_KEYS = ['agent_hp', 'agent_x', 'agent_y', 'agent_status', 'agent_victories']
    _ENEMY_KEYS = ['enemy_hp', 'enemy_x', 'enemy_y', 'enemy_status', 'enemy_victories']
    _META_KEYS = ['enemy_character', 'round_countdown', 'reset_countdown', 'score']

    NORM_SCALES = np.array([
        176.0, 512.0, 400.0, 1024.0, 2.0,
        176.0, 512.0, 400.0, 1024.0, 2.0,
        11.0, 40000.0, 255.0, 1e6,
    ], dtype=np.float32)

    _DEFAULT_AGENT_VALS = [176, 205, 192, 512, 0]
    _DEFAULT_ENEMY_VALS = [176, 307, 192, 512, 0]
    _DEFAULT_META_VALS = [0, 39208, 0, 0]

    def __init__(self, env, ego_is_left=True):
        super().__init__(env)
        if ego_is_left:
            self._info_keys = self._AGENT_KEYS + self._ENEMY_KEYS + self._META_KEYS
            default_raw = self._DEFAULT_AGENT_VALS + self._DEFAULT_ENEMY_VALS + self._DEFAULT_META_VALS
        else:
            self._info_keys = self._ENEMY_KEYS + self._AGENT_KEYS + self._META_KEYS
            default_raw = self._DEFAULT_ENEMY_VALS + self._DEFAULT_AGENT_VALS + self._DEFAULT_META_VALS
        n = len(self._info_keys)
        self.observation_space = Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)
        self._default_obs = np.array(default_raw, dtype=np.float32) / self.NORM_SCALES

    def _info_to_obs(self, info):
        raw = np.array([info.get(k, 0) for k in self._info_keys], dtype=np.float32)
        return np.clip(raw / self.NORM_SCALES, 0.0, 1.0)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._default_obs.copy()

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            _, reward, reward_other, done, info = result
            return self._info_to_obs(info), reward, reward_other, done, info
        else:
            _, reward, done, info = result
            return self._info_to_obs(info), reward, done, info


class RamObsWrapper(gym.Wrapper):
    """Replaces image observations with the raw emulator RAM.

    WHY THIS EXISTS (measured 2026-08-10, spar_Ry_Sa_6720000):

        distinct successors over 21 action-distinct joint actions, median
            pixels (any resolution, any frame count, any channel set)     1
            the 14 curated info variables, agent_status included          1
            RAM                                                          21

    The state that separates two different actions is NOT RENDERED -- almost
    certainly buffered input and animation counters during hitstun/blockstun
    lockout. No amount of resolution or frame coverage recovers it, so this is
    partial observability rather than a preprocessing defect.

    It is also PREDICTIVE, not transient: under an identical no-op continuation
    RAM holds 12 distinct futures out to 16 steps, while pixels reach 3 -- after
    the whole gamma=0.94 horizon has elapsed. So the information exists, matters,
    and arrives far too late through pixels.

    MASK. Most of the 65,536 bytes never change. `mask` is an index array of the
    bytes worth feeding the network (see build_ram_mask.py); None means the full
    RAM. Full RAM is the honest default; the mask exists because a 65,536-wide
    input is ~16.8M parameters in the first layer alone, nearly all of it
    reading constants.

    Must return the 2P 5-tuple like InfoObsWrapper, or the SPAR path breaks.
    """

    def __init__(self, env, mask=None, stack=1, stride=8):
        super().__init__(env)
        self._mask = None if mask is None else np.asarray(mask, dtype=np.int64)
        # STACK. A single RAM frame is Markov for the game's mechanical state IF
        # the mask kept the per-character state-machine bytes (move id, animation
        # frame counter, hitstun timer) -- and `vary` mode keeps exactly that
        # kind of byte, since counters change constantly. So stack=1 is the
        # honest default and this is opt-in.
        #
        # It changes the OBSERVATION WIDTH, so a checkpoint trained at one stack
        # cannot be loaded at another.
        self._stack = max(1, int(stack))
        # STRIDE, in EMULATOR frames. The default 8 equals num_step_frames, i.e.
        # one sample per agent step -- and at that stride every stacked frame
        # except the newest is SHARED by all 484 branches of a state, so stacking
        # cannot change one-step branch distinguishability at all. Only stride < 8
        # adds branch-discriminating information. That is the thing under test.
        self._stride = max(1, int(stride))
        # The buffer is in emulator frames; the observation samples every _stride
        # frames back from the newest.
        self._depth = (self._stack - 1) * self._stride + 1
        self._hist = deque(maxlen=self._depth)
        # Walk down to whatever actually owns get_ram(); the chain is
        # SFWrapper -> FrameStack -> RetroEnv and `unwrapped` is not reliable
        # through SFWrapper's custom __getattr__.
        e = env
        while not hasattr(e, "get_ram") and hasattr(e, "env"):
            e = e.env
        if not hasattr(e, "get_ram"):
            raise RuntimeError("RamObsWrapper: no get_ram() below this wrapper")
        self._retro = e
        n_full = int(np.asarray(e.get_ram()).size)
        if self._mask is not None:
            if self._mask.max() >= n_full or self._mask.min() < 0:
                raise ValueError(f"ram mask out of range for {n_full} bytes")
            n = int(self._mask.size)
        else:
            n = n_full
        self.ram_bytes_full = n_full
        self.ram_bytes_masked = n
        self.observation_space = Box(low=0.0, high=1.0,
                                     shape=(n * self._stack,), dtype=np.float32)
        if self._stack > 1:
            # Reach past this wrapper to SFWrapper, which owns the frame loop.
            sf = self.env
            while not hasattr(sf, "num_step_frames"):
                sf = sf.env
            sf.ram_tap = self._tap_frame
            self._sf = sf

    def _frame(self):
        ram = np.asarray(self._retro.get_ram())
        if self._mask is not None:
            ram = ram[self._mask]
        return (ram.astype(np.float32) / 255.0)

    def _tap_frame(self):
        self._hist.append(self._frame())

    def _ram_obs(self):
        if self._stack == 1:
            return self._frame()
        if not self._hist:                        # reset, or a restore into an
            f = self._frame()                     # empty history
            self._hist.extend([f] * self._depth)
        h = list(self._hist)
        # Sample every _stride frames back from the newest, oldest slot first.
        idx = [max(0, len(h) - 1 - i * self._stride) for i in range(self._stack)]
        return np.concatenate([h[i] for i in reversed(idx)])

    # The lbr_* methods live on SFWrapper, which this wrapper sits OUTSIDE of --
    # so a plain env_method("lbr_snapshot") would forward straight past this
    # history buffer and branches would inherit each other's frames. Intercept,
    # save the deque, then delegate.
    def lbr_snapshot(self, key="lbr_root"):
        self.__dict__.setdefault("_ram_hist_store", {})[key] = list(self._hist)
        return self.env.lbr_snapshot(key)

    def lbr_restore(self, key="lbr_root"):
        h = self.__dict__.get("_ram_hist_store", {}).get(key)
        if h is not None:
            self._hist = deque(h, maxlen=self._depth)
        return self.env.lbr_restore(key)

    def lbr_drop(self, key=None):
        st = self.__dict__.get("_ram_hist_store", {})
        if key is None:
            st.clear()
        else:
            st.pop(key, None)
        return self.env.lbr_drop(key)

    def ram_tail(self, n):
        """Last n masked emulator frames, oldest first. For offline (k, stride)
        sweeps: one rollout can be re-sampled at every combination."""
        h = list(self._hist)[-int(n):]
        return np.stack(h) if h else np.zeros((0, 0), dtype=np.float32)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self._hist.clear()
        return self._ram_obs()

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            _, reward, reward_other, done, info = result
            return self._ram_obs(), reward, reward_other, done, info
        _, reward, done, info = result
        return self._ram_obs(), reward, done, info


class EgoCentricImageWrapper(gym.Wrapper):
    """Horizontally flips image observations when ego is P2 (right controller),
    so the CNN always sees ego's character on the left side of the screen.
    """

    def __init__(self, env):
        super().__init__(env)
        print("EgoCentricImageWrapper active")

    def _flip(self, obs):
        return np.ascontiguousarray(obs[:, ::-1, :])

    def reset(self, **kwargs):
        return self._flip(self.env.reset(**kwargs))

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, reward_other, done, info = result
            return self._flip(obs), reward, reward_other, done, info
        else:
            obs, reward, done, info = result
            return self._flip(obs), reward, done, info
