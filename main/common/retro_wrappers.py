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

    def __init__(self, env, side, reset_type="round", init_level=1, rendering=False, num_stack=12, num_step_frames=8, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False):
        super(SFWrapper, self).__init__(env)
        self.env = FrameStack(env, num_stack=num_stack)

        assert side in ['left', 'right', 'both'], "side should be 'left', 'right' or 'both'"
        self.side = side

        self.num_stack = num_stack
        self.num_step_frames = num_step_frames

        self.aggresive_coeff = 1.0
        self.dense_coeff = 1.0

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_agent_hp = self.full_hp
        self.prev_enemy_hp = self.full_hp

        # self.observation_space = Box(low=0, high=255, shape=(100, 128, 3 * self.num_stack), dtype=np.uint8)
        self.observation_space = Box(low=0, high=255, shape=(100, 128, len(range(0, self.num_stack, self.num_step_frames // 2))), dtype=np.uint8)
        self.action_dim = 12 + 3 if (enable_combo or null_combo) else 12 # 3 bits for combos
        if transform_action:
            # self.action_space = MultiDiscrete([len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS) + len(COMBOS) for _ in range(self.players)])
            self.action_space = MultiDiscrete([len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS) + len(SF_COMBOS)]) if (enable_combo or null_combo) else MultiDiscrete([len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS)])
            def action_transformer(action):
                players_action = []
                for player_action in action:
                    if player_action >= len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS):
                        # if self.null_combo:
                        #     print(f"player_action = {player_action}, invalid for null combo", flush=True)
                        button_bits = [0 for _ in range(12)]
                        combo_bits = [int(i) for i in np.binary_repr(player_action - len(DIRECTIONS_BUTTONS) - len(ATTACKS_BUTTONS)).zfill(3)]
                    elif player_action >= len(DIRECTIONS_BUTTONS):
                        direction_buttons = []
                        attack_buttons = ATTACKS_BUTTONS[player_action - len(DIRECTIONS_BUTTONS)]
                        button_bits = [int(b in direction_buttons + attack_buttons) for b in BUTTONS]
                        combo_bits = [1 for _ in range(3)]
                    else:
                        direction_buttons = DIRECTIONS_BUTTONS[player_action]
                        attack_buttons = []
                        button_bits = [int(b in direction_buttons + attack_buttons) for b in BUTTONS]
                        combo_bits = [1 for _ in range(3)]
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

        self.reset_type = reset_type
        self.rendering = rendering

        self.init_level = init_level
        self.state_dir = state_dir
        self.verbose = verbose
        self.enable_combo = enable_combo
        self.null_combo = null_combo

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
            return np.stack([o[::2, ::2, i % 3] for (i, o) in enumerate(obs[::(self.num_step_frames // 2)])], axis=-1)

    def reset(self):
        obs = self.env.reset()

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
    
        return self._get_obs(obs)

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

    def step(self, action):
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
                combo_id = len(SF_COMBOS)
            if combo_id >= len(SF_COMBOS):
                action_seq = [np.hstack([action[:12], np.zeros_like(action[:12])]) for _ in range(self.num_step_frames)]
            else:
                combo = SF_COMBOS[combo_id]
                assert self.num_step_frames == len(combo)
                action_seq = combo
                action_seq = [np.hstack([combo[t], np.zeros_like(combo[t])]) for t in range(self.num_step_frames)]
        elif self.side == 'right':
            action[3] = 0 # Filter out the "START/PAUSE" button
            if self.enable_combo:
                combo_id = int(4 * action[-3] + 2 * action[-2] + action[-1])
            else:
                combo_id = len(SF_COMBOS)
            if combo_id >= len(SF_COMBOS):
                action_seq = [np.hstack([np.zeros_like(action[:12]), action[:12]]) for _ in range(self.num_step_frames)]
            else:
                combo = SF_COMBOS[combo_id]
                assert self.num_step_frames == len(combo)
                action_seq = combo
                action_seq = [np.hstack([np.zeros_like(combo[t]), combo[t]]) for t in range(self.num_step_frames)]
        else:
            action[3] = 0 # Filter out the "START/PAUSE" button
            action[self.action_dim + 3] = 0
            if self.enable_combo:
                combo_ids = [int(4 * action[self.action_dim - 3] + 2 * action[self.action_dim - 2] + action[self.action_dim - 1]), int(4 * action[-3] + 2 * action[-2] + action[-1])]
            else:
                combo_ids = [len(SF_COMBOS), len(SF_COMBOS)]
            action_seqs = []
            for player_id, combo_id in enumerate(combo_ids):
                if combo_id >= len(SF_COMBOS):
                    action_seq = [action[player_id * self.action_dim : player_id * self.action_dim + 12] for _ in range(self.num_step_frames)]
                else:
                    combo = SF_COMBOS[combo_id]
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
            if self.rendering:
                self.env.render()
                time.sleep(0.01)

        agent_hp = info['agent_hp']
        enemy_hp = info['enemy_hp']
        agent_victories = info['agent_victories']
        enemy_victories = info['enemy_victories']
        round_countdown = info['round_countdown']
        timesup = (round_countdown <= 0)

        self.total_timesteps += self.num_step_frames
        
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
                custom_reward = self.dense_coeff * (self.aggresive_coeff * (self.prev_enemy_hp - enemy_hp) - (self.prev_agent_hp - agent_hp))
                custom_reward_inverse = self.dense_coeff * (self.aggresive_coeff * (self.prev_agent_hp - agent_hp) - (self.prev_enemy_hp - enemy_hp))
                self.prev_agent_hp = agent_hp
                self.prev_enemy_hp = enemy_hp
                custom_done = False

        # if custom_reward != 0:
        #     print("reward:{}".format(custom_reward))

        info['level'] = self.level
        info['match'] = 'start' if self.match_status == START_STATUS else 'end'
        info['round'] = 'start' if self.round_status == START_STATUS else 'end'
        if custom_done:
            info['outcome'] = 'win' if (agent_hp > enemy_hp) else ('lose' if (agent_hp < enemy_hp) else 'draw')

        if self.side == 'left':
            return self._get_obs(obs), 0.001 * custom_reward, custom_done, info 
        elif self.side == 'right':
            return self._get_obs(obs), 0.001 * custom_reward_inverse, custom_done, info 
        else:
            return self._get_obs(obs), 0.001 * custom_reward, 0.001 * custom_reward_inverse, custom_done, info 


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
