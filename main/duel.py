"""
Head-resolved 1v1 evaluation between two trained agents.

Loads two models (possibly different families), routes the forward pass to the
matchup-specific head for the given (ego_char, adv_char), and plays N rounds.
Ego is locked to the left side; adv to the right.

Usage:
    python main/duel.py \\
        --ego_model_type spar --adv_model_type ippo \\
        --ego_char Guile --adv_char Ryu \\
        --num_rounds 10 \\
        --ego_model_file /abs/path/to/ego.task \\
        --adv_model_file /abs/path/to/adv.task
"""
import argparse
import os
import sys
import numpy as np
import retro
import torch
import torch as th

from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import obs_as_tensor

from common.const import sf_game
from common.retro_wrappers import SFWrapper, Monitor2P
from common.utils import SubprocVecEnv2P, VecTransposeImage2P
from common.algorithms import LeaguePPO
from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from common.justin.clean_derivative_free_spar_ippo import CleanDerivativeFreeSPARIPPO
from utils import agent_win, state2matchup


MODEL_TYPES = ["league", "spar", "ippo", "2timescale"]

# Canonical character names matching state-file naming (e.g. EHonda, ChunLi, MBison).
CHARACTERS = [
    "Ryu", "EHonda", "Blanka", "Guile", "Balrog", "Vega",
    "Ken", "ChunLi", "Zangief", "Dhalsim", "Sagat", "MBison",
]

SPAR_FAMILY = {"spar", "ippo", "2timescale"}


def make_env_factory(state, seed=0):
    """Returns a no-arg callable that builds a single retro+SF env at `state`."""
    def _init():
        env = retro.make(
            game=sf_game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=2,
        )
        env = SFWrapper(env, side="both", rendering=False, reset_type="round",
                        init_level=1, state_dir=None, verbose=False,
                        enable_combo=False, null_combo=False, transform_action=False)
        env = Monitor2P(env)
        env.seed(seed)
        return env
    return _init


def build_construction_vec_env(state_list):
    """Build a vec env shaped to the saved model's topology (one env per state)."""
    env_fns = [make_env_factory(s, seed=0) for s in state_list]
    return VecTransposeImage2P(SubprocVecEnv2P(env_fns))


def spar_class_for(model_type):
    if model_type == "spar":
        return CleanDerivativeFreeSPAR
    if model_type in ("ippo", "2timescale"):
        return CleanDerivativeFreeSPARIPPO
    raise ValueError(f"Not a SPAR-family type: {model_type}")


def load_spar_family(model_type, path, device):
    """Load a SPAR/IPPO/2timescale checkpoint and return (model, matchups)."""
    cls = spar_class_for(model_type)
    data, _, _ = load_from_zip_file(path, device="cpu")
    state_list = data["state_list"]
    env = build_construction_vec_env(state_list)
    model = cls.load(path, env=env, num_perturbed=1, device=device)
    matchups = list(getattr(model, "matchups", []) or [])
    if not matchups:
        # Defensive: fall back to deriving from state_list if attribute missing.
        matchups = [state2matchup(s) for s in state_list]
    return model, matchups


def load_league_agent_dict(path):
    """League checkpoints are torch-save style: {cls_name, kwargs:{agent_dict:{...}}}."""
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "kwargs" not in obj or "agent_dict" not in obj["kwargs"]:
        raise ValueError(
            f"League checkpoint {path} is not in the expected "
            f"torch-save format with kwargs/agent_dict."
        )
    return obj["kwargs"]["agent_dict"]


def build_league_model(duel_state, device):
    """Construct a fresh LeaguePPO with a single-env vec for inference."""
    env = VecTransposeImage2P(SubprocVecEnv2P([make_env_factory(duel_state, seed=0)]))
    return LeaguePPO(
        side="left",
        policy="CnnPolicy",
        env=env,
        device=device,
        verbose=0,
        n_steps=512,
        batch_size=1024,
        n_epochs=4,
        gamma=0.94,
        learning_rate=1e-4,
        clip_range=0.1,
        other_learning_rate=1e-4,
    )


def filter_left_keys(agent_dict):
    return {k: v for k, v in agent_dict.items()
            if k == "policy" or k.startswith("policy.")}


def filter_right_keys(agent_dict):
    return {k: v for k, v in agent_dict.items()
            if k == "policy_other" or k.startswith("policy_other.")}


def resolve_head_idx(matchups, matchup_key):
    if matchup_key in matchups:
        return matchups.index(matchup_key)
    if len(matchups) == 1:
        return 0
    raise ValueError(
        f"Model has no head trained for matchup '{matchup_key}'. "
        f"Available matchups: {matchups}"
    )


def make_spar_ego_action_fn(model, deterministic):
    """Ego head is shared across matchups; no buf_num needed."""
    @torch.no_grad()
    def _act(obs_t):
        actions, _ = model.policy.ego_forward(obs_t, deterministic=deterministic)
        return actions.cpu().numpy()
    return _act


def make_spar_adv_action_fn(model, head_idx, deterministic):
    @torch.no_grad()
    def _act(obs_t):
        actions, _ = model.policy.adv_forward(
            obs_t, buf_num=[head_idx], deterministic=deterministic
        )
        return actions.cpu().numpy()
    return _act


def make_league_action_fn(model, side, deterministic):
    policy = model.policy if side == "left" else model.policy_other

    @torch.no_grad()
    def _act(obs_t):
        # LeaguePPO's underlying CnnPolicy.predict() takes numpy.
        obs_np = obs_t.cpu().numpy() if isinstance(obs_t, th.Tensor) else obs_t
        actions, _ = policy.predict(obs_np, deterministic=deterministic)
        return actions
    return _act


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ego_model_type", required=True, choices=MODEL_TYPES,
                   help="Architecture family of the ego (left) model. Case-sensitive.")
    p.add_argument("--adv_model_type", required=True, choices=MODEL_TYPES,
                   help="Architecture family of the adversary (right) model. Case-sensitive.")
    p.add_argument("--num_rounds", type=int, required=True,
                   help="Number of rounds (episodes) to play.")
    p.add_argument("--ego_char", required=True, choices=CHARACTERS,
                   help="Protagonist character (left side).")
    p.add_argument("--adv_char", required=True, choices=CHARACTERS,
                   help="Antagonist character (right side).")
    p.add_argument("--ego_model_file", required=True,
                   help="Absolute path to the ego model checkpoint.")
    p.add_argument("--adv_model_file", required=True,
                   help="Absolute path to the adversary model checkpoint.")
    p.add_argument("--ego_side", default="left", choices=["left", "right"],
                   help="Ego side (must be 'left'; right is rejected).")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", default="True", choices=["True", "False"])
    return p.parse_args()


def resolve_device(spec):
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def main():
    args = parse_args()
    if args.ego_side != "left":
        raise ValueError("Ego must be on the left side. --ego_side right is not allowed.")
    for path, label in [(args.ego_model_file, "ego"), (args.adv_model_file, "adv")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} model file not found: {path}")

    deterministic = args.deterministic == "True"
    device = resolve_device(args.device)

    matchup_key = f"{args.ego_char}Vs{args.adv_char}"
    duel_state = (
        f"two_player/{args.ego_char}_left/"
        f"Champion.Level1.{args.ego_char}Vs{args.adv_char}.2Player.state"
    )

    same_file = args.ego_model_file == args.adv_model_file
    if same_file and args.ego_model_type != args.adv_model_type:
        raise ValueError(
            "Same file passed for ego and adv but model types differ. "
            "A single checkpoint cannot be two different model architectures."
        )

    # ---- Load ego model & build ego action fn ----
    if same_file:
        # Single instance shared across both sides.
        if args.ego_model_type in SPAR_FAMILY:
            ego_model, ego_matchups = load_spar_family(
                args.ego_model_type, args.ego_model_file, device
            )
            adv_model = ego_model
            adv_matchups = ego_matchups
        else:  # league
            ego_model = build_league_model(duel_state, device)
            agent_dict = load_league_agent_dict(args.ego_model_file)
            ego_model.set_parameters(agent_dict, exact_match=False, device=device)
            adv_model = ego_model
            ego_matchups = adv_matchups = []
    else:
        # Independent loads per side.
        if args.ego_model_type in SPAR_FAMILY:
            ego_model, ego_matchups = load_spar_family(
                args.ego_model_type, args.ego_model_file, device
            )
        else:
            ego_model = build_league_model(duel_state, device)
            ego_dict = load_league_agent_dict(args.ego_model_file)
            ego_model.set_parameters(filter_left_keys(ego_dict),
                                     exact_match=False, device=device)
            ego_matchups = []

        if args.adv_model_type in SPAR_FAMILY:
            adv_model, adv_matchups = load_spar_family(
                args.adv_model_type, args.adv_model_file, device
            )
        else:
            # If both sides are league, fold the adv weights into the ego LeaguePPO
            # so we end up with one instance whose .policy is ego-trained-left and
            # .policy_other is adv-trained-right (Q4(a)). Otherwise build standalone.
            adv_dict = load_league_agent_dict(args.adv_model_file)
            if args.ego_model_type == "league":
                ego_model.set_parameters(filter_right_keys(adv_dict),
                                         exact_match=False, device=device)
                adv_model = ego_model
            else:
                adv_model = build_league_model(duel_state, device)
                adv_model.set_parameters(filter_right_keys(adv_dict),
                                         exact_match=False, device=device)
            adv_matchups = []

    # ---- Resolve heads (SPAR family only) ----
    if args.ego_model_type in SPAR_FAMILY:
        ego_head_idx = resolve_head_idx(ego_matchups, matchup_key)
        ego_act = make_spar_ego_action_fn(ego_model, deterministic)
    else:
        ego_head_idx = None
        ego_act = make_league_action_fn(ego_model, "left", deterministic)

    if args.adv_model_type in SPAR_FAMILY:
        adv_head_idx = resolve_head_idx(adv_matchups, matchup_key)
        adv_act = make_spar_adv_action_fn(adv_model, adv_head_idx, deterministic)
    else:
        adv_head_idx = None
        adv_act = make_league_action_fn(adv_model, "right", deterministic)

    # ---- Build the actual duel env (single, non-vec) ----
    duel_env = make_env_factory(duel_state, seed=args.seed)().env

    # ---- Run rounds ----
    print(
        f"\nDuel: ego={args.ego_model_type}({args.ego_char}, head={ego_head_idx}) "
        f"vs adv={args.adv_model_type}({args.adv_char}, head={adv_head_idx})  "
        f"matchup={matchup_key}  rounds={args.num_rounds}  "
        f"shared_instance={same_file}"
    )

    ego_device = (ego_model.device if hasattr(ego_model, "device") else device)
    adv_device = (adv_model.device if hasattr(adv_model, "device") else device)

    wins = 0
    for r in range(1, args.num_rounds + 1):
        obs = duel_env.reset()
        done = False
        info = {}
        while not done:
            obs_ego = obs_as_tensor(obs, ego_device)
            obs_adv = obs_as_tensor(obs, adv_device) if adv_model is not ego_model else obs_ego
            left_action = ego_act(obs_ego)
            right_action = adv_act(obs_adv)
            obs, _reward, _reward_other, done, info = duel_env.step(
                np.hstack([left_action, right_action])
            )
        ego_won = bool(agent_win(info))
        wins += int(ego_won)
        print(
            f"  round {r}/{args.num_rounds}: ego_won={ego_won}  "
            f"ego_hp={info.get('agent_hp')}  adv_hp={info.get('enemy_hp')}"
        )

    duel_env.close()

    win_rate = wins / args.num_rounds
    print(
        f"\nfinal: ego_win_rate={win_rate:.4f}  ({wins}/{args.num_rounds})  "
        f"matchup={matchup_key}  ego={args.ego_model_type}@{os.path.basename(args.ego_model_file)}  "
        f"adv={args.adv_model_type}@{os.path.basename(args.adv_model_file)}"
    )


if __name__ == "__main__":
    main()
