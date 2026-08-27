import os, sys
import torch
import argparse
import multiprocessing
import re
from typing import Dict, List, Tuple
# multiprocessing.set_start_method('forkserver', force=True)
multiprocessing.set_start_method('spawn', force=True) #TODO: Changed it to spawn to help with CUDA, not sure if it works. Can revert back if needed.
from multiprocessing import Process

import retro

from common.const import *
from common.utils import SubprocVecEnv2P, VecTransposeImage2P
from common.algorithms import LeaguePPO
from common.retro_wrappers import SFWrapper, Monitor2P
from common.league import PayoffManager, League, FSPLeague, PSROLeague, Learner


# Default roster (can be overridden via CLI).
DEFAULT_PLAYERS = ["Ryu"]
DEFAULT_OPPONENTS = ["Guile", "MBison", "Dhalsim"]

# STATES is built from PLAYERS x OPPONENTS in main().
# Key format: "<player>_<opponent>" (lowercase/sanitized).
STATES = {}
# STATE = "Champion.RyuVsRyu.2Player.align"
# STATE = ["Champion.RyuVsRyu.2Player.align", "Champion.Level12.RyuVsBison.2Player", "Champion.Level13.RyuVsBison.2Player", "Champion.Level1.RyuVsRyu.2Player"]


def _sanitize_matchup_token(value: str) -> str:
    token = str(value).strip().lower()
    out = []
    for ch in token:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def _extract_chars_from_state_name(state_name: str) -> Tuple[str, str]:
    """
    Parse `LeftVsRight` from a retro state string.

    Examples:
      Champion.Level1.RyuVsGuile.2Player.state -> ("ryu", "guile")
      Champion.RyuVsRyu.2Player.align -> ("ryu", "ryu")
    """
    match = re.search(r"([A-Za-z]+)Vs([A-Za-z]+)", state_name)
    if not match:
        return "left", "right"
    return _sanitize_matchup_token(match.group(1)), _sanitize_matchup_token(match.group(2))


def _canonicalize_matchup_entry(raw_key: str, state_name: str) -> Tuple[str, str, str]:
    """
    Return `(canonical_key, left_char, right_char)` for one STATES entry.

    Accepted key styles:
      - "ryu_vs_sagat" (preferred)
      - "ryu" (legacy; left parsed from state name)
    """
    key_norm = _sanitize_matchup_token(raw_key)
    if "_vs_" in key_norm:
        left_char, right_char = key_norm.split("_vs_", 1)
    elif key_norm.count("_") == 1:
        # New preferred STATES key style: "<player>_<opponent>"
        left_char, right_char = key_norm.split("_", 1)
    else:
        left_char, right_char = _extract_chars_from_state_name(state_name)
        # Keep old behavior if key was a single right-character token.
        if key_norm not in ("", "unknown"):
            right_char = key_norm
    canonical_key = f"{left_char}_vs_{right_char}"
    return canonical_key, left_char, right_char


def _build_matchup_specs(states: Dict[str, str]) -> List[Dict[str, str]]:
    """Normalize STATES dict into deterministic matchup specs."""
    specs = []
    seen_keys = set()
    for raw_key, state_name in states.items():
        canonical_key, left_char, right_char = _canonicalize_matchup_entry(raw_key, state_name)
        if canonical_key in seen_keys:
            raise ValueError(f"Duplicate canonical matchup key generated: {canonical_key}")
        seen_keys.add(canonical_key)
        specs.append(
            {
                "canonical_key": canonical_key,
                "state_name": state_name,
                "left_char": left_char,
                "right_char": right_char,
            }
        )
    return specs


def _extract_matchup_from_log_name(log_name: str) -> str:
    """
    Extract matchup label from league agent names.

    Supports names like:
      MA0_right_m_02_ryu_vs_bison
      ME1_right_m02_ryu_vs_bison
      LE0_right_ryu_vs_bison
    """
    if not log_name:
        return ""
    name = _sanitize_matchup_token(log_name)
    # Historical checkpoints append `_historical_step_<n>`; strip this suffix so
    # matchup parsing works for both live and historical player names.
    name = re.sub(r"_historical_step_\d+$", "", name)
    # Preferred: ..._m_02_<matchup>
    m = re.search(r"_m_(?:\d+)_([a-z0-9_]+)$", name)
    if m:
        return m.group(1)
    # Backward-compatible compact form: ..._m02_<matchup>
    m = re.search(r"_m\d+_([a-z0-9_]+)$", name)
    if m:
        return m.group(1)
    # Fallback: everything after side token.
    m = re.search(r"_(?:left|right)_(.+)$", name)
    if m:
        return m.group(1)
    return ""


def _resolve_state_name(
    side: str,
    opponent: str,
    state_name: str,
    matchup_key: str,
    log_name: str,
) -> str:
    """
    Resolve a concrete Retro state string for constructor().

    Resolution priority:
      1) explicit state_name
      2) matchup_key / parsed log_name against STATES keys
      3) legacy opponent key
      4) left-side fallback to first STATES entry
    """
    if state_name is not None:
        return state_name

    candidates: List[str] = []
    if matchup_key:
        candidates.append(_sanitize_matchup_token(matchup_key))
    parsed_from_name = _extract_matchup_from_log_name(log_name)
    if parsed_from_name:
        candidates.append(parsed_from_name)
    if opponent:
        candidates.append(_sanitize_matchup_token(opponent))

    expanded_candidates: List[str] = []
    for cand in candidates:
        expanded_candidates.append(cand)
        if "_vs_" in cand:
            expanded_candidates.append(cand.replace("_vs_", "_"))
        elif cand.count("_") == 1:
            left_tok, right_tok = cand.split("_", 1)
            expanded_candidates.append(f"{left_tok}_vs_{right_tok}")

    for cand in expanded_candidates:
        if cand in STATES:
            return STATES[cand]

    # Left generalist is a single global agent, so its opponent tokens are bare
    # character names (e.g. "bison") while STATES keys are "<protagonist>_<opp>".
    # Map by opponent suffix so the left env FOLLOWS the opponent's character
    # instead of freezing on matchup-0. Require a unique match to stay safe under
    # multi-protagonist rosters (fall through to the old bootstrap otherwise).
    if side == "left":
        for cand in candidates:
            suffix = f"_{_sanitize_matchup_token(cand)}"
            suffix_matches = [key for key in STATES if key.endswith(suffix)]
            if len(suffix_matches) == 1:
                return STATES[suffix_matches[0]]

    if side == "left" and STATES:
        # Left MA is global; any valid state bootstraps env construction.
        return next(iter(STATES.values()))

    raise KeyError(
        f"Could not resolve state for side={side}, opponent={opponent}, "
        f"matchup_key={matchup_key}, log_name={log_name}. Known STATES keys={list(STATES.keys())[:8]}"
    )


def _build_states_from_roster(players: List[str], opponents: List[str], side: str) -> Dict[str, str]:
    """
    Build STATES mapping keyed by player-opponent pair.

    Example key:
      "ryu_sagat" -> "two_player/Ryu_left/Champion.Level1.RyuVsSagat.2Player.state"
    """
    # ippo.py state folder convention uses a concrete side token.
    roster_side = side if side in ("left", "right") else "left"
    states: Dict[str, str] = {}
    for player in players:
        player_clean = _sanitize_matchup_token(player)
        player_title = str(player).strip()
        player_folder_name = f"{player_title}_{roster_side}"
        for opponent in opponents:
            opp_clean = _sanitize_matchup_token(opponent)
            opponent_title = str(opponent).strip()
            key = f"{player_clean}_{opp_clean}"
            if key in states:
                raise ValueError(f"Duplicate STATES key generated: {key}")
            states[key] = (
                "two_player/%s/Champion.Level1.%sVs%s.2Player.state"
                % (player_folder_name, player_title, opponent_title)
            )
    return states


def _right_char_from_matchup_key(matchup_key: str) -> str:
    parts = _sanitize_matchup_token(matchup_key).split("_vs_", 1)
    return parts[1] if len(parts) == 2 else _sanitize_matchup_token(matchup_key)


def make_env(game, state_name: str, side, reset_type, rendering, init_level=1, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False, seed=0):
    def _init():
        players = 2
        env = retro.make(
            game=game, 
            state=state_name, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level, state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action)
        env = Monitor2P(env)
        env.seed(seed)
        return env
    return _init


def worker(idx, learner, total_steps, rollout_opponent_num):
    print(f"Starting worker for {learner.player.name} (type: {type(learner.player).__name__})") #TODO: DEBUG ONLY! Remove when done
    print(f"Agent exists: {learner.player.agent is not None}") #TODO: DEBUG ONLY! Remove when done

    print(f"worker {learner.player.name} start")
    with torch.cuda.device(idx % torch.cuda.device_count()):
        learner.player.construct_agent()
        learner.run(total_timesteps=total_steps, rollout_opponent_num=rollout_opponent_num)


def restore_worker(idx, learner, total_steps, rollout_opponent_num):
    print(f"restore_worker {learner.player.name} start")
    with torch.cuda.device(idx % torch.cuda.device_count()):
        learner.player.construct_agent()
        learner.player._initial_weights = learner.player._initial_weights_restore # restore the initial weights to the reset weights
        learner.run(total_timesteps=total_steps, rollout_opponent_num=rollout_opponent_num, reset_num_timesteps=False) # NOTE: do not reset num_timesteps so that the timesteps are restored

#Added the default opponent so the opponent can be added to the end to not change the order of varaibles.
def constructor(args, side, log_name=None, single_env=False, opponent: str="ryu", state_name: str=None, matchup_key: str=None):
    """
    Agent constructor for league players.

    Notes:
    - `state_name` and `matchup_key` are the new preferred inputs.
    - `opponent` is kept for backward compatibility with older call-sites.
    """
    num_env = 1 if single_env else args.num_env
    # Spawned workers re-import this module WITHOUT running main(), so module-global
    # STATES is empty there. Rebuild it from the roster on args so left-side opponent
    # resolution works in workers, not just in-process.
    global STATES
    if not STATES:
        STATES = _build_states_from_roster(
            getattr(args, "player", None) or DEFAULT_PLAYERS,
            getattr(args, "opponent_list", None) or DEFAULT_OPPONENTS,
            getattr(args, "side", "left"),
        )
    state_name = _resolve_state_name(
        side=side,
        opponent=opponent,
        state_name=state_name,
        matchup_key=matchup_key,
        log_name=log_name,
    )
    if matchup_key is None:
        _, _, right_char = _canonicalize_matchup_entry(opponent, state_name)
        matchup_key = f"left_vs_{right_char}"
    env = [make_env(sf_game, state_name=state_name, side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, seed=i) for i in range(num_env)]
    env = VecTransposeImage2P(SubprocVecEnv2P(env))
    league_ppo = LeaguePPO(
        side,
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=512,
        batch_size=1024, # 512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=1e-4, # lr_schedule,
        clip_range=0.1, # clip_range_schedule,
        tensorboard_log=None if log_name is None else os.path.join(args.log_dir, log_name),
        # seed=args.seed,
        other_learning_rate=1e-4, # other_lr_schedule,
    )
    league_ppo.constructor_args = args
    league_ppo.current_opponent = _right_char_from_matchup_key(matchup_key)
    return league_ppo


def main():
    parser = argparse.ArgumentParser(description='Reset game stats')
    parser.add_argument('--reset', choices=['round', 'match', 'game'], help='Reset stats for a round, a match, or the whole game', default='round')
    # parser.add_argument('--model-file', help='The model to continue to learn from')
    parser.add_argument('--save-dir', help='The directory to save the trained models', default="main/trained_models/ma")
    parser.add_argument('--log-dir', help='The directory to save logs', default="logs/ma")
    # parser.add_argument('--model-name-prefix', help='The prefix of the model names to save', default="ppo_ryu")
    # parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile', default=SF_DEFAULT_STATE)
    parser.add_argument('--side', help='The side for AI to control. By default both', default='both', choices=['left', 'right', 'both'])
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen')
    parser.add_argument('--num-env', type=int, help='How many envirorments to create', default=24)
    # parser.add_argument('--num-episodes', type=int, help='In evaluation, play how many episodes', default=20)
    # parser.add_argument('--num-epoch', type=int, help='Finetune how many epochs', default=50)
    parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=int(1e10)) # 1e5
    # parser.add_argument('--video-dir', help='The path to save videos', default='videos')
    # parser.add_argument('--finetune-dir', help='The path to save finetune results', default='finetune')
    # parser.add_argument('--init-level', type=int, help='Initial level to load from. By default 0, starting from pretrain', default=0)
    # parser.add_argument('--resume-epoch', type=int, help='Resume epoch. By default 0, starting from pretrain', default=0)
    parser.add_argument('--enable-combo', action='store_true', help='Enable special move action space for environment')
    parser.add_argument('--null-combo', action='store_true', help='Null action space for special move')
    parser.add_argument('--transform-action', action='store_true', help='Transform action space to MultiDiscrete')
    parser.add_argument('--seed', type=int, help='Seed', default=0)
    # parser.add_argument('--update-left', type=int, help='Update left policy', default=1)
    # parser.add_argument('--update-right', type=int, help='Update right policy', default=1)
    parser.add_argument('--left-model-file', help='The left model to continue to learn from')
    parser.add_argument('--right-model-file', help='The right model to continue to learn from')
    # parser.add_argument('--other-timescale', type=float, help='Other agent learning rate scale', default=1.0)
    # parser.add_argument('--fsp', action='store_true', help='Fictitious self-play')
    # parser.add_argument('--fsp-threshold', type=float, help='Fictitious self-play threshold', default=0.5)
    # parser.add_argument('--async-update', action='store_true', help='Update left and right asynchronously')
    parser.add_argument('--rollout-opponent-num', type=int, help='Numbers of opponents to interact for each update', default=5) # 2
    parser.add_argument('--fsp-league', action='store_true', help='Fictitious self-play league')
    parser.add_argument('--psro-league', action='store_true', help='PSRO league')
    parser.add_argument('--sync-save-interval', type=int, help='Steps between sync checkpoint saves (0 = save every sync)', default=5000)
    # Match ippo.py conventions for roster CLI.
    parser.add_argument('--player', type=str, nargs='+', default=DEFAULT_PLAYERS, help='Protagonist player(s).')
    parser.add_argument('--opponent-list', type=str, nargs='+', default=DEFAULT_OPPONENTS, help='List of opponent characters.')

    args = parser.parse_args()
    print("command line args:" + str(args))

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    # os.makedirs(args.video_dir, exist_ok=True)
    # os.makedirs(args.finetune_dir, exist_ok=True)

    global STATES
    STATES = _build_states_from_roster(
        players=args.player,
        opponents=args.opponent_list,
        side=args.side,
    )

    matchup_specs = _build_matchup_specs(STATES)
    if len(matchup_specs) == 0:
        raise ValueError("STATES must contain at least one matchup entry.")
    right_models = {}
    state_names = {}
    for spec in matchup_specs:
        right_models[spec["canonical_key"]] = constructor(
            args,
            "right",
            log_name=None,
            single_env=True,
            opponent=spec["right_char"],  # legacy arg for compatibility
            state_name=spec["state_name"],
            matchup_key=spec["canonical_key"],
        )
        state_names[spec["canonical_key"]] = spec["state_name"]

    # Left MA remains single/global; initialize using the first matchup state.
    left_model = constructor(
        args,
        "left",
        log_name=None,
        single_env=True,
        state_name=matchup_specs[0]["state_name"],
        matchup_key="left_vs_all",
    )

    #TODO: This should fail because right_model is no longer set.
    #TODO To Justin - what do you want to do with this block?
    if args.left_model_file and args.right_model_file:
        print("load model from " + args.left_model_file + " and " + args.right_model_file)
        left_model.set_parameters_2p(args.left_model_file, args.right_model_file)
        # Right side is now a matchup-keyed dict of models, not a single `right_model`.
        # Keep backward behavior by loading each right model with the provided 2p files.
        for model_key, model in right_models.items():
            print(f"loading right model parameters for {model_key}")
            model.set_parameters_2p(args.left_model_file, args.right_model_file)
    
    initial_agents = {
        'left': left_model,
        'right': right_models, #NOTE: This is no longer a single model
    }
    
    with PayoffManager() as manager:
        shared_payoff = manager.Payoff(args.save_dir)
        if args.fsp_league:
            league = FSPLeague(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1)
        elif args.psro_league:
            league = PSROLeague(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1, state_names=state_names)
        else:
            league = League(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1, main_exploiters=1, league_exploiters=2, state_names=state_names)
        #TODO: DEBUGGING ONLY!!! This is serial method for debugging instead of multipcroessing.
        #For some reason, it works but the parallel version doesn't.
        #After the debugging is done, return to multiprocessing.
        # for idx in range(league.size()):
        #     player = league.get_player(idx)
        #     learner = Learner(player)
        #     worker(idx, learner, args.total_steps, args.rollout_opponent_num)
        
        # TODO: This is the parallel version that doesn't work for some reason. Uncomment this block when done debugging the serial version.
        processes = []
        for idx in range(league.size()):
            player = league.get_player(idx)
            # player.constructor_fn = constructor #TODO: Delete when done
            learner = Learner(player)
            process = Process(target=worker, args=(idx, learner, args.total_steps, args.rollout_opponent_num))
            # process.daemon=True  # all processes closed when the main stops
            processes.append(process)
        for p in processes:
            p.start()
        for p in processes:
            p.join()


def restore():
    parser = argparse.ArgumentParser(description='Reset game stats')
    parser.add_argument('--reset', choices=['round', 'match', 'game'], help='Reset stats for a round, a match, or the whole game', default='round')
    # parser.add_argument('--model-file', help='The model to continue to learn from')
    parser.add_argument('--save-dir', help='The directory to save the trained models', default="trained_models/ma")
    parser.add_argument('--log-dir', help='The directory to save logs', default="logs/ma")
    # parser.add_argument('--model-name-prefix', help='The prefix of the model names to save', default="ppo_ryu")
    # parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile', default=SF_DEFAULT_STATE)
    parser.add_argument('--side', help='The side for AI to control. By default both', default='both', choices=['left', 'right', 'both'])
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen')
    parser.add_argument('--num-env', type=int, help='How many envirorments to create', default=24)
    # parser.add_argument('--num-episodes', type=int, help='In evaluation, play how many episodes', default=20)
    # parser.add_argument('--num-epoch', type=int, help='Finetune how many epochs', default=50)
    parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=int(1e10)) # 1e5
    # parser.add_argument('--video-dir', help='The path to save videos', default='videos')
    # parser.add_argument('--finetune-dir', help='The path to save finetune results', default='finetune')
    # parser.add_argument('--init-level', type=int, help='Initial level to load from. By default 0, starting from pretrain', default=0)
    # parser.add_argument('--resume-epoch', type=int, help='Resume epoch. By default 0, starting from pretrain', default=0)
    parser.add_argument('--enable-combo', action='store_true', help='Enable special move action space for environment')
    parser.add_argument('--null-combo', action='store_true', help='Null action space for special move')
    parser.add_argument('--transform-action', action='store_true', help='Transform action space to MultiDiscrete')
    parser.add_argument('--seed', type=int, help='Seed', default=0)
    # parser.add_argument('--update-left', type=int, help='Update left policy', default=1)
    # parser.add_argument('--update-right', type=int, help='Update right policy', default=1)
    parser.add_argument('--left-model-file', help='The left model to continue to learn from')
    parser.add_argument('--right-model-file', help='The right model to continue to learn from')
    # parser.add_argument('--other-timescale', type=float, help='Other agent learning rate scale', default=1.0)
    # parser.add_argument('--fsp', action='store_true', help='Fictitious self-play')
    # parser.add_argument('--fsp-threshold', type=float, help='Fictitious self-play threshold', default=0.5)
    # parser.add_argument('--async-update', action='store_true', help='Update left and right asynchronously')
    parser.add_argument('--rollout-opponent-num', type=int, help='Numbers of opponents to interact for each update', default=5) # 2
    parser.add_argument('--fsp-league', action='store_true', help='Fictitious self-play league')
    parser.add_argument('--psro-league', action='store_true', help='PSRO league')
    parser.add_argument('--sync-save-interval', type=int, help='Steps between sync checkpoint saves (0 = save every sync)', default=500000)

    args = parser.parse_args()
    print("command line args:" + str(args))

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    # os.makedirs(args.video_dir, exist_ok=True)
    # os.makedirs(args.finetune_dir, exist_ok=True)

    model_files = {
        "LE0_left": "trained_models/ma_231218/LE0_left_230214912.pt",
        "LE0_right": "trained_models/ma_231218/LE0_right_235170576.pt",
        "LE1_left": "trained_models/ma_231218/LE1_left_216107520.pt",
        "LE1_right": "trained_models/ma_231218/LE1_right_228268080.pt",
        "MA0_left": "trained_models/ma_231218/MA0_left_237694104.pt",
        "MA0_right": "trained_models/ma_231218/MA0_right_224946696.pt",
        "ME0_left": "trained_models/ma_231218/ME0_left_230108016.pt",
        "ME0_right": "trained_models/ma_231218/ME0_right_230112360.pt",
    }
    payoff_file = "trained_models/ma_231218/payoff_20231218_20_38.pt"
    
    left_model = constructor(args, "left", log_name=None, single_env=True)
    right_model = constructor(args, "right", log_name=None, single_env=True)

    if args.left_model_file and args.right_model_file:
        print("load model from " + args.left_model_file + " and " + args.right_model_file)
        left_model.set_parameters_2p(args.left_model_file, args.right_model_file)
        right_model.set_parameters_2p(args.left_model_file, args.right_model_file)
    
    initial_agents = {
        'left': left_model,
        'right': right_model,
    }
    
    with PayoffManager() as manager:
        shared_payoff = manager.Payoff(args.save_dir)
        if args.fsp_league:
            league = FSPLeague(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1)
        elif args.psro_league:
            league = PSROLeague(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1)
        else:
            league = League(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1, main_exploiters=1, league_exploiters=2)
        processes = []
        for idx in range(league.size()):
            player = league.get_player(idx)
            player.load(model_files[player.name])
        shared_payoff.load(payoff_file)
        for idx in range(league.size()):
            player = league.get_player(idx)
            learner = Learner(player)
            process = Process(target=restore_worker, args=(idx, learner, args.total_steps, args.rollout_opponent_num))
            # process.daemon=True  # all processes closed when the main stops
            processes.append(process)
        for p in processes:
            p.start()
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
    # restore() # NOTE: backup checkpoint before running this