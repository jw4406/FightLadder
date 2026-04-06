"""
BR worker. Parse --device before importing torch so ``--device cpu`` can hide GPUs from PyTorch.
"""
import os
import sys
from typing import Any, Dict, List, Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def _peek_torch_device_argv(argv):
    for i, a in enumerate(argv):
        if a == "--device" and i + 1 < len(argv):
            return argv[i + 1]
    return os.environ.get("BR_TORCH_DEVICE")


_peeked_dev = _peek_torch_device_argv(sys.argv[1:])
if _peeked_dev is not None and str(_peeked_dev).lower().startswith("cpu"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _ensure_cpu_only_env(device: str) -> None:
    """Multiprocessing ``spawn`` may not replay argv peek; call with the same ``device`` as training."""
    if str(device).lower().startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

from common.algorithms import Exploiter
import argparse
import time
import json
import random
import numpy as np
import wandb
import subprocess
import multiprocessing as mp
from stable_baselines3.common.preprocessing import is_image_space
from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR
from stable_baselines3.common.save_util import load_from_zip_file
from ippo import env_generator
from stable_baselines3.common.callbacks import ExploiterCheckpointCallback
from gymnasium.spaces import Box
from utils import state2matchup
# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
#print(current_dir)
TASK_DIR = os.path.join(current_dir, "trained_models/tasks")
#TASK_DIR = '/n/fs/magics/2415498/FightLadder/main/trained_models/tasks/'
PROCESSING_DIR = os.path.join(current_dir, "main/trained_models/tasks/processing")
DONE_DIR = os.path.join(current_dir, "main/trained_models/tasks/done")
ERROR_DIR = os.path.join(current_dir, "main/trained_models/tasks/error")
BR_MODEL_DIR = os.path.join(current_dir, "main/trained_models/tasks/br_models")
WR_STATS_DIR = os.path.join(current_dir, "main/trained_models/wr_stats")
MEAN_REW_STATS_DIR = os.path.join(current_dir, "main/trained_models/mean_rew_stats")
os.makedirs(BR_MODEL_DIR, exist_ok=True)
os.makedirs(TASK_DIR, exist_ok=True)
os.makedirs(PROCESSING_DIR, exist_ok=True)
os.makedirs(DONE_DIR, exist_ok=True)
os.makedirs(WR_STATS_DIR, exist_ok=True)
os.makedirs(MEAN_REW_STATS_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)

if not os.listdir(TASK_DIR):
    print("Warning: The TASK_DIR is empty. Please run ippo.py --player PLAYER to generate a task file.")

POLL_INTERVAL = 5  # Seconds to wait before checking for new tasks
BR_TRAINING_STEPS = 1000


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    """
    Remove duplicates while preserving first-seen order.

    This is intentionally deterministic: dedicated job generation and naming
    should not depend on dict/hash iteration order.
    """
    return list(dict.fromkeys(values).keys())


def _sanitize_for_filename(value: str) -> str:
    """
    Convert arbitrary matchup/state labels into safe filename fragments.

    We keep alphanumeric, underscore, and hyphen characters and replace all
    others with underscores so BR checkpoint names remain filesystem-friendly.
    """
    if value is None:
        return "unknown"
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def _extract_unique_states_from_task(task_file_path: str, device: str = "cuda") -> List[str]:
    """
    Read checkpoint/task metadata and return unique state strings.

    Why this helper exists:
    - Dedicated exploiter mode now schedules one BR job per unique matchup state.
    - We need to inspect the task checkpoint *before* launching subprocesses.
    - This helper centralizes metadata parsing so scheduling logic stays readable.
    """
    data, _, _ = load_from_zip_file(task_file_path, device=device)
    if "state_list" not in data:
        raise KeyError(f"Task/checkpoint {task_file_path} does not contain 'state_list'.")
    state_list = data["state_list"]
    if not isinstance(state_list, list) or len(state_list) == 0:
        raise ValueError(f"Task/checkpoint {task_file_path} has empty or invalid 'state_list'.")
    return _dedupe_preserve_order(state_list)


def _build_dedicated_job_specs(
    unique_states: List[str],
    replicates_per_matchup: int,
    run_eval_prot: bool,
    run_eval_adv: bool,
    launch_local_br_eval: bool,
) -> List[Dict[str, Any]]:
    """
    Build dedicated BR job specs:
      - one entry per (state, side, replicate)
      - side is controlled by eval_prot/eval_adv flags

    Notes:
    - "replicates_per_matchup" means if set to K, each matchup+side gets K
      independent subprocess jobs (different BR indices/checkpoints).
    - We keep a monotonic "job_index" for easy logging/debugging.
    """
    if replicates_per_matchup < 1:
        raise ValueError("replicates_per_matchup must be >= 1 for dedicated jobs.")

    job_specs: List[Dict[str, Any]] = []
    job_index = 0
    for state in unique_states:
        # Attempt to produce a concise human-readable matchup label from the
        # state string. If parsing fails, fall back to the raw state path.
        try:
            matchup_label_raw = state2matchup(state)
        except Exception:
            matchup_label_raw = state
        matchup_label = _sanitize_for_filename(matchup_label_raw)

        # eval_prot branch in this worker maps to exploiting='ego' downstream.
        # We retain that existing contract and only change scheduling granularity.
        if run_eval_prot:
            for rep in range(replicates_per_matchup):
                job_specs.append(
                    {
                        "job_index": job_index,
                        "eval_prot": True,
                        "state_subset": [state],
                        "matchup_label": matchup_label,
                        "replicate_idx": rep,
                        "launch_local_br_eval": launch_local_br_eval,
                    }
                )
                job_index += 1

        # eval_adv branch should schedule exploiter jobs on the other side
        # (downstream exploiting='adv'), so eval_prot=False here.
        if run_eval_adv:
            for rep in range(replicates_per_matchup):
                job_specs.append(
                    {
                        "job_index": job_index,
                        "eval_prot": False,
                        "state_subset": [state],
                        "matchup_label": matchup_label,
                        "replicate_idx": rep,
                        "launch_local_br_eval": launch_local_br_eval,
                    }
                )
                job_index += 1
    return job_specs


class _FixedMatchupPolicyAdapter:
    """
    Lightweight policy adapter that forces CDS forward passes to a single matchup head.

    Why this exists:
    - `CleanDerivativeFreeSPAR` is architected around a full matchup topology.
    - In dedicated BR runs we intentionally train on one matchup subset.
    - We do NOT want to modify CDS internals, so we adapt behavior at the worker layer.

    Behavior:
    - Delegates all unknown attributes to the wrapped policy.
    - Intercepts `__call__` and injects `network_keys=[fixed_idx]*batch_size`.
    - Preserves `.to(device)` semantics so existing device-transfer code keeps working.
    """

    def __init__(self, base_policy, fixed_matchup_idx: int):
        self._base_policy = base_policy
        self._fixed_matchup_idx = int(fixed_matchup_idx)

    def __call__(self, obs_tensor, *args, **kwargs):
        # Always route every sample in this mini-batch to the same matchup head.
        # This enforces "single-matchup dedicated evaluation/training" semantics
        # even when the original CDS model contains multiple matchup heads.
        batch_size = int(obs_tensor.shape[0])
        #kwargs = dict(kwargs)
        #kwargs["network_keys"] = [self._fixed_matchup_idx] * batch_size

        ego_forward = kwargs['ego_forward']
        adv_forward = kwargs['adv_forward']

        if ego_forward is True and adv_forward is True:
            raise ValueError("cannot have both ego and adv forward -- we can only exploit one at a time")

        if ego_forward:
            exploited_actions, exploited_log_probs = self._base_policy.ego_forward(obs_tensor)
        elif adv_forward:
            exploited_actions, exploited_log_probs = self._base_policy.adv_forward(obs_tensor, buf_num=[self._fixed_matchup_idx])
        else:
            raise ValueError(f"Invalid forward flag: {ego_forward} or {adv_forward}")

        return exploited_actions, exploited_log_probs

    def to(self, device):
        # Keep adapter identity stable while moving underlying policy modules.
        self._base_policy = self._base_policy.to(device)
        return self

    def __getattr__(self, name):
        # Delegate all other attributes/methods transparently.
        return getattr(self._base_policy, name)


def _resolve_matchup_index_for_state(loaded_model, dedicated_state: str) -> int:
    """
    Map a dedicated state string to a CDS matchup-head index.

    We prefer an exact state-index lookup using the checkpoint's unique state order,
    then fall back to matchup-label lookup if needed.
    """
    unique_states = getattr(loaded_model, "_worker_unique_states", None)
    if isinstance(unique_states, list) and dedicated_state in unique_states:
        return unique_states.index(dedicated_state)

    target_matchup = state2matchup(dedicated_state)
    envs_per_matchup = int(max(1, getattr(loaded_model, "envs_per_matchup", 1)))
    matchups = getattr(loaded_model, "matchups", None) or []
    for i in range(0, len(matchups), envs_per_matchup):
        if matchups[i] == target_matchup:
            return i // envs_per_matchup
    raise ValueError(
        f"Could not resolve matchup index for dedicated state '{dedicated_state}'."
    )


def load_spar_model(
    game_args: dict,
    task_file_path: str,
    n_envs: int = 2,
    device: str = "cuda",
    state_subset: Optional[List[str]] = None,
):
    worker_id = os.getpid()
    print(f"WORKER [{worker_id}]: Processing task: {os.path.basename(task_file_path)}")

    #try:
    # The task file IS the model checkpoint file, just renamed.
    checkpoint_path = task_file_path

    # Extract timestep from the checkpoint filename for wandb logging
    try:
        basename = os.path.basename(checkpoint_path)
        # Assumes format like '..._12345_steps.task'
        timestep_str = basename.replace('.task', '').split('_')[-2]
        ego_timestep = int(timestep_str)
    except (IndexError, ValueError):
        print(f"WORKER [{worker_id}]: Could not parse timestep from filename: {basename}. BR win rate will not be logged against a specific step.")
        ego_timestep = None

    
    # Default load_from_zip_file(..., device="auto") maps to CUDA and allocates GPU weights
    # even though we only need ``data`` here; always respect the worker device.
    data, _, _ = load_from_zip_file(checkpoint_path, device=device)
    # IMPORTANT:
    # Keep CDS model topology in its original full-matchup form. Even in
    # dedicated runs we should NOT shrink the loaded CDS state/matchup metadata,
    # because internal forward/update logic assumes the full topology.
    #
    # Dedicated isolation is handled later by:
    #   1) dedicated BR env construction (subset env for the exploiter), and
    #   2) `_FixedMatchupPolicyAdapter` (pins exploited CDS head selection).
    uniques = _dedupe_preserve_order(data["state_list"])
    # Build the concrete STATE list passed to env/model, repeating each unique
    # state n_envs times to match existing worker conventions.
    STATE = [state for state in uniques for _ in range(n_envs)]
    game_args = argparse.Namespace(**game_args)
    

    # argument to STATE should be the master list of states, including repeats for repeated matchups.
    # we should always follow this convention -- when there is a discrepancy
    # we should default to STATE.

    


    env = env_generator(game_args, STATE=STATE)
    #env.num_envs = 1 # HACKY FOR NOW!
    try:
        ftm = CleanDerivativeFreeSPAR.load(
            path=checkpoint_path, env=env, game_args=game_args, num_perturbed=1, device=device
        )
        #if ftm.policy.num_env_per_adv is None:
        #    ftm.policy.num_env_per_adv = ftm.envs_per_matchup
    except Exception as e:
        data, params, pytorch_variables = load_from_zip_file(
            checkpoint_path, device=device)
        ftm = CleanDerivativeFreeSPAR(
            "AACCnnPolicy",
            env,
            device=device,
            verbose=2,
            n_steps=256,
            batch_size=512,
            n_epochs=1,
            state_list=STATE,
            envs_per_matchup=1,
            env_generator_func=env_generator,
            num_adversaries=1,
            n_env_per_adv=1,
            seed= 0,
            target_kl=0.025,
            use_mirror=False
        )
        ftm.set_parameters(params, exact_match=True, device=ftm.device)
    # Keep a stable copy of the unique checkpoint state ordering for dedicated
    # matchup-index resolution. This is worker metadata only (no CDS changes).
    ftm._worker_unique_states = list(uniques)
    ftm._worker_full_state_list = list(STATE)
    return ftm


def train_best_response(
    game_args: dict,
    model_to_exploit,
    task_file_path: str,
    eval_prot: bool,
    use_mirror: bool,
    eval_only: bool,
    proj_name: str,
    analysis_upload_proj_name: str,
    n_envs: int,
    is_spar: bool = False,
    br_index: int = 0,
    from_scratch: bool = False,
    exploiter_save_freq: int = 100000,
    br_tracker_patience: int = 10,
    br_tracker_tolerance: float = 1e-4,
    br_tracker_window_size: int = 50,
    device: str = "cuda",
    dedicated_state_subset: Optional[List[str]] = None,
    matchup_label: Optional[str] = None,
    replicate_idx: Optional[int] = None,
    dedicated_job_id: Optional[int] = None,
    launch_local_br_eval: bool = False,
) -> None:
    """
    The core logic for a single best-response training run.

    Args:
        TODO: Complete this.
    """
    checkpoint_path = task_file_path
    done_model_checkpoint_path = os.path.join(DONE_DIR, os.path.basename(checkpoint_path))
    ftm = model_to_exploit
    # --- This is where your specific BR logic goes ---
    # 1. Load the frozen opponent
    # fixed_opponent = PPO.load(checkpoint_path)
    wandb.init(project=proj_name,
               entity='jw4406',
               group="br_workers",
               config={"eval_rew": 0,
                       "exploiter_rew": 0,
                       "epochs": 0,
                       "br_wr": 0,
                       "main_training_epoch": 0,
                       "torch_device": device,
                       })
    # 2. Create your environment, passing the frozen opponent to it
    #    so the BR agent can play against it.
    # env = YourStreetFighterEnv(opponent_policy=fixed_opponent)
    if is_spar == True:
        game_args = argparse.Namespace(**game_args)
        # In dedicated mode, train BR in a subset environment (usually one
        # matchup replicated `n_envs` times). This isolates BR rollouts while
        # leaving the exploited CDS model topology untouched.
        effective_state_list = ftm.state_list if dedicated_state_subset is None else dedicated_state_subset
        env = env_generator(game_args, STATE=effective_state_list, n_envs=n_envs)
        # if eval_prot is True: # we're training an optimal adversary
        #     dstb_action_space = Box(low=ftm.dstb_action_space.low, high=ftm.dstb_action_space.high, shape=ftm.dstb_action_space.shape)
        #     env.action_space = dstb_action_space
        # else:
        #     assert eval_prot is False 
        #     # we're training an optimal ego against the current adversary
        #     ego_action_space = Box(low=ftm.action_space.low, high=ftm.action_space.high, shape=ftm.action_space.shape)
        #     env.action_space = ego_action_space
    else:
        # NOT SURE WHAT TO DO HERE ABOUT LEAGUE MODELS
        env = env_generator(STATE=STATE)

    # 3. Create a new agent to be the best response
    br_agent = Exploiter(
        'CnnPolicy' if is_image_space(env.observation_space) else 'MlpPolicy',
        env,
        device=device,
        exploited=ftm,
        n_steps=2048,
        batch_size=512,
        n_epochs=5,
        exploiting='ego' if eval_prot is True else 'adv',
        br_tracker_patience=br_tracker_patience,
        br_tracker_tolerance=br_tracker_tolerance,
        br_tracker_window_size=br_tracker_window_size,
    )
    br_agent.is_spar = is_spar # TODO: This is a stupid hack to get the BR agent to know if it is a SPAR model or not. Remove this once we have a better way to do this.
    # 4. Train the BR agent
    #
    # Name each BR checkpoint with rich metadata so downstream analysis can
    # directly identify:
    # - which side was exploited (ego/adv),
    # - which matchup slice this dedicated run belongs to,
    # - which replicate index it is within that matchup.
    #
    # This makes your "which character matchup performs worst" analysis much
    # easier because filenames become self-describing.
    exploit_side = "ego" if eval_prot is True else "adv"
    dedicated_suffix_parts: List[str] = []
    if matchup_label is not None:
        dedicated_suffix_parts.append(f"matchup_{_sanitize_for_filename(matchup_label)}")
    if replicate_idx is not None:
        dedicated_suffix_parts.append(f"rep_{replicate_idx}")
    if dedicated_job_id is not None:
        dedicated_suffix_parts.append(f"job_{dedicated_job_id}")
    dedicated_suffix = "_".join(dedicated_suffix_parts)
    if dedicated_suffix != "":
        dedicated_suffix = f"_{dedicated_suffix}"
    br_model_name = (
        f"br{br_index}_to_{os.path.splitext(os.path.basename(checkpoint_path))[0]}"
        f"_exploiting_{exploit_side}{dedicated_suffix}.zip"
    )
    exploiter_callback = ExploiterCheckpointCallback(save_freq=exploiter_save_freq // n_envs, save_path=BR_MODEL_DIR, name_prefix=br_model_name)
     
    if eval_only == False:
        print("eval_only was passed as False. Training the BR agent.")
        if from_scratch == True:
            br_agent.learn(total_timesteps=BR_TRAINING_STEPS, callback=exploiter_callback)
        else:
            # if eval prot is True we are training an optimal adversary so we need to update the adversary
            # if eval prot is False we are training an optimal ego against the current adversary so we need to update the ego
            ftm.env = env # this is new (17:16)
            ftm.envs_per_matchup = ftm.envs_per_matchup
            ftm.policy.num_env_per_adv = ftm.envs_per_matchup
            ftm.policy.envs_per_matchup = ftm.envs_per_matchup
            ftm.exploited = None
            ftm.training_br = True
            ftm.br_tracker_patience = br_tracker_patience
            ftm.br_tracker_tolerance = br_tracker_tolerance
            ftm.br_tracker_window_size = br_tracker_window_size
            ftm.learn(total_timesteps=BR_TRAINING_STEPS, callback=exploiter_callback, update_ego=not eval_prot, update_adversary=eval_prot)
        #br_agent.learn(total_timesteps=BR_TRAINING_STEPS, callback=exploiter_callback)

        local_plot_and_eval_file = os.path.join(current_dir, "local_br_eval.py")
        
        br_interval_num = exploiter_callback.n_calls * env.num_envs // exploiter_callback.save_freq
        #br_model_path = os.path.join(BR_MODEL_DIR, f"{br_model_name}_{br_interval_num}000_steps.zip")
        br_model_path = exploiter_callback.model_path
        if launch_local_br_eval:
            subprocess.Popen(["python", local_plot_and_eval_file, 
            "--eval_prot", str(eval_prot),
            "--main_checkpoint_model_path", checkpoint_path,
            "--done_model_checkpoint_path", done_model_checkpoint_path,
            "--br_checkpoint_model_path", br_model_path,
            # Evaluate exactly the state slice used by this BR run.
            "--state_list", str(effective_state_list),
            "--exploiter_is_cds", str(not from_scratch),
            "--br_index", str(br_index),
            "--game_args", json.dumps(vars(game_args)),
            "--device", device,
            ])
        #agg_file = os.path.join(current_dir, "aggregate_to_wandb.py")
        #subprocess.Popen(["python", agg_file, "--read_from_proj_name", proj_name, "--upload_to_proj_name", analysis_upload_proj_name])


def run_br_for_task_in_subprocess(
    game_args: dict,
    task_file_path: str,
    eval_prot: bool,
    use_mirror: bool,
    eval_only: bool,
    proj_name: str,
    analysis_upload_proj_name: str,
    n_envs: int,
    is_spar: bool,
    br_index: int,
    from_scratch: bool = False,
    exploiter_save_freq: int = 100000,
    br_tracker_patience: int = 10,
    br_tracker_tolerance: float = 1e-4,
    br_tracker_window_size: int = 50,
    device: str = "cuda",
    state_subset: Optional[List[str]] = None,
    matchup_label: Optional[str] = None,
    replicate_idx: Optional[int] = None,
    dedicated_job_id: Optional[int] = None,
    launch_local_br_eval: bool = False,
) -> None:
    """
    Worker function for running a single BR training instance in a separate process.
    Each subprocess loads its own copy of the model to avoid pickling issues.
    """
    _ensure_cpu_only_env(device)
    if is_spar:
        # NOTE: Do not shrink CDS model topology based on state_subset.
        # We always load full topology and handle dedicated constraints via
        # env slicing + fixed-head policy adapter below.
        loaded_model = load_spar_model(
            game_args,
            task_file_path,
            n_envs=n_envs,
            device=device,
        )
        dedicated_state_list_for_env = None
        if state_subset is not None:
            if len(state_subset) == 0:
                raise ValueError("state_subset was provided but empty.")
            dedicated_state = state_subset[0]
            dedicated_state_list_for_env = [dedicated_state for _ in range(n_envs)]
            fixed_matchup_idx = _resolve_matchup_index_for_state(loaded_model, dedicated_state)

            # Wrap exploited CDS policy so forward calls always use the selected
            # matchup head. This avoids CDS full-topology assumptions breaking
            # while still allowing single-matchup BR specialization.
            loaded_model.policy = _FixedMatchupPolicyAdapter(
                loaded_model.policy, fixed_matchup_idx=fixed_matchup_idx
            )
            print(
                "Dedicated CDS adapter configured: "
                f"state={dedicated_state}, fixed_matchup_idx={fixed_matchup_idx}, "
                f"replicated_env_count={n_envs}"
            )
        if exploiter_save_freq * n_envs * len(loaded_model.matchups) > BR_TRAINING_STEPS:
            print("-------------------------------------------\n\n ")
            print("ERROR!")
            print("ERROR! Exploiter save frequency is greater than BR training steps. This will result in no exploiter checkpoints being saved AND AN ERROR RIGHT BEFORE LOCAL BR EVAL")
            print("ERROR!")
            print("-------------------------------------------\n\n ")
            quit() # QUIT THE PROGRAM
        loaded_model.c_learning_rate = 1e-4
        loaded_model.d_learning_rate = 2e-4
        loaded_model.v_learning_rate = 5e-4
        loaded_model.policy.ctrl_optimizer.param_groups[0]['lr'] = 1e-4
        loaded_model.policy.dstb_optimizer.param_groups[0]['lr'] = 1e-4
        loaded_model.policy.value_optimizer.param_groups[0]['lr'] = 2e-4
        loaded_model.use_lr_annealing = False
    else:
        raise NotImplementedError("Non-SPAR multiprocessing BR training is not implemented.")

    train_best_response(
        game_args,
        loaded_model,
        task_file_path,
        eval_prot=eval_prot,
        use_mirror=use_mirror,
        eval_only=eval_only,
        proj_name=proj_name,
        analysis_upload_proj_name=analysis_upload_proj_name,
        n_envs=n_envs,
        is_spar=is_spar,
        br_index=br_index,
        from_scratch = from_scratch,
        exploiter_save_freq=exploiter_save_freq,
        br_tracker_patience=br_tracker_patience,
        br_tracker_tolerance=br_tracker_tolerance,
        br_tracker_window_size=br_tracker_window_size,
        device=device,
        dedicated_state_subset=dedicated_state_list_for_env,
        matchup_label=matchup_label,
        replicate_idx=replicate_idx,
        dedicated_job_id=dedicated_job_id,
        launch_local_br_eval=launch_local_br_eval,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_prot", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--eval_adv", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--eval_only", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--proj_name", type=str, required=True)
    parser.add_argument("--analysis_upload_proj_name", type=str, required=True)
    parser.add_argument("--load_br", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--which_env", choices=['my_pendulum', 'my_walker2d_v4', 'my_mountain_car_continuous', 'my_half_cheetah', 'my_hopper', 'my_ant'], required=True)
    parser.add_argument("--is_league", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--use_mirror", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--task_dir", type=str, required=False)
    parser.add_argument("--num_brs", type=int, default=6, help="Number of independent BR agents to train per main checkpoint.")
    parser.add_argument("--n_envs", type=int, default=2, help="Number of environments to run in parallel.")
    parser.add_argument("--DEBUG", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--num_full_exploiters", type=int, default=4, help="Number of full exploiters to train.")
    parser.add_argument("--num_continue_exploiters", type=int, default=4, help="Number of continue exploiters to train.")
    parser.add_argument("--dedicated_exploiter", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--continue_exploiters", choices=['True', 'False'], default='False', required=True)
    parser.add_argument("--exploiter_save_freq", type=int, required=True, default=100000, help="Frequency of exploiter checkpoint saves.")
    parser.add_argument("--br_tracker_patience", type=int, default=10, help="Patience (in checks) for BR convergence early stopping.")
    parser.add_argument("--br_tracker_tolerance", type=float, default=1e-4, help="Tolerance for BR convergence stagnation checks.")
    parser.add_argument("--br_tracker_window_size", type=int, default=50, help="Window size used to smooth BR convergence metric.")

    parser.add_argument('--reset', choices=['round', 'match', 'game'],help='Reset stats for a round, a match, or the whole game', default='round')
    parser.add_argument("--side", type=str, help="Side", default="left", required=True, choices=["left", "right", "both"])

    parser.add_argument('--render', choices=['True', 'False'], help='Whether to render the game screen', default='False')
    parser.add_argument('--enable_combo', choices=['True', 'False'], help='Enable special move action space for environment', default='True')
    parser.add_argument('--null_combo', choices=['True', 'False'], help='Null action space for special move', default='False')
    parser.add_argument('--transform_action', choices=['True', 'False'], help='Transform action space to MultiDiscrete', default='False')
    parser.add_argument('--seed', type=int, help='Seed', default=0)
    parser.add_argument('--launch_local_br_eval', choices=['True', 'False'], help='Launch local br eval', default='False')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Torch device for policies and training (e.g. cpu, cuda, cuda:0)',
    )
    # whether or not we want to do full br training or continuing the CDS as exploiter
    args = parser.parse_args()
    _ensure_cpu_only_env(args.device)
    args.render = True if args.render == 'True' else False
    args.enable_combo = True if args.enable_combo == 'True' else False
    args.null_combo = True if args.null_combo == 'True' else False
    args.transform_action = True if args.transform_action == 'True' else False
    game_args = {
        "reset": args.reset,
        "side": args.side,
        "render": args.render,
        "enable_combo": args.enable_combo,
        "null_combo": args.null_combo,
        "transform_action": args.transform_action,
        "seed": args.seed,
    }
    args.DEBUG = args.DEBUG == 'True'
    args.dedicated_exploiter = args.dedicated_exploiter == 'True'
    args.continue_exploiters = args.continue_exploiters == 'True'
    args.launch_local_br_eval = args.launch_local_br_eval == 'True'


    args.eval_prot = args.eval_prot == 'True'
    args.eval_adv = args.eval_adv == 'True'
    if args.eval_only == 'True':
        print("WARNING!")
        print("This is an EVAL ONLY run. No exploiter training will be performed.")
        print("WARNING!")
    if not args.eval_prot or not args.eval_adv:
        print("WARNING!")
        training_type = "adversary" if args.eval_prot else "ego " # if eval_prot is True we are training an optimal adversary so we need to update the adversary
        print(f"WARNING! Only {training_type} training will be performed.")
    print("Dedicated exploiter: ", args.dedicated_exploiter)
    print("Number of full exploiters: ", args.num_full_exploiters)
    print("Continue exploiters: ", args.continue_exploiters)
    print("Number of continue exploiters: ", args.num_continue_exploiters)
    print("Number of environments: ", args.n_envs)
    print("Launch local br eval: ", args.launch_local_br_eval)
    if args.exploiter_save_freq * args.n_envs > BR_TRAINING_STEPS:
        print("WARNING! Exploiter save frequency is greater than BR training steps. This will result in no exploiter checkpoints being saved AND AN ERROR RIGHT BEFORE LOCAL BR EVAL")
    args.eval_only = args.eval_only == 'True'
    wandb.login(key='d95a51c4001b862123a34a3853fe0306906d2f07')
    todo_dir = os.path.join(TASK_DIR, "todo")

    if args.task_dir is not None and args.task_dir != "":
        print(f"WARNING: Using custom task directory: {args.task_dir}")
        todo_dir = os.path.join(args.task_dir, "todo")
    processing_dir = os.path.join(TASK_DIR, "processing")
    error_dir = os.path.join(TASK_DIR, "error")
    done_dir = os.path.join(TASK_DIR, "done")
    stop_file = os.path.join(TASK_DIR, "STOP")
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # if os.path.isfile(curr_dir + "/myfile.txt"):
    #     import json
    #     test = json.load(open(curr_dir + "/myfile.txt"))
    #     print(test)
    # else:
    #     print("myfile.txt does not exist")

    print(f"WORKER [{os.getpid()}]: Starting. Watching {todo_dir} for tasks. device={args.device}")
    if args.is_league == 'False' and args.load_br == 'False':
        while not os.path.exists(stop_file):
            tasks = [f for f in os.listdir(todo_dir) if f.endswith(".task")]

            if not tasks:
                time.sleep(POLL_INTERVAL)
                continue

            # Grab a random task to reduce the chance of multiple workers grabbing the same one
            task_filename = random.choice(tasks)
            todo_path = os.path.join(todo_dir, task_filename)
            processing_path = os.path.join(processing_dir, task_filename)
            error_path = os.path.join(error_dir, task_filename)

            try:
                # Atomically move the task file to claim it
                os.rename(todo_path, processing_path)

                # Now that we've claimed it, process it
                # if args.num_brs == 1:
                #     loaded_model = load_spar_model(processing_path)
                #     train_best_response(
                #         loaded_model,
                #         processing_path,
                #         eval_prot=args.eval_prot,
                #         use_mirror=args.use_mirror,
                #         eval_only=args.eval_only,
                #         proj_name=args.proj_name,
                #         analysis_upload_proj_name=args.analysis_upload_proj_name,
                #         is_spar=True,
                #         br_index=2,
                #         from_scratch=True,
                #     )
                # else:
                processes = []

                def _launch_job(
                    eval_prot_flag: bool,
                    br_idx: int,
                    from_scratch: bool,
                    state_subset: Optional[List[str]] = None,
                    matchup_label: Optional[str] = None,
                    replicate_idx: Optional[int] = None,
                    dedicated_job_id: Optional[int] = None,
                    launch_local_br_eval: bool = False,
                ) -> None:
                    """
                    Launch one BR subprocess (or run inline in DEBUG mode).

                    Why this helper is nested here:
                    - It captures current task-specific context (e.g. processing_path).
                    - It removes duplicated argument packing across the many loops.
                    - It ensures every launch path (continue + dedicated) uses the
                      same subprocess signature and consistent debug prints.
                    """
                    target = run_br_for_task_in_subprocess
                    training_args = (
                        game_args,
                        processing_path,
                        eval_prot_flag,
                        args.use_mirror,
                        args.eval_only,
                        args.proj_name,
                        args.analysis_upload_proj_name,
                        args.n_envs,
                        True,  # is_spar
                        br_idx,
                        from_scratch,
                        args.exploiter_save_freq,
                        args.br_tracker_patience,
                        args.br_tracker_tolerance,
                        args.br_tracker_window_size,
                        args.device,
                        state_subset,
                        matchup_label,
                        replicate_idx,
                        dedicated_job_id,
                        launch_local_br_eval,
                    )
                    if args.DEBUG:
                        print(
                            "DEBUG: Launching BR job "
                            f"idx={br_idx}, eval_prot_flag={eval_prot_flag}, "
                            f"from_scratch={from_scratch}, matchup={matchup_label}, "
                            f"replicate={replicate_idx}, dedicated_job_id={dedicated_job_id}, "
                            f"launch_local_br_eval={launch_local_br_eval}"
                        )
                        target(*training_args)
                    else:
                        p = mp.Process(target=target, args=training_args)
                        p.start()
                        processes.append(p)

                # Continue-exploiter mode (existing behavior): run a fixed number
                # of jobs against the full checkpoint state list.
                if args.continue_exploiters:
                    from_scratch = False
                    if args.eval_prot:
                        for br_idx in range(args.num_continue_exploiters):
                            _launch_job(eval_prot_flag=True, br_idx=br_idx, from_scratch=from_scratch, launch_local_br_eval=args.launch_local_br_eval)
                    if args.eval_adv:
                        # IMPORTANT: eval_adv jobs must set eval_prot_flag=False so
                        # train_best_response configures Exploiter(exploiting='adv').
                        for br_idx in range(args.num_continue_exploiters):
                            _launch_job(eval_prot_flag=False, br_idx=br_idx, from_scratch=from_scratch, launch_local_br_eval=args.launch_local_br_eval)

                # Dedicated mode (new behavior): schedule one job per unique
                # matchup state per side, with `num_full_exploiters` replicates
                # for each matchup+side pair.
                if args.dedicated_exploiter:
                    unique_states = _extract_unique_states_from_task(processing_path, device=args.device)
                    dedicated_specs = _build_dedicated_job_specs(
                        unique_states=unique_states,
                        replicates_per_matchup=args.num_full_exploiters,
                        run_eval_prot=args.eval_prot,
                        run_eval_adv=args.eval_adv,
                        launch_local_br_eval=args.launch_local_br_eval,
                    )
                    print(
                        "Dedicated exploiter scheduling:\n"
                        f"  unique_matchups={len(unique_states)}\n"
                        f"  replicates_per_matchup={args.num_full_exploiters}\n"
                        f"  total_jobs={len(dedicated_specs)}"
                    )
                    for spec in dedicated_specs:
                        if spec['job_index'] == 1:
                            print("hello")
                        # We use spec["job_index"] as br_idx so each dedicated run
                        # gets a globally unique BR index for this task.
                        _launch_job(
                            eval_prot_flag=spec["eval_prot"],
                            br_idx=spec["job_index"],
                            from_scratch=True,
                            state_subset=spec["state_subset"],
                            matchup_label=spec["matchup_label"],
                            replicate_idx=spec["replicate_idx"],
                            dedicated_job_id=spec["job_index"],
                            launch_local_br_eval=args.launch_local_br_eval,
                        )
                        print("Complete job: ", spec["job_index"])

                # Wait for all BR processes to finish before marking task as done.
                if not args.DEBUG:
                    for p in processes:
                        p.join()

                # Move it to 'done' when finished
                done_path = os.path.join(done_dir, task_filename)
                print("Finished processing task: ", task_filename)
                #os.rename(processing_path, done_path)

            except FileNotFoundError:
                # Another worker grabbed this file first. No problem.
                continue
            except Exception as e:
                print(f"WORKER [{os.getpid()}]: A critical error occurred. Error: {e}")
                # Move the failed task back to todo or to an error folder
                try:
                    os.rename(processing_path, error_path)
                except:
                    pass
    elif args.is_league == 'True' and args.load_br == 'False':
        model_files, payoff_path = load_league_models(model_dir=args.league_model_dir, character_names=["ryu", "bison", "guile"])
        loaded_league = instantiate_league_models(model_files, character_names=["ryu", "bison", "guile"])
        main_agent_left = loaded_league.get_player(0).agent
        train_best_response(main_agent_left, payoff_path, eval_prot=args.eval_prot, use_mirror=args.use_mirror, eval_only=args.eval_only, proj_name=args.proj_name, is_spar=False, analysis_upload_proj_name=args.analysis_upload_proj_name)
    else:
        while not os.path.exists(stop_file):
            tasks = [f for f in os.listdir(BR_MODEL_DIR) if f.endswith(".task")]

            if not tasks:
                time.sleep(POLL_INTERVAL)
                continue
            ep_info_buffers = []
            rew_arr = np.zeros(len(tasks))
            # Grab a random task to reduce the chance of multiple workers grabbing the same one
            for i in range(len(tasks)):
                loaded_model = Exploiter.load(os.path.join(BR_MODEL_DIR, tasks[i]), env=exploiter_env_generator(STATE=STATE))
                ep_info_buffers.append(loaded_model.ep_info_buffer)
                for j in range(len(loaded_model.ep_info_buffer)):
                    rew_arr[i] = rew_arr[i] + loaded_model.ep_info_buffer[j]['r']
                rew_arr[i] = rew_arr[i] / len(loaded_model.ep_info_buffer)
            
            print("hello")
    print(f"WORKER [{os.getpid()}]: Stop file detected. Shutting down.")
