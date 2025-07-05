import retro
import traceback
from stable_baselines3.common.vec_env import VecEnv
import os

# You'll need the check_env_alive function from the previous step
def check_env_alive(vec_env: VecEnv, step_name: str):
    """A helper function to check if the underlying retro emulator is alive."""
    try:
        raw_env = vec_env.envs[0]
        while hasattr(raw_env, 'env'):
            raw_env = raw_env.env
            if isinstance(raw_env, retro.RetroEnv):
                break

        if hasattr(raw_env, 'em') and raw_env.em is not None:
            print(f"--- [SUCCESS] ENV IS ALIVE after: '{step_name}' ---")
            return True
        else:
            print(f"!!! [FAILURE] ENV IS DEAD after: '{step_name}' !!!")
            return False
    except Exception as e:
        print(f"!!! [FAILURE] FAILED to check env status after '{step_name}': {e}")
        return False


#
# ---- THIS IS THE TEST FUNCTION ----
#
def run_minimal_test(args_tuple):
    # Unpack just what's needed for the environment
    ego_model_path, exploiter_model_path, current_num, player_arg, opponent_list_arg, side_arg, state_list_arg, all_args_dict = args_tuple

    class DummyArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    args = DummyArgs(**all_args_dict)
    eval_state = state_list_arg[0]

    # Use the same make_env function you already have
    def make_env_fn():
        return [
            make_env(sf_game, state=eval_state, side=args.side, reset_type=args.reset, rendering=args.render,
                     enable_combo=args.enable_combo, null_combo=args.null_combo,
                     transform_action=args.transform_action, seed=0)
        ]

    try:
        print(f"\n--- MINIMAL TEST STARTING IN WORKER {os.getpid()} ---")

        # === PHASE 1: Create and destroy the first environment ===
        print("\n[PHASE 1] Creating temporary environment...")
        temp_env = VecTransposeImage2P(DummyVecEnv2P(make_env_fn()))
        if not check_env_alive(temp_env, "temp env creation"): return "FAILED"

        print("[PHASE 1] Closing temporary environment...")
        temp_env.close()
        print("[PHASE 1] Temporary environment closed.")

        # === PHASE 2: Create the second environment ===
        print("\n[PHASE 2] Creating final evaluation environment...")
        final_env = VecTransposeImage2P(DummyVecEnv2P(make_env_fn()))
        if not check_env_alive(final_env, "final env creation"): return "FAILED"

        # === PHASE 3: Attempt to use the second environment ===
        print("\n[PHASE 3] Attempting to reset the final environment...")
        final_env.reset()
        print("--- [SUCCESS] Final environment was reset successfully. ---")

        final_env.close()
        return "SUCCESS"

    except Exception as e:
        print(f"\n--- MINIMAL TEST FAILED IN WORKER {os.getpid()} ---")
        traceback.print_exc()
        return f"FAILED"