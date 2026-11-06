"""End-to-end check for the new_br_worker eval-parity fix: load a REAL union/padded (69-wide) spar
checkpoint through load_spar_model and confirm it (a) infers capacity 6, (b) builds a 69-wide env, and
(c) loads without check_for_correct_spaces failing. Needs stable-retro + the ROM -> compute node."""
import sys
from new_br_worker import load_spar_model, _infer_combo_capacity


def main():
    CKPT = sys.argv[1] if len(sys.argv) > 1 else (
        "/n/fs/magics/4086402/FightLadder/main/"
        "minimax_phase0_vtoff_image_rs1.0_GuChVsRyGuCh_SMOKE/trained_models/tasks/todo/"
        "spar_GuCh_RyGuCh_240000_steps.task")

    # game_args mirrors new_br_worker.__main__ + the smoke run's env config.
    game_args = dict(
        reset="round", side="both", render=False, enable_combo=True, null_combo=False,
        transform_action=True, seed=0, obs_type="image", ram_mask="", reward_scale=1.0,
        aggresive_coeff=1.0, decision_timing="joint", dwell_frames=4,
        actionable_statuses="512,514,520", max_skip_frames=90,
    )

    # (1) inference straight off the saved data
    from stable_baselines3.common.save_util import load_from_zip_file
    data, _, _ = load_from_zip_file(CKPT, device="cpu")
    sp = data.get("action_space")
    cap = _infer_combo_capacity(sp)
    print(f"saved action_space={getattr(sp,'nvec',sp)}  inferred capacity={cap}")
    assert cap == 6, f"expected capacity 6 for the union smoke checkpoint, got {cap}"

    # (2)+(3) full load through the patched path (69-wide env; cls.load must pass check_for_correct_spaces)
    ftm = load_spar_model(game_args, CKPT, n_envs=1, device="cpu", use_wandb=False)
    nvec = list(int(x) for x in ftm.action_space.nvec)
    print(f"loaded main action_space nvec={nvec}  arch={getattr(ftm,'_worker_cds_arch','?')}  "
          f"matchups={len(getattr(ftm,'_worker_unique_states',[]))}")
    assert nvec == [69], f"expected 69-wide loaded main, got {nvec}"
    print("test_br_worker_capacity: OK")


# __main__ guard REQUIRED: env_generator's SubprocVecEnv uses spawn, which re-imports this module in
# each emulator child; without the guard the child re-runs the load and fork-bombs.
if __name__ == "__main__":
    main()
