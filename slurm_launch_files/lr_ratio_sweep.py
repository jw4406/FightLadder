#!/usr/bin/env python3
"""Two-timescale learning-rate *ratio* sweep for SPAR main-training + dedicated BR.

Fixes ``c_lr`` (ego / slow outer player) and sweeps the ratios that set the
two-timescale separation, enforcing ego < adversary < critic:

    d_lr = c_lr * m_d          (m_d > 1)   -> adversary faster than ego
    v_lr = d_lr * m_v          (m_v > 1)   -> critic faster than adversary

Per config (tag = ``md<m_d>_mv<m_v>``) everything is isolated under a
*deterministic* per-config tree ``$WORKDIR/lr_sweep/<tag>/FightLadder`` -- the
same shape as the existing ``$WORKDIR/$JOBID`` convention, but pre-known so BR
can be paired without chasing ``$SLURM_JOB_ID``. Tasks, checkpoints,
``br_rewards/`` and ``br_models/`` all live there, and ``MAIN_TRAINING_DIR=
lr_sweep/<tag>`` wires training -> orchestrator -> exploiters consistently.

Phases (both are pure ``sbatch`` submission -> login-node safe; the long-running
watchdog runs *inside* a job, never on the login node):

  --phase train : render + sbatch a main-training .slurm (repo copied into the
                  per-config tree; LRs + matchup substituted).
  --phase br    : render + sbatch a CPU-only orchestrator .slurm that runs
                  br_slurm_orchestrator.py on a compute node, watching that
                  config's tasks and firing exploiter jobs as checkpoints stream
                  in (exploitability curves).
  --phase both  : both.

SAFETY: ``--dry_run True`` (default) renders the .slurm files and prints the
exact ``sbatch`` commands, but submits nothing. Inspect, then ``--dry_run False``.
"""
import argparse
import os
import re
import subprocess
import sys


CLUSTER_BR_TEMPLATE = {
    "neuronic": "br_dedicated_template_NEURONIC.slurm",
    "della": "br_dedicated_template_DELLA.slurm",
}


def _fmt_lr(x):
    """Exact, compact LR string: 6 significant figures, trailing zeros stripped
    (e.g. 1e-5 -> '1e-05', 1.6e-4 -> '0.00016', 6.4e-4 -> '0.00064'). `.0e`
    would round multi-sig-fig ratios (1.6e-4 -> '2e-04'), corrupting the LR."""
    return f"{x:.6g}"


def _sanitize(lr_str):
    return lr_str.replace(".", "p").replace("-", "m").replace("+", "")


def compute_configs(c_lr, d_mults, v_mults, max_v_lr):
    """Yield dicts for each valid (m_d, m_v): enforces c<d<v and v_lr<=max_v_lr."""
    configs, skipped = [], []
    for m_d in d_mults:
        for m_v in v_mults:
            if m_d <= 1 or m_v <= 1:
                skipped.append((m_d, m_v, "multiplier must be > 1"))
                continue
            d_lr = c_lr * m_d
            v_lr = d_lr * m_v
            if not (c_lr < d_lr < v_lr):
                skipped.append((m_d, m_v, "ordering c<d<v violated"))
                continue
            if v_lr > max_v_lr:
                skipped.append((m_d, m_v, f"v_lr {v_lr:.2e} > max_v_lr {max_v_lr:.2e}"))
                continue
            configs.append({
                "m_d": m_d, "m_v": m_v, "tag": f"md{m_d:g}_mv{m_v:g}",
                "c_lr": c_lr, "d_lr": d_lr, "v_lr": v_lr,
            })
    return configs, skipped


def _apply(text, subs, what):
    for pattern, repl in subs:
        text, n = re.subn(pattern, repl, text)
        if n == 0:
            print(f"  [warn] pattern not found in {what}: {pattern}", file=sys.stderr)
    return text


def render_training_slurm(template_text, cfg, players, opponents, workdir,
                          sbatch_time, total_timesteps):
    """Substitute LRs / matchup / paths into the training template, and pin the
    run dir to a deterministic per-config tree (JOBID -> lr_sweep/<tag>) so
    training, tasks, checkpoints and BR outputs all share one location."""
    players_arr = " ".join(f'"{p}"' for p in players)
    opps_arr = " ".join(f'"{o}"' for o in opponents)
    subs = [
        (r"(?m)^(#SBATCH --job-name=)\S+", rf"\g<1>spar_{cfg['tag']}"),
        (r"(?m)^(#SBATCH --time=)\S+", rf"\g<1>{sbatch_time}"),
        (r"(?m)^WORKDIR=\S+$", f"WORKDIR={workdir}"),
        # Deterministic per-config run dir instead of $SLURM_JOB_ID.
        (r"(?m)^JOBID=\S+$", f"JOBID=lr_sweep/{cfg['tag']}"),
        # Avoid FightLadder/FightLadder nesting if the tree already exists (re-run).
        (r"(?m)^cp -r \$HOME/FightLadder/ \./$",
         "rm -rf ./FightLadder\ncp -r $HOME/FightLadder/ ./"),
        (r"(?m)^PLAYER=\(.*\)$", f"PLAYER=({players_arr})"),
        (r"(?m)^OPPONENTS=\(.*\)$", f"OPPONENTS=({opps_arr})"),
        (r'(?m)^C_LR=".*"$', f'C_LR="{_fmt_lr(cfg["c_lr"])}"'),
        (r'(?m)^D_LR=".*"$', f'D_LR="{_fmt_lr(cfg["d_lr"])}"'),
        (r'(?m)^V_LR=".*"$', f'V_LR="{_fmt_lr(cfg["v_lr"])}"'),
        (r'(?m)^TOTAL_TIMESTEPS=".*"$', f'TOTAL_TIMESTEPS="{total_timesteps}"'),
    ]
    return _apply(template_text, subs, "training template")


def render_orchestrator_job(template_text, cfg, cluster, workdir, br_job_time,
                            slurm_log_dir, br_training_steps, exploiter_save_freq,
                            step_stride, periodic_eval_freq, br_slurm_time, orch_dry_run):
    """Substitute per-config values into the CPU-only BR-orchestrator job template.
    Paths derive from $WORKDIR/$MAIN_TRAINING_DIR inside the template."""
    subs = [
        (r"(?m)^(#SBATCH --job-name=)\S+", rf"\g<1>brorch_{cfg['tag']}"),
        (r"(?m)^(#SBATCH --time=)\S+", rf"\g<1>{br_job_time}"),
        (r'(?m)^WORKDIR=".*"$', f'WORKDIR="{workdir}"'),
        (r'(?m)^MAIN_TRAINING_DIR=".*"$', f'MAIN_TRAINING_DIR="lr_sweep/{cfg["tag"]}"'),
        (r'(?m)^BR_DEDICATED_TEMPLATE=".*"$',
         f'BR_DEDICATED_TEMPLATE="$REPO_DIR/slurm_launch_files/{CLUSTER_BR_TEMPLATE[cluster]}"'),
        (r'(?m)^SLURM_LOG_DIR=".*"$', f'SLURM_LOG_DIR="{slurm_log_dir}"'),
        (r'(?m)^STEP_STRIDE=".*"$', f'STEP_STRIDE="{step_stride}"'),
        (r'(?m)^BR_TRAINING_STEPS=".*"$', f'BR_TRAINING_STEPS="{br_training_steps}"'),
        (r'(?m)^EXPLOITER_SAVE_FREQ=".*"$', f'EXPLOITER_SAVE_FREQ="{exploiter_save_freq}"'),
        (r'(?m)^PERIODIC_EVAL_FREQ=".*"$', f'PERIODIC_EVAL_FREQ="{periodic_eval_freq}"'),
        (r'(?m)^BR_SLURM_TIME=".*"$', f'BR_SLURM_TIME="{br_slurm_time}"'),
        (r'(?m)^ORCH_DRY_RUN=".*"$', f'ORCH_DRY_RUN="{orch_dry_run}"'),
    ]
    return _apply(template_text, subs, "BR orchestrator template")


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- ratio grid ---
    p.add_argument("--c_lr", type=float, default=1e-5, help="Fixed ego learning rate.")
    p.add_argument("--d_mults", type=float, nargs="+", default=[2, 4, 8, 16],
                   help="d_lr = c_lr * m_d for each m_d (>1).")
    p.add_argument("--v_mults", type=float, nargs="+", default=[2, 4, 8],
                   help="v_lr = d_lr * m_v for each m_v (>1).")
    p.add_argument("--max_v_lr", type=float, default=1e-3,
                   help="Skip configs whose v_lr exceeds this (critic-stability guard).")
    # --- matchup / training ---
    p.add_argument("--player", nargs="+", required=True, help="Protagonist(s).")
    p.add_argument("--opponent_list", nargs="+", required=True, help="Opponent characters.")
    p.add_argument("--main_training_steps", type=int, default=150_000_000)
    p.add_argument("--time", dest="sbatch_time", default="096:00:00",
                   help="#SBATCH --time for each training job (HH:MM:SS).")
    p.add_argument("--workdir", required=True,
                   help="Scratch WORKDIR (e.g. /scratch/gpfs/FISAC/jw4406/).")
    # --- BR ---
    p.add_argument("--cluster", choices=list(CLUSTER_BR_TEMPLATE), default="neuronic")
    p.add_argument("--br_job_time", default="096:00:00",
                   help="Walltime for the orchestrator job (>= training run).")
    p.add_argument("--br_training_steps", type=int, default=99999)
    p.add_argument("--exploiter_save_freq", type=int, default=1000)
    p.add_argument("--step_stride", type=int, default=0,
                   help="0 = BR every checkpoint; e.g. 40000 = every 40k env-steps (curve resolution).")
    p.add_argument("--periodic_eval_freq", type=int, default=500_000,
                   help="Env-steps between mid-training local_br_eval snapshots (curve points).")
    p.add_argument("--br_slurm_time", default="000:12:00",
                   help="#SBATCH --time for each individual exploiter job.")
    p.add_argument("--br_orch_dry_run", choices=["True", "False"], default="False",
                   help="Passed to br_slurm_orchestrator inside the job (False = really submit exploiters).")
    p.add_argument("--slurm_log_dir", default=os.path.expanduser("~/"))
    # --- plumbing ---
    p.add_argument("--phase", choices=["train", "br", "both"], default="both")
    p.add_argument("--template", default=os.path.join(here, "cds_style_template.slurm"))
    p.add_argument("--br_job_template", default=os.path.join(here, "br_orchestrator_job_template.slurm"))
    p.add_argument("--out_dir", default=os.path.join(here, "generated_lr_sweep"),
                   help="Where rendered per-config .slurm files are written.")
    p.add_argument("--dry_run", choices=["True", "False"], default="True",
                   help="Default True: render + print commands but submit nothing.")
    return p.parse_args()


def main():
    args = parse_args()
    dry_run = args.dry_run == "True"

    configs, skipped = compute_configs(args.c_lr, args.d_mults, args.v_mults, args.max_v_lr)
    if skipped:
        print("Skipped configs:")
        for m_d, m_v, why in skipped:
            print(f"  md{m_d:g}_mv{m_v:g}: {why}")
    if not configs:
        sys.exit("No valid configs after applying c<d<v and max_v_lr guards.")

    train_tpl = open(args.template).read() if args.phase in ("train", "both") else None
    br_tpl = open(args.br_job_template).read() if args.phase in ("br", "both") else None
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'MODE: DRY-RUN (nothing submitted)' if dry_run else 'MODE: LIVE'} | "
          f"phase={args.phase} | {len(configs)} configs | c_lr={args.c_lr:g}\n")
    for cfg in configs:
        tag = cfg["tag"]
        tree = f"{args.workdir.rstrip('/')}/lr_sweep/{tag}/FightLadder/main"
        print(f"[{tag}] c={cfg['c_lr']:g} d={cfg['d_lr']:g} v={cfg['v_lr']:g}  "
              f"(m_d={cfg['m_d']:g}, m_v={cfg['m_v']:g})")

        if train_tpl is not None:
            text = render_training_slurm(train_tpl, cfg, args.player, args.opponent_list,
                                         args.workdir, args.sbatch_time, args.main_training_steps)
            path = os.path.join(
                args.out_dir,
                f"main_training_spar_clr{_sanitize(_fmt_lr(cfg['c_lr']))}"
                f"_dlr{_sanitize(_fmt_lr(cfg['d_lr']))}"
                f"_vlr{_sanitize(_fmt_lr(cfg['v_lr']))}_{tag}.slurm")
            with open(path, "w") as f:
                f.write(text)
            print(f"  train : sbatch {path}")
            if not dry_run:
                subprocess.run(["sbatch", path], check=True)

        if br_tpl is not None:
            text = render_orchestrator_job(
                br_tpl, cfg, args.cluster, args.workdir, args.br_job_time,
                args.slurm_log_dir, args.br_training_steps, args.exploiter_save_freq,
                args.step_stride, args.periodic_eval_freq, args.br_slurm_time,
                args.br_orch_dry_run)
            path = os.path.join(args.out_dir, f"br_orchestrator_{tag}.slurm")
            with open(path, "w") as f:
                f.write(text)
            print(f"  br    : sbatch {path}")
            if not dry_run:
                subprocess.run(["sbatch", path], check=True)

        print(f"  tree  : {tree}   (tasks, checkpoints, br_rewards, br_models)")
        print()

    print("After runs produce checkpoints, plot per-config exploitability curves with:\n"
          "  python main/aggregate_local_eval_data.py "
          "--br_rewards_dir <WORKDIR>/lr_sweep/<tag>/FightLadder/main/br_rewards")


if __name__ == "__main__":
    main()
