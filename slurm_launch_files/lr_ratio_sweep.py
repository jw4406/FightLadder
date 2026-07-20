#!/usr/bin/env python3
"""Two-timescale learning-rate *ratio* sweep for SPAR main-training + dedicated BR.

Fixes ``c_lr`` (ego / slow outer player) and sweeps the ratios that set the
two-timescale separation, enforcing ego < adversary < critic:

    d_lr = c_lr * m_d          (m_d > 1)   -> adversary faster than ego
    v_lr = d_lr * m_v          (m_v > 1)   -> critic faster than adversary

For each valid ``(m_d, m_v)`` config the driver:

  1. Renders a per-config main-training ``.slurm`` from ``cds_style_template.slurm``
     (same template ``main_training_orchestrator.py`` uses), substituting the
     computed LRs, players/opponents, job-name and ``--time``, and injecting a
     *deterministic* ``SPAR_TASK_DIR`` so training + BR share one task subtree
     without chasing the ephemeral ``$SLURM_JOB_ID``. Submitted via ``sbatch``.

  2. Launches a dedicated ``br_slurm_orchestrator.py`` watchdog pointed at that
     config's task dirs, so BR exploiters fire on checkpoints as they stream in
     (yielding an exploitability *curve* per config, not a single endpoint).

BR reward ``.txt`` files land in ``br_rewards/<subdir>`` keyed by the run prefix
(the config tag rides in via the job-name / save path), and per-config curves are
plotted afterwards with ``aggregate_local_eval_data.py``.

SAFETY: ``--dry_run True`` is the default. In dry-run the driver writes the
rendered ``.slurm`` files and prints the exact ``sbatch`` / orchestrator commands
it *would* run, but submits nothing. Inspect them, then re-run with
``--dry_run False`` to go live.
"""
import argparse
import os
import re
import shlex
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


def _short(name):
    return name[:2]


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
            tag = f"md{m_d}_mv{m_v}"
            configs.append({
                "m_d": m_d, "m_v": m_v, "tag": tag,
                "c_lr": c_lr, "d_lr": d_lr, "v_lr": v_lr,
            })
    return configs, skipped


def render_training_slurm(template_text, cfg, players, opponents, workdir,
                          sbatch_time, total_timesteps, task_base):
    """Substitute LRs / matchup / paths into the training template.

    Mirrors main_training_orchestrator.py's substitutions, and additionally
    injects `export SPAR_TASK_DIR=<task_base>` and repoints SAVE_DIR at it so
    the per-config task subtree is a fixed, pre-known path.
    """
    players_arr = " ".join(f'"{p}"' for p in players)
    opps_arr = " ".join(f'"{o}"' for o in opponents)
    subs = [
        (r"(?m)^(#SBATCH --job-name=)\S+", rf"\g<1>spar_{cfg['tag']}"),
        (r"(?m)^(#SBATCH --time=)\S+", rf"\g<1>{sbatch_time}"),
        (r"(?m)^WORKDIR=\S+$", f"WORKDIR={workdir}"),
        (r"(?m)^PLAYER=\(.*\)$", f"PLAYER=({players_arr})"),
        (r"(?m)^OPPONENTS=\(.*\)$", f"OPPONENTS=({opps_arr})"),
        (r'(?m)^C_LR=".*"$', f'C_LR="{_fmt_lr(cfg["c_lr"])}"'),
        (r'(?m)^D_LR=".*"$', f'D_LR="{_fmt_lr(cfg["d_lr"])}"'),
        (r'(?m)^V_LR=".*"$', f'V_LR="{_fmt_lr(cfg["v_lr"])}"'),
        (r'(?m)^TOTAL_TIMESTEPS=".*"$', f'TOTAL_TIMESTEPS="{total_timesteps}"'),
        # Inject the deterministic per-config task dir + repoint SAVE_DIR at it.
        (r'(?m)^SAVE_DIR=".*"$',
         f'export SPAR_TASK_DIR="{task_base}"\nSAVE_DIR="$SPAR_TASK_DIR/todo/"'),
    ]
    text = template_text
    for pattern, repl in subs:
        text, n = re.subn(pattern, repl, text)
        if n == 0:
            print(f"  [warn] pattern not found in template: {pattern}", file=sys.stderr)
    return text


def br_orchestrator_cmd(cfg, repo_dir, cluster, workdir, task_base,
                        slurm_log_dir, br_training_steps, exploiter_save_freq,
                        step_stride, periodic_eval_freq, br_slurm_time, dry_run):
    """Compose the dedicated br_slurm_orchestrator.py command for one config.

    Mirrors launch_br_orchestrators_<CLUSTER>.sh but with explicit per-config
    task dirs so each config's BR is fully isolated.
    """
    br_template = os.path.join(repo_dir, "slurm_launch_files", CLUSTER_BR_TEMPLATE[cluster])
    return [
        sys.executable, "-u", os.path.join(repo_dir, "main", "br_slurm_orchestrator.py"),
        "--br_dedicated_sh_template", br_template,
        "--workdir", workdir,
        "--main_training_dir", f"lr_sweep/{cfg['tag']}",
        "--todo_dir", os.path.join(task_base, "todo"),
        "--processing_dir", os.path.join(task_base, "slurm_processing"),
        "--done_dir", os.path.join(task_base, "slurm_done"),
        "--stop_file", os.path.join(task_base, "STOP_SLURM"),
        "--local_plot_dir", os.path.join(task_base, "local_entropy_plots"),
        "--slurm_log_dir", slurm_log_dir,
        "--step_stride", str(step_stride),
        "--br_training_steps", str(br_training_steps),
        "--slurm_time", br_slurm_time,
        "--exploiter_save_freq", str(exploiter_save_freq),
        "--launch_local_br_eval", "True",
        "--periodic_eval_freq", str(periodic_eval_freq),
        "--dry_run", str(dry_run),
    ]


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_default = os.path.dirname(here)
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
                   help="Scratch WORKDIR substituted into the template (e.g. /scratch/gpfs/FISAC/jw4406/).")
    # --- BR ---
    p.add_argument("--cluster", choices=list(CLUSTER_BR_TEMPLATE), default="neuronic")
    p.add_argument("--br_training_steps", type=int, default=99999)
    p.add_argument("--exploiter_save_freq", type=int, default=1000)
    p.add_argument("--step_stride", type=int, default=0,
                   help="0 = BR every checkpoint; e.g. 40000 = every 40k env-steps (curve resolution).")
    p.add_argument("--periodic_eval_freq", type=int, default=500_000,
                   help="Env-steps between mid-training local_br_eval snapshots (curve points).")
    p.add_argument("--br_slurm_time", default="000:12:00")
    p.add_argument("--slurm_log_dir", default=os.path.expanduser("~/"))
    # --- plumbing ---
    p.add_argument("--repo_dir", default=repo_default, help="FightLadder checkout root.")
    p.add_argument("--template", default=os.path.join(here, "cds_style_template.slurm"))
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
            print(f"  md{m_d}_mv{m_v}: {why}")
    if not configs:
        sys.exit("No valid configs after applying c<d<v and max_v_lr guards.")

    with open(args.template) as f:
        template_text = f.read()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'MODE: DRY-RUN (nothing submitted)' if dry_run else 'MODE: LIVE'} "
          f"| {len(configs)} configs | c_lr={args.c_lr:g}\n")
    for cfg in configs:
        tag = cfg["tag"]
        task_base = f"{args.workdir.rstrip('/')}/lr_sweep/{tag}/tasks"
        slurm_text = render_training_slurm(
            template_text, cfg, args.player, args.opponent_list, args.workdir,
            args.sbatch_time, args.main_training_steps, task_base,
        )
        out_path = os.path.join(
            args.out_dir,
            f"main_training_spar_clr{_sanitize(_fmt_lr(cfg['c_lr']))}"
            f"_dlr{_sanitize(_fmt_lr(cfg['d_lr']))}"
            f"_vlr{_sanitize(_fmt_lr(cfg['v_lr']))}_{tag}.slurm",
        )
        with open(out_path, "w") as f:
            f.write(slurm_text)

        br_cmd = br_orchestrator_cmd(
            cfg, args.repo_dir, args.cluster, args.workdir, task_base,
            args.slurm_log_dir, args.br_training_steps, args.exploiter_save_freq,
            args.step_stride, args.periodic_eval_freq, args.br_slurm_time, dry_run,
        )

        print(f"[{tag}] c={cfg['c_lr']:g} d={cfg['d_lr']:g} v={cfg['v_lr']:g}  "
              f"(m_d={cfg['m_d']:g}, m_v={cfg['m_v']:g})")
        print(f"  train : sbatch {out_path}")
        print(f"  br    : {' '.join(shlex.quote(c) for c in br_cmd)}")
        print(f"  tasks : {task_base}")

        if not dry_run:
            subprocess.run(["sbatch", out_path], check=True)
            log = os.path.join(args.slurm_log_dir, f"br_orch_{tag}.log")
            with open(log, "w") as lf:
                subprocess.Popen(br_cmd, stdout=lf, stderr=subprocess.STDOUT)
            print(f"  -> submitted; BR orchestrator log: {log}")
        print()

    print("After runs produce checkpoints, plot per-config exploitability curves with:\n"
          "  python main/aggregate_local_eval_data.py --br_rewards_dir <br_rewards> --run <tag>")


if __name__ == "__main__":
    main()
