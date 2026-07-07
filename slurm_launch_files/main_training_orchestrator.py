#!/usr/bin/env python3
"""Generate per-arch-type SLURM scripts from a template and submit them via sbatch."""

import argparse
import os
import re
import subprocess
import sys


ARCH_CHOICES = ["league", "spar", "ippo", "2timescale"]
# Arch types with a dedicated ego value head -> require --ego_value_head_lr.
EGO_VALUE_HEAD_ARCHES = ("ippo", "2timescale")


def _short(name: str) -> str:
    return name[:2]


def _concat_short(names) -> str:
    return "".join(_short(n) for n in names)


def _sanitize_lr(lr: str) -> str:
    return lr.replace(".", "p")


def _bash_array(values) -> str:
    return " ".join(f'"{v}"' for v in values)


def _render(text: str, arch_type: str, c_lr: str, d_lr: str, v_lr: str,
            players, opponents, main_training_steps: str,
            ego_value_head_lr: str = None) -> str:
    jobname = f"{arch_type}_{_concat_short(players)}_{_concat_short(opponents)}"

    subs = [
        (r"(?m)^(#SBATCH --job-name=)\S+", rf"\g<1>{jobname}"),
        (r"(?m)^PLAYER=\(.*\)$", f"PLAYER=({_bash_array(players)})"),
        (r"(?m)^OPPONENTS=\(.*\)$", f"OPPONENTS=({_bash_array(opponents)})"),
        (r'(?m)^C_LR=".*"$', f'C_LR="{c_lr}"'),
        (r'(?m)^D_LR=".*"$', f'D_LR="{d_lr}"'),
        (r'(?m)^V_LR=".*"$', f'V_LR="{v_lr}"'),
        (r'(?m)^MODEL_ARCH_TYPE=".*"$', f'MODEL_ARCH_TYPE="{arch_type}"'),
        (r'(?m)^TOTAL_TIMESTEPS=".*"$', f'TOTAL_TIMESTEPS="{main_training_steps}"'),
    ]

    for pattern, repl in subs:
        new_text, n = re.subn(pattern, repl, text)
        if n == 0:
            raise RuntimeError(f"Template substitution failed for pattern: {pattern}")
        text = new_text

    # Ego value-head LR: only substituted for arch types that have a dedicated ego
    # value head. For spar/league the template default (EGO_VALUE_HEAD_LR="") is left
    # as-is, so the template omits --ego_value_head_lr and ippo.py treats it as optional.
    if arch_type in EGO_VALUE_HEAD_ARCHES:
        pattern = r'(?m)^EGO_VALUE_HEAD_LR=".*"$'
        new_text, n = re.subn(pattern, f'EGO_VALUE_HEAD_LR="{ego_value_head_lr}"', text)
        if n == 0:
            raise RuntimeError(f"Template substitution failed for pattern: {pattern}")
        text = new_text

    return text


def _output_path(template_path: str, arch_type: str, c_lr: str, d_lr: str,
                 v_lr: str, players, opponents) -> str:
    out_dir = os.path.dirname(os.path.abspath(template_path))
    fname = (
        f"main_training_{arch_type}"
        f"_clr{_sanitize_lr(c_lr)}"
        f"_dlr{_sanitize_lr(d_lr)}"
        f"_vlr{_sanitize_lr(v_lr)}"
        f"_p-{_concat_short(players)}"
        f"_o-{_concat_short(opponents)}"
        f".slurm"
    )
    return os.path.join(out_dir, fname)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main_training_sh_template", required=True,
                   help="Path to the SLURM template script.")
    p.add_argument("--main_training_model_arch_types", required=True, nargs="+",
                   choices=ARCH_CHOICES,
                   help="One or more model arch types; one slurm script per type.")
    p.add_argument("--c_lr", required=True)
    p.add_argument("--d_lr", required=True)
    p.add_argument("--v_lr", required=True)
    # Ego value-head LR. Conditionally required (see validation in main): needed when
    # any requested arch type is ippo/2timescale; optional (unused) otherwise.
    p.add_argument("--ego_value_head_lr", default=None,
                   help="Ego value-head learning rate. REQUIRED when any "
                        "--main_training_model_arch_types is 'ippo' or '2timescale'.")
    p.add_argument("--player", required=True, nargs="+",
                   help="One or more player character names.")
    p.add_argument("--opponent-list", required=True, nargs="+", dest="opponent_list",
                   help="One or more opponent character names.")
    p.add_argument("--main_training_steps", required=True,
                   help="Value to substitute for TOTAL_TIMESTEPS in the template.")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate scripts but skip sbatch submission.")
    return p.parse_args()


def main():
    args = parse_args()

    # ego_value_head_lr is required whenever any requested arch has an ego value head
    # (ippo/2timescale); optional otherwise. Mirrors the check in ippo.py.
    if (any(a in EGO_VALUE_HEAD_ARCHES for a in args.main_training_model_arch_types)
            and args.ego_value_head_lr is None):
        print(
            "ERROR: --ego_value_head_lr is required when any "
            "--main_training_model_arch_types is 'ippo' or '2timescale'",
            file=sys.stderr,
        )
        sys.exit(2)

    template_path = os.path.abspath(args.main_training_sh_template)
    if not os.path.isfile(template_path):
        print(f"ERROR: template not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    with open(template_path, "r") as f:
        template_text = f.read()

    for arch_type in args.main_training_model_arch_types:
        rendered = _render(
            template_text,
            arch_type=arch_type,
            c_lr=args.c_lr,
            d_lr=args.d_lr,
            v_lr=args.v_lr,
            players=args.player,
            opponents=args.opponent_list,
            main_training_steps=args.main_training_steps,
            ego_value_head_lr=args.ego_value_head_lr,
        )
        out_path = _output_path(
            template_path, arch_type, args.c_lr, args.d_lr, args.v_lr,
            args.player, args.opponent_list,
        )
        with open(out_path, "w") as f:
            f.write(rendered)
        os.chmod(out_path, 0o755)
        print(f"[generated] {out_path}")

        cmd = ["sbatch", out_path]
        if args.dry_run:
            print(f"[dry-run]   {' '.join(cmd)}")
        else:
            print(f"[submit]    {' '.join(cmd)}")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
