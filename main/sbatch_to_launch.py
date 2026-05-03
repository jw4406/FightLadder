"""
Convert an orchestrator-generated `.sbatch` file into a VS Code
``launch.json`` debug configuration entry, so a single per-job invocation
of `br_single_matchup.py` or `br_single_continue.py` can be stepped
through interactively.

Usage:
    python main/sbatch_to_launch.py --sbatch path/to/job.sbatch
    python main/sbatch_to_launch.py --sbatch path/to/job.sbatch --name "my debug"
    python main/sbatch_to_launch.py --sbatch path/to/job.sbatch \
        --workspace_folder /home/jw4406/codebase/FightLadder

The generated entry is printed to stdout — paste it into the
``configurations`` array in ``.vscode/launch.json``. We deliberately do
NOT auto-edit launch.json: it's JSONC (with comments and trailing commas)
that python's stdlib json can't round-trip safely.

How it works:
  - Parses the ``CMD=( ... )`` array the orchestrator writes (one --flag
    per line via build_python_cmd in br_slurm_common.py).
  - shlex.split unquotes args, so paths-with-spaces and the embedded
    --shared_config_json blob round-trip cleanly.
  - The first ``.py`` token in the CMD array is the runner script and
    becomes the launch entry's ``program``; everything after it is
    forwarded as ``args``.
  - The ``cd <repo_dir>`` line in the sbatch script becomes ``cwd`` in
    the launch entry so debugging matches what SLURM (or local bash)
    would have run.
  - The ``#SBATCH --job-name=`` directive becomes the launch entry name
    (overridable via --name).
"""
import argparse
import json
import os
import re
import shlex
import sys
from typing import Dict, List, Optional, Tuple


def _read_sbatch(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"sbatch file not found: {path}")
    with open(path, "r") as f:
        return f.read()


def _extract_sbatch_directive(content: str, key: str) -> Optional[str]:
    """
    Pull ``#SBATCH --<key>=<value>`` out of *content*. Returns None if
    the directive is missing.
    """
    m = re.search(rf"^\s*#SBATCH\s+--{re.escape(key)}=(.+?)\s*$", content, re.MULTILINE)
    return m.group(1) if m else None


def _extract_cd_target(content: str) -> Optional[str]:
    """
    Pull the ``cd <path>`` line out of the sbatch body (the orchestrator
    inserts exactly one between the SBATCH header and the python invocation).
    Returns None if absent.
    """
    m = re.search(r"^\s*cd\s+(\S.*?)\s*$", content, re.MULTILINE)
    return m.group(1) if m else None


def _extract_cmd_block(content: str) -> str:
    """
    Return the body of the ``CMD=( ... )`` array as a single string,
    suitable for shlex.split.
    """
    m = re.search(
        r"^\s*CMD=\(\s*\n(.*?)^\s*\)\s*$",
        content,
        re.MULTILINE | re.DOTALL,
    )
    if m is None:
        raise ValueError(
            "No CMD=( ... ) block found. Is this a current-format sbatch "
            "produced by br_slurm_common.build_python_cmd?"
        )
    return m.group(1)


def _split_program_and_args(tokens: List[str]) -> Tuple[str, List[str]]:
    """
    The CMD array is shaped like ``["python", "-u", "/abs/runner.py",
    "--flag", "value", ...]``. Find the first ``.py`` token; treat
    everything before it as the interpreter (we drop it — VS Code uses
    its own python via the type=debugpy runtime) and everything after
    it as forwarded args.
    """
    py_idx = next((i for i, t in enumerate(tokens) if t.endswith(".py")), None)
    if py_idx is None:
        raise ValueError(
            "No .py runner script token found inside CMD array. Expected "
            "something like 'python -u /abs/path/br_single_*.py ...'."
        )
    program = tokens[py_idx]
    args = tokens[py_idx + 1:]
    return program, args


def _maybe_substitute_workspace(path: str, workspace_folder: Optional[str]) -> str:
    """
    If *path* lives under *workspace_folder*, swap that prefix for the
    VS Code ``${workspaceFolder}`` variable so the launch entry stays
    portable when the repo gets moved.
    """
    if workspace_folder is None:
        return path
    abs_ws = os.path.abspath(workspace_folder).rstrip("/")
    abs_p = os.path.abspath(path).rstrip("/")
    if abs_p == abs_ws:
        return "${workspaceFolder}"
    if abs_p.startswith(abs_ws + os.sep):
        return "${workspaceFolder}" + abs_p[len(abs_ws):]
    return path


def parse_sbatch(path: str) -> Dict[str, object]:
    """
    Parse an orchestrator-generated .sbatch file and return a dict with
    the pieces a launch.json entry needs:
      program (str), args (List[str]), cwd (Optional[str]),
      name (str — from #SBATCH --job-name), out_log/err_log (for context).
    """
    content = _read_sbatch(path)
    body = _extract_cmd_block(content)
    tokens = shlex.split(body)
    program, args = _split_program_and_args(tokens)
    return {
        "program": program,
        "args": args,
        "cwd": _extract_cd_target(content),
        "name": _extract_sbatch_directive(content, "job-name"),
        "out_log": _extract_sbatch_directive(content, "output"),
        "err_log": _extract_sbatch_directive(content, "error"),
    }


def build_launch_entry(
    parsed: Dict[str, object],
    *,
    name_override: Optional[str] = None,
    workspace_folder: Optional[str] = None,
) -> Dict[str, object]:
    """
    Translate the parsed sbatch dict into a VS Code launch.json entry.
    """
    program = _maybe_substitute_workspace(str(parsed["program"]), workspace_folder)
    cwd = parsed.get("cwd")
    if cwd:
        cwd = _maybe_substitute_workspace(str(cwd), workspace_folder)

    base_name = name_override or parsed.get("name") or "br_single_debug"
    entry: Dict[str, object] = {
        "name": f"Python Debugger: {base_name}",
        "type": "debugpy",
        "request": "launch",
        "program": program,
        "console": "integratedTerminal",
        "args": list(parsed["args"]),
    }
    if cwd:
        entry["cwd"] = cwd
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an orchestrator-generated .sbatch into a "
                    "VS Code launch.json entry."
    )
    parser.add_argument("--sbatch", type=str, required=True,
                        help="Path to the .sbatch file (produced under "
                             "slurm_logs/<task_stem>/<job_name>.sbatch).")
    parser.add_argument("--name", type=str, default=None,
                        help="Override the launch entry's display name. "
                             "Defaults to the SBATCH --job-name directive.")
    parser.add_argument("--workspace_folder", type=str, default=None,
                        help="If the program/cwd path lives under this dir, "
                             "swap the prefix for ${workspaceFolder} so the "
                             "entry stays repo-relative.")
    parser.add_argument("--show_logs", action="store_true",
                        help="Also print the SBATCH --output / --error log "
                             "paths as a comment before the entry, so you "
                             "know where to look if the job already ran.")
    args = parser.parse_args()

    try:
        parsed = parse_sbatch(args.sbatch)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.show_logs:
        out_log = parsed.get("out_log")
        err_log = parsed.get("err_log")
        if out_log:
            print(f"// SBATCH --output: {out_log}")
        if err_log:
            print(f"// SBATCH --error : {err_log}")
        print()

    entry = build_launch_entry(
        parsed,
        name_override=args.name,
        workspace_folder=args.workspace_folder,
    )

    # Pretty-print at indent=4 to match the .vscode/launch.json style.
    # Trailing comma is fine — launch.json is JSONC and VS Code accepts it,
    # and it's convenient for the user to drop the entry into the middle
    # of the configurations array.
    print(json.dumps(entry, indent=4) + ",")


if __name__ == "__main__":
    main()
