"""Render self-play rollouts of a SPAR checkpoint to mp4 (ego=ctrl head vs
adversary=dstb head, the exact self-play the diagnostics analyse).

The policy acts on the RAM observation (make_lbr_env with obs_type=ram); the VIDEO
is the emulator screen grabbed via env.render('rgb_array'). A single non-vec env
is used so there is no auto-reset -- each round is captured cleanly start to KO/
timeout. One mp4 per round, named with the outcome and end HP.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _overlay(frame, lines):
    """Draw a few status lines top-left. No-op if PIL is missing."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return frame
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    y = 2
    for ln in lines:
        d.text((3, y), ln, fill=(255, 255, 0))
        y += 11
    return __import__("numpy").asarray(im)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ram_mask", default="ram_mask.npy")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=1200,
                    help="agent-step cap per round (rs1 rounds run to ~517).")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--scale", type=int, default=3, help="integer upscale of the 200x256 screen")
    ap.add_argument("--deterministic", action="store_true",
                    help="argmax actions instead of sampling (default: sample, as trained)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="videos")
    ap.add_argument("--tag", default="rs1", help="basename prefix for the mp4s")
    a = ap.parse_args(argv)

    import numpy as np
    import subprocess
    from local_best_response import (build_lbr_venv, make_lbr_env, load_checkpoint,
                                     PolicyOps, resolve_matchups, infer_obs_kwargs)
    from stable_baselines3.common.save_util import load_from_zip_file

    data = load_from_zip_file(a.ckpt, device="cpu")[0]
    head_idx, label, st = resolve_matchups(data, "all")[0]
    kw = infer_obs_kwargs(data, a.ram_mask or None)

    # a 1-env vec only to give load_checkpoint the spaces; render on a raw env.
    load_venv = build_lbr_venv(st, 1, **kw)
    L = load_checkpoint(a.ckpt, load_venv, a.device)
    model = L[0] if isinstance(L, tuple) else L
    ops = PolicyOps(model, head_idx=head_idx, lbr_is_adv=False)

    env = make_lbr_env(st, seed=a.seed, **kw)()
    rng = np.random.RandomState(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    def act(obs):
        o = np.asarray(obs)[None]
        if a.deterministic:
            ego = ops.ego_probs(o).argmax(-1)
            adv = ops.adv_probs(o).argmax(-1)
        else:
            ego = ops.sample_ego(o, rng)
            adv = ops.sample_adv(o, rng)
        return ops.joint(ego, adv)[0]           # [left, right]

    summary = []
    for r in range(1, a.rounds + 1):
        obs = env.reset()
        frames = []
        done = False
        info = {}
        t = 0
        ah = eh = 176
        while not done and t < a.max_steps:
            scr = env.render(mode="rgb_array")
            if a.scale > 1:
                scr = np.repeat(np.repeat(scr, a.scale, axis=0), a.scale, axis=1)
            frames.append(_overlay(scr, [f"round {r}  t={t}",
                                         f"Ryu(ego) hp {ah:>4}",
                                         f"Sagat(adv) hp {eh:>4}"]))
            out = env.step(act(obs))
            obs, done, info = out[0], out[3], out[4]
            ah = int(info.get("agent_hp", ah)); eh = int(info.get("enemy_hp", eh))
            t += 1
        oc = info.get("outcome") or ("win" if ah > eh else "lose" if ah < eh else "draw")
        # one last frame of the end state
        scr = env.render(mode="rgb_array")
        if a.scale > 1:
            scr = np.repeat(np.repeat(scr, a.scale, axis=0), a.scale, axis=1)
        for _ in range(a.fps):  # ~1s hold on the final frame
            frames.append(_overlay(scr, [f"round {r}  END  {oc.upper()}",
                                         f"Ryu(ego) hp {ah:>4}",
                                         f"Sagat(adv) hp {eh:>4}"]))
        fn = os.path.join(a.out_dir, f"{a.tag}_round{r}_{oc}_ego{ah}_adv{eh}_t{t}.mp4")
        H, W = frames[0].shape[:2]
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", fn],
            stdin=subprocess.PIPE)
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed to encode {fn}")
        print(f"  round {r}: {oc:5s}  ego_hp={ah:>4} adv_hp={eh:>4}  steps={t:>4}  -> {fn}")
        summary.append((r, oc, ah, eh, t, fn))
    env.close(); load_venv.close()

    if not summary:
        raise RuntimeError("no rounds rendered -- nothing produced")
    print(f"\n  wrote {len(summary)} mp4(s) to {a.out_dir}/")
    for r, oc, ah, eh, t, fn in summary:
        print(f"    {os.path.basename(fn)}")


if __name__ == "__main__":
    main()
