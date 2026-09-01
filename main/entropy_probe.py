"""Direct policy-entropy probe at a checkpoint. Rolls out on-policy self-play and
records H(pi_ego(.|s)) and H(pi_adv(.|s)) at each visited state, using the SAME
distribution path LBR uses (pi_ctrl/pi_dstb feature extractors -> mlp_extractor
-> _get_*_action_dist_from_latent). Reports mean +/- sd in nats and as % of ln(n).

Fails LOUDLY: asserts the dist has >=1 categorical component and prints the count,
and cross-checks .entropy() against a manual -sum p log p on component 0.
"""
import os, sys, argparse, json
_MAIN = "/home/jw4406/codebase/FightLadder/main"
sys.path.insert(0, _MAIN); os.chdir(_MAIN)
import numpy as np, torch as th
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.preprocessing import preprocess_obs
import duel


def _x(p, obs, device):
    t = th.as_tensor(obs).to(device)
    return preprocess_obs(t, p.observation_space, normalize_images=p.normalize_images)

def ego_dist(p, obs, device):
    x = _x(p, obs, device)
    f = p.pi_ctrl_features_extractor(x)
    latent = p.mlp_extractor.ego_forward(f)
    return p._get_ego_action_dist_from_latent(latent)

def adv_dist(p, obs, device, head):
    x = _x(p, obs, device)
    f = p.pi_dstb_features_extractor(x)
    latent = p.mlp_extractor.adv_forward(f, side_flag=None)
    return p._get_adv_action_dist_from_latent(latent, buf_num=[head], evaluate=True)[0]

def dist_entropy(dist):
    """Joint entropy (nats) per batch element, plus n_components. Uses .entropy();
    cross-checks against manual -sum p log p summed over all components."""
    comps = dist.distribution
    ncomp = len(comps)
    # manual joint entropy = sum over independent components of -sum p log p
    manual = 0.0
    for c in comps:
        pr = c.probs
        manual = manual + (-(pr.clamp_min(1e-12).log() * pr).sum(-1))
    H = dist.entropy()
    # loud consistency check
    if not th.allclose(H, manual, atol=1e-4, rtol=1e-3):
        raise SystemExit(f"[FAIL] dist.entropy() != manual: "
                         f"{H.mean().item():.5f} vs {manual.mean().item():.5f} (ncomp={ncomp})")
    return H.detach().cpu().numpy(), ncomp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", default="ippo", choices=["spar", "ippo", "2timescale"])
    ap.add_argument("--model_file", required=True)
    ap.add_argument("--ego_char", default="Ryu")
    ap.add_argument("--adv_char", default="Sagat")
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--out_prefix", required=True)
    a = ap.parse_args()

    duel._OBS_CFG["obs_type"] = "ram"; duel._OBS_CFG["ram_mask"] = "ram_mask.npy"
    duel._OBS_CFG["transform_action"] = True
    device = duel.resolve_device("auto")
    model, matchups = duel.load_spar_family(a.model_type, a.model_file, device)
    key = f"{a.ego_char}Vs{a.adv_char}"
    head = duel.resolve_head_idx(matchups, key)
    adv_head = head // model.envs_per_matchup
    p = model.policy
    dev = model.device

    ego_act = duel.make_spar_ego_action_fn(model, deterministic=False)
    adv_act = duel.make_spar_adv_action_fn(model, adv_head, deterministic=False)
    duel_state = f"two_player/{a.ego_char}_left/Champion.Level1.{key}.2Player.state"
    env = duel.env_generator(duel.env_args(), STATE=[duel_state])

    n_actions = int(model.action_space.nvec[0])
    Hmax = float(np.log(n_actions))
    print(f"[probe] model={a.model_type} n_actions(nvec[0])={n_actions} "
          f"nvec={list(model.action_space.nvec)} Hmax_percomp=ln={Hmax:.4f}", flush=True)

    ego_H, adv_H = [], []
    ncomp_seen = set()
    obs = env.reset()
    for t in range(a.max_steps):
        obs_t = obs_as_tensor(obs, dev)
        with th.no_grad():
            eH, ne = dist_entropy(ego_dist(p, obs, dev))
            aH, na = dist_entropy(adv_dist(p, obs, dev, head))
        ncomp_seen |= {ne, na}
        ego_H.extend(np.asarray(eH).reshape(-1).tolist())
        adv_H.extend(np.asarray(aH).reshape(-1).tolist())
        la = ego_act(obs_t); ra = adv_act(obs_t)
        obs, _r, _ro, done, info = env.step(np.hstack([la, ra]))
    env.close()

    ego_H = np.asarray(ego_H); adv_H = np.asarray(adv_H)
    ncomp = max(ncomp_seen)
    Hmax_joint = Hmax * ncomp  # joint max over independent components
    summ = dict(
        model_type=a.model_type, model=os.path.basename(a.model_file),
        n_samples=int(ego_H.size), n_actions=n_actions, n_components=int(ncomp),
        Hmax_joint_nats=round(Hmax_joint, 4),
        ego_H_mean=round(float(ego_H.mean()), 4), ego_H_sd=round(float(ego_H.std()), 4),
        adv_H_mean=round(float(adv_H.mean()), 4), adv_H_sd=round(float(adv_H.std()), 4),
        ego_pct_of_max=round(100 * float(ego_H.mean()) / Hmax_joint, 2),
        adv_pct_of_max=round(100 * float(adv_H.mean()) / Hmax_joint, 2),
    )
    with open(a.out_prefix + "_entropy.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2), flush=True)
    print("[probe] DONE", flush=True)


if __name__ == "__main__":
    main()
