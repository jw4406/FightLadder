"""
Diagnostic: measure the value head's Adam second moment vs eps at a given reward_scale.

Tests the claim in retro_wrappers.py that reward_scale=0.001 knocks the value head out of
Adam's adaptive regime (sqrt(v) ~1e-9 < eps=1e-8). This reads the OPTIMIZER STATE directly,
so it is NOT confounded by opponent stochasticity (unlike explained_variance).

Run once per scale in a SEPARATE process (retro allows one emulator per process):
    python main/diag_value_adam_secondmoment.py 0.001
    python main/diag_value_adam_secondmoment.py 1.0
"""
import sys, math
from types import SimpleNamespace

import numpy as np
import torch

import train_ma
from train_ma import constructor, _build_states_from_roster


def main():
    assert len(sys.argv) == 2, "usage: diag_value_adam_secondmoment.py <reward_scale>"
    rs = float(sys.argv[1])

    STATES = _build_states_from_roster(["Ryu"], ["Guile"], "both")
    train_ma.STATES = STATES
    state = STATES["ryu_guile"]

    args = SimpleNamespace(
        reset="round", side="both", render=False, num_env=1,
        enable_combo=True, null_combo=False, transform_action=True, seed=0,
        reward_scale=rs, player=["Ryu"], opponent_list=["Guile"],
        log_dir=None, save_dir="main/trained_models/_rsdiag",
    )

    # Right agent: train() optimizes self.policy_other (the trained value head lives there).
    agent = constructor(args, "right", log_name=None, single_env=True,
                        opponent="guile", state_name=state, matchup_key="ryu_vs_guile")
    trained = agent.policy_other  # side=="right" -> trained net is policy_other
    opt = trained.optimizer

    # Frozen self-play opponent (a left-side policy) drives the other side; no-op coordinate/sync.
    def get_kwargs():
        return {"policy": agent.policy, "coordinate_fn": (lambda *a, **k: None),
                "sync_fn": (lambda *a, **k: None)}

    # ~11 iterations * (n_epochs=4) ~= 44 Adam steps -> stable second-moment estimate.
    agent.learn(total_timesteps=12000, rollout_opponent_num=2, get_kwargs_fn=get_kwargs, log_interval=5)

    eps = float(opt.param_groups[0]["eps"])
    beta2 = float(opt.param_groups[0]["betas"][1])

    sqrt_vhat, sqrt_raw, nsteps = [], [], None
    for p in trained.value_net.parameters():
        st = opt.state.get(p, None)
        assert st is not None and "exp_avg_sq" in st, \
            "FAIL LOUD: value_net param has NO Adam state -> value head was not updated"
        v = st["exp_avg_sq"]
        t = st["step"]
        t = float(t.item() if torch.is_tensor(t) else t)
        nsteps = t
        assert t > 0, "FAIL LOUD: Adam step count is 0 on value head"
        vhat = v / (1.0 - beta2 ** t)          # bias-corrected 2nd moment (what the denom uses)
        sqrt_vhat.append(torch.sqrt(vhat).flatten())
        sqrt_raw.append(torch.sqrt(v).flatten())

    a = torch.cat(sqrt_vhat).detach().cpu().numpy()
    r = torch.cat(sqrt_raw).detach().cpu().numpy()
    frac_below = float((a < eps).mean())

    print("=" * 60)
    print(f"RESULT reward_scale={rs}")
    print(f"  value-head params measured: {a.size} scalars over {sum(1 for _ in trained.value_net.parameters())} tensors")
    print(f"  Adam steps on value head: {nsteps:.0f}   eps={eps:.1e}   beta2={beta2}")
    print(f"  sqrt(v_hat) [bias-corrected] : median={np.median(a):.3e}  mean={a.mean():.3e}  min={a.min():.3e}  max={a.max():.3e}")
    print(f"  sqrt(exp_avg_sq) [raw]       : median={np.median(r):.3e}")
    print(f"  fraction of value-head sqrt(v_hat) < eps: {frac_below:.1%}")
    verdict = "OUT OF adaptive regime (eps dominates)" if np.median(a) < eps else "IN adaptive regime"
    print(f"  VERDICT: median sqrt(v_hat)={np.median(a):.3e} vs eps={eps:.1e}  ->  value head {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
