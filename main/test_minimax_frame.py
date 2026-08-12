"""The minimax head's target must be in EGO frame on BOTH seats.

THE BUG THIS LOCKS DOWN. `_minimax_q_update` negated the target
unconditionally:

    target = -rollout_data.returns

on the reasoning that it "only ever runs on the ADVERSARY pass". It does not.
The ego pass uses `self.rollout_buffer` (a Q_RolloutBuffer), whose samples are
Q_RolloutBufferSamples -- and that NamedTuple carries `adv_actions`, which the
method's own getattr fallback picks up. So the update ran on both passes, and
because only the adversary buffer stores ADV-frame returns (it is fed -rewards
and last_values=-values), the ego pass was regressing the head onto -G_ego:
the exact negative of what the adversary pass was teaching it. Half the updates
undid the other half for the whole of Phase 0.

`skipped_no_adv_actions` -- the counter that would have fired had the ego pass
really returned early -- never appeared in a single log. That was the evidence.

WHY A BEHAVIORAL TEST AND NOT AN ASSERTION ON `target`. Reaching into the
method to inspect its local `target` would pass just as happily if the sign were
applied to the wrong quantity downstream. Instead this runs the REAL method to
convergence on a constant-return batch and asks what the head ends up
predicting. MinimaxHead is ego-payoff by construction, so:

    ego-frame  batch, returns +G  ->  head must converge to  +G
    adv-frame  batch, returns +G  ->  head must converge to  -G

The pre-fix code fails the ego case and passes the adv case, which is precisely
the asymmetry that made the aggregate look merely mediocre (target_corr +0.414)
rather than broken.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch as th
import torch.nn as nn

from stable_baselines3.common.clean_new_policies import MinimaxHead
from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR

LATENT, NE, NA, B = 16, 22, 22, 64
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


def run_frame(adv_frame, G=0.05, steps=400):
    """Drive the REAL _minimax_q_update to convergence and return what the head
    predicts at the played cell. Everything the method touches is stubbed with a
    working object, so the code path exercised is the shipped one."""
    th.manual_seed(0)
    head = MinimaxHead(nn.Sequential(nn.Linear(LATENT, LATENT), nn.LeakyReLU()),
                       LATENT, n_ego=NE, n_adv=NA)
    latent = th.randn(B, LATENT)

    policy = types.SimpleNamespace(
        minimax_head_for=lambda buf_num: head,
        minimax_matrices=lambda obs, buf_num=None, side_flag=None, stop_grad=True: head(latent),
        minimax_optimizer=th.optim.Adam(head.parameters(), lr=3e-3),
    )
    # minimax_stat_every large so the inner MWU solve is skipped after the first
    # call: it is diagnostic here and dominates the runtime of this test.
    agent = types.SimpleNamespace(minimax_q=True, policy=policy,
                                  minimax_stop_grad=True, minimax_stat_every=10**9,
                                  minimax_iters=64, minimax_eta=0.5, _mm_calls=0)

    a_ego = th.randint(0, NE, (B,))
    a_adv = th.randint(0, NA, (B,))
    # Constant returns: the head can fit them exactly, so the converged
    # prediction is the target itself and the sign is unambiguous.
    data = types.SimpleNamespace(observations=th.zeros(B, 1), actions=a_ego,
                                 adv_actions=a_adv, returns=th.full((B,), G))

    for _ in range(steps):
        CleanDerivativeFreeSPAR._minimax_q_update(agent, data, [0], adv_frame=adv_frame)

    with th.no_grad():
        M = head(latent)
        return float(M[th.arange(B), a_ego, a_adv].mean())


def main():
    G = 0.05

    q_ego = run_frame(adv_frame=False, G=G)
    check("EGO-frame batch converges to +G (NOT negated)",
          q_ego > 0.5 * G, f"returns {G:+.3f} -> head {q_ego:+.4f}")

    q_adv = run_frame(adv_frame=True, G=G)
    check("ADV-frame batch converges to -G (negated to ego frame)",
          q_adv < -0.5 * G, f"returns {G:+.3f} -> head {q_adv:+.4f}")

    # The two seats must DISAGREE in sign on identically-signed returns. If they
    # agree, one frame is wrong regardless of which -- and this is the check the
    # old code fails: it produced -G on both.
    check("the two frames produce OPPOSITE signs",
          q_ego * q_adv < 0, f"ego {q_ego:+.4f}  adv {q_adv:+.4f}")

    # adv_frame is keyword-only with no default: omitting it must be a hard
    # error, not a silent inherit of the wrong sign.
    try:
        CleanDerivativeFreeSPAR._minimax_q_update(
            types.SimpleNamespace(minimax_q=False), None, [0])
        missing_raises = False
    except TypeError:
        missing_raises = True
    check("omitting adv_frame raises TypeError", missing_raises)

    print("\n  ALL PASS" if OK else "\n  FAILURES PRESENT")
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
