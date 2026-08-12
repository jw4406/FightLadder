"""Does the minimax head touch ANYTHING outside itself?

THE CLAIM THIS DEFENDS. Every Phase 0 result -- the payoff ANOVA, the gate, the
q_decompose comparison, 18.5M steps of divergence-free option B -- rests on the
head being INERT: it trains, and it moves no parameter that the policy uses. If
that is false, those runs were not measuring a passive instrument and their
policy trajectories are not comparable to anything.

WHY RE-CHECK IT NOW. It WAS verified, as max|shared - initial| == 0.0 across
optimizer steps -- but under OPTION A, before three code paths existed that did
not exist then:

    * the option-B target, which runs a second forward through the encoder
      (minimax_latent) and an MWU solve to build r + gamma*V_mm(s')
    * _minimax_frozen_head, a deepcopy of the head refreshed per rollout
    * _minimax_bootstrap, which reads the encoder over every buffer state

Each is written to be gradient-free. "Written to be" is not a measurement, and
an argument that three separate no_grad barriers all hold is exactly the kind of
thing that is wrong once and silent forever.

TWO INDEPENDENT ASSERTIONS, because either alone can pass while leaking:

  parameters unchanged   the shipped consequence. Bitwise, not allclose.
  NO GRADIENT accumulated on shared parameters. Strictly stronger: a leak shows
                         up here even when the optimizer that would have applied
                         it is scoped elsewhere, so this catches a leak that has
                         not yet had a chance to do damage.

Run against the REAL CleanActorActorCriticPolicy, not a stub -- a stub encoder
cannot leak through the path being tested.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch as th
from gym import spaces

from common.justin.clean_derivative_free_spar import CleanDerivativeFreeSPAR

OBS, NA, B, T, NENV = 2105, 22, 32, 4, 4
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


def build(head_kind="factored", target="minimax", kappa=0.0, gamma=0.94, seed=0):
    from stable_baselines3.common.clean_new_policies import CleanActorActorCriticPolicy
    th.manual_seed(seed)
    pol = CleanActorActorCriticPolicy(
        spaces.Box(0.0, 1.0, (OBS,), dtype=np.float32), spaces.MultiDiscrete([NA]),
        lr_schedule=[lambda _: 3e-4] * 8,
        minimax_q=True, minimax_head=head_kind, minimax_rank=4,
        minimax_n_ego=NA, minimax_n_adv=NA,
        num_adversaries=1, matchups=["RyuVsSagat"], envs_per_matchup=4).to("cpu")
    ag = types.SimpleNamespace(
        minimax_q=True, vtrace_enabled=False, policy=pol, gamma=gamma,
        minimax_stop_grad=True, minimax_stat_every=10**9,
        minimax_iters=64, minimax_eta=0.5, _mm_calls=0,
        minimax_target=target, num_timesteps=1000,
        minimax_bootstrap_kappa=kappa, minimax_bootstrap_warmup=0,
        num_adversaries=1, n_env_per_adv=NENV, device="cpu")
    for m in ("_minimax_q_update", "_minimax_frozen_head", "_minimax_kappa",
              "_minimax_values_for", "_minimax_bootstrap"):
        setattr(ag, m, types.MethodType(getattr(CleanDerivativeFreeSPAR, m), ag))
    return ag, pol


def split(pol):
    """(inside the head, everything else) -- by name, so it cannot drift."""
    ins, outs = {}, {}
    for n, p in pol.named_parameters():
        (ins if "minimax_net" in n else outs)[n] = p
    return ins, outs


class Buf:
    def __init__(self, seed=0):
        r = np.random.RandomState(seed)
        self.observations = r.rand(T, NENV, OBS).astype(np.float32)
        self.values = (r.randn(T, NENV) * 0.01).astype(np.float32)


def rollout_data(seed=0):
    r = np.random.RandomState(seed)
    return types.SimpleNamespace(
        observations=th.as_tensor(r.rand(B, OBS).astype(np.float32)),
        next_observations=th.as_tensor(r.rand(B, OBS).astype(np.float32)),
        actions=th.randint(0, NA, (B,)), adv_actions=th.randint(0, NA, (B,)),
        returns=th.randn(B) * 0.02, rewards=th.randn(B) * 0.05,
        dones=th.zeros(B))


def main():
    for head_kind in ("matrix", "factored"):
        print(f"\n--- head = {head_kind} ---")

        # ---- OPTION B update: frozen head + solve + encoder forward --------
        ag, pol = build(head_kind=head_kind, target="minimax")
        ins, outs = split(pol)
        before = {n: p.detach().clone() for n, p in outs.items()}
        n_in_before = {n: p.detach().clone() for n, p in ins.items()}
        data = rollout_data()
        for t in range(6):
            ag.num_timesteps = 1000 + t * 100      # force frozen-head refreshes
            ag._minimax_q_update(data, [0], adv_frame=False)

        moved = [n for n, p in outs.items()
                 if not th.equal(p.detach(), before[n])]
        check(f"option B: NO parameter outside minimax_net moved",
              not moved, f"{len(outs)} checked" if not moved else f"MOVED: {moved[:4]}")
        grads = [n for n, p in outs.items()
                 if p.grad is not None and float(p.grad.abs().max()) > 0.0]
        check("option B: NO gradient accumulated on shared parameters",
              not grads, f"{len(outs)} checked" if not grads else f"GRADS: {grads[:4]}")
        head_moved = sum(1 for n, p in ins.items()
                         if not th.equal(p.detach(), n_in_before[n]))
        check("option B: the HEAD itself DID train (else the test is vacuous)",
              head_moved > 0, f"{head_moved}/{len(ins)} head tensors changed")

        # ---- PHASE 1 bootstrap: encoder read over every buffer state -------
        ag2, pol2 = build(head_kind=head_kind, target="minimax", kappa=1.0)
        ins2, outs2 = split(pol2)
        before2 = {n: p.detach().clone() for n, p in outs2.items()}
        head_before2 = {n: p.detach().clone() for n, p in ins2.items()}
        buf, advb = Buf(), [Buf(1)]
        lv = np.zeros(NENV, np.float32)
        pre_vals = buf.values.copy()
        ag2._minimax_bootstrap(buf, advb, np.random.rand(NENV, OBS).astype(np.float32), lv)
        check("bootstrap: it actually RAN (values replaced)",
              not np.allclose(buf.values, pre_vals))
        moved2 = [n for n, p in outs2.items() if not th.equal(p.detach(), before2[n])]
        check("bootstrap: NO parameter outside minimax_net moved",
              not moved2, f"{len(outs2)} checked" if not moved2 else f"MOVED: {moved2[:4]}")
        hm = [n for n, p in ins2.items() if not th.equal(p.detach(), head_before2[n])]
        check("bootstrap: does not move the HEAD either (it only READS it)", not hm)
        g2 = [n for n, p in list(outs2.items()) + list(ins2.items())
              if p.grad is not None and float(p.grad.abs().max()) > 0.0]
        check("bootstrap: NO gradient anywhere (it is entirely no_grad)", not g2,
              "" if not g2 else f"GRADS: {g2[:4]}")

    print("\n  ALL PASS" if OK else "\n  FAILURES PRESENT")
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
