"""Structural guarantees of FactoredMinimaxHead.

    Q(s,i,j) = V(s) + A_ego(s,i) + A_adv(s,j) + e_ego(i)^T W(s) e_adv(j)

The value of this parameterization is that the four terms are ORTHOGONAL and the
interaction is confined to the correct subspace BY CONSTRUCTION rather than by
training. These tests check the construction, because if it does not hold the
head is just a differently-shaped free matrix and every argument for it collapses.

What is checked, and why each one matters:

  double-centring    sum_i gamma_ij = 0 and sum_j gamma_ij = 0 to float
                     precision. gamma must live in the 21x21 = 441-dim space; if
                     it can leak into mu/alpha/beta it competes with V and the
                     advantages for the same directions, which is the exact
                     identifiability problem centring exists to remove.
  zero-init          at step 0 the output must equal the ADDITIVE head EXACTLY.
                     This is what makes the interaction opt-in: the head cannot
                     start worse than separable.
  recovery           fit a synthetic additive matrix -> gamma stays ~0; fit a
                     synthetic rank-2 gamma -> it is recovered. A head that
                     "recovers" interaction from additive data is fitting noise.
  density            ONE played cell must produce nonzero gradient on EVERY
                     output. This is the whole point against the 484-cell head,
                     whose density is 1/484 = 0.207%.
  interface          output shape and method set identical to MinimaxHead, so
                     minimax_matrices / solve_matrix_game / LBR / the six
                     diagnostics keep working untouched.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3.common.clean_new_policies import (FactoredMinimaxHead,
                                                         MinimaxHead)

LATENT, NE, NA, R = 32, 22, 22, 4
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


def make(rank=R, seed=0, w_init=0.0):
    """w_init defaults to 0.0 HERE so the structural guarantees below (exact
    additivity at init) are still tested; the SHIPPED default is 0.01 and gets
    its own checks."""
    th.manual_seed(seed)
    trunk = nn.Sequential(nn.Linear(LATENT, LATENT), nn.LeakyReLU(),
                          nn.Linear(LATENT, LATENT), nn.LeakyReLU())
    return FactoredMinimaxHead(trunk, LATENT, NE, NA, rank=rank, w_init=w_init)


def main():
    h = make()
    x = th.randn(16, LATENT)

    # ---- double-centring -------------------------------------------------
    # Perturb W away from its zero init first: at init gamma is identically 0,
    # which would satisfy the property trivially and prove nothing.
    with th.no_grad():
        h.w_out.weight.normal_(0, 0.5)
        h.w_out.bias.normal_(0, 0.5)
    _, _, _, g = h.components(x)
    check("gamma rows sum to zero", g.sum(1).abs().max().item() < 1e-5,
          f"max |sum_i| = {g.sum(1).abs().max().item():.2e}")
    check("gamma cols sum to zero", g.sum(2).abs().max().item() < 1e-5,
          f"max |sum_j| = {g.sum(2).abs().max().item():.2e}")
    check("gamma is non-trivial once W != 0", g.abs().max().item() > 1e-3,
          f"max |gamma| = {g.abs().max().item():.4f}")

    v, ae, aa, _ = h.components(x)
    check("A_ego is centred", ae.sum(-1).abs().max().item() < 1e-5)
    check("A_adv is centred", aa.sum(-1).abs().max().item() < 1e-5)

    # ---- zero-init => exactly additive -----------------------------------
    h0 = make()
    v0, ae0, aa0, g0 = h0.components(x)
    M0 = h0(x)
    add = v0[:, None, None] + ae0[:, :, None] + aa0[:, None, :]
    check("W is zero-initialised", h0.w_out.weight.abs().max().item() == 0.0)
    check("gamma == 0 at init", g0.abs().max().item() == 0.0)
    check("output is EXACTLY the additive head at init",
          th.allclose(M0, add, atol=0, rtol=0))

    # ---- recovery --------------------------------------------------------
    # Same trunk input for every sample so the target is a single fixed matrix;
    # this isolates whether the head can represent it, not whether it can
    # generalise.
    def fit(target, steps=1500, rank=R):
        hh = make(rank=rank, seed=1)
        xx = th.randn(1, LATENT).repeat(256, 1)
        opt = th.optim.Adam(hh.parameters(), lr=3e-3)
        tgt = target.expand(256, NE, NA)
        for _ in range(steps):
            # one RANDOM cell per sample -- the real training signal, not the
            # full matrix, so this also exercises the density claim
            i = th.randint(0, NE, (256,)); j = th.randint(0, NA, (256,))
            q = hh.played(xx, i, j)
            b = th.arange(256)
            loss = ((q - tgt[b, i, j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with th.no_grad():
            _, _, _, gg = hh.components(xx[:1])
        return hh, gg[0]

    th.manual_seed(3)
    mu, al, be = 0.5, th.randn(NE), th.randn(NA)
    al = al - al.mean(); be = be - be.mean()
    additive = (mu + al[:, None] + be[None, :]).unsqueeze(0)
    _, g_add = fit(additive)
    check("additive target -> gamma stays ~0",
          g_add.abs().max().item() < 0.05,
          f"max |gamma| = {g_add.abs().max().item():.4f}")

    u = th.randn(NE, 2); w = th.randn(NA, 2)
    u = u - u.mean(0); w = w - w.mean(0)
    g_true = u @ w.T
    inter = (mu + al[:, None] + be[None, :] + g_true).unsqueeze(0)
    _, g_rec = fit(inter)
    err = (g_rec - g_true).norm() / g_true.norm()
    check("rank-2 interaction target -> gamma recovered",
          err.item() < 0.25, f"relative error {err.item():.3f}")

    # ---- gradient density ------------------------------------------------
    hd = make(seed=2)
    with th.no_grad():
        hd.w_out.weight.normal_(0, 0.1)      # else dW is zero by symmetry
    xd = th.randn(1, LATENT)
    hd.played(xd, th.tensor([3]), th.tensor([7])).sum().backward()
    named = {n: p for n, p in hd.named_parameters() if p.grad is not None}
    live = {n: float(p.grad.abs().sum()) for n, p in named.items()}
    dead = [n for n, v in live.items() if v == 0.0]
    check("one played cell -> gradient on EVERY parameter tensor",
          not dead, f"dead: {dead}" if dead else f"{len(live)} tensors live")
    ge = hd.a_ego_out.weight.grad.abs().sum(1)
    check("all 22 A_ego outputs get gradient from one cell",
          int((ge > 0).sum()) == NE, f"{int((ge > 0).sum())}/{NE}")

    # ---- interface parity -------------------------------------------------
    trunk = nn.Sequential(nn.Linear(LATENT, LATENT), nn.LeakyReLU())
    mm = MinimaxHead(trunk, LATENT, NE, NA)
    check("output shape matches MinimaxHead", h(x).shape == mm(x).shape,
          f"{tuple(h(x).shape)}")
    check("method set matches",
          all(hasattr(h, m) for m in ("forward", "played", "note_visits",
                                      "coverage", "cell_visits")))
    h.note_visits(th.tensor([1, 2]), th.tensor([3, 4]))
    check("note_visits / coverage work", h.coverage() > 0)

    # ---- W INIT SCALE: the embedding-gradient fix -------------------------
    # gamma_ij = sum_rc e_ego[i,r] W[r,c] e_adv[j,c], so d(gamma)/d(e_ego) is
    # PROPORTIONAL TO W. At W==0 the embeddings get literally no gradient and sit
    # at their random init through the cold start. Measured consequence at 14.4M:
    # only 4.93% of the true interaction lay inside the learned embedding
    # subspace, against 56.43% reachable at the same rank (3.63% = random).
    # This asserts the mechanism directly rather than the symptom.
    for wi, want_grad in ((0.0, False), (0.01, True)):
        hw = make(seed=5, w_init=wi)
        xw = th.randn(8, LATENT)
        hw.played(xw, th.randint(0, NE, (8,)), th.randint(0, NA, (8,))).sum().backward()
        ge = float(hw.e_ego.grad.abs().max()) if hw.e_ego.grad is not None else 0.0
        gw = float(hw.w_out.weight.grad.abs().max())
        got = ge > 0.0
        check(f"w_init={wi}: embeddings {'DO' if want_grad else 'do NOT'} get gradient",
              got == want_grad, f"|d/d e_ego| = {ge:.3e}   |d/d w_out| = {gw:.3e}")
    check("w_init=0.0 still gives EXACTLY the additive head",
          make(seed=5, w_init=0.0).components(th.randn(4, LATENT))[3].abs().max().item() == 0.0)
    hd_ = make(seed=5, w_init=0.01)
    g_small = hd_.components(th.randn(4, LATENT))[3].abs().max().item()
    check("w_init=0.01 gives a SMALL but nonzero interaction",
          0.0 < g_small < 0.05, f"max|gamma| at init = {g_small:.2e}")

    st = h.interaction_stats(x)
    check("interaction_stats returns the live readouts",
          all(k in st for k in ("w_norm", "gamma_share", "anti_share", "noop_emb")),
          f"w_norm={st['w_norm']:.4f} gamma_share={st['gamma_share']:.4f} "
          f"anti={st['anti_share']:.3f}")

    n_f = sum(p.numel() for p in make().parameters())
    n_m = sum(p.numel() for p in MinimaxHead(
        nn.Sequential(nn.Linear(LATENT, LATENT), nn.LeakyReLU()), LATENT, NE, NA).parameters())
    print(f"\n  params (LATENT={LATENT}): factored {n_f:,}   matrix {n_m:,}")
    print(f"  outputs: factored {1 + NE + NA + R*R}   matrix {NE*NA}")

    print("\n  ALL PASS" if OK else "\n  FAILURES PRESENT")
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
