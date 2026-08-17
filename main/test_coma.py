"""The COMA counterfactual baseline: inert by default, unbiased, correctly signed.

FOUR PROPERTIES, EACH OF WHICH FAILS SILENTLY IF BROKEN.

1. INERTNESS. coma_coef=0 with coma_diag off must leave advantages BITWISE
   unchanged, or every existing baseline is invalidated by a flag nobody set.

2. UNBIASEDNESS. The correction must be mean-zero under the opponent's policy:
   E_j~pi_adv[beta_hat_j(s)] = 0 by construction. If it is not, the "baseline"
   has a non-zero expectation and DOES bias the policy gradient -- which is the
   entire safety argument for preferring this over a counterfactual VALUE.

3. SIGN. The head is EGO-perspective, so the adversary's own main effect enters
   negated. A flipped sign here is the single most dangerous line in the method:
   the same mistake on the ego target invalidated all of Phase 0 and was only
   caught weeks later. Under a symmetric payoff the two seats' corrections must
   be exact negatives.

4. IT ACTUALLY REMOVES BETA. On a synthetic payoff built as mu+alpha+beta+gamma
   with known parts, the ego correction must equal beta_j and nothing else.

These are checked on the ARITHMETIC directly rather than through a live trainer,
because the arithmetic is where the sign and centring errors live, and a test
that needs a GPU and an emulator will not be run.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EXPECTED_CHECKS = 9
NC = 0


def chk(name, cond):
    global NC
    NC += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(f"FAILED: {name}")


rng = np.random.RandomState(0)
B, na = 400, 22

# Synthetic payoff with KNOWN ANOVA parts, so the correction can be checked
# against ground truth rather than against itself.
mu = rng.randn(B) * 0.05
al = rng.randn(B, na) * 0.03; al -= al.mean(1, keepdims=True)
be = rng.randn(B, na) * 0.03; be -= be.mean(1, keepdims=True)
ga = rng.randn(B, na, na) * 0.01
ga -= ga.mean(1, keepdims=True); ga -= ga.mean(2, keepdims=True)
Q = mu[:, None, None] + al[:, :, None] + be[:, None, :] + ga

pe = rng.dirichlet(np.ones(na), B)
pa = rng.dirichlet(np.ones(na), B)
j = np.array([rng.choice(na, p=pa[b]) for b in range(B)])
i = np.array([rng.choice(na, p=pe[b]) for b in range(B)])

# EXACTLY the arithmetic in _coma_correction
b_all = np.einsum("bi,bij->bj", pe, Q)          # mu + beta_j  (+ pe-weighted al/ga)
a_all = np.einsum("bij,bj->bi", Q, pa)          # mu + alpha_i
grand = np.einsum("bj,bj->b", b_all, pa)        # mu
d_beta = b_all[np.arange(B), j] - grand
d_alpha = a_all[np.arange(B), i] - grand

# 2. UNBIASEDNESS: mean-zero under the opponent's own policy
chk("beta_hat is mean-zero under pi_adv (unbiased baseline)",
    float(np.abs(np.einsum("bj,bj->b", b_all - grand[:, None], pa)).max()) < 1e-10)
chk("alpha_hat is mean-zero under pi_ego (unbiased baseline)",
    float(np.abs(np.einsum("bi,bi->b", a_all - grand[:, None], pe)).max()) < 1e-10)

# 4. It removes beta PLUS the policy-weighted gamma marginal -- not beta alone.
# gamma is centred over i in the UNWEIGHTED sense, so SUM_i pi_ego(i) gamma_ij is
# only zero when pi_ego is uniform; in general it is a function of j and rides
# along with beta. That is harmless (it still does not depend on the ego's own
# action, so the baseline stays unbiased) but it is NOT what I first claimed,
# and the difference is exactly what this check pins down.
gm_e = np.einsum("bi,bij->bj", pe, ga)                 # pe-weighted gamma marginal
want_b = (be + gm_e)[np.arange(B), j] - np.einsum("bj,bj->b", be + gm_e, pa)
chk("ego correction == beta_j + pi_ego-weighted gamma marginal (not alpha)",
    float(np.abs(d_beta - want_b).max()) < 1e-10)
gm_a = np.einsum("bij,bj->bi", ga, pa)
want_a = (al + gm_a)[np.arange(B), i] - np.einsum("bi,bi->b", al + gm_a, pe)
chk("adv correction == alpha_i + pi_adv-weighted gamma marginal (not beta)",
    float(np.abs(d_alpha - want_a).max()) < 1e-10)
# and with a UNIFORM opponent policy the gamma marginal vanishes and it is
# exactly beta -- the clean case, verified separately.
pu = np.full((B, na), 1.0 / na)
b_u = np.einsum("bi,bij->bj", pu, Q)
g_u = np.einsum("bj,bj->b", b_u, pa)
chk("under UNIFORM pi_ego the ego correction is exactly beta_j",
    float(np.abs((b_u[np.arange(B), j] - g_u)
                 - (be[np.arange(B), j] - np.einsum("bj,bj->b", be, pa))).max()) < 1e-10)

# 3. SIGN, under a symmetric (antisymmetric-payoff) construction: if the game is
# mirrored, ego's beta and adversary's alpha swap roles and the corrections must
# be exact negatives of each other.
Qs = mu[:, None, None] + al[:, :, None] - al[:, None, :]     # beta = -alpha
b2 = np.einsum("bi,bij->bj", pe, Qs)
a2 = np.einsum("bij,bj->bi", Qs, pe)                          # same policy both seats
g2 = np.einsum("bj,bj->b", b2, pe)
k = np.array([rng.choice(na, p=pe[b]) for b in range(B)])
chk("mirrored payoff: ego and adv corrections are exact negatives",
    float(np.abs((b2[np.arange(B), k] - g2) + (a2[np.arange(B), k] - g2)).max()) < 1e-10)

# 1. INERTNESS at coef 0.
# A must be built to CONTAIN beta, the way a real advantage does -- the first
# version of this used independent noise, so both the real and shuffled
# reductions were ~0 and which one won was a coin flip. A control that cannot
# lose is not a control.
A = (al[np.arange(B), i] + be[np.arange(B), j] + ga[np.arange(B), i, j]
     + rng.randn(B) * 0.01)
chk("coef=0 leaves advantages bitwise unchanged", np.array_equal(A - 0.0 * d_beta, A))
chk("coef=1 actually changes them", not np.array_equal(A - 1.0 * d_beta, A))

red = 1.0 - np.var(A - d_beta) / np.var(A)
red_shuf = 1.0 - np.var(A - d_beta[rng.permutation(B)]) / np.var(A)
chk("removes real variance, and beats its own shuffled control",
    red > 0.10 and red_shuf < red / 2)
print(f"        (synthetic advantage containing beta: real reduction {red:+.3f}, "
      f"shuffled {red_shuf:+.3f})")

if NC != EXPECTED_CHECKS:
    raise SystemExit(f"FAILED: ran {NC} checks, expected {EXPECTED_CHECKS} -- a "
                     f"check that does not run is indistinguishable from one "
                     f"that passes")
print(f"ALL {NC} PASS")
