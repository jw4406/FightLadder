"""Batched zero-sum matrix-game solver for minimax-Q.

Given Q(s, a_ego, a_adv) as a (B, n, m) batch of payoff matrices (ROW player =
ego = maximizer, zero-sum), compute for each state

    V(s) = max_{p in simplex(n)} min_{q in simplex(m)} p^T M q

which is the value of the game. The equilibrium policies p, q fall out of the
same solve.

WHY NOT AN LP: this is exact and tiny per state (23 variables at n=m=22), but
scipy.linprog is serial CPU at ~1 ms each. A rollout is 12,288 states => ~12 s,
against ~2.7 s of env time for the same rollout. It would more than quadruple
iteration time to compute a critic target, and it is not differentiable.

WHAT THIS DOES INSTEAD: multiplicative weights (mirror descent on the simplex),
which is two batched matmuls and two softmaxes per iteration and runs on the
GPU. Time-averaged iterates converge to a Nash equilibrium of the matrix game.

    p_{t+1} ∝ p_t * exp(+eta * (M q_t))
    q_{t+1} ∝ q_t * exp(-eta * (M^T p_t))

THE SCALE TRAP -- read before touching `eta`/`tau`:
`eta` trades off against the MAGNITUDE of M, so it is NOT scale-free. Measured
return scale on this task is G_std ~ 0.0166, so raw Q entries sit around 0.02.
With eta small relative to that, the exponent is ~0 every iteration, the weights
never move, and the solver silently returns UNIFORM play with a large duality
gap -- a wrong answer that looks like a working solver. `normalize=True`
(default) rescales each matrix to unit std before iterating and reports V in the
ORIGINAL units, which makes eta scale-free. Do not disable it without a reason.

ALWAYS CHECK THE GAP. Matrix games hand you a free convergence certificate:

    gap = max_a (M q)_a  -  min_o (p^T M)_o   >= 0,  == 0 at equilibrium

If the median gap is not near zero the solve did not converge and every V
downstream is meaningless. `solve_matrix_game` returns it; log it.
"""
from typing import NamedTuple, Optional

import torch as th


class MinimaxSolution(NamedTuple):
    V: th.Tensor        # (B,)    game value, in the ORIGINAL units of M
    p: th.Tensor        # (B, n)  row (ego) equilibrium strategy
    q: th.Tensor        # (B, m)  column (adversary) equilibrium strategy
    gap: th.Tensor      # (B,)    duality gap, original units; ->0 at equilibrium


def _row_value(M: th.Tensor, q: th.Tensor) -> th.Tensor:
    """(M q)_a -- ego's payoff for each pure action against mixed q. (B, n)"""
    return th.bmm(M, q.unsqueeze(-1)).squeeze(-1)


def _col_value(M: th.Tensor, p: th.Tensor) -> th.Tensor:
    """(p^T M)_o -- ego's payoff from mixed p against each pure adv action. (B, m)"""
    return th.bmm(p.unsqueeze(1), M).squeeze(1)


def duality_gap(M: th.Tensor, p: th.Tensor, q: th.Tensor) -> th.Tensor:
    """max_a (M q)_a - min_o (p^T M)_o.

    Non-negative, and zero exactly at a Nash equilibrium: the first term is the
    best ego can do against q, the second is the worst adv can hold p to. This
    is the only honest convergence check available, so it is returned always.
    """
    return _row_value(M, q).max(dim=-1).values - _col_value(M, p).min(dim=-1).values


def solve_matrix_game(
    M: th.Tensor,
    iters: int = 1024,
    eta: float = 0.5,
    normalize: bool = True,
    average: bool = True,
    eps: float = 1e-8,
    p0: Optional[th.Tensor] = None,
    q0: Optional[th.Tensor] = None,
) -> MinimaxSolution:
    """Solve a batch of zero-sum matrix games by multiplicative weights.

    Args:
        M: (B, n, m) or (n, m). Row player maximizes. Zero-sum, so the column
           player's payoff is -M.
        iters: optimistic-MWU iterations -- the accuracy dial. Measured on
           12,288 unit-variance 22x22 games (a full rollout) on one 4090:
               256   18 ms   median duality gap 2.1e-02
               1024  70 ms   median duality gap 5.1e-03
           1024 is the default: a rollout is ~2.7 s of env time, so 70 ms is
           ~2.5% overhead for a 4x tighter convergence certificate. Lower it
           only if the logged gap stays acceptable.
        eta: step size, applied to the NORMALIZED matrix when normalize=True.
        normalize: rescale each matrix to zero mean / unit std before iterating,
           and report V and gap in original units. See the scale trap above.
        average: return time-averaged iterates. This is what gives MWU its Nash
           guarantee -- the LAST iterate can cycle (matching pennies is the
           textbook example, and it appears in this game too).
        p0, q0: optional warm starts, (B, n) / (B, m). Uniform if omitted.

    Returns:
        MinimaxSolution(V, p, q, gap), all in the original units of M.
    """
    squeeze_batch = (M.dim() == 2)
    if squeeze_batch:
        M = M.unsqueeze(0)
    if M.dim() != 3:
        raise ValueError(f"M must be (B,n,m) or (n,m); got {tuple(M.shape)}")
    B, n, m = M.shape

    # Solve on a scaled copy so `eta` means the same thing regardless of the
    # reward scale; V and gap are computed against the ORIGINAL M at the end.
    if normalize:
        flat = M.reshape(B, -1)
        mu = flat.mean(dim=1).view(B, 1, 1)
        sd = flat.std(dim=1).view(B, 1, 1).clamp_min(eps)
        Mw = (M - mu) / sd
    else:
        Mw = M

    # Work in log space: MWU is additive on logits, and softmax subtracts the
    # max, so this stays stable even when eta*iters drives logits large.
    lp = th.zeros(B, n, dtype=M.dtype, device=M.device)
    lq = th.zeros(B, m, dtype=M.dtype, device=M.device)
    if p0 is not None:
        lp = th.log(p0.clamp_min(eps))
    if q0 is not None:
        lq = th.log(q0.clamp_min(eps))

    p = th.softmax(lp, dim=-1)
    q = th.softmax(lq, dim=-1)
    p_sum = th.zeros_like(p)
    q_sum = th.zeros_like(q)

    # OPTIMISTIC MWU (Rakhlin & Sridharan): step on (2*g_t - g_{t-1}) rather
    # than g_t, i.e. extrapolate the opponent's next move from their last one.
    # Plain MWU converges as O(sqrt(log n / T)) and measured a median duality
    # gap of 0.056 on unit-variance 22x22 games at 4096 iterations -- ~5% of the
    # matrix scale, too loose to certify anything. Optimism gives O(1/T) and
    # drops that by orders of magnitude at the same cost per iteration (the two
    # extra tensors below are the entire overhead).
    gp_prev = th.zeros_like(p)
    gq_prev = th.zeros_like(q)

    for _ in range(iters):
        # Simultaneous: q's step uses the PRE-update p, so this is a genuine
        # simultaneous mirror-descent step rather than a Gauss-Seidel sweep.
        gp = _row_value(Mw, q)          # (B, n)  ego payoff per row
        gq = _col_value(Mw, p)          # (B, m)  ego payoff per column
        lp = lp + eta * (2.0 * gp - gp_prev)
        lq = lq - eta * (2.0 * gq - gq_prev)    # adv minimizes ego's payoff
        gp_prev, gq_prev = gp, gq
        p = th.softmax(lp, dim=-1)
        q = th.softmax(lq, dim=-1)
        p_sum += p
        q_sum += q

    if average:
        p = p_sum / iters
        q = q_sum / iters

    V = th.einsum("ba,bao,bo->b", p, M, q)
    gap = duality_gap(M, p, q)

    if squeeze_batch:
        return MinimaxSolution(V.squeeze(0), p.squeeze(0), q.squeeze(0), gap.squeeze(0))
    return MinimaxSolution(V, p, q, gap)


@th.no_grad()
def minimax_values(M: th.Tensor, **kw) -> th.Tensor:
    """V only, no grad. The common case: computing GAE bootstraps for a rollout."""
    return solve_matrix_game(M, **kw).V
