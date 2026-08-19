"""Does Adam eps (1e-8) suppress the VALUE head's updates at the x0.001 reward scale?

GROUND TRUTH, NOT A PROXY. An earlier read inferred "eps binds" from the value
GRADIENT RMS and got the ratio backwards (grad_rms >> eps means eps does NOT
bind). This reads the ACTUAL Adam second moment sqrt(v_hat) per parameter from
the saved value_optimizer state, which is what the denominator sqrt(v_hat)+eps is
built from.

THE QUANTITY. Adam's step is lr * m_hat / (sqrt(v_hat) + eps). Relative to eps=0
the step is multiplied by sqrt(v_hat)/(sqrt(v_hat)+eps); eps SUPPRESSES it by

    suppression = eps / (sqrt(v_hat) + eps)

eps binds on the LOW TAIL of sqrt(v_hat), not the RMS -- so the decisive number
is the FRACTION OF PARAMETERS suppressed by more than a material amount, not the
mean.

AUDIT, per CLAUDE.md line 8:
  * regime-dependent?  swept across balanced / one-sided / seat-asymmetric /
    early / late checkpoints. If eps binds only in one regime it is a confound.
  * baseline-less?     the value optimizer is compared to the POLICY optimizers
    (ctrl/dstb) on the same checkpoint -- is any binding value-SPECIFIC or does
    it hit every head?
  * scale?             every value parameter (~1-2M elements), >=5 checkpoints.

CLAIM SURVIVES ONLY IF a material fraction of value params are suppressed >20%
in the regimes training actually occupies. Otherwise eps is a non-effect and the
earlier concern is closed.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def opt_sqrt_vhat(opt_state, beta2=0.999):
    """sqrt(v_hat) for every element across all params in an optimizer state_dict."""
    import numpy as np
    import torch as th
    vs = []
    st = opt_state.get("state", {})
    for pid, s in st.items():
        if "exp_avg_sq" not in s:
            continue
        v = s["exp_avg_sq"].detach().float()
        t = float(s.get("step", 0))
        if hasattr(t, "item"):
            t = float(t.item())
        bc = 1.0 - beta2 ** t if t > 0 else 1.0
        vhat = v / max(bc, 1e-30)
        vs.append(th.sqrt(vhat.clamp_min(0)).reshape(-1).cpu().numpy())
    return np.concatenate(vs) if vs else np.zeros(0)


def summarize(sv, eps, label, out=None):
    import numpy as np
    if sv.size == 0:
        print(f"  {label:>34}  (no state)"); return
    supp = eps / (sv + eps)
    med = np.median(sv)
    frac20 = float((supp > 0.20).mean())
    frac50 = float((supp > 0.50).mean())
    print(f"  {label:>34}  n={sv.size:>8}  med_sqrtv={med:.2e}  "
          f"med_supp={np.median(supp)*100:5.1f}%  "
          f"supp>20%: {frac20*100:5.1f}%  supp>50%: {frac50*100:5.1f}%")
    if out is not None:
        out.append(dict(label=label, n=int(sv.size), med_sqrtv=float(med),
                        med_supp=float(np.median(supp)),
                        frac_supp_gt20=frac20, frac_supp_gt50=frac50))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="label=path pairs, e.g. balanced=.../x.task")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--out", default="eps_bind.json")
    a = ap.parse_args(argv)

    import numpy as np
    from stable_baselines3.common.save_util import load_from_zip_file

    print(f"  Adam eps = {a.eps:.0e}.  suppression = eps/(sqrt(v_hat)+eps) per element.\n")
    print(f"  VALUE optimizer, swept across regimes:")
    out = []
    for spec in a.ckpts:
        label, path = spec.split("=", 1)
        _, params, _ = load_from_zip_file(path, device="cpu")
        sv = opt_sqrt_vhat(params["policy.value_optimizer"])
        summarize(sv, a.eps, label, out)

    print(f"\n  BASELINE -- policy optimizers on the LAST checkpoint (is binding value-specific?):")
    label, path = a.ckpts[-1].split("=", 1)
    _, params, _ = load_from_zip_file(path, device="cpu")
    for opt in ("policy.ctrl_optimizer", "policy.dstb_optimizer", "policy.value_optimizer"):
        summarize(opt_sqrt_vhat(params[opt]), a.eps, f"{label}:{opt.split('.')[1]}")

    with open(a.out, "w") as f:
        json.dump({"eps": a.eps, "value_sweep": out}, f, indent=2)
    mx = max((r["frac_supp_gt20"] for r in out), default=0.0)
    print(f"\n  VERDICT: max fraction of value params suppressed >20% across regimes "
          f"= {mx*100:.1f}%")
    if mx < 0.02:
        print("  => eps does NOT bind in any regime. Item-2 eps concern is CLOSED.")
    elif mx < 0.1:
        print("  => eps binds a small tail; marginal. Weigh against LR retuning.")
    else:
        print("  => eps binds a MATERIAL fraction; a smaller eps is worth testing.")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
