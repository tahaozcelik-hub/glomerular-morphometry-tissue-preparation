"""Preparation effect on glomerulus-level morphometry: mixed models (donor and section random effects)
and paired donor-level tests on section medians.

usage: python paired_analysis.py --input pred_glomerulus_morphometry.csv --tag _pred7 --cortex_only --no_vertex_cov \
           --paired 66999,67026,67112,67347,67548,68304,68555
"""
import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from config import RES

PAIRED = [66999, 67026, 67347, 67548]
DISCORDANT = 67548
FEATURES = [   # (column, transform, covariates); manual-contour shape features are adjusted for vertex density
    ("area_um2", "log", ""),
    ("eq_diam_um", "log", ""),
    ("feret_max_um", "log", ""),
    ("aspect_ratio", "log", ""),
    ("eccentricity", "none", ""),
    ("circularity", "none", " + np.log(vertices_per_100um)"),
    ("solidity", "none", " + np.log(vertices_per_100um)"),
]


def fit(d, col, tr, cov):
    d = d.assign(y=np.log(d[col]) if tr == "log" else d[col], ffpe=(d.prep == "FFPE").astype(int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("y ~ ffpe" + cov, d, groups="donor", re_formula="1",
                        vc_formula={"section": "0 + C(img_id)"}).fit(reml=True, method="lbfgs")
    v_donor, v_sec, v_res = float(m.cov_re.iloc[0, 0]), float(m.vcomp[0]), float(m.scale)
    tot = v_donor + v_sec + v_res
    b, se = m.params["ffpe"], m.bse["ffpe"]
    out = dict(feature=col, transform=tr, n_glom=len(d), n_sections=d.img_id.nunique(), n_donors=d.donor.nunique(),
               ffpe_coef=b, ffpe_se=se, ffpe_wald_p=m.pvalues["ffpe"],
               var_donor=v_donor, var_section=v_sec, var_residual=v_res,
               pct_donor=100 * v_donor / tot, pct_section=100 * v_sec / tot, pct_residual=100 * v_res / tot,
               converged=m.converged)
    if tr == "log":
        out.update(ffpe_pct_change=100 * (np.exp(b) - 1),
                   ffpe_pct_ci_lo=100 * (np.exp(b - 1.96 * se) - 1), ffpe_pct_ci_hi=100 * (np.exp(b + 1.96 * se) - 1))
    return out


def sign_flip_p(d):
    """Exact two-sided sign-flip permutation P value of the mean."""
    flips = np.array([np.mean(d * s) for s in itertools.product([1, -1], repeat=len(d))])
    return np.mean(np.abs(flips) >= abs(d.mean()) - 1e-12)


def paired_tests(df, paired):
    rows = []
    for col, tr, _ in FEATURES:
        sec = df[df.donor.isin(paired)].groupby(["donor", "prep", "img_id"])[col].median()
        sec = np.log(sec) if tr == "log" else sec
        med = sec.groupby(["donor", "prep"]).mean().unstack()   # donors with two sections: mean of the two
        diff = med.FFPE - med.fresh_frozen
        for label, dd in ((f"{len(paired)} donors", diff), (f"without {DISCORDANT}", diff.drop(DISCORDANT))):
            t = stats.ttest_1samp(dd, 0)
            ci = t.confidence_interval()
            r = dict(feature=col, set=label, n_donors=len(dd), mean_diff=dd.mean(), t_p=t.pvalue,
                     perm_p=sign_flip_p(dd.values), ci_lo=ci.low, ci_hi=ci.high)
            if tr == "log":
                r.update(pct_change=100 * (np.exp(dd.mean()) - 1), pct_ci_lo=100 * (np.exp(ci.low) - 1),
                         pct_ci_hi=100 * (np.exp(ci.high) - 1))
            r.update({f"donor_{k}": v for k, v in dd.items()})
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="glomerulus_morphometry.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--paired", default=",".join(map(str, PAIRED)))
    ap.add_argument("--cortex_only", action="store_true")
    ap.add_argument("--no_vertex_cov", action="store_true", help="for model contours, which have uniform vertex density")
    a = ap.parse_args()
    paired = [int(d) for d in a.paired.split(",")]
    df = pd.read_csv(os.path.join(RES, a.input))
    if a.cortex_only:
        df = df[df.region.str.contains("Cortex")]
    feats = [(c, t, "" if a.no_vertex_cov else cov) for c, t, cov in FEATURES]
    sets = {f"A_paired_{len(paired)}donors": df[df.donor.isin(paired)],
            f"B_paired_without_{DISCORDANT}": df[df.donor.isin([d for d in paired if d != DISCORDANT])],
            f"C_all_{df.img_id.nunique()}_wsis": df}
    mm = pd.DataFrame([dict(model=name, **fit(d, c, t, cov)) for name, d in sets.items() for c, t, cov in feats])
    mm.to_csv(os.path.join(RES, f"mixed_model_results{a.tag}.csv"), index=False, float_format="%.4g")
    paired_tests(df, paired).to_csv(os.path.join(RES, f"paired_donor_comparison{a.tag}.csv"), index=False,
                                    float_format="%.4g")


if __name__ == "__main__":
    main()
