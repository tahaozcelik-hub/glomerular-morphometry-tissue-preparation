"""Clinical associations in the frozen-section cohort (challenge and HuBMAP extension donors).

Donor-level MGV and glomerular density (geometric mean over a donor's frozen sections with >= 30 glomeruli),
log-linear OLS with HC3 errors: unadjusted, scanner-adjusted and multivariable.
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import RES

TERMS = {"age10": "Age, per 10 years", "bmi5": "BMI, per 5 kg/m²", "htn": "Hypertension", "C(sex)[T.M]": "Male sex"}


def flag(s, key):
    return s.fillna("").str.contains(key).astype(int)


def sections():
    k = pd.read_csv(os.path.join(RES, "clinical_metrics_pred.csv"))
    t1 = pd.read_csv(os.path.join(RES, "table1_donors.csv")).set_index("donor")
    k["donor"] = k.donor.astype(str)
    t1.index = t1.index.astype(str)
    k = k.assign(source="Kaggle", scanner="Leica SCN400 (Kaggle)",
                 htn=k.donor.map(t1.hypertension.map({"yes": 1, "no": 0})),
                 t2dm=k.donor.map(t1.diabetes.map({"yes": 1, "no": 0})), race=k.donor.map(t1.race))
    e = pd.read_csv(os.path.join(RES, "extra_section_metrics.csv"))
    e = e.assign(source="HuBMAP", htn=flag(e.medical_history, "Hypertension"), t2dm=flag(e.medical_history, "Diabetes"),
                 scanner=e.scanner_group, race=e.race.str.contains("Black").map({True: "Black", False: "White"}))
    cols = ["img_id", "donor", "source", "prep", "scanner", "age", "sex", "race", "bmi", "htn", "t2dm",
            "n_glom", "cortex_mm2", "glom_density_mm2", "mgv_um3"]
    s = pd.concat([k[cols], e[cols]], ignore_index=True)
    s["sex"] = s.sex.str[0]
    return s


def donors(s):
    f = s[(s.prep == "fresh_frozen") & (s.n_glom >= 30)]
    g = f.groupby("donor")
    d = g.agg(source=("source", "first"), scanner=("scanner", "first"), age=("age", "first"), sex=("sex", "first"),
              race=("race", "first"), bmi=("bmi", "first"), htn=("htn", "first"), t2dm=("t2dm", "first"),
              n_sections=("img_id", "size"), n_glom=("n_glom", "sum"), cortex_mm2=("cortex_mm2", "sum"))
    d["mgv"] = g.mgv_um3.apply(lambda v: np.exp(np.log(v).mean()))
    d["density"] = g.glom_density_mm2.apply(lambda v: np.exp(np.log(v).mean()))
    return d.reset_index()


def fit(d, outcome, terms):
    d = d.dropna(subset=[t for t in terms.replace("C(", "").replace(")", "").split(" + ")] + [outcome]).copy()
    d["y"] = np.log(d[outcome])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.ols(f"y ~ {terms}", d).fit(cov_type="HC3"), len(d)


def row(outcome, model, m, n, key):
    b, lo, hi = m.params[key], *m.conf_int().loc[key]
    return dict(outcome=outcome, model=model, term=key, n=n, pct=100 * (np.exp(b) - 1),
                ci_lo=100 * (np.exp(lo) - 1), ci_hi=100 * (np.exp(hi) - 1), p=m.pvalues[key])


def models(d):
    rows = []
    for outcome in ("mgv", "density"):
        for terms in ("age10", "bmi5", "htn", "t2dm", "C(sex)", "C(race)"):
            for adj, extra in (("unadjusted", ""), ("scanner-adjusted", " + C(scanner)")):
                m, n = fit(d, outcome, terms + extra)
                rows.append(row(outcome, adj, m, n, [p for p in m.params.index if p.startswith(terms)][0]))
        m, n = fit(d, outcome, "age10 + C(sex) + bmi5 + htn + C(scanner)")
        rows += [row(outcome, "multivariable", m, n, k) for k in m.params.index if k != "Intercept" and "scanner" not in k]
    return pd.DataFrame(rows)


def summary_table(r):
    r = r[r.term.isin(TERMS)].copy()
    r["cell"] = [f"{p:+.1f} % ({lo:+.1f} to {hi:+.1f}), p={pv:.3f}" for p, lo, hi, pv in zip(r.pct, r.ci_lo, r.ci_hi, r.p)]
    t = r.pivot_table(index=["outcome", "term"], columns="model", values="cell", aggfunc="first").reset_index()
    t["Outcome"] = t.outcome.map({"mgv": "MGV", "density": "Glomerular density"})
    t["Term"] = t.term.map(TERMS)
    t = t.rename(columns={"unadjusted": "Unadjusted", "scanner-adjusted": "Scanner-adjusted", "multivariable": "Multivariable"})
    return t.sort_values(["Outcome", "Term"])[["Outcome", "Term", "Unadjusted", "Scanner-adjusted", "Multivariable"]]


def main():
    d = donors(sections())
    d.to_csv(os.path.join(RES, "cohort_donors.csv"), index=False, float_format="%.5g")
    d["age10"], d["bmi5"] = d.age / 10, d.bmi / 5
    path = os.path.join(RES, "cohort_models.csv")
    models(d).to_csv(path, index=False, float_format="%.4g")
    summary_table(pd.read_csv(path)).to_csv(os.path.join(RES, "table3_cohort.csv"), index=False)


if __name__ == "__main__":
    main()
