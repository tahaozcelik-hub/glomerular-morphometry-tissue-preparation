"""Bias transfer of an embedding-based size regressor between preparations, without alignment,
after per-preparation mean-centring and after CORAL, followed by the probes of probe.py.

usage: python mitigation.py <glomerulus_csv> <emb_prefix> model [model ...]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probe import RES, probe


def coral(Z, groups):
    """Whiten each group and recolour it with the pooled covariance."""
    Z = Z.copy()
    eye = np.eye(Z.shape[1]) * 1e-3
    w, V = np.linalg.eigh(np.cov(Z, rowvar=False) + eye)
    col, mu = V @ np.diag(np.sqrt(w)) @ V.T, Z.mean(0)
    for g in np.unique(groups):
        m = groups == g
        w, V = np.linalg.eigh(np.cov(Z[m], rowvar=False) + eye)
        Z[m] = (Z[m] - Z[m].mean(0)) @ (V @ np.diag(1 / np.sqrt(w)) @ V.T) @ col + mu
    return Z


def bias_transfer(Z, df):
    """Median error (%) of a log-area regressor trained on one preparation and applied to the other, by held-out donor."""
    y = np.log(df.area_um2.values)
    out = {}
    for src, dst in (("FFPE", "fresh_frozen"), ("fresh_frozen", "FFPE")):
        res = []
        for d in df.donor.unique():
            tr = ((df.prep == src) & (df.donor != d)).values
            te = ((df.prep == dst) & (df.donor == d)).values
            if te.sum():
                m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 12))).fit(Z[tr], y[tr])
                res.append(m.predict(Z[te]) - y[te])
        out[f"size_bias_{src}_to_{dst}_pct"] = 100 * (np.exp(np.median(np.concatenate(res))) - 1)
    return out


def main(csv, prefix, *models):
    df = pd.read_csv(csv)
    df = df[df.region.str.contains("Cortex")].reset_index()
    rows = []
    for m in models:
        Z = np.load(f"{prefix}_{m}_fixed.npy").astype(np.float32)[df["index"].values]
        Zp = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(Z))
        rows.append(dict(model=m, alignment="none", **bias_transfer(Z, df), **probe(Zp, df)))
        centred = Zp.copy()
        for g in ("FFPE", "fresh_frozen"):
            sel = (df.prep == g).values
            centred[sel] -= centred[sel].mean(0)
        rows.append(dict(model=m, alignment="prep-mean-centring", **bias_transfer(centred, df), **probe(centred, df)))
        cz = coral(Zp, df.prep.values)
        rows.append(dict(model=m, alignment="CORAL", **bias_transfer(cz, df), **probe(cz, df)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "fm_mitigation.csv"), index=False, float_format="%.4f")


if __name__ == "__main__":
    main(*sys.argv[1:])
