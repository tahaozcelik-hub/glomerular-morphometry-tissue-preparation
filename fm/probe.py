"""Preparation, donor and size information in glomerulus embeddings (and in handcrafted morphometry).

  prep_auc_lodo         FFPE vs frozen, logistic regression, leave one donor out
  ss_prep/donor/section/resid   nested shares of variance in the first 50 principal components
  donor_xprep_top1/knn_acc      retrieval of the same donor across preparations, 7 paired donors (chance 1/7)
  r2_log_area           ridge regression of log area, leave one donor out

usage: python probe.py <glomerulus_csv> <emb_prefix> <tag> model [model ...]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RES  # noqa: E402

PAIRED = [66999, 67026, 67112, 67347, 67548, 68304, 68555]
MORPH = ["area_um2", "eq_diam_um", "feret_max_um", "feret_min_um", "perimeter_um", "circularity", "solidity",
         "aspect_ratio", "eccentricity"]


def ss_decomp(Z, df):
    """Preparation > donor within preparation > section within donor > residual."""
    Z = Z - Z.mean(0)
    tot = (Z ** 2).sum()
    out, prev = {}, np.zeros_like(Z)
    for name, keys in (("prep", ["prep"]), ("donor", ["prep", "donor"]), ("section", ["prep", "donor", "img_id"])):
        means = pd.DataFrame(Z).groupby([df[k].values for k in keys]).transform("mean").values
        out[f"ss_{name}"] = ((means - prev) ** 2).sum() / tot
        prev = means
    out["ss_resid"] = ((Z - prev) ** 2).sum() / tot
    return out


def probe(Z, df):
    res = {}
    ffpe = (df.prep == "FFPE").values.astype(int)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    p = cross_val_predict(lr, Z, ffpe, groups=df.donor, cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
    res["prep_auc_lodo"] = roc_auc_score(ffpe, p)

    pz = (PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(Z)) if Z.shape[1] > 50
          else StandardScaler().fit_transform(Z))
    res.update(ss_decomp(pz, df))

    m = df.donor.isin(PAIRED).values
    zc, dc = pz[m], df[m]
    cent = pd.DataFrame(zc).groupby([dc.prep.values, dc.donor.values]).mean()
    hits, acc = [], []
    for src, dst in (("FFPE", "fresh_frozen"), ("fresh_frozen", "FFPE")):
        A, B = cent.loc[src], cent.loc[dst]
        hits += list(B.index.values[(normalize(A.values) @ normalize(B.values).T).argmax(1)] == A.index.values)
        knn = KNeighborsClassifier(15, metric="cosine").fit(zc[dc.prep == src], dc.donor[dc.prep == src])
        acc.append((knn.predict(zc[dc.prep == dst]) == dc.donor[dc.prep == dst]).mean())
    res["donor_xprep_top1"] = np.mean(hits)
    res["donor_xprep_knn_acc"] = np.mean(acc)

    y = np.log(df.area_um2.values)
    pr = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 12))), Z, y,
                           groups=df.donor, cv=LeaveOneGroupOut())
    res["r2_log_area"] = r2_score(y, pr)
    return res


def main(csv, prefix, tag, *models):
    df = pd.read_csv(csv)
    df = df[df.region.str.contains("Cortex")].reset_index()
    Zm = df[MORPH].values.astype(float)
    Zm[:, :5] = np.log(Zm[:, :5])
    rows = [dict(model="morphometry", dim=Zm.shape[1], **probe(Zm, df))]
    for m in models:
        Z = np.load(f"{prefix}_{m}_fixed.npy").astype(np.float32)[df["index"].values]
        rows.append(dict(model=m, dim=Z.shape[1], **probe(Z, df)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"fm_probe{tag}.csv"), index=False, float_format="%.4f")


if __name__ == "__main__":
    main(*sys.argv[1:])
