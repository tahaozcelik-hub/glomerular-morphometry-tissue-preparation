"""Global-sclerosis classifier: logistic regression on AIDPATH embeddings, applied to the HuBMAP glomeruli.

usage:
  python sclerosis.py cv model [model ...]                         slide-grouped 5-fold CV on AIDPATH
  python sclerosis.py apply <model> <glomerulus_csv> <emb_prefix>  adds p_sclerosis -> <csv>_scl.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMB, RES  # noqa: E402


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))


def aidpath(model):
    a = pd.read_csv(os.path.join(EMB, "aidpath_index.csv"))
    return a, np.load(os.path.join(EMB, f"aidpath_{model}_bbox.npy")).astype(np.float32)


def cv(*models):
    rows = []
    for m in models:
        a, X = aidpath(m)
        y = a.label.values
        p = cross_val_predict(clf(), X, y, groups=a.slide, cv=GroupKFold(5), method="predict_proba")[:, 1]
        mixed = a.groupby("slide").label.transform(lambda s: 0 < s.mean() < 1).astype(bool).values
        rows.append(dict(model=m, auc=roc_auc_score(y, p), bal_acc=balanced_accuracy_score(y, p >= 0.5),
                         auc_within_mixed=roc_auc_score(y[mixed], p[mixed])))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "sclerosis_cv_models.csv"), index=False)


def apply(model, hub_csv, hub_prefix):
    a, X = aidpath(model)
    final = clf().fit(X, a.label.values)
    df = pd.read_csv(hub_csv)
    df["p_sclerosis"] = final.predict_proba(np.load(f"{hub_prefix}_{model}_bbox.npy").astype(np.float32))[:, 1]
    df.to_csv(hub_csv.replace(".csv", "_scl.csv"), index=False, float_format="%.4g")


if __name__ == "__main__":
    {"cv": cv, "apply": apply}[sys.argv[1]](*sys.argv[2:])
