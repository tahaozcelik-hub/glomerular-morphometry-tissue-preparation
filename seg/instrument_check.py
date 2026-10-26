"""Sensitivity of the paired area effect to the ensemble: re-measure the unannotated WSIs with each fold model.

Run first: for K in 0 1 2 3: python infer.py 2ec3f1bb9 3589adb90 57512b7f1 aa05346ff d488c759a --force_fold K
"""
import os

import numpy as np
import pandas as pd
from matplotlib.path import Path

from common import PRED, index
from config import RES
from morphometry import anatomy
from validate import load_polys

TEST = ["2ec3f1bb9", "3589adb90", "57512b7f1", "aa05346ff", "d488c759a"]
PAIRS = {66999: ("8242609fa", ["4ef6695ce"]), 67026: ("b2dc8411c", ["e79de561c"]), 67112: ("2ec3f1bb9", ["c68fe75ea"]),
         67347: ("b9a3865fc", ["afa5e8098", "aa05346ff"]), 68304: ("3589adb90", ["26dc41664"]),
         68555: ("2f6ecfcdf", ["57512b7f1"])}   # donor: (FFPE, frozen sections); 67548 excluded


def cortical_median_area(img_id, pred_dir, px_um):
    cortex = [Path(np.asarray(p.vertices)) for n, p in anatomy("test", img_id) if n == "Cortex"]
    a = []
    for p in load_polys(os.path.join(pred_dir, f"{img_id}.json")):
        if any(c.contains_point(p.mean(0)) for c in cortex):
            x, y = p[:, 0] * px_um, p[:, 1] * px_um
            a.append(0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])))
    return np.median(a)


def main():
    idx = index().set_index("id")
    t = pd.DataFrame({tag: {i: cortical_median_area(i, d, idx.loc[i].px_um) for i in TEST}
                      for tag, d in [("ensemble", PRED)] + [(f"fold{f}", PRED + f"_f{f}") for f in range(4)]})
    t.index.name = "img_id"
    folds = [f"fold{f}" for f in range(4)]
    t["single_fold_spread_pct"] = 100 * (t[folds].max(1) / t[folds].min(1) - 1)
    t["ensemble_vs_singlemean_pct"] = 100 * (t.ensemble / t[folds].mean(1) - 1)
    t.to_csv(os.path.join(RES, "instrument_check_test_wsis.csv"))

    p = pd.read_csv(os.path.join(RES, "pred_glomerulus_morphometry.csv"))
    med = p[p.region.str.contains("Cortex")].groupby("img_id").area_um2.median()
    out = []
    for model in ["ensemble"] + folds:
        L = np.log(med.copy().astype(float))
        L[TEST] = np.log(t[model])
        v = np.array([L[ff] - np.mean([L[f] for f in fz]) for ff, fz in PAIRS.values()])
        out.append(dict(test_wsi_model=model, effect_pct=100 * (np.exp(v.mean()) - 1), n_negative=(v < 0).sum(),
                        min_pct=100 * (np.exp(v.max()) - 1)))
    pd.DataFrame(out).to_csv(os.path.join(RES, "instrument_sensitivity_paired6.csv"), index=False)


if __name__ == "__main__":
    main()
