"""Out-of-fold accuracy of the mixed, FFPE-only and frozen-only models on both preparations."""
import os

import numpy as np
import pandas as pd

from common import KAGGLE_DIR, PRED, index
from config import RES
from validate import load_polys, match, pixel_dice

VARIANTS = {"mixed": "", "FFPE-only": "_FFPE", "frozen-only": "_fresh_frozen"}


def area(p):
    return 0.5 * abs(np.dot(p[:-1, 0], p[1:, 1]) - np.dot(p[1:, 0], p[:-1, 1]))


def main():
    idx = index()
    rows = []
    for _, r in idx[idx.split == "train"].iterrows():
        prep = "FFPE" if r.px_um == 0.65 else "fresh_frozen"
        man = [p * r.px_um for p in load_polys(os.path.join(KAGGLE_DIR, "train", f"{r.id}.json"))]
        for name, suf in VARIANTS.items():
            d = PRED + suf
            if not os.path.exists(os.path.join(d, f"{r.id}.json")):
                continue
            pred = [p * r.px_um for p in load_polys(os.path.join(d, f"{r.id}.json"))]
            pairs = match(man, pred)
            tp = len(pairs)
            lr = [np.log(area(pred[j]) / area(man[i])) for i, j, _ in pairs]
            rows.append(dict(img_id=r.id, donor=r.donor, target_prep=prep, model=name, n_manual=len(man), n_pred=len(pred),
                             recall=tp / len(man), precision=tp / max(len(pred), 1), f1=2 * tp / (len(man) + len(pred)),
                             pixel_dice=pixel_dice(r.id, d),
                             area_bias_pct=100 * (np.exp(np.median(lr)) - 1) if lr else np.nan,
                             median_iou=np.median([v for *_, v in pairs]) if pairs else np.nan))
    w = pd.DataFrame(rows)
    w.to_csv(os.path.join(RES, "xprep_per_wsi.csv"), index=False, float_format="%.4f")
    s = w.groupby(["model", "target_prep"])[["recall", "precision", "f1", "pixel_dice", "median_iou", "area_bias_pct"]].median()
    s["n_wsi"] = w.groupby(["model", "target_prep"]).size()
    s.to_csv(os.path.join(RES, "xprep_summary.csv"), float_format="%.4f")


if __name__ == "__main__":
    main()
