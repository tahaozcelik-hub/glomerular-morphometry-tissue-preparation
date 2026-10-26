"""Section-level metrics of the challenge WSIs: cortex area, glomerular density, MGV and classifier flags.

usage: python clinical_metrics.py <glomerulus_csv> <out_csv>
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

from config import CACHE, KAGGLE_DIR

STEP = 16                # tissue-mask grid (µm/pixel)
BETA, D = 1.382, 1.01    # Weibel-Gomez shape and size-distribution coefficients


def cortex_mm2(img_id, split, px_um):
    tissue = np.load(os.path.join(CACHE, f"{img_id}_tissue.npy"))
    cortex = np.zeros(tissue.shape, np.uint8)
    for f in json.load(open(os.path.join(KAGGLE_DIR, split, f"{img_id}-anatomical-structure.json"))):
        if f["properties"]["classification"]["name"] != "Cortex":
            continue
        g = f["geometry"]
        rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [r[0] for r in g["coordinates"]]
        for r in rings:
            p = np.asarray(r, float) * px_um / STEP
            cv2.fillPoly(cortex, [np.round(p * 8).astype(np.int32)], 1, shift=3)
    return (cortex.astype(bool) & tissue).sum() * STEP ** 2 / 1e6


def mgv(a):
    return BETA / D * np.mean(a) ** 1.5 if len(a) else np.nan


def main(inp, out):
    df = pd.read_csv(inp)
    df = df[df.region.str.contains("Cortex")]
    idx = pd.read_csv(os.path.join(CACHE, "index.csv")).set_index("id")
    rows = []
    for img_id, g in df.groupby("img_id"):
        r = idx.loc[img_id]
        cm = cortex_mm2(img_id, r.split, r.px_um)
        row = dict(img_id=img_id, split=r.split, donor=g.donor.iloc[0], prep=g.prep.iloc[0], age=g.age.iloc[0],
                   sex=g.sex.iloc[0], bmi=g.bmi.iloc[0], n_glom=len(g), cortex_mm2=cm, glom_density_mm2=len(g) / cm,
                   mean_area_um2=g.area_um2.mean(), median_area_um2=g.area_um2.median(),
                   mgv_um3=mgv(g.area_um2), mean_eq_diam_um=g.eq_diam_um.mean())
        if "p_sclerosis" in g:
            scl = g.p_sclerosis >= 0.5
            row.update(n_scl=int(scl.sum()), gs_pct=100 * scl.mean(), mgv_nonscl_um3=mgv(g.area_um2[~scl]),
                       nonscl_density_mm2=(~scl).sum() / cm)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.5g")


if __name__ == "__main__":
    main(*sys.argv[1:])
