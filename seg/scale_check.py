"""Nuclear-size ruler on the 1 µm/pixel cache, with predicted glomeruli as anchors.

usage: python scale_check.py <out_csv> id [id ...]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from common import CACHE, PRED, index
from nuclear_ruler import N_WIN, WIN_UM, nuclei


def main(out, *ids):
    idx = index(include_extra=True).set_index("id")
    rows = []
    for i in ids:
        r = idx.loc[i]
        img = np.load(os.path.join(CACHE, f"{i}_img.npy"), mmap_mode="r")
        polys = [np.asarray(f["geometry"]["coordinates"][0]) * r.px_um
                 for f in json.load(open(os.path.join(PRED, f"{i}.json")))]
        rng = np.random.default_rng(0)
        nuc, used = [], 0
        for k in rng.permutation(len(polys)):
            if used == N_WIN:
                break
            p = polys[k]
            cx, cy = p.mean(0)
            off = np.hypot(*(p - [cx, cy]).T).max() + WIN_UM * 0.75
            ang = rng.uniform(0, 2 * np.pi)
            x, y = int(cx + off * np.cos(ang)), int(cy + off * np.sin(ang))
            crop = np.asarray(img[max(y - WIN_UM // 2, 0):y + WIN_UM // 2, max(x - WIN_UM // 2, 0):x + WIN_UM // 2])
            if crop.shape[:2] != (WIN_UM, WIN_UM) or (crop.mean(-1) > 220).mean() > 0.3 or (crop.mean(-1) < 15).mean() > 0.05:
                continue
            nuc += nuclei(crop, 1.0)
            used += 1
        a = np.array(nuc)
        rows.append(dict(img_id=i, split=r.split, px_um=r.px_um, windows=used, n_nuclei=len(a),
                         nuc_area_median_um2=np.median(a[:, 0]) if len(a) else np.nan,
                         nuc_minor_axis_median_um=np.median(a[:, 1]) if len(a) else np.nan))
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4g")


if __name__ == "__main__":
    main(*sys.argv[1:])
