"""Out-of-fold agreement with the manual outlines and morphometry of all predicted glomeruli (challenge WSIs)."""
import json
import os

import numpy as np
import pandas as pd
from skimage.draw import polygon as raster_polygon

from common import CACHE, KAGGLE_DIR, PRED, index
from config import RES
from morphometry import add_nn_dist, anatomy, load_meta, measure

FEATS = ["area_um2", "eq_diam_um", "feret_max_um", "circularity", "solidity", "aspect_ratio", "eccentricity"]


def load_polys(path):
    return [np.asarray(f["geometry"]["coordinates"][0], float) for f in json.load(open(path))]


def iou(p, q):
    x0, y0 = np.floor(np.minimum(p.min(0), q.min(0))).astype(int)
    x1, y1 = np.ceil(np.maximum(p.max(0), q.max(0))).astype(int) + 1
    m = np.zeros((2, y1 - y0, x1 - x0), bool)
    for k, r in enumerate((p, q)):
        rr, cc = raster_polygon(r[:, 1] - y0, r[:, 0] - x0, m.shape[1:])
        m[k, rr, cc] = True
    return (m[0] & m[1]).sum() / max((m[0] | m[1]).sum(), 1)


def match(man, pred, thr=0.5):
    """Greedy one-to-one matching on IoU; polygons at 1 µm/pixel."""
    bm = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in man])
    bp = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in pred])
    cand = []
    for i in range(len(man)):
        js = np.where((bp[:, 0] < bm[i, 2]) & (bp[:, 2] > bm[i, 0]) & (bp[:, 1] < bm[i, 3]) & (bp[:, 3] > bm[i, 1]))[0]
        cand += [(iou(man[i], pred[j]), i, j) for j in js]
    pairs, used_i, used_j = [], set(), set()
    for v, i, j in sorted(cand, reverse=True):
        if v >= thr and i not in used_i and j not in used_j:
            pairs.append((i, j, v))
            used_i.add(i)
            used_j.add(j)
    return pairs


def pixel_dice(img_id, pred_dir=PRED):
    mask = np.load(os.path.join(CACHE, f"{img_id}_mask.npy"), mmap_mode="r")
    prob = np.load(os.path.join(pred_dir, f"{img_id}_prob.npy"), mmap_mode="r")
    inter = tot = 0
    for y in range(0, mask.shape[0], 2048):
        m, p = mask[y:y + 2048] > 0, prob[y:y + 2048] >= 128
        inter += np.count_nonzero(m & p)
        tot += np.count_nonzero(m) + np.count_nonzero(p)
    return 2 * inter / tot


def main():
    idx = index()
    meta = load_meta()
    pred_rows, match_rows, wsi_rows = [], [], []
    for _, r in idx.iterrows():
        ps = r.px_um
        regions = anatomy(r.split, r.id)
        pred = load_polys(os.path.join(PRED, f"{r.id}.json"))
        pf = measure(pred, r.id, ps, r.H, r.W, regions, meta[r.id], overlaps=False)
        for row in pf:
            row.update(split=r.split, fold=r.fold, glom_id=row["glom_id"].replace(r.id, r.id + "_pred"))
        pred_rows += pf
        if r.split != "train":
            continue
        man = load_polys(os.path.join(KAGGLE_DIR, "train", f"{r.id}.json"))
        mf = measure(man, r.id, ps, r.H, r.W, regions, meta[r.id], overlaps=False)
        pairs = match([p * ps for p in man], [p * ps for p in pred])
        rows = []
        for i, j, v in pairs:
            row = dict(img_id=r.id, donor=r.donor, prep=mf[i]["prep"], fold=r.fold, iou=v,
                       manual_region=mf[i]["region"], vertices_per_100um=mf[i]["vertices_per_100um"])
            for f in FEATS:
                row[f"manual_{f}"], row[f"pred_{f}"] = mf[i][f], pf[j][f]
            rows.append(row)
        match_rows += rows
        mm, tp = pd.DataFrame(rows), len(pairs)
        wsi_rows.append(dict(
            img_id=r.id, donor=r.donor, prep=mf[0]["prep"], fold=r.fold, n_manual=len(man), n_pred=len(pred), tp=tp,
            precision=tp / max(len(pred), 1), recall=tp / len(man), f1=2 * tp / (len(man) + len(pred)),
            pixel_dice=pixel_dice(r.id), median_iou=mm.iou.median(),
            area_ratio_median=np.exp(np.median(np.log(mm.pred_area_um2 / mm.manual_area_um2))),
            manual_area_median=np.median([m["area_um2"] for m in mf]), pred_area_median=np.median([p["area_um2"] for p in pf]),
            manual_vertices_per_100um=np.median([m["vertices_per_100um"] for m in mf])))
        print(wsi_rows[-1], flush=True)
    add_nn_dist(pd.DataFrame(pred_rows)).to_csv(os.path.join(RES, "pred_glomerulus_morphometry.csv"), index=False,
                                                float_format="%.4g")
    pd.DataFrame(match_rows).to_csv(os.path.join(RES, "seg_validation_matches.csv"), index=False, float_format="%.4g")
    pd.DataFrame(wsi_rows).to_csv(os.path.join(RES, "seg_validation_per_wsi.csv"), index=False, float_format="%.4g")


if __name__ == "__main__":
    main()
