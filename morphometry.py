"""Glomerulus shape features in µm; run directly to measure the manual outlines of the 15 annotated WSIs."""
import csv
import json
import os
import re

import numpy as np
import pandas as pd
import tifffile
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull, cKDTree
from skimage.draw import polygon as raster_polygon

from config import KAGGLE_DIR, PREP_BY_PX, RES

RESAMPLE_UM = 1.0   # contour resampling step
SMOOTH_UM = 2.0     # Gaussian sigma along the contour


def pixel_size(tiff_path):
    with tifffile.TiffFile(tiff_path) as t:
        desc = t.pages[0].tags["ImageDescription"].value
        shape = t.pages[0].shape
    # raster size from the page shape; the OME axis order is wrong in two files
    if len(shape) == 2:
        h, w = shape
    elif shape[-1] == 3:
        h, w = shape[:2]
    else:
        h, w = shape[1:]
    return float(re.search(r'PhysicalSizeX="([\d.]+)"', desc).group(1)), h, w


def ring_moments(p):
    """Area, centroid and central second moments of a closed polygon."""
    x, y = p[:-1, 0], p[:-1, 1]
    x1, y1 = p[1:, 0], p[1:, 1]
    cr = x * y1 - x1 * y
    a = cr.sum() / 2
    cx = ((x + x1) * cr).sum() / (6 * a)
    cy = ((y + y1) * cr).sum() / (6 * a)
    sxx = ((y * y + y * y1 + y1 * y1) * cr).sum() / 12
    syy = ((x * x + x * x1 + x1 * x1) * cr).sum() / 12
    sxy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / 24
    return abs(a), cx, cy, syy / a - cx * cx, sxx / a - cy * cy, sxy / a - cx * cy


def perimeter(p):
    return np.hypot(*np.diff(p, axis=0).T).sum()


def smooth_contour(p):
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    n = max(int(s[-1] / RESAMPLE_UM), 16)
    t = np.linspace(0, s[-1], n, endpoint=False)
    q = np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])
    q = gaussian_filter1d(q, SMOOTH_UM / RESAMPLE_UM, axis=0, mode="wrap")
    return np.vstack([q, q[:1]])


def feret(hull_pts):
    fmax = np.sqrt(((hull_pts[:, None] - hull_pts[None]) ** 2).sum(-1)).max()
    e = np.roll(hull_pts, -1, axis=0) - hull_pts
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    widths = [(hull_pts @ n).max() - (hull_pts @ n).min() for n in np.column_stack([-e[:, 1], e[:, 0]])]
    return fmax, min(widths)


def self_intersects(p):
    a, b = p[:-1], p[1:]
    n = len(a)
    if n < 4:
        return False
    d1 = b - a
    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))
    i, j = i[keep], j[keep]

    def orient(o, d, q):
        return np.sign(d[:, 0] * (q[:, 1] - o[:, 1]) - d[:, 1] * (q[:, 0] - o[:, 0]))
    o1, o2 = orient(a[i], d1[i], a[j]), orient(a[i], d1[i], b[j])
    o3, o4 = orient(a[j], d1[j], a[i]), orient(a[j], d1[j], b[i])
    return bool(np.any((o1 * o2 < 0) & (o3 * o4 < 0)))


def anatomy(split, img_id):
    regions = []
    for f in json.load(open(os.path.join(KAGGLE_DIR, split, f"{img_id}-anatomical-structure.json"))):
        g = f["geometry"]
        rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [r[0] for r in g["coordinates"]]
        regions += [(f["properties"]["classification"]["name"], Path(np.asarray(r))) for r in rings]
    return regions


def pairwise_overlaps(polys_px):
    boxes = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys_px])
    over = np.zeros(len(polys_px))
    for i in range(len(polys_px)):
        cand = np.where((boxes[:, 0] < boxes[i, 2]) & (boxes[:, 2] > boxes[i, 0]) &
                        (boxes[:, 1] < boxes[i, 3]) & (boxes[:, 3] > boxes[i, 1]))[0]
        for j in cand[cand > i]:
            x0, y0 = np.floor(np.minimum(boxes[i, :2], boxes[j, :2])).astype(int)
            x1, y1 = np.ceil(np.maximum(boxes[i, 2:], boxes[j, 2:])).astype(int) + 1
            m = np.zeros((2, y1 - y0, x1 - x0), bool)
            for k, p in enumerate((polys_px[i], polys_px[j])):
                rr, cc = raster_polygon(p[:, 1] - y0, p[:, 0] - x0, m.shape[1:])
                m[k, rr, cc] = True
            inter = (m[0] & m[1]).sum()
            if inter:
                over[i] = max(over[i], inter / m[0].sum())
                over[j] = max(over[j], inter / m[1].sum())
    return over


def load_meta():
    path = os.path.join(KAGGLE_DIR, "HuBMAP-20-dataset_information.csv")
    return {r["image_file"][:-5]: r for r in csv.DictReader(open(path))}


def measure(polys_px, img_id, ps, H, W, regions, m, overlaps=True):
    """One feature row per polygon (native pixel coordinates, pixel size ps in µm)."""
    overlap = pairwise_overlaps(polys_px) if overlaps else np.full(len(polys_px), np.nan)
    rows = []
    for k, p_px in enumerate(polys_px):
        p = p_px * ps
        a, cx, cy, mu20, mu02, mu11 = ring_moments(p)
        hull = p[ConvexHull(p).vertices]
        hull_closed = np.vstack([hull, hull[:1]])
        hull_area = ring_moments(hull_closed)[0]
        hull_perim = perimeter(hull_closed)
        per_raw = perimeter(p)
        per = perimeter(smooth_contour(p))
        ev = np.sort(np.linalg.eigvalsh([[mu20, mu11], [mu11, mu02]]))[::-1]
        major, minor = 4 * np.sqrt(ev[0]), 4 * np.sqrt(max(ev[1], 0))
        fmax, fmin = feret(hull)
        orient = np.degrees(0.5 * np.arctan2(-2 * mu11, mu20 - mu02))   # counter-clockwise on screen
        reg = [name for name, path in regions if path.contains_point((cx / ps, cy / ps))]
        rows.append(dict(
            glom_id=f"{img_id}_{k:04d}", img_id=img_id, donor=m["patient_number"],
            prep=PREP_BY_PX.get(ps), px_um=ps, sex=m["sex"], age=int(m["age"]),
            bmi=float(m["bmi_kg/m^2"]) if m["bmi_kg/m^2"] else np.nan, laterality=m["laterality"],
            region="|".join(sorted(set(reg))) if reg else "outside",
            centroid_x_px=cx / ps, centroid_y_px=cy / ps, centroid_x_um=cx, centroid_y_um=cy,
            area_um2=a, eq_diam_um=np.sqrt(4 * a / np.pi),
            perimeter_um=per, perimeter_raw_um=per_raw,
            convex_area_um2=hull_area, convex_perimeter_um=hull_perim,
            circularity=4 * np.pi * a / per ** 2, circularity_raw=4 * np.pi * a / per_raw ** 2,
            solidity=a / hull_area, convexity=hull_perim / per,
            feret_max_um=fmax, feret_min_um=fmin, feret_ratio=fmin / fmax,
            major_axis_um=major, minor_axis_um=minor, aspect_ratio=major / minor,
            eccentricity=np.sqrt(1 - ev[1] / ev[0]), orientation_deg=orient,
            n_vertices=len(p_px) - 1, vertices_per_100um=100 * (len(p_px) - 1) / per_raw,
            max_overlap_frac=overlap[k], self_intersecting=self_intersects(p_px),
            touches_border=bool(p_px[:, 0].min() <= 2 or p_px[:, 1].min() <= 2 or
                                p_px[:, 0].max() >= W - 3 or p_px[:, 1].max() >= H - 3),
        ))
    return rows


def add_nn_dist(df):
    for _, g in df.groupby("img_id"):
        d, _ = cKDTree(g[["centroid_x_um", "centroid_y_um"]].values).query(g[["centroid_x_um", "centroid_y_um"]].values, k=2)
        df.loc[g.index, "nn_dist_um"] = d[:, 1]
    return df


def main():
    os.makedirs(RES, exist_ok=True)
    meta = load_meta()
    rows = []
    train = os.path.join(KAGGLE_DIR, "train")
    for fn in sorted(f for f in os.listdir(train) if f.endswith(".tiff")):
        img_id = fn[:-5]
        ps, H, W = pixel_size(os.path.join(train, fn))
        polys_px = [np.asarray(f["geometry"]["coordinates"][0], float)
                    for f in json.load(open(os.path.join(train, f"{img_id}.json")))]
        rows += measure(polys_px, img_id, ps, H, W, anatomy("train", img_id), meta[img_id])
        print(img_id, len(polys_px), flush=True)
    df = add_nn_dist(pd.DataFrame(rows))
    df.to_csv(os.path.join(RES, "glomerulus_morphometry.csv"), index=False, float_format="%.4g")
    print(f"{len(df)} glomeruli")


if __name__ == "__main__":
    main()
