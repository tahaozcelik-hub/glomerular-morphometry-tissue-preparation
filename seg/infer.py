"""Glomerulus segmentation of every cached WSI; polygons are saved in native pixel coordinates.

Annotated WSIs are predicted out of fold (the model that did not see the donor); all others by the four-model ensemble.
usage: python infer.py [ids ...] [--suffix _FFPE] [--force_fold K]
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours, label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from common import CACHE, MODELS, PRED, TISSUE_STEP, index, make_model, normalize

TILE, MARGIN = 1024, 128
STEP = TILE - 2 * MARGIN
MIN_AREA_UM2 = 2500     # smallest manual glomerulus: 3,274 µm²
SPLIT_SOLIDITY = 0.92   # manual glomeruli: median solidity 0.993
SPLIT_H_UM = 8
BATCH = 4


@torch.no_grad()
def predict_tiles(models, img, tissue, device):
    H, W = img.shape[:2]
    prob = np.zeros((H, W), np.uint8)
    coords = []
    for y in range(-MARGIN, H - MARGIN, STEP):
        for x in range(-MARGIN, W - MARGIN, STEP):
            ty0, tx0 = max(y + MARGIN, 0) // TISSUE_STEP, max(x + MARGIN, 0) // TISSUE_STEP
            if tissue[ty0:(y + TILE - MARGIN) // TISSUE_STEP + 1, tx0:(x + TILE - MARGIN) // TISSUE_STEP + 1].any():
                coords.append((y, x))
    for b in range(0, len(coords), BATCH):
        batch = []
        for y, x in coords[b:b + BATCH]:
            t = np.full((TILE, TILE, 3), 255, np.uint8)
            ys, xs, ye, xe = max(y, 0), max(x, 0), min(y + TILE, H), min(x + TILE, W)
            t[ys - y:ye - y, xs - x:xe - x] = img[ys:ye, xs:xe]
            batch.append(t)
        xb = normalize(torch.from_numpy(np.stack(batch)), device)
        p = 0
        for m in models:
            for flip in ((), (3,), (2,)):   # flip test-time augmentation
                xi = torch.flip(xb, flip) if flip else xb
                with torch.autocast("cuda", dtype=torch.float16):
                    o = torch.sigmoid(m(xi).float())
                p = p + (torch.flip(o, flip) if flip else o)
        p = (p / (3 * len(models)))[:, 0].cpu().numpy()
        for (y, x), pt in zip(coords[b:b + BATCH], p):
            ys, xs = max(y + MARGIN, 0), max(x + MARGIN, 0)
            ye, xe = min(y + TILE - MARGIN, H), min(x + TILE - MARGIN, W)
            prob[ys:ye, xs:xe] = np.round(pt[ys - y:ye - y, xs - x:xe - x] * 255).astype(np.uint8)
    return prob


def split_touching(own):
    """Watershed split of a low-solidity object with at least two distance-transform maxima."""
    cs, _ = cv2.findContours(own, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    if hull == 0 or cv2.contourArea(c) / hull >= SPLIT_SOLIDITY:
        return [own]
    dist = distance_transform_edt(own)
    markers = label(h_maxima(dist, SPLIT_H_UM))
    if markers.max() < 2:
        return [own]
    lab = watershed(-dist, markers, mask=own.astype(bool))
    parts = [(lab == k).astype(np.uint8) for k in range(1, lab.max() + 1)]
    return [p for p in parts if p.sum() >= MIN_AREA_UM2] or [own]


def polygons(prob, ps, split=True):
    """Sub-pixel 0.5 contours of each glomerulus, returned in native pixels."""
    binary = (prob >= 128).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < MIN_AREA_UM2:
            continue
        y0, x0 = max(y - 2, 0), max(x - 2, 0)
        crop = prob[y0:y + h + 2, x0:x + w + 2].astype(np.float32)
        own = np.zeros(crop.shape, np.uint8)
        cv2.drawContours(own, [c - [x0, y0]], -1, 1, thickness=-1)
        for part in (split_touching(own) if split else [own]):
            m = cv2.dilate(own, np.ones((3, 3), np.uint8)) if part is own else part   # mask out neighbours
            cs = find_contours(np.pad(crop * m, 1), 127.5)
            if not cs:
                continue
            rc = max(cs, key=len)
            poly = np.column_stack([rc[:, 1] - 1 + x0, rc[:, 0] - 1 + y0])
            if not np.allclose(poly[0], poly[-1]):
                poly = np.vstack([poly, poly[:1]])
            area = 0.5 * abs(np.dot(poly[:-1, 0], poly[1:, 1]) - np.dot(poly[1:, 0], poly[:-1, 1]))
            if area >= MIN_AREA_UM2:
                out.append(poly / ps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--suffix", default="", help="model variant, e.g. _FFPE loads fold{k}_FFPE.pt")
    ap.add_argument("--train_only", action="store_true")
    ap.add_argument("--from_prob", action="store_true", help="re-extract polygons from saved probability maps")
    ap.add_argument("--no_split", action="store_true")
    ap.add_argument("--force_fold", type=int, help="use one fold model for every WSI")
    a = ap.parse_args()
    out_dir = PRED + a.suffix + (f"_f{a.force_fold}" if a.force_fold is not None else "")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda")
    cache = {}

    def load(fold):
        if fold not in cache:
            m = make_model(pretrained=False)
            m.load_state_dict(torch.load(os.path.join(MODELS, f"fold{fold}{a.suffix}.pt"), map_location="cpu",
                                         weights_only=True))
            cache[fold] = m.to(device).eval()
        return cache[fold]

    for _, r in index(include_extra=True).iterrows():
        if (a.ids and r.id not in a.ids) or (a.train_only and r.split != "train"):
            continue
        folds = [r.fold] if r.split == "train" else [0, 1, 2, 3]
        if a.force_fold is not None:
            folds = [a.force_fold]
        prob_path = os.path.join(out_dir, f"{r.id}_prob.npy")
        if a.from_prob:
            prob = np.load(prob_path)
        else:
            img = np.load(os.path.join(CACHE, f"{r.id}_img.npy"), mmap_mode="r")
            tissue = np.load(os.path.join(CACHE, f"{r.id}_tissue.npy"))
            prob = predict_tiles([load(f) for f in folds], img, tissue, device)
            np.save(prob_path, prob)
        polys = polygons(prob, r.px_um, split=not a.no_split)
        feats = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [np.round(p, 2).tolist()]},
                  "properties": {"classification": {"name": "glomerulus"}, "models": [int(f) for f in folds]}}
                 for p in polys]
        json.dump(feats, open(os.path.join(out_dir, f"{r.id}.json"), "w"))
        print(r.id, "folds", folds, "glomeruli", len(polys), flush=True)


if __name__ == "__main__":
    main()
