"""Resample the challenge WSIs to 1 µm/pixel, with tissue masks (16 µm/pixel) and manual glomerulus masks."""
import json
import os

import cv2
import numpy as np
import pandas as pd

from common import CACHE, KAGGLE_DIR
from morphometry import pixel_size
from wsi_io import WSI

TARGET_UM = 1.0
STRIP = 1024
TISSUE_STEP = 16


def resample(wsi, ps, path):
    f = ps / TARGET_UM
    H1, W1 = int(round(wsi.height * f)), int(round(wsi.width * f))
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(H1, W1, 3))
    for t0 in range(0, H1, STRIP):
        t1 = min(t0 + STRIP, H1)
        y0, y1 = int(np.floor(t0 / f)), min(int(np.ceil(t1 / f)), wsi.height)
        src = wsi.read_region(y0, 0, y1 - y0, wsi.width)
        out[t0:t1] = cv2.resize(src, (W1, t1 - t0), interpolation=cv2.INTER_AREA)
    out.flush()
    return out


def tissue_mask(img):
    small = img[TISSUE_STEP // 2::TISSUE_STEP, TISSUE_STEP // 2::TISSUE_STEP]
    hsv = cv2.cvtColor(np.ascontiguousarray(small), cv2.COLOR_RGB2HSV)
    m = (hsv[..., 1] > 25) & (hsv[..., 2] < 245)
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)


def main():
    os.makedirs(CACHE, exist_ok=True)
    rows = []
    for split in ("train", "test"):
        for fn in sorted(f for f in os.listdir(os.path.join(KAGGLE_DIR, split)) if f.endswith(".tiff")):
            img_id = fn[:-5]
            tif = os.path.join(KAGGLE_DIR, split, fn)
            ps, H, W = pixel_size(tif)
            ip = os.path.join(CACHE, f"{img_id}_img.npy")
            if not os.path.exists(ip):
                wsi = WSI(tif)
                resample(wsi, ps, ip + ".tmp.npy")   # returned memmap is released before the rename
                wsi.close()
                os.replace(ip + ".tmp.npy", ip)
            img = np.load(ip, mmap_mode="r")
            H1, W1 = img.shape[:2]
            np.save(os.path.join(CACHE, f"{img_id}_tissue.npy"), tissue_mask(img))
            gj = os.path.join(KAGGLE_DIR, split, f"{img_id}.json")
            if os.path.exists(gj):
                mask = np.zeros((H1, W1), np.uint8)
                for feat in json.load(open(gj)):
                    p = np.asarray(feat["geometry"]["coordinates"][0], float) * ps / TARGET_UM
                    cv2.fillPoly(mask, [np.round(p * 8).astype(np.int32)], 1, shift=3)
                np.save(os.path.join(CACHE, f"{img_id}_mask.npy"), mask)
            rows.append(dict(id=img_id, split=split, px_um=ps, H=H, W=W, H1=H1, W1=W1))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(CACHE, "index.csv"), index=False)


if __name__ == "__main__":
    main()
