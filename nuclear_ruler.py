"""Median nuclear size next to annotated glomeruli in each WSI, as an internal check of the stated pixel size."""
import os

import numpy as np
import pandas as pd
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, disk

from config import KAGGLE_DIR, RES
from wsi_io import WSI

WIN_UM = 150
N_WIN = 60


def nuclei(rgb, ps):
    h = rgb2hed(rgb)[..., 0]
    mask = binary_opening(h > threshold_otsu(h), disk(max(1, int(round(1 / ps)))))
    out = []
    for r in regionprops(label(mask)):
        a = r.area * ps ** 2
        if 8 <= a <= 120 and r.solidity > 0.9 and r.eccentricity < 0.9:
            out.append((a, r.axis_minor_length * ps))
    return out


def main():
    df = pd.read_csv(os.path.join(RES, "glomerulus_morphometry.csv"))
    rows = []
    for img_id, g in df.groupby("img_id"):
        ps = g.px_um.iloc[0]
        wsi = WSI(os.path.join(KAGGLE_DIR, "train", f"{img_id}.tiff"))
        rng = np.random.default_rng(0)
        side = int(round(WIN_UM / ps))
        nuc, used = [], 0
        for _, r in g.sample(len(g), random_state=0).iterrows():
            if used == N_WIN:
                break
            ang = rng.uniform(0, 2 * np.pi)   # window in the tubulointerstitium beside the glomerulus
            off = (r.feret_max_um / 2 + WIN_UM * 0.75) / ps
            cx, cy = r.centroid_x_px + off * np.cos(ang), r.centroid_y_px + off * np.sin(ang)
            rgb = wsi.read_region(int(cy) - side // 2, int(cx) - side // 2, side, side)
            if (rgb.mean(-1) > 220).mean() > 0.3:
                continue
            nuc += nuclei(rgb, ps)
            used += 1
        wsi.close()
        a = np.array(nuc)
        rows.append(dict(img_id=img_id, donor=g.donor.iloc[0], prep=g.prep.iloc[0], px_um=ps, windows=used,
                         n_nuclei=len(a), nuc_area_median_um2=np.median(a[:, 0]),
                         nuc_minor_axis_median_um=np.median(a[:, 1]), nuc_area_median_px2=np.median(a[:, 0]) / ps ** 2))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "nuclear_ruler.csv"), index=False, float_format="%.4g")


if __name__ == "__main__":
    main()
