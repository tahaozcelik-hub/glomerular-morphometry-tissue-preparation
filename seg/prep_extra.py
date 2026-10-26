"""Resample the HuBMAP extension pyramids (metadata/extension_sections.csv) to 1 µm/pixel."""
import os
import re

import numpy as np
import pandas as pd
import tifffile

from common import CACHE
from config import EXTRA_DIR, META
from prep_1um import resample, tissue_mask
from wsi_io import WSI

# The OME header states 0.50 µm/pixel, but the HuBMAP dataset metadata and both internal rulers
# (median nuclear and median glomerular area) agree on 0.65 µm/pixel.
PX_OVERRIDE = {"HBM933.LDKX.448": 0.65, "HBM395.HZGV.525": 0.65, "HBM925.FFHK.438": 0.65}


def ome_pixel_size(path):
    with tifffile.TiffFile(path) as t:
        return float(re.search(r'PhysicalSizeX="([\d.eE-]+)"', t.ome_metadata).group(1))


def main():
    sections = pd.read_csv(os.path.join(META, "extension_sections.csv")).fillna({"exclude": ""})
    rows = []
    for r in sections[sections.exclude == ""].itertuples():
        path = os.path.join(EXTRA_DIR, r.file)
        ps = PX_OVERRIDE.get(r.id, ome_pixel_size(path))
        ip = os.path.join(CACHE, f"{r.id}_img.npy")
        wsi = WSI(path)
        H, W = wsi.height, wsi.width
        if not os.path.exists(ip):
            resample(wsi, ps, ip + ".tmp.npy")
            os.replace(ip + ".tmp.npy", ip)
        wsi.close()
        img = np.load(ip, mmap_mode="r")
        np.save(os.path.join(CACHE, f"{r.id}_tissue.npy"), tissue_mask(img))
        rows.append(dict(id=r.id, split="extra", px_um=ps, H=H, W=W, H1=img.shape[0], W1=img.shape[1],
                         donor=r.donor, file=r.file))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(CACHE, "extra_index.csv"), index=False)


if __name__ == "__main__":
    main()
