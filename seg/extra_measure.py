"""Morphometry and section metrics of the HuBMAP extension sections, using the predicted cortex map.

A cortex map is accepted only if at least 90% of the detected glomeruli lie inside it; otherwise MGV uses
all glomeruli and density is not reported.
"""
import json
import os

import cv2
import numpy as np
import pandas as pd

from common import CACHE, PRED, index
from clinical_metrics import BETA, D
from config import C8, META, RES
from morphometry import add_nn_dist, measure

QC_MIN = 0.9


def cortex_mask(img_id):
    prob = np.load(os.path.join(C8, f"{img_id}_cortex_prob.npy"))
    tissue = np.load(os.path.join(CACHE, f"{img_id}_tissue.npy")).astype(np.uint8)
    tissue = cv2.resize(tissue, prob.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    return (prob >= 128) & tissue, tissue


def main():
    idx = index(include_extra=True)
    idx = idx[idx.split == "extra"]
    sec_meta = pd.read_csv(os.path.join(META, "extension_sections.csv")).set_index("id")
    rows, sec = [], []
    for r in idx.itertuples():
        mr = sec_meta.loc[r.id]
        meta = {"patient_number": mr.donor, "sex": mr.sex, "age": mr.age,
                "bmi_kg/m^2": "" if pd.isna(mr.bmi) else mr.bmi, "laterality": ""}
        polys = [np.asarray(f["geometry"]["coordinates"][0], float)
                 for f in json.load(open(os.path.join(PRED, f"{r.id}.json")))]
        feats = measure(polys, r.id, r.px_um, r.H, r.W, [], meta, overlaps=False) if polys else []
        cmask, tissue = cortex_mask(r.id)
        for f in feats:
            y, x = int(f["centroid_y_um"] // 8), int(f["centroid_x_um"] // 8)
            inside = 0 <= y < cmask.shape[0] and 0 <= x < cmask.shape[1] and cmask[y, x]
            f.update(region="Cortex" if inside else "outside", prep=mr.prep, scanner_group=mr.scanner_group,
                     medical_history=mr.medical_history, race=mr.race, glom_id=f["glom_id"].replace(r.id, r.id + "_pred"))
        rows += feats
        qc = sum(f["region"] == "Cortex" for f in feats) / max(len(feats), 1)
        cort = [f for f in feats if f["region"] == "Cortex"] if qc >= QC_MIN else feats
        a = np.array([f["area_um2"] for f in cort])
        cm2 = cmask.sum() * 64 / 1e6 if qc >= QC_MIN else np.nan
        sec.append(dict(img_id=r.id, donor=mr.donor, prep=mr.prep, scanner_group=mr.scanner_group, px_um=r.px_um,
                        age=mr.age, sex=mr.sex, race=mr.race, bmi=mr.bmi, medical_history=mr.medical_history,
                        n_glom_all=len(feats), n_glom=len(cort), cortex_qc_frac=qc, cortex_qc_pass=qc >= QC_MIN,
                        tissue_mm2=tissue.sum() * 64 / 1e6, cortex_mm2=cm2,
                        glom_density_mm2=len(cort) / cm2 if cm2 else np.nan,
                        median_area_um2=np.median(a) if len(a) else np.nan,
                        mean_area_um2=a.mean() if len(a) else np.nan,
                        mgv_um3=BETA / D * a.mean() ** 1.5 if len(a) else np.nan))
    add_nn_dist(pd.DataFrame(rows)).to_csv(os.path.join(RES, "extra_glomerulus_morphometry.csv"), index=False,
                                           float_format="%.4g")
    pd.DataFrame(sec).to_csv(os.path.join(RES, "extra_section_metrics.csv"), index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
