"""Self-contained HTML tool for the blinded review: one PAS crop per case, preparation and classifier output hidden.

usage: python make_review.py [out.html]   (cases from results/histology_review.csv)
"""
import base64
import io
import json
import os
import sys

import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fm"))
from embed import hubmap_crops  # noqa: E402
from config import RES  # noqa: E402

CROP_SCALE = 1.6   # crop side relative to the maximum Feret diameter


def main(out=os.path.join(HERE, "glomerulus_review.html")):
    cases = pd.read_csv(os.path.join(RES, "histology_review.csv"))
    g = pd.read_csv(os.path.join(RES, "pred_glomerulus_morphometry_scl.csv")).set_index("glom_id")
    wide = g.loc[cases.glom_id].reset_index()
    wide["feret_max_um"] *= CROP_SCALE
    imgs = {}
    for case, crop in zip(cases.case, hubmap_crops(wide, "bbox")):
        b = io.BytesIO()
        Image.fromarray(crop).resize((360, 360), Image.LANCZOS).save(b, "JPEG", quality=92)
        imgs[case] = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    html = open(os.path.join(HERE, "review_template.html"), encoding="utf-8").read()
    html = html.replace('const CASES = Array.from({ length: 128 }, (_, i) => "G" + String(i + 1).padStart(3, "0"));',
                        "const IMG = " + json.dumps(imgs) + ";\nconst CASES = Object.keys(IMG);")
    html = html.replace('src="img/${c}.jpg"', 'src="${IMG[c]}"').replace(
        'document.getElementById("zoomimg").src = `img/${c}.jpg`', 'document.getElementById("zoomimg").src = IMG[c]')
    open(out, "w", encoding="utf-8").write(html)


if __name__ == "__main__":
    main(*sys.argv[1:])
