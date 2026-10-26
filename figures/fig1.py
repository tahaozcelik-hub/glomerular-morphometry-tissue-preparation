"""Figure 1. Study overview, paired whole-slide images and paired glomeruli."""
import os

import cv2
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from style import (C8, CACHE, FFPE, FROZEN, INK, PRED, RES, W_FULL, arrow, box, img_ax, letter, plt, polys, save,
                   scalebar, tag)

from common import index  # noqa: E402

PAIR = {"fresh_frozen": "4ef6695ce", "FFPE": "8242609fa"}   # donor 66999
EXAMPLES = [66999, 67026, 67347, 68555]
PREP_TXT = {"fresh_frozen": "Frozen", "FFPE": "FFPE"}
NAVY = "#3d5a80"


def hex_rgb(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def wsi_overlay(img_id, color, step=2):
    """8 µm/px thumbnail with predicted glomeruli filled, cropped to the tissue."""
    r = index(include_extra=True).set_index("id").loc[img_id]
    im = np.load(os.path.join(C8, f"{img_id}_img.npy"))
    im = np.ascontiguousarray(im[::step, ::step]).copy()
    im[im.max(-1) < 30] = 255                                   # black scanner padding -> background
    ps = [np.round(p).astype(np.int32) for p in polys(os.path.join(PRED, f"{img_id}.json"), r.px_um / 8 / step)]
    fill = im.copy()
    cv2.fillPoly(fill, ps, hex_rgb(color))
    im = cv2.addWeighted(fill, 0.6, im, 0.4, 0)
    ys, xs = np.where(im.min(-1) < 215)
    pad = 15
    return im[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad], 8 * step, len(ps)


def crop_with_outline(img_id, cx, cy, side, color, px_um, lw=3):
    img = np.load(os.path.join(CACHE, f"{img_id}_img.npy"), mmap_mode="r")
    x0, y0 = int(cx) - side // 2, int(cy) - side // 2
    c = np.full((side, side, 3), 255, np.uint8)
    ys, xs, ye, xe = max(y0, 0), max(x0, 0), min(y0 + side, img.shape[0]), min(x0 + side, img.shape[1])
    c[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    ps = polys(os.path.join(PRED, f"{img_id}.json"), px_um)
    cv2.polylines(c, [np.round(p - [x0, y0]).astype(np.int32) for p in ps], True, hex_rgb(color), lw, cv2.LINE_AA)
    return c


def header(ax, x, w, text):
    ax.add_patch(Rectangle((x, 0.90), w, 0.10, fc=NAVY, ec="none", transform=ax.transAxes))
    ax.text(x + w / 2, 0.95, text, transform=ax.transAxes, ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="white")


def schematic(ax):
    ax.set_axis_off()
    X = [0.005, 0.262, 0.519, 0.776]
    for x, t in zip(X, ["1  Tissue and donors", "2  Image standardisation", "3  Automated instruments",
                        "4  Analyses"]):
        header(ax, x, 0.219, t)
    rows = [0.60, 0.31, 0.02]
    col1 = [("HuBMAP adult kidneys\n38 donors\nPAS whole-slide images", "#eef4fb"),
            ("Paired set · 7 donors\nfrozen and FFPE sections\nfrom different blocks", "#fdf1ea"),
            ("Frozen cohort · 35 donors\n3 scanner types\nage · BMI · hypertension", "#eaf7f1")]
    col2 = ["Pixel-size audit\nnuclear and glomerular rulers\n3 header errors corrected",
            "Resampling\n1 µm/pixel (glomeruli)\n8 µm/pixel (cortex)",
            "Donor-grouped 4-fold split\nno donor in training and test\nout-of-fold measurement"]
    col3 = ["Glomerulus U-Net\nrecall 0.99 · Dice 0.94\nwatershed instance split",
            "Cortex U-Net\nDice 0.94 · quality control\nglomeruli per mm² cortex",
            "Sclerosis classifier\ntrained on FFPE glomeruli\nfoundation-model features"]
    for y, (t, fc) in zip(rows, col1):
        box(ax, X[0] + 0.007, y, 0.205, 0.26, t, fc=fc)
    for y, t in zip(rows, col2):
        box(ax, X[1] + 0.007, y, 0.205, 0.26, t)
    for y, t in zip(rows, col3):
        box(ax, X[2] + 0.007, y, 0.205, 0.26, t)
    outs = ["Paired preparation effect\narea · MGV · density", "Classifier transfer\nblinded histologist review",
            "Segmentation transfer\nFFPE-only vs frozen-only", "Foundation-model probing\nsix encoders",
            "Clinical associations\nfrozen cohort"]
    for k, t in enumerate(outs):
        box(ax, X[3] + 0.007, 0.708 - k * 0.172, 0.205, 0.152, t, fc="white", fontsize=7, dy=0.022)
    for y in (0.73, 0.44, 0.15):
        arrow(ax, X[1] - 0.038, y, X[1] + 0.005, y)
        arrow(ax, X[2] - 0.038, y, X[2] + 0.005, y)
    for k in range(5):
        arrow(ax, X[3] - 0.038, 0.44, X[3] + 0.005, 0.708 - k * 0.172 + 0.076, lw=0.7)


def main():
    fig = plt.figure(figsize=(W_FULL, 8.4))
    axA = fig.add_axes([0.03, 0.735, 0.95, 0.245])
    schematic(axA)
    letter(fig, axA, "A", dx=0.0)

    # B: paired WSIs of one kidney
    for j, (prep, img_id) in enumerate(PAIR.items()):
        col = FROZEN if prep != "FFPE" else FFPE
        im, um_px, n = wsi_overlay(img_id, col)
        ax = fig.add_axes([0.03 + j * 0.485, 0.39, 0.465, 0.30])
        img_ax(ax, im, f"{PREP_TXT[prep]} section, donor 66999 ({img_id})", color=col)
        scalebar(ax, 2000, 1 / um_px, "2 mm", color=INK)
        tag(ax, f"{n} glomeruli detected", loc="ur")
        if j == 0:
            letter(fig, ax, "B", dx=0.0)

    # C: paired glomeruli of four donors (near-median size in each section)
    idx = index().set_index("id")
    g = pd.read_csv(os.path.join(RES, "pred_glomerulus_morphometry.csv"))
    g = g[g.region.str.contains("Cortex")]
    rng = np.random.default_rng(1)
    for j, d in enumerate(EXAMPLES):
        for i, prep in enumerate(("fresh_frozen", "FFPE")):
            s = g[(g.donor == d) & (g.prep == prep)]
            s = s[(s.area_um2 > s.area_um2.quantile(.4)) & (s.area_um2 < s.area_um2.quantile(.6))]
            r = s.iloc[rng.integers(len(s))]
            col = FROZEN if prep != "FFPE" else FFPE
            c = crop_with_outline(r.img_id, r.centroid_x_um, r.centroid_y_um, 300, col, idx.loc[r.img_id].px_um)
            ax = fig.add_axes([0.03 + j * 0.2425, 0.175 - i * 0.165, 0.225, 0.16])
            img_ax(ax, c, f"Donor {d}" if i == 0 else None)
            tag(ax, PREP_TXT[prep], color="white", bg=col, loc="ul")
            if i == 1 and j == 0:
                scalebar(ax, 100, 1.0)
            if i == 0 and j == 0:
                letter(fig, ax, "C", dx=0.0)
    save(fig, "Figure1")


if __name__ == "__main__":
    main()
