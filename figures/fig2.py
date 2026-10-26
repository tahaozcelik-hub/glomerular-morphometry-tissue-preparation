"""Figure 2. Automated glomerulus segmentation: image / manual label / model output / agreement, and accuracy."""
import os

import cv2
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from style import (CACHE, GREEN, INK, INK2, KAGGLE_DIR, MAGENTA, PRED, PREP_COL, PREP_LAB, RES, WORK_DIR, W_FULL,
                   YELLOW, img_ax, letter, plt, save, scalebar, tag)

from common import index  # noqa: E402
from validate import load_polys, match  # noqa: E402

FIELDS = {"fresh_frozen": "4ef6695ce", "FFPE": "8242609fa"}
FW, FH = 1050, 700          # field of view (µm) for panel A
CACHE_MATCH = os.path.join(WORK_DIR, "fig2_matches.pkl")


def hex_rgb(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def all_matches():
    """Manual and predicted polygons (1 µm grid) with TP / FP / FN status, for the 15 annotated WSIs."""
    if os.path.exists(CACHE_MATCH):
        return pd.read_pickle(CACHE_MATCH)
    idx = index()
    rows = []
    idx["prep"] = np.where(np.isclose(idx.px_um, 0.65), "FFPE", "fresh_frozen")
    for _, r in idx[idx.split == "train"].iterrows():
        man = [p * r.px_um for p in load_polys(os.path.join(KAGGLE_DIR, "train", f"{r.id}.json"))]
        pred = [p * r.px_um for p in load_polys(os.path.join(PRED, f"{r.id}.json"))]
        pairs = match(man, pred)
        mi, pj = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
        rows += [dict(img_id=r.id, prep=r.prep, kind="manual", status="TP" if i in mi else "FN", poly=p)
                 for i, p in enumerate(man)]
        rows += [dict(img_id=r.id, prep=r.prep, kind="pred", status="TP" if j in pj else "FP", poly=p)
                 for j, p in enumerate(pred)]
    df = pd.DataFrame(rows)
    df.to_pickle(CACHE_MATCH)
    return df


def best_field(img_id, m):
    """Field with many manual glomeruli, preferring one that contains an error."""
    d = m[m.img_id == img_id].copy()
    d["cx"] = d.poly.map(lambda p: p[:, 0].mean())
    d["cy"] = d.poly.map(lambda p: p[:, 1].mean())
    best, score = None, -1
    img = np.load(os.path.join(CACHE, f"{img_id}_img.npy"), mmap_mode="r")
    for _, e in pd.concat([d[d.status != "TP"], d[d.kind == "manual"].sample(150, random_state=0)]).iterrows():
        for ox in (0.2, 0.5, 0.8):
            for oy in (0.3, 0.5, 0.7):
                x0, y0 = int(e.cx - ox * FW), int(e.cy - oy * FH)
                if x0 < 0 or y0 < 0 or x0 + FW > img.shape[1] or y0 + FH > img.shape[0]:
                    continue
                inside = d[(d.cx > x0 + 60) & (d.cx < x0 + FW - 60) & (d.cy > y0 + 60) & (d.cy < y0 + FH - 60)]
                n_man = (inside.kind == "manual").sum()
                s = n_man + 3 * min((inside.status != "TP").sum(), 1)
                if s > score:
                    best, score = (x0, y0), s
    return best


def field_panels(img_id, m, x0, y0):
    img = np.ascontiguousarray(np.load(os.path.join(CACHE, f"{img_id}_img.npy"), mmap_mode="r")[y0:y0 + FH, x0:x0 + FW])
    mask = np.load(os.path.join(CACHE, f"{img_id}_mask.npy"), mmap_mode="r")[y0:y0 + FH, x0:x0 + FW] > 0
    prob = np.load(os.path.join(PRED, f"{img_id}_prob.npy"), mmap_mode="r")[y0:y0 + FH, x0:x0 + FW] / 255.0
    d = m[m.img_id == img_id]
    ov = img.copy()
    loc = lambda p: np.round(p - [x0, y0]).astype(np.int32)  # noqa: E731
    for _, e in d.iterrows():
        if e.kind == "pred" and e.status == "TP":
            cv2.polylines(ov, [loc(e.poly)], True, hex_rgb(GREEN), 4, cv2.LINE_AA)
        elif e.kind == "pred":
            cv2.polylines(ov, [loc(e.poly)], True, hex_rgb(MAGENTA), 4, cv2.LINE_AA)
        elif e.status == "FN":
            cv2.polylines(ov, [loc(e.poly)], True, hex_rgb(YELLOW), 4, cv2.LINE_AA)
    lab = np.zeros((FH, FW, 3), np.uint8) + 20
    lab[mask] = 235
    return img, lab, prob, ov


def tile(img_id, poly, color, side=300):
    img = np.load(os.path.join(CACHE, f"{img_id}_img.npy"), mmap_mode="r")
    cx, cy = poly.mean(0).astype(int)
    x0, y0 = cx - side // 2, cy - side // 2
    c = np.full((side, side, 3), 255, np.uint8)
    ys, xs, ye, xe = max(y0, 0), max(x0, 0), min(y0 + side, img.shape[0]), min(x0 + side, img.shape[1])
    c[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    cv2.polylines(c, [np.round(poly - [x0, y0]).astype(np.int32)], True, hex_rgb(color), 4, cv2.LINE_AA)
    return c


def main():
    m = all_matches()
    fig = plt.figure(figsize=(W_FULL, 7.6))

    # A: image / label / output / agreement, frozen and FFPE
    titles = ["PAS image", "Manual label", "U-Net probability", "Agreement"]
    for i, (prep, img_id) in enumerate(FIELDS.items()):
        x0, y0 = best_field(img_id, m)
        ims = field_panels(img_id, m, x0, y0)
        for j, im in enumerate(ims):
            ax = fig.add_axes([0.075 + j * 0.232, 0.775 - i * 0.165, 0.222, 0.155])
            if j == 2:
                ax.imshow(im, cmap="magma", vmin=0, vmax=1, interpolation="lanczos")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color(INK2)
            else:
                img_ax(ax, im)
            if i == 0:
                ax.set_title(titles[j], fontsize=8.5, pad=3)
            if j == 0:
                scalebar(ax, 200, 1.0, "200 µm")
                ax.set_ylabel(f"{PREP_LAB[prep]}\n{img_id}", fontsize=8, color=PREP_COL[prep], fontweight="bold",
                              labelpad=4)
                if i == 0:
                    letter(fig, ax, "A", dx=-0.035)
            if j == 2 and i == 1:
                cax = fig.add_axes([0.075 + 2 * 0.232 + 0.03, 0.6085, 0.16, 0.008])
                cb = fig.colorbar(ax.images[0], cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
                cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
                cb.outline.set_linewidth(0.4)
    handles = [Line2D([], [], color=GREEN, lw=2, label="Detected (true positive)"),
               Line2D([], [], color=MAGENTA, lw=2, label="Prediction without manual outline"),
               Line2D([], [], color=YELLOW, lw=2, label="Manual outline missed by U-Net")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.598), ncol=3, fontsize=7.5,
               handlelength=1.8, columnspacing=1.5)

    # B: per-WSI detection metrics
    v = pd.read_csv(os.path.join(RES, "seg_validation_per_wsi.csv"))
    axB = fig.add_axes([0.075, 0.255, 0.25, 0.255])
    mets = [("recall", "Recall"), ("precision", "Precision"), ("pixel_dice", "Dice"), ("median_iou", "IoU")]
    rng = np.random.default_rng(0)
    for k, (c, lab) in enumerate(mets):
        for o, prep in ((-0.17, "fresh_frozen"), (0.17, "FFPE")):
            y = v[v.prep == prep][c].values
            axB.scatter(k + o + rng.uniform(-0.06, 0.06, len(y)), y, s=11, color=PREP_COL[prep], alpha=0.85,
                        lw=0.3, ec="white", zorder=3)
            axB.plot([k + o - 0.12, k + o + 0.12], [np.median(y)] * 2, color=INK, lw=1.2, zorder=4)
    axB.set_xticks(range(4)); axB.set_xticklabels([l for _, l in mets])
    axB.set_ylim(0.825, 1.007); axB.set_ylabel("Out-of-fold agreement per WSI")
    axB.grid(axis="y", lw=0.3, color="#dddddd"); axB.set_axisbelow(True)
    axB.legend(handles=[Line2D([], [], marker="o", ls="", color=PREP_COL[p], label=PREP_LAB[p], ms=4)
                        for p in PREP_COL], loc="lower left", fontsize=7, handletextpad=0.2)
    letter(fig, axB, "B", dx=-0.045)

    # C: manual vs predicted area
    mt = pd.read_csv(os.path.join(RES, "seg_validation_matches.csv"))
    axC = fig.add_axes([0.40, 0.255, 0.25, 0.255])
    for prep in ("fresh_frozen", "FFPE"):
        s = mt[mt.prep == prep]
        axC.scatter(s.manual_area_um2 / 1e3, s.pred_area_um2 / 1e3, s=2.5, color=PREP_COL[prep], alpha=0.35, lw=0,
                    rasterized=True, label=f"{PREP_LAB[prep]} (n = {len(s):,})")
    lim = [4, 75]
    axC.plot(lim, lim, color=INK, lw=0.8, ls="--")
    axC.set_xlim(lim); axC.set_ylim(lim)
    axC.set_xlabel("Manual area (×10³ µm²)"); axC.set_ylabel("U-Net area (×10³ µm²)")
    r = np.corrcoef(mt.manual_area_um2, mt.pred_area_um2)[0, 1]
    axC.text(0.04, 0.96, f"r = {r:.3f}\n{len(mt):,} matched glomeruli", transform=axC.transAxes, va="top", fontsize=7)
    axC.legend(loc="lower right", fontsize=6.8, markerscale=4, handletextpad=0.1)
    letter(fig, axC, "C", dx=-0.045)

    # D: Bland–Altman (percentage difference)
    axD = fig.add_axes([0.73, 0.255, 0.25, 0.255])
    for prep in ("fresh_frozen", "FFPE"):
        s = mt[mt.prep == prep]
        mean = (s.manual_area_um2 + s.pred_area_um2) / 2
        diff = 100 * (s.pred_area_um2 - s.manual_area_um2) / mean
        axD.scatter(mean / 1e3, diff, s=2.5, color=PREP_COL[prep], alpha=0.35, lw=0, rasterized=True)
        med = np.median(diff)
        axD.axhline(med, color=PREP_COL[prep], lw=1.1)
        axD.text(76, med + (1.5 if prep == "FFPE" else -1.5), f"{med:+.1f}%", color=PREP_COL[prep], fontsize=7,
                 ha="right", va="bottom" if prep == "FFPE" else "top", fontweight="bold")
    d_all = 100 * (mt.pred_area_um2 - mt.manual_area_um2) / ((mt.pred_area_um2 + mt.manual_area_um2) / 2)
    for q in np.percentile(d_all, [2.5, 97.5]):
        axD.axhline(q, color=INK2, lw=0.6, ls=":")
    axD.axhline(0, color=INK, lw=0.5)
    axD.set_xlim(4, 78); axD.set_ylim(-45, 45)
    axD.set_xlabel("Mean area (×10³ µm²)"); axD.set_ylabel("Difference, U-Net − manual (%)")
    axD.text(0.97, 0.04, "dotted: 95% limits", transform=axD.transAxes, ha="right", fontsize=6.5, color=INK2)
    letter(fig, axD, "D", dx=-0.045)

    # E: discrepancies
    fp = m[(m.kind == "pred") & (m.status == "FP")].sample(frac=1, random_state=3)
    fn = m[(m.kind == "manual") & (m.status == "FN")].sample(frac=1, random_state=3)
    fp["area"] = fp.poly.map(lambda p: 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1))))
    fp = fp[fp.area > 8000]
    pick = [("pred", r) for r in fp.groupby("prep").head(2).itertuples()] + \
           [("manual", r) for r in fn.groupby("prep").head(2).itertuples()]
    for k, (kind, r) in enumerate(pick):
        ax = fig.add_axes([0.075 + k * 0.1165, 0.03, 0.11, 0.1026])
        img_ax(ax, tile(r.img_id, r.poly, MAGENTA if kind == "pred" else YELLOW))
        tag(ax, PREP_LAB[r.prep], color="white", bg=PREP_COL[r.prep], loc="ul", fontsize=6.5)
        if k == 0:
            letter(fig, ax, "E", dx=-0.035)
            scalebar(ax, 100, 1.0, fontsize=6)
    fig.text(0.075 + 1.5 * 0.1165 + 0.055, 0.14, "Predictions without a manual outline", ha="center", fontsize=8,
             fontweight="bold", color=MAGENTA)
    fig.text(0.075 + 5.5 * 0.1165 + 0.055, 0.14, "Manual outlines missed by the U-Net", ha="center", fontsize=8,
             fontweight="bold", color="#b07800")
    save(fig, "Figure2")


if __name__ == "__main__":
    main()
