"""Supplementary Figure S2. Cortex segmentation: out-of-fold examples, accuracy, and the quality-control rule."""
import os

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from style import (C8, GREEN, INK, PRED, PREP_COL, PREP_LAB, RED, RES, W_FULL, YELLOW, img_ax, letter, plt, polys,
                   save, scalebar)

from common import index  # noqa: E402
from cortex import all_ids, kaggle_folds, load, predict_one  # noqa: E402

FAIL = "HBM933.LDKX.448"


def clean(im):
    im = im.copy()
    im[im.max(-1) < 30] = 255
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    im[(hsv[..., 1] < 30) & (hsv[..., 2] > 170)] = 255
    return im


def crop_tissue(*arrs, ref):
    ys, xs = np.where(ref.min(-1) < 215)
    sl = (slice(max(ys.min() - 8, 0), ys.max() + 8), slice(max(xs.min() - 8, 0), xs.max() + 8))
    return [a[sl] for a in arrs]


def contours(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cs if cv2.contourArea(c) > 150]


def oof_example(img_id, fold, dev):
    img = np.load(os.path.join(C8, f"{img_id}_img.npy"))
    gt = np.load(os.path.join(C8, f"{img_id}_cortex_gt.npy")) > 0
    pr = predict_one(load(fold, dev), img, dev) >= 128
    pr = cv2.morphologyEx(pr.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    im = clean(img)
    ov = im.copy()
    ov[pr] = (0.88 * ov[pr] + 0.12 * np.array([27, 175, 122])).astype(np.uint8)
    cv2.drawContours(ov, contours(gt), -1, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.drawContours(ov, contours(gt), -1, (237, 161, 0), 2, cv2.LINE_AA)
    cv2.drawContours(ov, contours(pr), -1, (27, 175, 122), 2, cv2.LINE_AA)
    return crop_tissue(ov, ref=im)[0]


def qc_example():
    r = index(include_extra=True).set_index("id").loc[FAIL]
    im = clean(np.load(os.path.join(C8, f"{FAIL}_img.npy"))[::2, ::2])
    prob = np.load(os.path.join(C8, f"{FAIL}_cortex_prob.npy"))[::2, ::2]
    cortex = prob >= 128 if prob.dtype == np.uint8 else prob >= 0.5
    ov = im.copy()
    ov[cortex] = (0.85 * ov[cortex] + 0.15 * np.array([27, 175, 122])).astype(np.uint8)
    cv2.drawContours(ov, contours(cortex), -1, (27, 175, 122), 2, cv2.LINE_AA)
    nin = 0
    ps = polys(os.path.join(PRED, f"{FAIL}.json"), r.px_um / 16)
    for p in ps:
        cx, cy = p.mean(0).astype(int)
        inside = cortex[min(cy, cortex.shape[0] - 1), min(cx, cortex.shape[1] - 1)]
        nin += inside
        cv2.circle(ov, (cx, cy), 6, (237, 161, 0) if inside else (227, 73, 72), -1, cv2.LINE_AA)
        cv2.circle(ov, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)
    return crop_tissue(ov, ref=im)[0], nin, len(ps)


def main():
    dev = torch.device("cuda")
    idx, _ = all_ids()
    folds = kaggle_folds(idx)
    oof = pd.read_csv(os.path.join(RES, "cortex_oof.csv"))
    px = index().set_index("id").px_um
    oof["prep"] = np.where(np.isclose(oof.img_id.map(px), 0.65), "FFPE", "fresh_frozen")
    o = oof.sort_values("dice")
    picks = [o.iloc[-1].img_id, o.iloc[len(o) // 2].img_id, o.iloc[0].img_id]

    fig = plt.figure(figsize=(W_FULL, 5.2))
    x = 0.03
    for k, i in enumerate(picks):
        im = oof_example(i, folds[i], dev)
        w = 0.30
        h = min(w * im.shape[0] / im.shape[1] * W_FULL / 5.2, 0.38)
        ax = fig.add_axes([x, 0.58, w, h])
        img_ax(ax, im)
        r = oof.set_index("img_id").loc[i]
        ax.set_title(f"{PREP_LAB[r.prep]} {i} · Dice {r.dice:.2f}", fontsize=7.5, pad=3, color=PREP_COL[r.prep])
        scalebar(ax, 2000, 1 / 8, "2 mm", color=INK, fontsize=6.5)
        if k == 0:
            letter(fig, ax, "A", dx=-0.005)
        x += w + 0.025
    fig.legend(handles=[Line2D([], [], color=YELLOW, lw=2, label="Manual cortex (challenge annotation)"),
                        Line2D([], [], color=GREEN, lw=2, label="U-Net cortex, out of fold")],
               loc="upper center", bbox_to_anchor=(0.5, 0.575), ncol=2, fontsize=7)

    axB = fig.add_axes([0.07, 0.08, 0.17, 0.34])
    rng = np.random.default_rng(0)
    for k, p in enumerate(("fresh_frozen", "FFPE")):
        d = oof[oof.prep == p].dice
        axB.scatter(k + rng.uniform(-0.1, 0.1, len(d)), d, s=16, color=PREP_COL[p], ec="white", lw=0.3, zorder=3)
        axB.plot([k - 0.2, k + 0.2], [d.median()] * 2, color=INK, lw=1.3)
    axB.set_xticks([0, 1]); axB.set_xticklabels(["Frozen", "FFPE"]); axB.set_xlim(-0.5, 1.5)
    axB.set_ylabel("Cortex Dice (out of fold)")
    axB.text(0.97, 0.97, f"median {oof.dice.median():.2f}", transform=axB.transAxes, ha="right", va="top", fontsize=7)
    letter(fig, axB, "B", dx=-0.05)

    e = pd.read_csv(os.path.join(RES, "extra_section_metrics.csv")).sort_values("cortex_qc_frac")
    axC = fig.add_axes([0.33, 0.08, 0.25, 0.34])
    cols = [RED if not q else GREEN for q in e.cortex_qc_pass]
    axC.bar(range(len(e)), 100 * e.cortex_qc_frac, color=cols, width=0.75)
    axC.axhline(90, color=INK, lw=0.8, ls="--")
    axC.text(len(e) - 0.5, 84, "QC threshold 90%", ha="right", fontsize=6.8, bbox=dict(fc="white", ec="none", pad=1))
    axC.set_xticks([]); axC.set_xlabel("HuBMAP extension sections (sorted)")
    axC.set_ylabel("Glomeruli inside predicted cortex (%)")
    axC.set_ylim(0, 105)
    axC.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREEN, label="Pass: density reported"),
                        plt.Rectangle((0, 0), 1, 1, color=RED, label="Fail: density not reported")],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.4, handlelength=1.0, columnspacing=0.8)
    letter(fig, axC, "C", dx=-0.05)

    im, nin, n = qc_example()
    h = 0.44
    w = im.shape[1] / im.shape[0] * h * 5.2 / W_FULL
    axD = fig.add_axes([0.64, 0.05, w, h])
    img_ax(axD, im)
    frac = e.set_index("img_id").loc[FAIL].cortex_qc_frac
    axD.set_title(f"QC failure: {FAIL} ({100 * frac:.0f}% of glomeruli inside)", fontsize=7.5, pad=3)
    scalebar(axD, 2000, 1 / 16, "2 mm", color=INK, fontsize=6.5)
    axD.legend(handles=[Line2D([], [], marker="o", ls="", color=YELLOW, ms=4, label="Inside cortex"),
                        Line2D([], [], marker="o", ls="", color=RED, ms=4, label="Outside cortex")],
               loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.6, handletextpad=0.1)
    letter(fig, axD, "D", dx=-0.005)
    save(fig, "FigureS2")


if __name__ == "__main__":
    main()
