"""Supplementary Figure S3. Foundation-model controls: stain normalisation, compression, and embedding alignment."""
import os

import numpy as np
import pandas as pd

from style import (FFPE, GREEN, INK, INK2, PREP_COL, PREP_LAB, RES, W_FULL, img_ax, letter, plt, save)

from embed import hubmap_crops, jpeg, lab_stats, reinhard  # noqa: E402

ENC = [("imagenet_vitb", "ImageNet ViT"), ("dinov2", "DINOv2"), ("lunit", "Lunit-DINO"), ("phikon", "Phikon"),
       ("phikon2", "Phikon-v2"), ("midnight", "Midnight")]
COND = [("raw", "Original", "#3d5a80"), ("reinhard", "Reinhard normalised", "#98c1d9"), ("jpeg", "JPEG q = 75", "#eda100")]


def examples(fig, x0, y0, w, h):
    g = pd.read_csv(os.path.join(RES, "pred_glomerulus_morphometry.csv"))
    g = g[g.region.str.contains("Cortex") & (g.donor == 66999)]
    ids = g.img_id.unique()
    stats = {i: lab_stats(i) for i in ids}
    tgt = (np.mean([s[0] for s in stats.values()], 0), np.mean([s[1] for s in stats.values()], 0))
    first = None
    for i, prep in enumerate(("fresh_frozen", "FFPE")):
        s = g[g.prep == prep]
        r = s.iloc[(s.area_um2 - s.area_um2.median()).abs().argsort().iloc[:1]]
        c = next(hubmap_crops(r, "fixed"))
        ims = [c, reinhard(c, stats[r.img_id.iloc[0]], tgt), jpeg(c, 75)]
        for j, im in enumerate(ims):
            ax = fig.add_axes([x0 + j * w / 3, y0 + h / 2 * (1 - i), w / 3 * 0.95, h / 2 * 0.9])
            img_ax(ax, im)
            if i == 0:
                ax.set_title(COND[j][1], fontsize=7.5, pad=3)
            if j == 0:
                ax.set_ylabel(PREP_LAB[prep], fontsize=7.5, color=PREP_COL[prep], fontweight="bold")
                first = first or ax
    return first


def grouped(ax, col, ylabel, scale=1.0, chance=None):
    data = {c: pd.read_csv(os.path.join(RES, f"fm_probe_{c}.csv")).set_index("model") for c, _, _ in COND}
    wbar = 0.26
    for j, (c, lab, color) in enumerate(COND):
        vals = [data[c].loc[m, col] * scale if m in data[c].index else np.nan for m, _ in ENC]
        ax.bar(np.arange(len(ENC)) + (j - 1) * wbar, vals, width=wbar, color=color, label=lab, ec="white", lw=0.3)
    if chance is not None:
        ax.axhline(chance, color=INK, lw=0.7, ls="--")
        ax.text(len(ENC) - 0.5, chance, "chance", fontsize=6.5, ha="right", va="bottom")
    ax.set_xticks(range(len(ENC))); ax.set_xticklabels([l for _, l in ENC], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)


def mitigation(ax, col, ylabel):
    m = pd.read_csv(os.path.join(RES, "fm_mitigation.csv"))
    al = [("none", "None", FFPE), ("prep-mean-centring", "Preparation-mean centring", GREEN), ("CORAL", "CORAL", "#8a5ac2")]
    wbar = 0.26
    for j, (a, lab, color) in enumerate(al):
        vals = [m[(m.model == e) & (m.alignment == a)][col].iloc[0] for e, _ in ENC]
        ax.bar(np.arange(len(ENC)) + (j - 1) * wbar, vals, width=wbar, color=color, label=lab, ec="white", lw=0.3)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(ENC))); ax.set_xticklabels([l for _, l in ENC], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)


def main():
    fig = plt.figure(figsize=(W_FULL, 6.4))
    ax0 = examples(fig, 0.06, 0.56, 0.36, 0.38)
    letter(fig, ax0, "A", dx=-0.03)

    axB = fig.add_axes([0.53, 0.62, 0.20, 0.29])
    grouped(axB, "ss_prep", "Preparation share of variance (%)", 100)
    h, l = axB.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.755, 1.0), ncol=3, fontsize=6.8, handlelength=1.0)
    axB.text(0.02, 0.97, "JPEG control run for\nLunit-DINO and Phikon", transform=axB.transAxes, fontsize=6.3, va="top",
             color=INK2)
    letter(fig, axB, "B", dx=-0.05)
    axC = fig.add_axes([0.80, 0.62, 0.18, 0.29])
    grouped(axC, "donor_xprep_knn_acc", "Cross-preparation donor kNN accuracy", chance=1 / 7)
    letter(fig, axC, "C", dx=-0.05)

    axD = fig.add_axes([0.08, 0.12, 0.25, 0.29])
    mitigation(axD, "r2_log_area", "Size decoded after alignment, R²")
    axD.set_ylim(0, 1)
    letter(fig, axD, "D", dx=-0.06)
    axE = fig.add_axes([0.41, 0.12, 0.25, 0.29])
    mitigation(axE, "size_bias_FFPE_to_fresh_frozen_pct", "Size bias, FFPE-trained → frozen (%)")
    letter(fig, axE, "E", dx=-0.06)
    axF = fig.add_axes([0.74, 0.12, 0.24, 0.29])
    mitigation(axF, "size_bias_fresh_frozen_to_FFPE_pct", "Size bias, frozen-trained → FFPE (%)")
    axE.set_ylim(-35, 12); axF.set_ylim(-35, 12)
    letter(fig, axF, "F", dx=-0.05)
    h, l = axE.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.53, 0.50), ncol=3, fontsize=6.8, handlelength=1.0)
    save(fig, "FigureS3")


if __name__ == "__main__":
    main()
