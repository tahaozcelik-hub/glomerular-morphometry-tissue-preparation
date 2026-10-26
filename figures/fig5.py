"""Figure 5. Preparation as a domain shift: segmentation transfer between preparations and foundation-model embeddings."""
import os

import cv2
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from style import (CACHE, EMB, FFPE, GREEN, INK, INK2, KAGGLE_DIR, LIGHT, PRED, PREP_COL, PREP_LAB, RES, WORK_DIR,
                   W_FULL, YELLOW, img_ax, letter, plt, save, scalebar, tag)

from common import index  # noqa: E402
from validate import load_polys, match  # noqa: E402

ROWS = [("afa5e8098", "fresh_frozen"), ("b9a3865fc", "FFPE")]      # donor 67347, both preparations
FW, FH = 1000, 1000
ENC = [("morphometry", "Morphometry"), ("imagenet_vitb", "ImageNet ViT"), ("dinov2", "DINOv2"),
       ("lunit", "Lunit-DINO"), ("phikon", "Phikon"), ("phikon2", "Phikon-v2"), ("midnight", "Midnight")]
PAIRED = [66999, 67026, 67112, 67347, 67548, 68304, 68555]
DONOR_COL = ["#1b4f8a", "#e34948", "#1baf7a", "#eda100", "#8a5ac2", "#00a3c4", "#c2185b"]


def hex_rgb(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def transfer_field(img_id, suffix, x0, y0, man):
    r = index().set_index("id").loc[img_id]
    pred = [p * r.px_um for p in load_polys(os.path.join(PRED + suffix, f"{img_id}.json"))]
    pairs = match(man, pred)
    hit = {i for i, _, _ in pairs}
    im = np.ascontiguousarray(np.load(os.path.join(CACHE, f"{img_id}_img.npy"), mmap_mode="r")[y0:y0 + FH, x0:x0 + FW])
    loc = lambda p: np.round(p - [x0, y0]).astype(np.int32)  # noqa: E731
    for i, p in enumerate(man):
        cv2.polylines(im, [loc(p)], True, hex_rgb(GREEN if i in hit else YELLOW), 5, cv2.LINE_AA)
    inside = [i for i, p in enumerate(man) if (x0 < p[:, 0].mean() < x0 + FW) and (y0 < p[:, 1].mean() < y0 + FH)]
    rec = np.mean([i in hit for i in inside]) if inside else np.nan
    return im, rec


def pick_field(img_id, man, suffix, miss_weight=1.5):
    """Field rich in glomeruli where the cross-preparation model misses some."""
    r = index().set_index("id").loc[img_id]
    pred = [p * r.px_um for p in load_polys(os.path.join(PRED + suffix, f"{img_id}.json"))]
    hit = {i for i, _, _ in match(man, pred)}
    c = np.array([p.mean(0) for p in man])
    miss = np.array([i not in hit for i in range(len(man))])
    shape = np.load(os.path.join(CACHE, f"{img_id}_img.npy"), mmap_mode="r").shape
    best, score = None, -1
    for cx, cy in c:
        x0, y0 = int(cx - FW / 2), int(cy - FH / 2)
        if x0 < 0 or y0 < 0 or x0 + FW > shape[1] or y0 + FH > shape[0]:
            continue
        ins = (c[:, 0] > x0 + 80) & (c[:, 0] < x0 + FW - 80) & (c[:, 1] > y0 + 80) & (c[:, 1] < y0 + FH - 80)
        s = ins.sum() + miss_weight * (ins & miss).sum()
        if s > score:
            best, score = (x0, y0), s
    return best


def umap_xy():
    cache = os.path.join(WORK_DIR, "fig5_umap_phikon.npy")
    g = pd.read_csv(os.path.join(RES, "pred_glomerulus_morphometry.csv"))
    keep = g.region.str.contains("Cortex").values
    if not os.path.exists(cache):
        import umap
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        Z = np.load(os.path.join(EMB, "pred_phikon_fixed.npy")).astype(np.float32)[keep]
        Z = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(Z))
        np.save(cache, umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Z))
    return g[keep].reset_index(drop=True), np.load(cache)


def recall_panel(ax):
    x = pd.read_csv(os.path.join(RES, "xprep_per_wsi.csv"))
    pooled = pd.read_csv(os.path.join(RES, "xprep_summary.csv")).set_index(["model", "target_prep"]).recall
    models = ["FFPE-only", "frozen-only", "mixed"]
    labs = ["FFPE-trained", "Frozen-trained", "Mixed"]
    for t, (prep, off) in enumerate((("fresh_frozen", 0), ("FFPE", 4))):
        s = x[x.target_prep == prep].pivot(index="img_id", columns="model", values="recall")[models]
        for _, r in s.iterrows():
            ax.plot(np.arange(3) + off, r.values, color=PREP_COL[prep], lw=0.6, alpha=0.5)
        for k, mdl in enumerate(models):
            ax.scatter(np.full(len(s), k + off), s[mdl], s=12, color=PREP_COL[prep], ec="white", lw=0.3, zorder=3)
            ax.plot([k + off - 0.25, k + off + 0.25], [pooled[(mdl, prep)]] * 2, color=INK, lw=1.4, zorder=4)
        ax.text(off + 1, 1.045, f"Target: {PREP_LAB[prep]} sections", ha="center", fontsize=7.5,
                color=PREP_COL[prep], fontweight="bold")
    ax.set_xticks(list(range(3)) + list(range(4, 7)))
    ax.set_xticklabels(labs * 2, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0.38, 1.07); ax.set_ylabel("Recall against manual outlines")
    ax.axvline(3, color=LIGHT, lw=0.8)
    ax.text(0.3, 0.77, "0.77", fontsize=7, ha="left", va="center", fontweight="bold", color=INK)


def variance_panel(ax):
    p = pd.read_csv(os.path.join(RES, "fm_probe_raw.csv")).set_index("model")
    cols = [("ss_prep", "Preparation", "#d1495b"), ("ss_donor", "Donor", "#3d5a80"),
            ("ss_section", "Section", "#98c1d9"), ("ss_resid", "Within section", "#e0e0dc")]
    for k, (m, lab) in enumerate(ENC):
        r = p.loc[m]
        left = 0
        for c, _, col in cols:
            v = 100 * r[c]
            ax.barh(k, v, left=left, color=col, height=0.66, ec="white", lw=0.4)
            if v >= 9:
                ax.text(left + v / 2, k, f"{v:.0f}", ha="center", va="center", fontsize=6.6,
                        color="white" if c in ("ss_prep", "ss_donor") else INK)
            left += v
        ax.text(105, k, f"{r.prep_auc_lodo:.2f}", va="center", fontsize=6.8)
        ax.text(128, k, f"{r.donor_xprep_knn_acc:.2f}", va="center", fontsize=6.8)
    ax.text(105, -0.75, "Prep.\nAUC", fontsize=6.5, va="bottom", fontweight="bold")
    ax.text(128, -0.75, "Donor\nkNN", fontsize=6.5, va="bottom", fontweight="bold")
    ax.set_yticks(range(len(ENC))); ax.set_yticklabels([l for _, l in ENC])
    ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.axhline(0.5, color=INK2, lw=0.5, ls=":")
    ax.set_xlabel("Share of embedding variance (%)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for _, l, c in cols], ncol=4,
              loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=6.8, handlelength=1.0, columnspacing=0.8)


def bias_panel(ax):
    m = pd.read_csv(os.path.join(RES, "fm_mitigation.csv"))
    enc = [e for e in ENC if e[0] != "morphometry"]
    for k, (e, lab) in enumerate(enc):
        r0 = m[(m.model == e) & (m.alignment == "none")].iloc[0]
        r1 = m[(m.model == e) & (m.alignment == "prep-mean-centring")].iloc[0]
        a, b = r0.size_bias_FFPE_to_fresh_frozen_pct, r1.size_bias_FFPE_to_fresh_frozen_pct
        ax.annotate("", xy=(b, k), xytext=(a, k), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8,
                                                                 mutation_scale=7, shrinkA=3, shrinkB=3))
        ax.scatter(a, k, s=26, color=FFPE, zorder=3, ec="white", lw=0.4)
        ax.scatter(b, k, s=26, color=GREEN, zorder=3, ec="white", lw=0.4)
    ax.axvspan(-8, 8, color=LIGHT, lw=0, zorder=0)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(range(len(enc))); ax.set_yticklabels([l for _, l in enc])
    ax.invert_yaxis(); ax.set_xlim(-36, 10)
    ax.set_xlabel("Size bias on frozen glomeruli of an\nFFPE-trained size regressor (%)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=FFPE, label="Raw embeddings", ms=4.5),
                       Line2D([], [], marker="o", ls="", color=GREEN, label="Preparation-mean centred", ms=4.5)],
              loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=6.8, handletextpad=0.1)


def main():
    fig = plt.figure(figsize=(W_FULL, 8.6))
    idx = index().set_index("id")
    titles = ["Manual outlines", "FFPE-trained U-Net", "Frozen-trained U-Net"]
    for i, (img_id, prep) in enumerate(ROWS):
        man = [p * idx.loc[img_id].px_um for p in load_polys(os.path.join(KAGGLE_DIR, "train", f"{img_id}.json"))]
        x0, y0 = pick_field(img_id, man, "_FFPE" if prep != "FFPE" else "_fresh_frozen", 1.5 if prep != "FFPE" else 0)
        raw = np.ascontiguousarray(np.load(os.path.join(CACHE, f"{img_id}_img.npy"), mmap_mode="r")[y0:y0 + FH, x0:x0 + FW])
        for p in man:
            cv2.polylines(raw, [np.round(p - [x0, y0]).astype(np.int32)], True, (255, 255, 255), 4, cv2.LINE_AA)
        panels = [(raw, None)] + [transfer_field(img_id, s, x0, y0, man) for s in ("_FFPE", "_fresh_frozen")]
        for j, (im, rec) in enumerate(panels):
            ax = fig.add_axes([0.075 + j * 0.172, 0.775 - i * 0.18, 0.165, 0.17])
            img_ax(ax, im)
            if i == 0:
                ax.set_title(titles[j], fontsize=8, pad=3)
            if rec is not None:
                tag(ax, f"recall {rec:.2f}", loc="ur", fontsize=6.8)
            if j == 0:
                ax.set_ylabel(f"{PREP_LAB[prep]} target\n{img_id}", fontsize=7.5, color=PREP_COL[prep],
                              fontweight="bold", labelpad=3)
                scalebar(ax, 200, 1.0, "200 µm", fontsize=6.5)
                if i == 0:
                    letter(fig, ax, "A", dx=-0.035)
    fig.legend(handles=[Line2D([], [], color=GREEN, lw=2, label="Detected"),
                        Line2D([], [], color=YELLOW, lw=2, label="Missed")],
               loc="upper center", bbox_to_anchor=(0.33, 0.597), ncol=2, fontsize=7)

    axB = fig.add_axes([0.66, 0.66, 0.32, 0.28])
    recall_panel(axB)
    letter(fig, axB, "B", dx=-0.05)

    g, xy = umap_xy()
    axC = fig.add_axes([0.04, 0.32, 0.23, 0.23])
    o = np.random.default_rng(0).permutation(len(g))
    axC.scatter(xy[o, 0], xy[o, 1], s=1.2, c=[PREP_COL[p] for p in g.prep.values[o]], lw=0, alpha=0.7, rasterized=True)
    axC.set_xticks([]); axC.set_yticks([]); axC.set_xlabel("UMAP 1"); axC.set_ylabel("UMAP 2 (Phikon)")
    axC.set_title("Coloured by preparation", fontsize=8, pad=3)
    axC.legend(handles=[Line2D([], [], marker="o", ls="", color=PREP_COL[p], label=PREP_LAB[p], ms=4)
                        for p in PREP_COL], loc="lower right", fontsize=6.8, handletextpad=0.1)
    letter(fig, axC, "C", dx=-0.01)

    axD = fig.add_axes([0.30, 0.32, 0.23, 0.23])
    other = ~g.donor.isin(PAIRED).values
    axD.scatter(xy[other, 0], xy[other, 1], s=1.0, color="#d8d8d4", lw=0, rasterized=True)
    for d, col in zip(PAIRED, DONOR_COL):
        for prep, mk in (("fresh_frozen", "o"), ("FFPE", "^")):
            s = ((g.donor == d) & (g.prep == prep)).values
            axD.scatter(xy[s, 0], xy[s, 1], s=1.6 if mk == "o" else 2.2, color=col, marker=mk, lw=0, alpha=0.8,
                        rasterized=True)
    axD.set_xticks([]); axD.set_yticks([]); axD.set_xlabel("UMAP 1")
    axD.set_title("Coloured by donor (paired donors)", fontsize=8, pad=3)
    axD.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=str(d), ms=3.5) for d, c in zip(PAIRED, DONOR_COL)],
               loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.3, handletextpad=0.1, labelspacing=0.3)
    letter(fig, axD, "D", dx=-0.01)

    axE = fig.add_axes([0.715, 0.32, 0.17, 0.23])
    variance_panel(axE)
    letter(fig, axE, "E", dx=-0.085)

    axF = fig.add_axes([0.12, 0.05, 0.33, 0.15])
    bias_panel(axF)
    letter(fig, axF, "F", dx=-0.085)

    axG = fig.add_axes([0.60, 0.05, 0.36, 0.15])
    r2_panel(axG)
    letter(fig, axG, "G", dx=-0.06)
    save(fig, "Figure5")


def r2_panel(ax):
    """How much size the embeddings carry, versus how much preparation they carry."""
    p = pd.read_csv(os.path.join(RES, "fm_probe_raw.csv")).set_index("model")
    for (m, lab), col in zip(ENC[1:], ["#9a9992", "#b07800", "#1baf7a", "#1b4f8a", "#4f8fd1", "#8a5ac2"]):
        r = p.loc[m]
        ax.scatter(100 * r.ss_prep, r.r2_log_area, s=30, color=col, ec="white", lw=0.4, zorder=3)
        ax.annotate(lab, (100 * r.ss_prep, r.r2_log_area), xytext=(4, -3), textcoords="offset points", fontsize=6.6)
    ax.set_xlabel("Preparation share of embedding variance (%)")
    ax.set_ylabel("Size decoded, R²")
    ax.set_xlim(8, 34); ax.set_ylim(0.68, 0.95)
    ax.grid(lw=0.3, color="#e5e5e5"); ax.set_axisbelow(True)


if __name__ == "__main__":
    main()
