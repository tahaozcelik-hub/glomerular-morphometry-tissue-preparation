"""Figure 3. Paired effect of tissue preparation on glomerular morphometry (automated instrument)."""
import os

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from style import (FFPE, FROZEN, GREY, INK, INK2, LIGHT, PREP_COL, RED, RES, W_FULL, letter, plt, save)

PAIRED = [66999, 67026, 67112, 67347, 68304, 68555, 67548]
OUTLIER = 67548


def donor_level(c, col):
    s = c[c.donor.isin(PAIRED)].copy()
    s["v"] = np.log(s[col])
    return np.exp(s.groupby(["donor", "prep"]).v.mean().unstack())


def slope(ax, c, col, scale, ylabel, fmt):
    d = donor_level(c, col) / scale
    for donor, r in d.iterrows():
        out = donor == OUTLIER
        ax.plot([0, 1], [r.fresh_frozen, r.FFPE], color=RED if out else INK2, lw=1.0 if out else 0.8,
                ls="--" if out else "-", alpha=0.95, zorder=2)
        for x, p in ((0, "fresh_frozen"), (1, "FFPE")):
            ax.scatter(x, r[p], s=18, color=PREP_COL[p], ec="white", lw=0.4, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Frozen", "FFPE"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel(ylabel)
    six = d.drop(OUTLIER)
    eff = 100 * (np.exp(np.mean(np.log(six.FFPE / six.fresh_frozen))) - 1)
    ax.set_title(fmt.format(eff).replace("-", "−"), fontsize=8, pad=3)


def forest(ax):
    pdc = pd.read_csv(os.path.join(RES, "paired_donor_comparison_pred7.csv"))
    pcm = pd.read_csv(os.path.join(RES, "paired_clinical_metrics_pred7.csv"))
    rows = []
    for f, lab in (("area_um2", "Area"), ("eq_diam_um", "Equivalent diameter"), ("feret_max_um", "Maximum Feret"),
                   ("aspect_ratio", "Aspect ratio")):
        for s, k in (("without 67548", 6), ("7 donors", 7)):
            r = pdc[(pdc.feature == f) & (pdc.set == s)].iloc[0]
            rows.append((lab, k, r["pct_change"], r.pct_ci_lo, r.pct_ci_hi, r.perm_p))
    for f, lab in (("mgv_um3", "Mean glomerular volume"), ("glom_density_mm2", "Glomerular density")):
        for s, k in (("6 (no 67548)", 6), ("7", 7)):
            r = pcm[(pcm.metric == f) & (pcm.donors.astype(str) == s)].iloc[0]
            rows.append((lab, k, r.effect, r.ci_lo, r.ci_hi, r.perm_p))
    labs = list(dict.fromkeys(r[0] for r in rows))
    for lab, k, e, lo, hi, p in rows:
        y = labs.index(lab) + (-0.16 if k == 6 else 0.16)
        col = INK if k == 6 else GREY
        ax.plot([lo, hi], [y, y], color=col, lw=1.2, solid_capstyle="butt")
        ax.scatter(e, y, s=22 if k == 6 else 16, marker="o" if k == 6 else "D", color=col if k == 6 else "white",
                   ec=col, lw=0.9, zorder=3)
        if k == 6:
            ax.text(1.02, y, f"{e:+.1f}%".replace("-", "−") + f"   p = {p:.3f}", va="center", ha="left", fontsize=6.8,
                    color=INK, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    ax.invert_yaxis()
    ax.set_xlim(-30, 72)
    ax.text(1.02, -0.85, "6 donors", fontsize=6.8, fontweight="bold", transform=ax.get_yaxis_transform())
    ax.set_xlabel("FFPE vs frozen, within donor (%, 95% CI)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([], [], marker="o", color=INK, ls="-", ms=4, label="6 donors (67548 excluded)"),
                       Line2D([], [], marker="D", color=GREY, mfc="white", ls="-", ms=3.5, label="All 7 donors")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.8, handlelength=1.6)


def violins(ax, g):
    order = [66999, 67026, 67112, 67347, 68304, 68555, 67548]
    for k, d in enumerate(order):
        for side, prep in ((-1, "fresh_frozen"), (1, "FFPE")):
            a = g[(g.donor == d) & (g.prep == prep)].area_um2.values / 1e3
            xs = np.linspace(5, 60, 300)
            dens = gaussian_kde(a, bw_method=0.25)(xs)
            dens = dens / dens.max() * 0.42
            keep = dens > 0.012
            xs, dens = xs[keep], dens[keep]
            ax.fill_betweenx(xs, k, k + side * dens, color=PREP_COL[prep], alpha=0.55, lw=0)
            ax.plot(k + side * dens, xs, color=PREP_COL[prep], lw=0.6)
            q1, med, q3 = np.percentile(a, [25, 50, 75])
            ax.plot([k, k + side * 0.30], [med, med], color=INK, lw=1.3)
            ax.plot([k + side * 0.03] * 2, [q1, q3], color=INK, lw=0.6)
        if d == OUTLIER:
            ax.axvspan(k - 0.5, k + 0.5, color=LIGHT, zorder=0, lw=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{d}" + ("\n(different blocks,\nexcluded)" if d == OUTLIER else "") for d in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(5, 58)
    ax.set_ylabel("Glomerular area (×10³ µm²)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=FROZEN, alpha=0.6, label="Frozen (left half)"),
                       plt.Rectangle((0, 0), 1, 1, color=FFPE, alpha=0.6, label="FFPE (right half)"),
                       Line2D([], [], color=INK, lw=1.3, label="Median")], ncol=3, loc="upper left", fontsize=7)


def quantile_shift(ax, g):
    qs = np.arange(0.05, 0.96, 0.05)
    allv = []
    for d in [x for x in PAIRED if x != OUTLIER]:
        a = g[(g.donor == d) & (g.prep == "fresh_frozen")].area_um2
        b = g[(g.donor == d) & (g.prep == "FFPE")].area_um2
        v = 100 * (np.quantile(b, qs) / np.quantile(a, qs) - 1)
        allv.append(v)
        ax.plot(qs * 100, v, color=GREY, lw=0.7, alpha=0.8)
    m = np.mean(allv, 0)
    ax.plot(qs * 100, m, color=FFPE, lw=2.0, label="Mean of 6 donors")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("Percentile of the frozen-section area distribution")
    ax.set_ylabel("FFPE vs frozen at\nthe same percentile (%)")
    ax.set_xlim(5, 95); ax.set_ylim(-25, 5)
    ax.legend(handles=[Line2D([], [], color=GREY, lw=0.8, label="Individual donors"),
                       Line2D([], [], color=FFPE, lw=2, label="Mean of 6 donors")], loc="lower right", fontsize=7)


def variance(ax):
    mm = pd.read_csv(os.path.join(RES, "mixed_model_results_pred7.csv"))
    feats = [("area_um2", "Area"), ("eq_diam_um", "Equiv. diameter"), ("feret_max_um", "Maximum Feret"),
             ("aspect_ratio", "Aspect ratio"), ("circularity", "Circularity"), ("solidity", "Solidity")]
    cols = [("pct_donor", "Donor", "#3d5a80"), ("pct_section", "Section (block, preparation)", "#98c1d9"),
            ("pct_residual", "Glomerulus within section", "#e0e0dc")]
    for k, (f, lab) in enumerate(feats):
        r = mm[(mm.model == "A_paired_7donors") & (mm.feature == f)].iloc[0]
        left = 0
        for c, _, col in cols:
            ax.barh(k, r[c], left=left, color=col, height=0.62, ec="white", lw=0.4)
            if r[c] > 8:
                ax.text(left + r[c] / 2, k, f"{r[c]:.0f}%", ha="center", va="center", fontsize=6.8, color=INK)
            left += r[c]
        ax.text(101, k, f"{r.pct_donor + r.pct_section:.1f}%", va="center", fontsize=6.8, color="#3d5a80")
    ax.set_yticks(range(len(feats))); ax.set_yticklabels([l for _, l in feats])
    ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel("Share of variance, 7 paired donors (%)")
    ax.text(101, -0.75, "donor +\nsection", fontsize=6.5, color="#3d5a80", va="bottom")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for _, l, c in cols], ncol=3,
              loc="upper center", bbox_to_anchor=(0.45, -0.25), fontsize=6.8, handlelength=1.0, columnspacing=0.9)


def main():
    c = pd.read_csv(os.path.join(RES, "clinical_metrics_pred.csv"))
    g = pd.read_csv(os.path.join(RES, "pred_glomerulus_morphometry.csv"))
    g = g[g.region.str.contains("Cortex") & g.donor.isin(PAIRED)]
    fig = plt.figure(figsize=(W_FULL, 7.4))

    specs = [("median_area_um2", 1e3, "Median area (×10³ µm²)", "Area  {:+.1f}%"),
             ("mgv_um3", 1e6, "MGV (×10⁶ µm³)", "MGV  {:+.1f}%"),
             ("glom_density_mm2", 1, "Glomeruli per mm² cortex", "Density  {:+.1f}%")]
    for k, (col, sc, yl, fmt) in enumerate(specs):
        ax = fig.add_axes([0.07 + k * 0.165, 0.70, 0.115, 0.26])
        slope(ax, c, col, sc, yl, fmt)
        if k == 0:
            letter(fig, ax, "A", dx=-0.04)
    fig.legend(handles=[Line2D([], [], color=INK2, lw=0.8, label="Donor (n = 6)"),
                        Line2D([], [], color=RED, lw=1, ls="--", label="Donor 67548")],
               loc="upper left", bbox_to_anchor=(0.10, 0.655), ncol=2, fontsize=7)

    axB = fig.add_axes([0.69, 0.72, 0.20, 0.24])
    forest(axB)
    letter(fig, axB, "B", dx=-0.15)

    axC = fig.add_axes([0.07, 0.37, 0.91, 0.24])
    violins(axC, g)
    letter(fig, axC, "C", dx=-0.04)

    axD = fig.add_axes([0.08, 0.07, 0.36, 0.20])
    quantile_shift(axD, g)
    letter(fig, axD, "D", dx=-0.05)

    axE = fig.add_axes([0.62, 0.10, 0.32, 0.17])
    variance(axE)
    letter(fig, axE, "E", dx=-0.10)
    save(fig, "Figure3")


if __name__ == "__main__":
    main()
