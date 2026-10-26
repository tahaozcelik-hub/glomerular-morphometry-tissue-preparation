"""Figure 6. Frozen-section cohort: automated cortex and glomerulus measurement and clinical associations."""
import os

import cv2
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D

from style import (C8, CACHE, GREEN, INK, INK2, PRED, RES, W_FULL, YELLOW, img_ax, letter, plt, polys, save,
                   scalebar, tag)

from common import index  # noqa: E402

EXAMPLES = [("HBM758.PCQK.344", "Zeiss Axio Scan"), ("HBM574.HBKW.654", "Leica SCN400, IMS block"),
            ("HBM925.FFHK.438", "Leica SCN400, registered")]
PURPLE = "#8a5ac2"
SCAN_COL = {"Zeiss Axio Scan": "#3d5a80", "Leica SCN400 (Kaggle)": "#e34948", "Leica SCN400 (IMS)": "#1baf7a",
            "Leica SCN400 (registered)": "#c2185b"}
SCAN_LAB = {"Zeiss Axio Scan": "Zeiss (HuBMAP)", "Leica SCN400 (Kaggle)": "Leica (Kaggle)",
            "Leica SCN400 (IMS)": "Leica (HuBMAP IMS)", "Leica SCN400 (registered)": "Leica (HuBMAP reg.)"}


def wsi_panel(img_id, step=2):
    r = index(include_extra=True).set_index("id").loc[img_id]
    im = np.load(os.path.join(C8, f"{img_id}_img.npy"))[::step, ::step].copy()
    im[im.max(-1) < 30] = 255
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    im[(hsv[..., 1] < 30) & (hsv[..., 2] > 170)] = 255      # unscanned / glass background -> white
    prob = np.load(os.path.join(C8, f"{img_id}_cortex_prob.npy"))[::step, ::step]
    prob = prob.astype(np.float32) / (255.0 if prob.dtype == np.uint8 else 1.0)
    cortex = cv2.morphologyEx((prob >= 0.5).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    shade = im.copy()
    shade[cortex > 0] = (0.75 * shade[cortex > 0] + 0.25 * np.array([27, 175, 122])).astype(np.uint8)
    cs, _ = cv2.findContours(cortex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(shade, [c for c in cs if cv2.contourArea(c) > 200], -1, (27, 175, 122), 2, cv2.LINE_AA)
    ps = polys(os.path.join(PRED, f"{img_id}.json"), r.px_um / 8 / step)
    for p in ps:
        cx, cy = p.mean(0)
        cv2.circle(shade, (int(cx), int(cy)), 3, (237, 161, 0), -1, cv2.LINE_AA)
    ys, xs = np.where(im.min(-1) < 215)
    pad = 10
    sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
    return shade[sl], 8 * step, len(ps)


def zoom_tile(img_id, side=600):
    r = index(include_extra=True).set_index("id").loc[img_id]
    img = np.load(os.path.join(CACHE, f"{img_id}_img.npy"), mmap_mode="r")
    ps = polys(os.path.join(PRED, f"{img_id}.json"), r.px_um)
    c = np.array([p.mean(0) for p in ps])
    best, score = None, -1
    for cx, cy in c[np.random.default_rng(0).permutation(len(c))[:200]]:
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        if x0 < 0 or y0 < 0 or x0 + side > img.shape[1] or y0 + side > img.shape[0]:
            continue
        n = ((c[:, 0] > x0 + 60) & (c[:, 0] < x0 + side - 60) & (c[:, 1] > y0 + 60) & (c[:, 1] < y0 + side - 60)).sum()
        if n > score:
            best, score = (x0, y0), n
    x0, y0 = best
    t = np.ascontiguousarray(img[y0:y0 + side, x0:x0 + side])
    cv2.polylines(t, [np.round(p - [x0, y0]).astype(np.int32) for p in ps], True, (237, 161, 0), 3, cv2.LINE_AA)
    return t


def assoc(ax, d, x, y, scale, xl, yl, row):
    d = d.dropna(subset=[x, y])
    for s, col in SCAN_COL.items():
        q = d[d.scanner == s]
        ax.scatter(q[x], q[y] / scale, s=20, color=col, ec="white", lw=0.4, zorder=3)
    fit = smf.ols(f"np.log({y}) ~ {x}", d).fit(cov_type="HC3")
    xs = np.linspace(d[x].min(), d[x].max(), 100)
    pr = fit.get_prediction(pd.DataFrame({x: xs})).summary_frame(alpha=0.05)
    ax.plot(xs, np.exp(pr["mean"]) / scale, color=INK, lw=1.2)
    ax.fill_between(xs, np.exp(pr.mean_ci_lower) / scale, np.exp(pr.mean_ci_upper) / scale, color=INK, alpha=0.1, lw=0)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.38 * (hi - lo))
    ax.text(0.97, 0.97, f"{row.pct:+.1f}% per {'decade' if x == 'age' else '5 kg/m²'}\n"
            f"95% CI {row.ci_lo:+.1f} to {row.ci_hi:+.1f}\np = {row.p:.3f}   n = {len(d)}".replace("-", "−"),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8)


def forest(ax):
    m = pd.read_csv(os.path.join(RES, "cohort_models.csv"))
    rows = [("density", "age10", "Density ~ age (per decade)"), ("density", "bmi5", "Density ~ BMI (per 5 kg/m²)"),
            ("mgv", "age10", "MGV ~ age (per decade)"), ("mgv", "bmi5", "MGV ~ BMI (per 5 kg/m²)"),
            ("mgv", "htn", "MGV ~ hypertension")]
    mods = [("unadjusted", "o", INK), ("scanner-adjusted", "s", "#3d5a80"), ("multivariable", "D", PURPLE)]
    for k, (o, t, lab) in enumerate(rows):
        for j, (mod, mk, col) in enumerate(mods):
            r = m[(m.outcome == o) & (m.term == t) & (m.model == mod)]
            if not len(r):
                continue
            r = r.iloc[0]
            y = k + (j - 1) * 0.22
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.0)
            ax.scatter(r.pct, y, marker=mk, s=16, color=col if mod != "multivariable" else "white", ec=col, lw=0.9,
                       zorder=3)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(-25, 12)
    ax.set_xlabel("Change (%, 95% CI)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([], [], marker=mk, color=col, mfc=col if mod != "multivariable" else "white", ls="-",
                              ms=3.5, label=mod.capitalize()) for mod, mk, col in mods],
              loc="lower center", bbox_to_anchor=(0.2, 1.0), ncol=3, fontsize=6.3, handlelength=1.3,
              columnspacing=0.8)




def main():
    fig = plt.figure(figsize=(W_FULL, 7.3))
    ex = pd.read_csv(os.path.join(RES, "extra_section_metrics.csv")).set_index("img_id")
    pans = [wsi_panel(i) for i, _ in EXAMPLES]
    x = 0.03
    tot = sum(p[0].shape[1] / p[0].shape[0] for p in pans)
    W, H = 0.95, 0.27
    hw = H * 7.3 / (W_FULL)          # height in width units
    scale = min(1.0, (W - 0.04) / (tot * hw))
    for k, ((im, um_px, n), (img_id, lab)) in enumerate(zip(pans, EXAMPLES)):
        w = im.shape[1] / im.shape[0] * hw * scale
        ax = fig.add_axes([x, 0.68, w, H])
        img_ax(ax, im)
        r = ex.loc[img_id]
        dens = "density not reported (cortex QC)" if np.isnan(r.glom_density_mm2) else f"{r.glom_density_mm2:.2f} /mm² cortex"
        ax.set_title(f"{lab}\n{n} glomeruli · {dens}", fontsize=7.5, pad=3)
        scalebar(ax, 2000, 1 / um_px, "2 mm", color=INK, fontsize=6.5)
        if k == 0:
            letter(fig, ax, "A", dx=-0.005)
        x += w + 0.02
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#bfe3d3", ec=GREEN, label="Predicted cortex"),
                        Line2D([], [], marker="o", ls="", color=YELLOW, ms=4, label="Detected glomerulus")],
               loc="upper center", bbox_to_anchor=(0.5, 0.668), ncol=2, fontsize=7)

    for k, (img_id, lab) in enumerate(EXAMPLES):
        ax = fig.add_axes([0.03 + k * 0.137, 0.42, 0.13, 0.13 * W_FULL / 7.3])
        img_ax(ax, zoom_tile(img_id))
        if k == 0:
            scalebar(ax, 200, 1.0, "200 µm", fontsize=6)
            letter(fig, ax, "B", dx=-0.005)
        tag(ax, lab.split(",")[0].replace(" Axio Scan", "").replace(" SCN400", ""), loc="ul", fontsize=6.2)

    c = pd.read_csv(os.path.join(RES, "cohort_donors.csv"))
    m = pd.read_csv(os.path.join(RES, "cohort_models.csv"))
    get = lambda o, t: m[(m.outcome == o) & (m.term == t) & (m.model == "unadjusted")].iloc[0]  # noqa: E731
    axC = fig.add_axes([0.52, 0.33, 0.20, 0.24])
    assoc(axC, c, "age", "density", 1, "Age (years)", "Glomeruli per mm² cortex", get("density", "age10"))
    letter(fig, axC, "C", dx=-0.05)
    axD = fig.add_axes([0.79, 0.33, 0.20, 0.24])
    assoc(axD, c, "bmi", "density", 1, "BMI (kg/m²)", "Glomeruli per mm² cortex", get("density", "bmi5"))
    letter(fig, axD, "D", dx=-0.05)
    axE = fig.add_axes([0.08, 0.05, 0.25, 0.20])
    assoc(axE, c, "bmi", "mgv", 1e6, "BMI (kg/m²)", "MGV (×10⁶ µm³)", get("mgv", "bmi5"))
    letter(fig, axE, "E", dx=-0.06)
    axE.legend(handles=[Line2D([], [], marker="o", ls="", color=col, label=SCAN_LAB[s], ms=4)
                        for s, col in SCAN_COL.items()], loc="upper left", bbox_to_anchor=(1.03, 1.0), fontsize=6.8,
               handletextpad=0.1, title="Scanner (frozen sections)", title_fontsize=6.8)

    axF = fig.add_axes([0.72, 0.05, 0.26, 0.17])
    forest(axF)
    letter(fig, axF, "F", dx=-0.19)
    save(fig, "Figure6")


if __name__ == "__main__":
    main()
