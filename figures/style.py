"""Shared figure style (Times New Roman, 180 mm width, bold panel letters) and helpers."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "seg", "fm"):
    sys.path.insert(0, os.path.join(REPO, sub))
from config import AIDPATH_DIR, C8, CACHE, EMB, FIG_OUT, KAGGLE_DIR, PRED, RES, WORK_DIR  # noqa: E402,F401

os.makedirs(FIG_OUT, exist_ok=True)

W_FULL = 180 / 25.4
FROZEN, FFPE = "#2a78d6", "#eb6834"
GREEN, YELLOW, RED, MAGENTA = "#1baf7a", "#eda100", "#e34948", "#c2185b"
GREY, LIGHT, INK, INK2 = "#9a9992", "#e6e5e0", "#111111", "#4a4946"
PREP_COL = {"fresh_frozen": FROZEN, "FFPE": FFPE}
PREP_LAB = {"fresh_frozen": "Frozen", "FFPE": "FFPE"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titleweight": "bold", "savefig.dpi": 400, "figure.dpi": 150,
})


def letter(fig, ax, s, dx=-0.01, dy=0.005, size=13):
    b = ax.get_position()
    fig.text(b.x0 + dx, b.y1 + dy, s, fontsize=size, fontweight="bold", va="bottom", ha="right", color=INK)


def img_ax(ax, im, title=None, color=None):
    ax.imshow(im, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8 if color else 0.4); sp.set_color(color or INK2)
    if title:
        ax.set_title(title, fontsize=8, pad=2, color=color or INK)


def scalebar(ax, um, px_per_um, label=None, color="white", loc="lr", lw=2.5, fontsize=7):
    h, w = ax.images[0].get_array().shape[:2]
    L = um * px_per_um
    x0 = w * 0.06 if loc.endswith("l") else w * 0.94 - L
    y0 = h * 0.92
    ax.plot([x0, x0 + L], [y0, y0], color=color, lw=lw, solid_capstyle="butt")
    ax.text(x0 + L / 2, y0 - h * 0.03, label or f"{um:g} µm", color=color, ha="center", va="bottom", fontsize=fontsize,
            fontweight="bold")


def tag(ax, text, color=INK, bg="white", loc="ul", fontsize=7):
    x, y, ha, va = {"ul": (0.03, 0.97, "left", "top"), "ur": (0.97, 0.97, "right", "top"),
                    "ll": (0.03, 0.03, "left", "bottom"), "lr": (0.97, 0.03, "right", "bottom")}[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc=bg, ec="none", alpha=0.85))


def box(ax, x, y, w, h, text, fc="#f4f3ef", ec=INK2, fontsize=7.5, lw=0.7, color=INK, dy=0.035):
    """Rounded box; the first line of text is bold."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015", fc=fc, ec=ec, lw=lw,
                                transform=ax.transAxes))
    lines = text.split("\n")
    ax.text(x + w / 2, y + h - dy, lines[0], transform=ax.transAxes, ha="center", va="top", fontsize=fontsize + 0.5,
            fontweight="bold", color=color)
    if len(lines) > 1:
        ax.text(x + w / 2, y + h - 0.11, "\n".join(lines[1:]), transform=ax.transAxes, ha="center", va="top",
                fontsize=fontsize, color=color, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=9))


def polys(path, scale):
    return [np.asarray(f["geometry"]["coordinates"][0], float) * scale for f in json.load(open(path))]


def save(fig, name):
    """PNG (400 dpi), PDF and LZW TIFF."""
    import cv2
    png = os.path.join(FIG_OUT, f"{name}.png")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_OUT, f"{name}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    cv2.imwrite(os.path.join(FIG_OUT, f"{name}.tif"), cv2.imread(png),
                [cv2.IMWRITE_TIFF_COMPRESSION, 5, cv2.IMWRITE_TIFF_XDPI, 400, cv2.IMWRITE_TIFF_YDPI, 400])
