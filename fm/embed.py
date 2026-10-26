"""Frozen encoder embeddings of glomerulus crops.

Crops: fixed = 224 × 224 µm centred on the glomerulus (absolute size kept); bbox = 1.15 × maximum Feret
diameter resized to 224 pixels (as in AIDPATH).

usage:
  python embed.py hubmap|hubmap_norm|hubmap_jpeg <glomerulus_csv> <out_prefix> model [model ...]
  python embed.py aidpath <out_prefix> model [model ...]
"""
import io
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AIDPATH_DIR, CACHE  # noqa: E402

IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def load_model(name):
    """Returns (module, forward(normalised NCHW) -> (N, D), (mean, std))."""
    from transformers import AutoModel
    cls = lambda m: (lambda x: m(pixel_values=x).last_hidden_state[:, 0])  # noqa: E731
    if name in ("phikon", "phikon2", "dinov2"):
        m = AutoModel.from_pretrained({"phikon": "owkin/phikon", "phikon2": "owkin/phikon-v2",
                                       "dinov2": "facebook/dinov2-base"}[name])
        return m, cls(m), IMNET
    if name == "midnight":
        m = AutoModel.from_pretrained("kaiko-ai/midnight")

        def f(x):   # class token concatenated with the mean patch token, as in the model card
            h = m(pixel_values=x).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)
        return m, f, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    import timm
    if name == "lunit":
        m = timm.create_model("hf-hub:1aurent/vit_small_patch8_224.lunit_dino", pretrained=True, num_classes=0)
        return m, m, ((0.70322989, 0.53606487, 0.66096631), (0.21716536, 0.26081574, 0.20723464))
    if name == "imagenet_vitb":
        m = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True, num_classes=0)
        return m, m, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    raise ValueError(name)


def lab_stats(img_id, n=20000, seed=0):
    """LAB mean and standard deviation of tissue pixels of one WSI."""
    img = np.load(os.path.join(CACHE, f"{img_id}_img.npy"), mmap_mode="r")
    pts = np.argwhere(np.load(os.path.join(CACHE, f"{img_id}_tissue.npy")))
    pts = pts[np.random.default_rng(seed).choice(len(pts), min(n, len(pts)), replace=False)] * 16 + 8
    pts = pts[np.argsort(pts[:, 0])]
    px = np.stack([img[y, x] for y, x in pts])[None].astype(np.float32) / 255
    lab = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0]
    return lab.mean(0), lab.std(0)


def reinhard(crop, src, tgt):
    lab = cv2.cvtColor(crop.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    lab = (lab - src[0]) / src[1] * tgt[1] + tgt[0]
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255, 0, 255).astype(np.uint8)


def jpeg(crop, q=75):
    b = io.BytesIO()
    Image.fromarray(crop).save(b, "JPEG", quality=q)
    return np.asarray(Image.open(b).convert("RGB"))


def hubmap_crops(df, kind, stain_norm=False, jpeg_q=None):
    mm = {}
    if stain_norm:
        stats = {i: lab_stats(i) for i in df.img_id.unique()}
        tgt = (np.mean([s[0] for s in stats.values()], 0), np.mean([s[1] for s in stats.values()], 0))
    for _, r in df.iterrows():
        if r.img_id not in mm:
            mm[r.img_id] = np.load(os.path.join(CACHE, f"{r.img_id}_img.npy"), mmap_mode="r")
        img = mm[r.img_id]
        side = 224 if kind == "fixed" else int(max(64, round(1.15 * r.feret_max_um)))
        cy, cx = int(round(r.centroid_y_um)), int(round(r.centroid_x_um))
        y0, x0 = cy - side // 2, cx - side // 2
        out = np.full((side, side, 3), 255, np.uint8)
        ys, xs, ye, xe = max(y0, 0), max(x0, 0), min(y0 + side, img.shape[0]), min(x0 + side, img.shape[1])
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
        if side != 224:
            out = np.asarray(Image.fromarray(out).resize((224, 224), Image.BICUBIC))
        if jpeg_q:
            out = jpeg(out, jpeg_q)
        yield reinhard(out, stats[r.img_id], tgt) if stain_norm else out


def aidpath_slide(name):
    """Slide of an AIDPATH image; the NOR/SCL token marks the cohort, not the glomerulus label."""
    if "Aperio" in name:
        s = name.split("Aperio")[0] + "Aperio"
    elif "PAS" in name:
        s = name.split("PAS")[0] + "PAS"
    else:
        s = "_".join(name.split("_")[:3])
    return re.sub(r"_(NOR|SCL)_", "_", s)


def aidpath_files():
    return pd.DataFrame([dict(file=os.path.join(c, f), label=int(c == "sclerosed"), slide=aidpath_slide(f))
                         for c in ("normal", "sclerosed") for f in sorted(os.listdir(os.path.join(AIDPATH_DIR, c)))])


def aidpath_crops(df):
    for f in df.file:
        yield np.asarray(Image.open(os.path.join(AIDPATH_DIR, f)).convert("RGB").resize((224, 224), Image.BICUBIC))


@torch.no_grad()
def run(model_name, crops, n, out_path, bs=64):
    model, fwd, (mean, std) = load_model(model_name)
    model = model.cuda().eval().half()
    mean = torch.tensor(mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(std, device="cuda").view(1, 3, 1, 1)
    feats, buf = [], []

    def flush():
        x = torch.from_numpy(np.stack(buf)).cuda().permute(0, 3, 1, 2).float() / 255
        feats.append(fwd(((x - mean) / std).half()).float().cpu().numpy().astype(np.float16))
        buf.clear()
    for c in crops:
        buf.append(c)
        if len(buf) == bs:
            flush()
    if buf:
        flush()
    e = np.concatenate(feats)
    assert len(e) == n, (len(e), n)
    np.save(out_path, e)
    del model
    torch.cuda.empty_cache()


def main():
    mode = sys.argv[1]
    if mode == "aidpath":
        df = aidpath_files()
        prefix, models = sys.argv[2], sys.argv[3:]
        df.to_csv(prefix + "_index.csv", index=False)
        for m in models:
            run(m, aidpath_crops(df), len(df), f"{prefix}_{m}_bbox.npy")
        return
    df = pd.read_csv(sys.argv[2])
    prefix, models = sys.argv[3], sys.argv[4:]
    kinds = ("fixed", "bbox") if mode == "hubmap" else ("fixed",)
    for m in models:   # use a separate prefix for the controls, e.g. emb/prednorm and emb/predjpeg
        for kind in kinds:
            path = f"{prefix}_{m}_{kind}.npy"
            if not os.path.exists(path):
                run(m, hubmap_crops(df, kind, stain_norm=mode == "hubmap_norm",
                                    jpeg_q=75 if mode == "hubmap_jpeg" else None), len(df), path)


if __name__ == "__main__":
    main()
