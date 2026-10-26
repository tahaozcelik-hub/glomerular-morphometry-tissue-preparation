"""Cortex segmentation at 8 µm/pixel, trained on the cortex outlines of the 20 challenge WSIs.

usage:
  python cortex.py prep              8 µm/pixel images and manual cortex masks
  python cortex.py train --fold K    K = 0..3 donor-grouped, K = -1 all WSIs (final model)
  python cortex.py eval              out-of-fold Dice
  python cortex.py predict           cortex probability maps for every cached WSI
"""
import argparse
import json
import math
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from common import CACHE, KAGGLE_DIR, MODELS, index, make_model, normalize  # noqa: E402
from config import C8, RES  # noqa: E402
from train import dice_loss  # noqa: E402

STEP = 8
PATCH = 512


def all_ids():
    ex = os.path.join(CACHE, "extra_index.csv")
    return index(), pd.read_csv(ex) if os.path.exists(ex) else pd.DataFrame(columns=["id"])


def kaggle_folds(idx):
    f = idx.set_index("id").fold.copy()
    f[f == -1] = 3   # donor 68138 has no manual glomerulus outlines; its cortex joins fold 3
    return f


def prep():
    os.makedirs(C8, exist_ok=True)
    idx, ex = all_ids()
    for img_id in list(idx.id) + list(ex.id):
        out = os.path.join(C8, f"{img_id}_img.npy")
        if not os.path.exists(out):
            img = np.load(os.path.join(CACHE, f"{img_id}_img.npy"), mmap_mode="r")
            H, W = img.shape[0] // STEP, img.shape[1] // STEP
            small = np.zeros((H, W, 3), np.uint8)
            for y in range(0, H, 256):
                y1 = min(y + 256, H)
                small[y:y1] = cv2.resize(np.ascontiguousarray(img[y * STEP:y1 * STEP, :W * STEP]), (W, y1 - y),
                                         interpolation=cv2.INTER_AREA)
            np.save(out, small)
        if img_id in set(idx.id):
            r = idx.set_index("id").loc[img_id]
            mask = np.zeros(np.load(out, mmap_mode="r").shape[:2], np.uint8)
            for f in json.load(open(os.path.join(KAGGLE_DIR, r.split, f"{img_id}-anatomical-structure.json"))):
                if f["properties"]["classification"]["name"] != "Cortex":
                    continue
                g = f["geometry"]
                rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
                for ring in rings:
                    p = np.asarray(ring, float) * r.px_um / STEP
                    cv2.fillPoly(mask, [np.round(p * 8).astype(np.int32)], 1, shift=3)
            np.save(os.path.join(C8, f"{img_id}_cortex_gt.npy"), mask)


class CortexSet(Dataset):
    """Random 512-pixel crops with strong colour augmentation, so that other scanners and stains are covered."""

    def __init__(self, ids, n):
        self.ids, self.n = list(ids), n
        self.imgs = {i: np.load(os.path.join(C8, f"{i}_img.npy")) for i in self.ids}
        self.masks = {i: np.load(os.path.join(C8, f"{i}_cortex_gt.npy")) for i in self.ids}
        self.aug = A.Compose([
            A.PadIfNeeded(PATCH, PATCH, border_mode=cv2.BORDER_CONSTANT, fill=255, fill_mask=0),
            A.RandomCrop(PATCH, PATCH), A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
            A.HueSaturationValue(40, 40, 25, p=0.9),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.8),
            A.ToGray(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        i = self.ids[np.random.randint(len(self.ids))]
        o = self.aug(image=self.imgs[i], mask=self.masks[i])
        return torch.from_numpy(np.ascontiguousarray(o["image"])), torch.from_numpy(np.ascontiguousarray(o["mask"])).float()


def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)


def train(fold, steps=3000, bs=12):
    idx, _ = all_ids()
    folds = kaggle_folds(idx)
    tr = [i for i in idx.id if fold == -1 or folds[i] != fold]
    dev = torch.device("cuda")
    model = make_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    scaler = torch.amp.GradScaler()
    dl = DataLoader(CortexSet(tr, steps * bs), batch_size=bs, num_workers=2, drop_last=True, worker_init_fn=seed_worker)
    run = 0.0
    for s, (x, y) in enumerate(dl):
        x, y = normalize(x, dev), y.to(dev).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.float16):
            lo = model(x)
            loss = F.binary_cross_entropy_with_logits(lo, y) + dice_loss(lo.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run += loss.item()
        if (s + 1) % 500 == 0:
            print(f"fold {fold} step {s + 1} loss {run / 500:.4f}", flush=True)
            run = 0.0
    torch.save(model.state_dict(), os.path.join(MODELS, f"cortex_fold{fold}.pt"))


@torch.no_grad()
def predict_one(model, img, dev, tile=1024, margin=128):
    H, W = img.shape[:2]
    out = np.zeros((H, W), np.uint8)
    step = tile - 2 * margin
    for y in range(-margin, H - margin, step):
        for x in range(-margin, W - margin, step):
            t = np.full((tile, tile, 3), 255, np.uint8)
            ys, xs, ye, xe = max(y, 0), max(x, 0), min(y + tile, H), min(x + tile, W)
            t[ys - y:ye - y, xs - x:xe - x] = img[ys:ye, xs:xe]
            xb = normalize(torch.from_numpy(t[None]), dev)
            p = 0
            for flip in ((), (3,), (2,)):
                xi = torch.flip(xb, flip) if flip else xb
                with torch.autocast("cuda", dtype=torch.float16):
                    o = torch.sigmoid(model(xi).float())
                p = p + (torch.flip(o, flip) if flip else o)
            p = (p / 3)[0, 0].cpu().numpy()
            ys2, xs2 = max(y + margin, 0), max(x + margin, 0)
            ye2, xe2 = min(y + tile - margin, H), min(x + tile - margin, W)
            out[ys2:ye2, xs2:xe2] = np.round(p[ys2 - y:ye2 - y, xs2 - x:xe2 - x] * 255)
    return out


def load(fold, dev):
    m = make_model(pretrained=False)
    m.load_state_dict(torch.load(os.path.join(MODELS, f"cortex_fold{fold}.pt"), map_location="cpu", weights_only=True))
    return m.to(dev).eval()


def evaluate():
    idx, _ = all_ids()
    folds = kaggle_folds(idx)
    dev = torch.device("cuda")
    rows = []
    for f in range(4):
        m = load(f, dev)
        for i in [i for i in idx.id if folds[i] == f]:
            gt = np.load(os.path.join(C8, f"{i}_cortex_gt.npy")) > 0
            tissue = cv2.resize(np.load(os.path.join(CACHE, f"{i}_tissue.npy")).astype(np.uint8), gt.shape[::-1],
                                interpolation=cv2.INTER_NEAREST) > 0
            pr = (predict_one(m, np.load(os.path.join(C8, f"{i}_img.npy")), dev) >= 128) & tissue
            g = gt & tissue
            rows.append(dict(img_id=i, fold=f, dice=2 * (pr & g).sum() / max(pr.sum() + g.sum(), 1),
                             cortex_frac_gt=g.sum() / max(tissue.sum(), 1), cortex_frac_pred=pr.sum() / max(tissue.sum(), 1)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "cortex_oof.csv"), index=False, float_format="%.4f")


def predict_all():
    idx, ex = all_ids()
    dev = torch.device("cuda")
    m = load(-1, dev)
    for i in list(idx.id) + list(ex.id):
        np.save(os.path.join(C8, f"{i}_cortex_prob.npy"), predict_one(m, np.load(os.path.join(C8, f"{i}_img.npy")), dev))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "train", "eval", "predict"])
    ap.add_argument("--fold", type=int, default=-1)
    a = ap.parse_args()
    {"prep": prep, "eval": evaluate, "predict": predict_all}.get(a.mode, lambda: train(a.fold))()
