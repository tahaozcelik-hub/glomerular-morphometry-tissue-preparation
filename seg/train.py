"""Train the glomerulus U-Net for one donor-grouped fold.

usage: python train.py --fold K [--prep FFPE|fresh_frozen]   (K = 0..3 is held out)
"""
import argparse
import math
import os
import time

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from common import CACHE, MODELS, TISSUE_STEP, index, make_model, normalize  # noqa: E402
from config import RES  # noqa: E402

PATCH = 512


class PatchSet(Dataset):
    """Random 512-µm patches; 60% centred near a manual glomerulus, the rest anywhere in tissue."""

    def __init__(self, ids, n, pos_frac=0.6, augment=True, seed=0):
        self.ids, self.n, self.pos_frac, self.seed = list(ids), n, pos_frac, seed
        morph = pd.read_csv(os.path.join(RES, "glomerulus_morphometry.csv"))
        self.cent = {i: morph.loc[morph.img_id == i, ["centroid_y_um", "centroid_x_um"]].values for i in self.ids}
        self.tissue = {i: np.argwhere(np.load(os.path.join(CACHE, f"{i}_tissue.npy"))) for i in self.ids}
        self.mm = None
        self.aug = A.Compose([
            A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
            A.Affine(scale=(0.9, 1.1), rotate=(-30, 30), fill=255, fill_mask=0, p=0.5),
            A.HueSaturationValue(10, 20, 15, p=0.7),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.7),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.GaussNoise(std_range=(0.02, 0.05))], p=0.3),
        ]) if augment else None

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        if self.mm is None:
            self.mm = {i: (np.load(os.path.join(CACHE, f"{i}_img.npy"), mmap_mode="r"),
                           np.load(os.path.join(CACHE, f"{i}_mask.npy"), mmap_mode="r")) for i in self.ids}
        rng = np.random.default_rng((self.seed, k, torch.initial_seed() % 2**31))
        i = self.ids[rng.integers(len(self.ids))]
        img, mask = self.mm[i]
        if rng.random() < self.pos_frac:
            cy, cx = self.cent[i][rng.integers(len(self.cent[i]))] + rng.uniform(-200, 200, 2)
        else:
            ty, tx = self.tissue[i][rng.integers(len(self.tissue[i]))]
            cy, cx = (ty + rng.random()) * TISSUE_STEP, (tx + rng.random()) * TISSUE_STEP
        H, W = mask.shape
        y0 = int(np.clip(cy - PATCH // 2, 0, H - PATCH))
        x0 = int(np.clip(cx - PATCH // 2, 0, W - PATCH))
        x = np.array(img[y0:y0 + PATCH, x0:x0 + PATCH])
        y = np.array(mask[y0:y0 + PATCH, x0:x0 + PATCH])
        if self.aug is not None:
            o = self.aug(image=x, mask=y)
            x, y = o["image"], o["mask"]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)).float()


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = tot = 0.0
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(normalize(x, device)) > 0
        y = y.to(device).unsqueeze(1) > 0.5
        inter += (p & y).sum().item()
        tot += p.sum().item() + y.sum().item()
    model.train()
    return 2 * inter / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per_epoch", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prep", choices=["FFPE", "fresh_frozen"], help="train on one preparation only")
    a = ap.parse_args()

    idx = index()
    idx["prep"] = np.where(idx.px_um == 0.65, "FFPE", "fresh_frozen")
    tr = idx[(idx.split == "train") & (idx.fold != a.fold) & ((idx.prep == a.prep) if a.prep else True)].id
    va = idx[(idx.split == "train") & (idx.fold == a.fold)].id
    tag = f"fold{a.fold}" + (f"_{a.prep}" if a.prep else "")
    device = torch.device("cuda")
    torch.manual_seed(a.fold + 100)
    dl = DataLoader(PatchSet(tr, a.per_epoch * a.epochs, seed=a.fold), batch_size=a.bs, shuffle=False,
                    num_workers=a.workers, pin_memory=True, persistent_workers=True, drop_last=True)
    vl = DataLoader(PatchSet(va, 400, augment=False, seed=999), batch_size=a.bs, num_workers=4) if len(va) else None

    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps, warm = a.epochs * a.per_epoch // a.bs, 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler()
    os.makedirs(MODELS, exist_ok=True)
    log, run, t0 = [], 0.0, time.time()
    per_ep = a.per_epoch // a.bs
    for step, (x, y) in enumerate(dl):
        x, y = normalize(x, device), y.to(device).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run += loss.item()
        if (step + 1) % per_ep == 0:
            ep = (step + 1) // per_ep
            row = dict(epoch=ep, loss=run / per_ep, lr=sched.get_last_lr()[0], min=(time.time() - t0) / 60)
            if vl is not None and (ep % 5 == 0 or ep == a.epochs):
                row["heldout_patch_dice"] = evaluate(model, vl, device)   # monitoring only, no model selection
            log.append(row)
            print(row, flush=True)
            run = 0.0
    torch.save(model.state_dict(), os.path.join(MODELS, f"{tag}.pt"))
    pd.DataFrame(log).to_csv(os.path.join(MODELS, f"{tag}_log.csv"), index=False)


if __name__ == "__main__":
    main()
