"""Shared WSI index, donor-grouped folds and U-Net factory."""
import os
import sys

import pandas as pd
import segmentation_models_pytorch as smp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CACHE, KAGGLE_DIR, MODELS, PRED  # noqa: E402,F401

TISSUE_STEP = 16
FOLD_DONORS = {   # no donor in more than one fold; balanced on glomerulus count and preparation
    0: [66999, 67177],
    1: [67347, 68304, 65631],
    2: [67548, 68250, 67112],
    3: [67026, 63921, 68555],
}
RESNET34_URL = "https://download.pytorch.org/models/resnet34-b627a593.pth"


def index(include_extra=False):
    """WSI table with donor and fold (-1 for donors without manual outlines); optionally with the HuBMAP sections."""
    idx = pd.read_csv(os.path.join(CACHE, "index.csv"))
    meta = pd.read_csv(os.path.join(KAGGLE_DIR, "HuBMAP-20-dataset_information.csv"))
    meta["id"] = meta.image_file.str[:-5]
    idx = idx.merge(meta[["id", "patient_number"]], on="id").rename(columns={"patient_number": "donor"})
    d2f = {d: f for f, ds in FOLD_DONORS.items() for d in ds}
    idx["fold"] = idx.donor.map(d2f).fillna(-1).astype(int)
    extra = os.path.join(CACHE, "extra_index.csv")
    if include_extra and os.path.exists(extra):
        ex = pd.read_csv(extra).assign(fold=-1)
        idx = pd.concat([idx, ex[["id", "split", "px_um", "H", "W", "H1", "W1", "donor", "fold"]]], ignore_index=True)
    return idx


def make_model(pretrained=True):
    model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    if pretrained:
        sd = torch.hub.load_state_dict_from_url(RESNET34_URL, map_location="cpu")
        model.encoder.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")})
    return model


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def normalize(x_uint8_nhwc, device):
    x = x_uint8_nhwc.to(device, non_blocking=True).permute(0, 3, 1, 2).float()
    return (x - MEAN.to(device)) / STD.to(device)
