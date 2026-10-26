"""Input and output locations. Override any of them with the environment variable of the same name."""
import os

REPO = os.path.dirname(os.path.abspath(__file__))


def _env(name, default):
    return os.environ.get(name, default)


KAGGLE_DIR = _env("KAGGLE_DIR", os.path.join(REPO, "data", "hubmap-kidney-segmentation"))  # train/, test/, *.csv
EXTRA_DIR = _env("EXTRA_DIR", os.path.join(REPO, "data", "hubmap_extra"))                  # HuBMAP OME-TIFF pyramids
AIDPATH_DIR = _env("AIDPATH_DIR", os.path.join(REPO, "data", "aidpath"))                   # normal/, sclerosed/
WORK_DIR = _env("WORK_DIR", os.path.join(REPO, "work"))

CACHE = os.path.join(WORK_DIR, "cache_1um")   # images resampled to 1 µm/pixel
C8 = os.path.join(CACHE, "c8")                # 8 µm/pixel images and cortex maps
MODELS = os.path.join(WORK_DIR, "models")
PRED = os.path.join(WORK_DIR, "pred")         # variants: pred_FFPE, pred_fresh_frozen, pred_f0 ...
EMB = os.path.join(WORK_DIR, "emb")
RES = _env("RES_DIR", os.path.join(REPO, "results"))
META = os.path.join(REPO, "metadata")
FIG_OUT = _env("FIG_OUT", os.path.join(REPO, "figures", "out"))

PREP_BY_PX = {0.65: "FFPE", 0.5: "fresh_frozen"}   # challenge images: preparation follows the scanner resolution
