"""Download the open encoder weights from the Hugging Face Hub (with retries)."""
import os
import sys
import time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
from huggingface_hub import snapshot_download  # noqa: E402

MODELS = sys.argv[1:] or ["owkin/phikon", "owkin/phikon-v2", "1aurent/vit_small_patch8_224.lunit_dino",
                          "kaiko-ai/midnight", "facebook/dinov2-base"]

for m in MODELS:
    for attempt in range(20):
        try:
            print("ok", m, snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt"], max_workers=4))
            break
        except Exception as e:
            print("retry", m, attempt, repr(e)[:200])
            time.sleep(5)
    else:
        print("failed", m)
