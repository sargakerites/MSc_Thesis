import os
import json
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path(os.getenv("SCRATCH_ROOT", f"/scratch/{os.getenv('USER', 'user')}"))

os.environ["HF_HOME"] = str(SCRATCH_ROOT / ".hf_home")
os.environ["HF_HUB_CACHE"] = str(SCRATCH_ROOT / ".hf_home" / "hub")
os.environ["HF_DATASETS_CACHE"] = str(SCRATCH_ROOT / ".hf_datasets_cache")

Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


DATASET_NAME = "openlanguagedata/flores_plus"
OUTPUT_PATH = PROJECT_ROOT / "Evaluation" / "Data" / "flores_plus_hun_eng_devtest.jsonl"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

ds_hu = load_dataset(DATASET_NAME, "hun_Latn", split="devtest")
ds_en = load_dataset(DATASET_NAME, "eng_Latn", split="devtest")

assert len(ds_hu) == len(ds_en)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for en_row, hu_row in zip(ds_en, ds_hu):
        obj = {
            "eng_Latn": en_row["text"],
            "hun_Latn": hu_row["text"],
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Created: {OUTPUT_PATH}")