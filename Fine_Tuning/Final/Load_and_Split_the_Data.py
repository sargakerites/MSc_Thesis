import os
import json
import glob
from typing import Optional, List, Dict, Tuple
from sklearn.model_selection import train_test_split

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
CHUNK_SIZE = 500

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"


def extract_best_translation(item: Dict) -> Optional[Dict[str, str]]:
    """
    Extract one EN->HU training pair from a raw corpus item.

    Expected raw structure:
    {
        "sentence": "<Hungarian sentence>",
        "translation": {
            "translations": [
                {"translation": "..."},
                {"translation": "..."},
                {"translation": "..."}
            ],
            "quality_order": [0, 2, 1]
        }
    }

    Interpretation:
    quality_order[0] gives the index of the best translation candidate.
    """
    target_hu = item.get("sentence", "")
    if not isinstance(target_hu, str):
        return None
    target_hu = target_hu.strip()
    if not target_hu:
        return None

    translation_block = item.get("translation")
    if not isinstance(translation_block, dict):
        return None

    translations = translation_block.get("translations")
    quality_order = translation_block.get("quality_order")

    if not isinstance(translations, list) or len(translations) == 0:
        return None
    if not isinstance(quality_order, list) or len(quality_order) == 0:
        return None

    best_idx = quality_order[0]
    if not isinstance(best_idx, int):
        return None
    if best_idx < 0 or best_idx >= len(translations):
        return None

    best_item = translations[best_idx]
    if not isinstance(best_item, dict):
        return None

    source_en = best_item.get("translation", "")
    if not isinstance(source_en, str):
        return None
    source_en = source_en.strip()
    if not source_en:
        return None

    return {
        "source": source_en,  
        "target": target_hu,   
    }


def load_pairs_from_raw_folder(raw_data_dir: str) -> List[Dict[str, str]]:
    """
    Read all parallel_corpus_*.json files and extract valid EN->HU pairs.
    No deduplication is performed.
    """
    pattern = os.path.join(raw_data_dir, "parallel_corpus_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    print(f"Found {len(files)} raw corpus files.")

    pairs: List[Dict[str, str]] = []
    total_records = 0
    skipped_records = 0

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping file due to read error: {file_path} | {e}")
            continue

        if not isinstance(data, list):
            print(f"Skipping non-list JSON file: {file_path}")
            continue

        for item in data:
            total_records += 1
            pair = extract_best_translation(item)
            if pair is None:
                skipped_records += 1
                continue
            pairs.append(pair)

    print(f"Total raw records: {total_records}")
    print(f"Valid extracted pairs: {len(pairs)}")
    print(f"Skipped invalid records: {skipped_records}")

    if not pairs:
        raise ValueError("No valid training pairs were extracted.")

    return pairs


def split_pairs(
    pairs: List[Dict[str, str]],
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split extracted pairs into train/val/test.
    """
    train_pairs, temp_pairs = train_test_split(
        pairs,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=seed,
        shuffle=True,
    )

    val_relative = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=(1.0 - val_relative),
        random_state=seed,
        shuffle=True,
    )

    total = len(pairs)
    print(f"Train size: {len(train_pairs)} ({len(train_pairs) / total:.2%})")
    print(f"Val size:   {len(val_pairs)} ({len(val_pairs) / total:.2%})")
    print(f"Test size:  {len(test_pairs)} ({len(test_pairs) / total:.2%})")

    return train_pairs, val_pairs, test_pairs


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_chunked_json(
    pairs: List[Dict[str, str]],
    split_dir: str,
    split_name: str,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """
    Save one split as chunked JSON files:
    train/train_0000.json, train/train_0001.json, ...
    """
    ensure_dir(split_dir)

    num_files = 0
    for start_idx in range(0, len(pairs), chunk_size):
        chunk = pairs[start_idx:start_idx + chunk_size]
        chunk_id = start_idx // chunk_size
        out_path = os.path.join(split_dir, f"{split_name}_{chunk_id:04d}.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

        num_files += 1

    print(f"Saved {len(pairs)} pairs into {num_files} file(s) under: {split_dir}")


def create_and_save_splits(
    raw_data_dir: str,
    split_root_dir: str,
    seed: int,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """
    Full pipeline:
    raw files -> extracted pairs -> split -> chunked JSON files
    """
    print("Creating dataset splits from raw corpus...")
    print(f"Raw input folder: {raw_data_dir}")
    print(f"Split output folder: {split_root_dir}")

    train_dir = os.path.join(split_root_dir, "train")
    val_dir = os.path.join(split_root_dir, "val")
    test_dir = os.path.join(split_root_dir, "test")

    ensure_dir(split_root_dir)
    ensure_dir(train_dir)
    ensure_dir(val_dir)
    ensure_dir(test_dir)

    pairs = load_pairs_from_raw_folder(raw_data_dir)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, seed=seed)

    save_chunked_json(train_pairs, train_dir, "train", chunk_size=chunk_size)
    save_chunked_json(val_pairs, val_dir, "val", chunk_size=chunk_size)
    save_chunked_json(test_pairs, test_dir, "test", chunk_size=chunk_size)

    print("Finished creating dataset splits.")

if __name__ == "__main__":
    print("Starting dataset split creation...")

    create_and_save_splits(
        raw_data_dir="/home/nr_syne/nr_synthpar/1_M_Translated_Sentences",
        split_root_dir="/home/nr_syne/nr_synthpar/Training_Dataset_Split",
        seed=42,
        chunk_size=500,
    )

    print("Dataset split creation finished.")