import json
from pathlib import Path
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_DIR = PROJECT_ROOT / "1_M_Annotated_Sentences"
FILLER_DIR = PROJECT_ROOT / "Elte_Rk_Annotated_Sentences"

TARGET_SIZE = 500


def load_json(path: Path):
    """Load and return JSON data from a UTF-8 encoded file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    """Save data to a JSON file using UTF-8 encoding and indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_sentence_text(item):
    """Extract the sentence text from an item if it is a dictionary with a valid sentence field."""
    if isinstance(item, dict):
        sentence = item.get("sentence")
        if isinstance(sentence, str):
            return sentence
    return None


def normalize_sentence(sentence: str) -> str:
    """Normalize a sentence by collapsing whitespace and trimming leading/trailing spaces."""
    return " ".join(sentence.split()).strip()


def build_filler_pool(filler_dir: Path):
    """Build a pool of unique filler sentences from JSON files in the filler directory."""
    filler_files = sorted(filler_dir.glob("*.json"))

    pool = []
    seen_sentences = set()

    print(f"Number of filler files: {len(filler_files)}")

    for fp in tqdm(filler_files, desc="Building filler pool"):
        try:
            data = load_json(fp)
        except Exception as e:
            print(f"Error while reading file: {fp} -> {e}")
            continue

        if not isinstance(data, list):
            continue

        for item in data:
            sent = get_sentence_text(item)
            if not sent:
                continue

            sent_norm = normalize_sentence(sent)
            if not sent_norm:
                continue

            if sent_norm not in seen_sentences:
                seen_sentences.add(sent_norm)
                pool.append(item)

    print(f"Total unique filler sentences: {len(pool)}")
    return pool


def build_used_sentences_from_targets(target_files):
    """Collect normalized sentences already present in the target files to avoid duplicates."""
    used_sentences = set()

    for tf in tqdm(target_files, desc="Indexing existing target sentences"):
        try:
            data = load_json(tf)
        except Exception as e:
            print(f"Error while reading file: {tf} -> {e}")
            continue

        if not isinstance(data, list):
            continue

        for item in data:
            sent = get_sentence_text(item)
            if not sent:
                continue

            sent_norm = normalize_sentence(sent)
            if sent_norm:
                used_sentences.add(sent_norm)

    print(f"Existing unique sentences in target dataset: {len(used_sentences)}")
    return used_sentences


def main():
    """Fill incomplete target JSON files with unique filler sentences until each file reaches the target size."""
    target_files = sorted(TARGET_DIR.glob("500_hungarian_sentences_*.json"))
    if not target_files:
        print("No target files found in the 1_M_Annotated_Sentences directory.")
        return

    filler_pool = build_filler_pool(FILLER_DIR)
    used_sentences = build_used_sentences_from_targets(target_files)

    filler_idx = 0
    total_missing = 0
    total_added = 0
    modified_files = 0

    for tf in tqdm(target_files, desc="Filling target files"):
        try:
            data = load_json(tf)
        except Exception as e:
            print(f"Error while reading file: {tf} -> {e}")
            continue

        if not isinstance(data, list):
            print(f"File content is not a list: {tf}")
            continue

        current_len = len(data)

        if current_len > TARGET_SIZE:
            print(f"File contains more than {TARGET_SIZE} items, skipping: {tf} ({current_len})")
            continue

        if current_len == TARGET_SIZE:
            continue

        missing = TARGET_SIZE - current_len
        total_missing += missing

        existing_sentences = set()
        for item in data:
            sent = get_sentence_text(item)
            if sent:
                sent_norm = normalize_sentence(sent)
                if sent_norm:
                    existing_sentences.add(sent_norm)

        additions = []
        while len(additions) < missing and filler_idx < len(filler_pool):
            candidate = filler_pool[filler_idx]
            filler_idx += 1

            cand_sent = get_sentence_text(candidate)
            if not cand_sent:
                continue

            cand_sent_norm = normalize_sentence(cand_sent)
            if not cand_sent_norm:
                continue

            if cand_sent_norm in existing_sentences:
                continue

            if cand_sent_norm in used_sentences:
                continue

            additions.append(candidate)
            existing_sentences.add(cand_sent_norm)
            used_sentences.add(cand_sent_norm)

        if len(additions) < missing:
            print(
                f"Not enough unique filler sentences. "
                f"{tf.name}: missing {missing}, but only {len(additions)} can be added."
            )
            continue

        data.extend(additions)
        save_json(tf, data)

        modified_files += 1
        total_added += len(additions)

    print("\n===== DONE =====")
    print(f"Number of modified files: {modified_files}")
    print(f"Total missing sentences: {total_missing}")
    print(f"Actually added sentences: {total_added}")
    print(f"Number of used filler pool items: {filler_idx}")


if __name__ == "__main__":
    main()