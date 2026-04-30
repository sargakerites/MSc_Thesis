import json
import gzip
import re
from pathlib import Path
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "news"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "1_M_Sentences"),
    )
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--max_files", type=int, default=1000)
    parser.add_argument("--max_sentences", type=int, default=6000)

    return parser.parse_args()


args = parse_args()

DATA_DIR = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir)
CHUNK_SIZE = args.chunk_size
MAX_FILES = args.max_files
MAX_SENTENCES = args.max_sentences


ENCODING_FIXES = {
    "û": "ű",
    "õ": "ő",
    "Û": "Ű",
    "Õ": "Ő",
}


def fix_encoding(text: str) -> str:
    for bad, good in ENCODING_FIXES.items():
        text = text.replace(bad, good)
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = fix_encoding(text)
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\xad", "")
    text = re.sub(r"-\n\s*", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text: str):
    if not text:
        return []

    text = clean_text(text)

    parts = re.split(
        r'(?<=[.!?])\s+(?=(?:[-"„”]\s*)*[A-ZÁÉÍÓÖŐÚÜŰ])',
        text
    )

    sentences = []

    for s in parts:
        s = s.strip()
        if not s:
            continue

        s = re.sub(r'^\-\s*', '', s)
        s = re.sub(r'(?<=\w)\d+(?=\w)', '', s)

        if re.fullmatch(r"\d+\.", s):
            continue

        if len(s) < 20:
            continue

        strange_ratio = sum(
            not (c.isalnum() or c.isspace() or c in ".,:;!?-–—'\"()")
            for c in s
        ) / len(s)

        if strange_ratio > 0.2:
            continue

        sentences.append(s)

    return sentences


def open_maybe_gzip(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "r", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_sentences = []

    files = list(DATA_DIR.rglob("*.jsonl")) + \
            list(DATA_DIR.rglob("*.jsonl.gz"))

    files = files[:MAX_FILES]

    print(f"Testing on {len(files)} files")

    for file in files:
        print("Processing:", file)

        with open_maybe_gzip(file) as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)
                text = obj.get("text", "")

                for s in split_sentences(text):
                    all_sentences.append({"sentence": s})

                    if len(all_sentences) >= MAX_SENTENCES:
                        break

                if len(all_sentences) >= MAX_SENTENCES:
                    break

        if len(all_sentences) >= MAX_SENTENCES:
            break

    for chunk_start in range(0, len(all_sentences), CHUNK_SIZE):
        chunk = all_sentences[chunk_start:chunk_start + CHUNK_SIZE]
        idx = chunk_start // CHUNK_SIZE

        out_path = OUTPUT_DIR / f"{CHUNK_SIZE}_hungarian_sentences_{idx}.json"

        if out_path.exists():
            print(f"Skipping existing file: {out_path}")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

        print(f"Wrote {len(chunk)} sentences -> {out_path}")


if __name__ == "__main__":
    main()