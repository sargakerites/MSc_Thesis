import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

import huspacy
from datasketch import MinHash, MinHashLSH

from concurrent.futures import ProcessPoolExecutor, as_completed


NGRAM_SIZE = 5
NUM_PERM = 128
DEDUP_THRESHOLD = 0.93

MAX_WORKERS = 30
CHUNK_SIZE = 200

_NLP = None
_EKSZ_DEFS = None


def normalize_text(text: str) -> str:
    """Normalize a sentence by collapsing repeated whitespace and trimming leading/trailing spaces."""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_token_entries(doc, eksz_defs):
    """Create token-level annotation entries with word form, lemma, POS tag, and dictionary definitions."""

    tokens = []
    for t in doc:
        word = t.text.strip()
        if not word:
            continue

        lemma = t.lemma_
        pos = t.pos_
        definitions = eksz_defs.get(lemma, [])

        tokens.append({
            "word": word,
            "lemma": lemma,
            "pos": pos,
            "definitions": definitions
        })
    return tokens


def char_ngrams(text, n=5):
    """Convert text into a set of lowercase character n-grams."""

    text = f" {text.lower()} "
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def sentence_to_minhash(sentence, n=5, num_perm=128):
    """Create a MinHash signature from a sentence using character n-grams."""

    m = MinHash(num_perm=num_perm)
    for ng in char_ngrams(sentence, n):
        m.update(ng.encode("utf-8"))
    return m


def deduplicate_sentences(
    annotated_sentences,
    threshold=0.93,
    ngram_size=5,
    num_perm=128,
):
    """Remove near-duplicate annotated sentences using MinHash LSH similarity search."""

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    unique = []
    counter = 0

    for item in annotated_sentences:
        sentence = item.get("sentence", "").strip()
        if not sentence:
            continue

        mh = sentence_to_minhash(sentence, n=ngram_size, num_perm=num_perm)
        matches = lsh.query(mh)

        if not matches:
            key = f"s_{counter}"
            counter += 1
            lsh.insert(key, mh)
            unique.append(item)

    return unique


def _chunk_list(lst: List[Any], chunk_size: int):
    """Split a list into consecutive chunks of a given size."""

    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def _mp_init(model_name: str, dict_path: str):
    """Initialize each worker process by loading the HuSpaCy model and dictionary definitions."""

    global _NLP, _EKSZ_DEFS
    _NLP = huspacy.load(model_name)
    with open(dict_path, "r", encoding="utf-8") as f:
        _EKSZ_DEFS = json.load(f)


def _annotate_chunk(args: Tuple[int, List[Dict[str, Any]]]) -> Tuple[int, List[Dict[str, Any]]]:
    """Annotate one chunk of sentences with HuSpaCy and dictionary definitions."""

    global _NLP, _EKSZ_DEFS
    chunk_id, items = args

    out = []
    for item in items:
        s = normalize_text(item.get("sentence", ""))
        if not s:
            continue

        doc = _NLP(s)
        tokens = build_token_entries(doc, _EKSZ_DEFS)
        out.append({"sentence": s, "tokens": tokens})

    return chunk_id, out


def annotate_sentences_parallel(
    executor: ProcessPoolExecutor,
    sentences: List[Dict[str, Any]],
    chunk_size: int,
) -> List[Dict[str, Any]]:
    """Annotate sentences in parallel chunks and restore the original chunk order."""

    chunks = list(_chunk_list(sentences, chunk_size))
    tasks = [(i, chunk) for i, chunk in enumerate(chunks)]

    results_by_id: Dict[int, List[Dict[str, Any]]] = {}

    futs = [executor.submit(_annotate_chunk, t) for t in tasks]

    for fut in tqdm(as_completed(futs), total=len(futs), desc="Annotating (chunks)"):
        chunk_id, annotated = fut.result()
        results_by_id[chunk_id] = annotated

    combined: List[Dict[str, Any]] = []
    for i in range(len(chunks)):
        combined.extend(results_by_id.get(i, []))

    return combined


def parse_args():
    """Parse command-line arguments for input/output paths, dictionary path, worker count, chunk size, and HuSpaCy model."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dict_path", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=30)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--model_name", type=str, default="hu_core_news_lg")
    return parser.parse_args()


def main():
    """Run the annotation pipeline: load files, annotate sentences in parallel, deduplicate them, and save outputs."""
    
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    dict_path = Path(args.dict_path)

    output_dir.mkdir(exist_ok=True, parents=True)

    files = sorted(input_dir.glob("*.json"))
    if not files:
        print(f"No input files found in {input_dir}")
        return

    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=_mp_init,
        initargs=(args.model_name, str(dict_path)),
    ) as executor:

        for file in files:

            out_path = output_dir / file.name

            if out_path.exists():
                print(f"Skipping (already annotated): {out_path}")
                continue


            print("\nProcessing:", file)

            with open(file, "r", encoding="utf-8") as f:
                sentences = json.load(f)

            annotated = annotate_sentences_parallel(
                executor=executor,
                sentences=sentences,
                chunk_size=args.chunk_size,
            )

            print("Deduplicating...")
            annotated = deduplicate_sentences(
                annotated,
                threshold=DEDUP_THRESHOLD,
                ngram_size=NGRAM_SIZE,
                num_perm=NUM_PERM,
            )

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(annotated, f, ensure_ascii=False, indent=2)

            print("Saved:", out_path)


if __name__ == "__main__":
    main()