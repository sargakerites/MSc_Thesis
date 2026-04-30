#!/usr/bin/env python3

import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu.metrics import BLEU, CHRF
from rouge_score import rouge_scorer


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Loads:
      - .jsonl files (one JSON object per line)
      - .json files containing either:
          * a list of JSON objects
          * a single JSON object
    Returns a flat list of dicts.
    """
    records = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError(f"Line {line_no} is not a JSON object.")
                    records.append(obj)
                except Exception as e:
                    raise ValueError(f"Failed to parse JSONL line {line_no} in {path}: {e}") from e

    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for i, obj in enumerate(data):
                if not isinstance(obj, dict):
                    raise ValueError(f"Element {i} in {path} is not a JSON object.")
                records.append(obj)
        elif isinstance(data, dict):
            records.append(data)
        else:
            raise ValueError(f"Unsupported JSON structure in {path}.")
    else:
        raise ValueError(f"Unsupported file type: {path}")

    return records


def collect_dataset_records(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Supports:
      - a single .json or .jsonl file
      - a directory with multiple .json / .jsonl files (recursively)
    """
    p = Path(dataset_path)

    if not p.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    all_records = []

    if p.is_file():
        all_records.extend(load_json_or_jsonl(p))
    else:
        files = sorted(list(p.rglob("*.jsonl")) + list(p.rglob("*.json")))
        if not files:
            raise ValueError(f"No .json or .jsonl files found under directory: {dataset_path}")

        for file_path in files:
            all_records.extend(load_json_or_jsonl(file_path))

    if not all_records:
        raise ValueError(f"No records found in dataset: {dataset_path}")

    return all_records


def normalize_record(record: Dict[str, Any]) -> Tuple[str, str]:
    """
    Supports both formats:
      1) {"eng_Latn": ..., "hun_Latn": ...}
      2) {"source": ..., "target": ...}
    Returns: (source_english, target_hungarian)
    """
    if "eng_Latn" in record and "hun_Latn" in record:
        src = record["eng_Latn"]
        tgt = record["hun_Latn"]
    elif "source" in record and "target" in record:
        src = record["source"]
        tgt = record["target"]
    else:
        raise ValueError(
            f"Unsupported record structure. Expected "
            f"{{'eng_Latn','hun_Latn'}} or {{'source','target'}}, got keys: {list(record.keys())}"
        )

    if not isinstance(src, str) or not isinstance(tgt, str):
        raise ValueError(f"Source/target must be strings. Got: {type(src)}, {type(tgt)}")

    return src.strip(), tgt.strip()


def load_dataset_pairs(dataset_path: str) -> Tuple[List[str], List[str]]:
    records = collect_dataset_records(dataset_path)

    sources = []
    references = []

    for i, rec in enumerate(records):
        try:
            src, tgt = normalize_record(rec)
            if src and tgt:
                sources.append(src)
                references.append(tgt)
        except Exception as e:
            raise ValueError(f"Error while processing record #{i}: {e}") from e

    if not sources:
        raise ValueError("No valid source-reference pairs found.")

    return sources, references


def resolve_tgt_lang_id(tokenizer, tgt_lang: str) -> int:
    """
    NLLB-style tokenizer handling.
    """
    if hasattr(tokenizer, "lang_code_to_id") and tgt_lang in tokenizer.lang_code_to_id:
        return tokenizer.lang_code_to_id[tgt_lang]

    token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Could not resolve target language token ID for: {tgt_lang}")

    return token_id


def generate_translations(
    model_name_or_path: str,
    sources: List[str],
    batch_size: int = 8,
    max_source_length: int = 256,
    max_target_length: int = 256,
    num_beams: int = 4,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "hun_Latn",
    local_files_only: bool = False,
) -> List[str]:
    """
    Loads model+tokenizer and generates translations.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[INFO] Loading tokenizer from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        local_files_only=local_files_only,
        use_fast=True,
    )

    print(f"[INFO] Loading model from: {model_name_or_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    forced_bos_token_id = resolve_tgt_lang_id(tokenizer, tgt_lang)

    predictions = []

    total = len(sources)
    num_batches = math.ceil(total / batch_size)

    print(f"[INFO] Running generation on {total} examples in {num_batches} batches.")
    print(f"[INFO] Device: {device}")

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_sources = sources[start:end]

            inputs = tokenizer(
                batch_sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_source_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generated_ids = model.generate(
                **inputs,
                max_length=max_target_length,
                num_beams=num_beams,
                forced_bos_token_id=forced_bos_token_id,
            )

            batch_predictions = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            predictions.extend([pred.strip() for pred in batch_predictions])

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"[INFO] Finished batch {batch_idx + 1}/{num_batches}")

    return predictions


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes:
      - BLEU (13a)
      - spBLEU (spBLEU-1K tokenizer)
      - chrF
      - chrF++
      - ROUGE-1 / ROUGE-2 / ROUGE-L
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    refs_for_sacrebleu = [references]

    bleu_metric = BLEU(tokenize="13a")
    spbleu_metric = BLEU(tokenize="spBLEU-1K")
    chrf_metric = CHRF(word_order=0)      
    chrfpp_metric = CHRF(word_order=2)  

    bleu = bleu_metric.corpus_score(predictions, refs_for_sacrebleu).score
    spbleu = spbleu_metric.corpus_score(predictions, refs_for_sacrebleu).score
    chrf = chrf_metric.corpus_score(predictions, refs_for_sacrebleu).score
    chrfpp = chrfpp_metric.corpus_score(predictions, refs_for_sacrebleu).score

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    rougeL = sum(rougeL_scores) / len(rougeL_scores)

    return {
        "BLEU": bleu,
        "spBLEU": spbleu,
        "chrF": chrf,
        "chrF++": chrfpp,
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL,
    }


def save_outputs(
    output_dir: str,
    predictions: List[str],
    references: List[str],
    sources: List[str],
    metrics: Dict[str, float],
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"

    with preds_path.open("w", encoding="utf-8") as f:
        for src, pred, ref in zip(sources, predictions, references):
            obj = {
                "source": src,
                "prediction": pred,
                "reference": ref,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved predictions to: {preds_path}")
    print(f"[INFO] Saved metrics to: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EN->HU translation model on JSON/JSONL datasets.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or local model/checkpoint path."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset file (.json/.jsonl) or directory containing them."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save metrics and predictions."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Generation batch size."
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help="Max source length."
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="Max target length used in generation."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam size for generation."
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng_Latn",
        help="Source language code."
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="hun_Latn",
        help="Target language code."
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load model/tokenizer from local files."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Loading dataset...")
    sources, references = load_dataset_pairs(args.dataset)
    print(f"[INFO] Loaded {len(sources)} source-reference pairs.")

    predictions = generate_translations(
        model_name_or_path=args.model,
        sources=sources,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_beams=args.num_beams,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        local_files_only=args.local_files_only,
    )

    print("[INFO] Computing metrics...")
    metrics = compute_metrics(predictions, references)

    print("\n===== EVALUATION RESULTS =====")
    for k, v in metrics.items():
        if k.startswith("ROUGE"):
            print(f"{k:8s}: {v:.6f}")
        else:
            print(f"{k:8s}: {v:.4f}")

    save_outputs(
        output_dir=args.output_dir,
        predictions=predictions,
        references=references,
        sources=sources,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()