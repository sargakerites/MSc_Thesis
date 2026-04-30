import os
import json
import re
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
from collections import defaultdict

import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


OPENAI_JUDGE_MODEL = "gpt-5.2"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_ROOT = Path(os.getenv("SCRATCH_ROOT", f"/scratch/{os.getenv('USER', 'user')}"))

MODELS = {
    "base_model": "NYTK/translation-nllb-200-3.3B-multi12-hungarian",
    "fine_tuned_model": str(
        SCRATCH_ROOT / "Fine-Tuning" / "Final" / "Checkpoints" / "final_bs1024_e6"
    ),
}

DATASETS = {
    "flores_plus_hun_eng_devtest": {
        "path": str(
            PROJECT_ROOT
            / "Evaluation"
            / "Data"
            / "flores_plus_hun_eng_devtest.jsonl"
        ),
        "type": "flores",
    },
    "training_test_split": {
        "path": str(
            PROJECT_ROOT / "Training_Dataset_Split" / "test"
        ),
        "type": "train_split",
    },
}

MAX_SENTENCES_PER_DATASET = 1000
SAVE_EVERY = 50
BATCH_SIZE = 8
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256
NUM_BEAMS = 5

SRC_LANG = "eng_Latn"
TGT_LANG = "hun_Latn"

OUTPUT_DIR = PROJECT_ROOT / "Evaluation" / "Semantic_Eval_Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clean_text(text: str) -> str:
    """Normalize text by collapsing whitespace and trimming. Converts input to string; returns empty string if input is None."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def save_json(data: Dict, path: Path) -> None:
    """Save a dictionary to a JSON file using UTF-8 encoding with indentation."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_jsonl(path: Path) -> Iterable[Dict]:
    """Iterate over a JSONL file and yield each non-empty line as a parsed dictionary."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_flores_pairs(path: Path, limit: int) -> List[Dict[str, str]]:
    """Load and clean English-Hungarian sentence pairs from a FLORES+ JSONL file up to a given limit."""
    pairs = []

    for item in iter_jsonl(path):
        source_en = clean_text(item.get("eng_Latn", ""))
        reference_hu = clean_text(item.get("hun_Latn", ""))

        if not source_en or not reference_hu:
            continue

        pairs.append({
            "source_en": source_en,
            "reference_hu": reference_hu,
        })

        if len(pairs) >= limit:
            break

    return pairs


def load_train_split_pairs(path: Path, limit: int) -> List[Dict[str, str]]:
    """Load English-Hungarian pairs from JSON/JSONL train/test split files, supporting multiple files and a limit."""
    pairs = []

    if path.is_dir():
        candidate_files = sorted(list(path.glob("*.json")) + list(path.glob("*.jsonl")))
    else:
        candidate_files = [path]

    for file_path in candidate_files:
        if file_path.suffix == ".jsonl":
            items = iter_jsonl(file_path)
        else:
            with file_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                items = loaded
            else:
                items = [loaded]

        for item in items:
            source_en = clean_text(item.get("source", ""))
            reference_hu = clean_text(item.get("target", ""))

            if not source_en or not reference_hu:
                continue

            pairs.append({
                "source_en": source_en,
                "reference_hu": reference_hu,
            })

            if len(pairs) >= limit:
                return pairs

    return pairs


def load_dataset_pairs(dataset_name: str, dataset_cfg: Dict, limit: int) -> List[Dict[str, str]]:
    """Dispatch dataset loading based on dataset type configuration."""
    path = Path(dataset_cfg["path"])
    dtype = dataset_cfg["type"]

    if dtype == "flores":
        return load_flores_pairs(path, limit)
    elif dtype == "train_split":
        return load_train_split_pairs(path, limit)
    else:
        raise ValueError(f"Unknown dataset type for {dataset_name}: {dtype}")


def load_translation_model(model_path: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """Load a tokenizer and seq2seq model from a given path, move to device, and set eval mode."""
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def translate_batch(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    texts: List[str],
) -> List[str]:
    """Translate a batch of source texts using the model with beam search and return cleaned outputs."""
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = SRC_LANG

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    generated = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=MAX_TARGET_LENGTH,
        num_beams=NUM_BEAMS,
    )

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [clean_text(x) for x in decoded]


def translate_all(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    texts: List[str],
    batch_size: int = 8,
) -> List[str]:
    """Translate a list of texts in batches with progress logging."""
    outputs = []

    for start_idx in range(0, len(texts), batch_size):
        batch = texts[start_idx:start_idx + batch_size]
        batch_outputs = translate_batch(tokenizer, model, batch)
        outputs.extend(batch_outputs)

        end_idx = start_idx + len(batch)
        print(f"Translated {end_idx}/{len(texts)}")

    return outputs


def judge_pair(
    client: OpenAI,
    judge_model: str,
    source_en: str,
    translation_hu: str,
) -> Dict[str, int]:
    """Evaluate a single translation using an LLM judge and return metric scores as a JSON dict."""
    system_prompt = """
    You are a STRICT and CONSISTENT translation judge for English → Hungarian.

    Your task is to evaluate the Hungarian translation of an English sentence using
    THREE INDEPENDENT criteria.

    CRITICAL RULES:
    - Each criterion must be judged independently.
    - Do NOT let one criterion influence another.
    - Use a CONTINUOUS score between -1.0 and 1.0.
    - You MAY use decimals (e.g., 0.7, -0.3, 0.95).
    - If unsure between two scores, choose the LOWER one.
    - Output ONLY valid JSON.
    - Do NOT include explanations, reasoning, or extra text.

    CRITERIA DEFINITIONS:

    1) content_accuracy
    - Evaluate ONLY semantic meaning.
    - Ignore grammar, style, and fluency.
    - 1.0  = all meaning preserved, no omissions or additions
    - 0.0  = minor nuance loss, but core meaning intact
    - -1.0 = missing, distorted, or misleading meaning
    - Intermediate values allowed (e.g., 0.5, -0.2)

    2) grammatical_correctness
    - Evaluate ONLY grammar and syntax.
    - Ignore wording elegance and meaning accuracy.
    - 1.0  = grammatically correct
    - 0.0  = minor grammatical issues that do not confuse meaning
    - -1.0 = serious grammar errors or broken sentence structure
    - Intermediate values allowed

    3) fluency
    - Evaluate ONLY naturalness and readability.
    - Ignore grammar errors already penalized above.
    - Ignore content accuracy.
    - 1.0  = natural, native-like Hungarian
    - 0.0  = awkward but understandable
    - -1.0 = very unnatural, robotic, or hard to read
    - Intermediate values allowed

    OUTPUT FORMAT:
    Return EXACTLY this JSON object, with NO additional keys:

    {
    "content_accuracy": 0.0,
    "grammatical_correctness": 0.0,
    "fluency": 0.0
    }
    """.strip()

    user_prompt = f"""
Evaluate the following translation.

English sentence:
{source_en}

Hungarian translation:
{translation_hu}

Return ONLY the JSON score object.
""".strip()

    response = client.responses.create(
        model=judge_model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return json.loads(response.output_text)


def judge_translations(
    client: OpenAI,
    judge_model: str,
    dataset_name: str,
    model_name: str,
    pairs: List[Dict[str, str]],
    translations: List[str],
) -> Dict:
    """Evaluate all translations, compute average scores, and periodically save detailed and summary results."""
    score_sums = defaultdict(float)
    score_counts = defaultdict(int)
    items = []

    output_path = OUTPUT_DIR / f"{dataset_name}__{model_name}_detailed.json"
    summary_path = OUTPUT_DIR / f"{dataset_name}__{model_name}_summary.json"

    for idx, (pair, translation_hu) in enumerate(zip(pairs, translations), start=1):
        source_en = pair["source_en"]
        reference_hu = pair["reference_hu"]

        try:
            scores = judge_pair(
                client=client,
                judge_model=judge_model,
                source_en=source_en,
                translation_hu=translation_hu,
            )

            for k, v in scores.items():
                score_sums[k] += v
                score_counts[k] += 1

            item = {
                "index": idx,
                "source_en": source_en,
                "reference_hu": reference_hu,
                "model_translation_hu": translation_hu,
                "scores": scores,
            }

        except Exception as e:
            item = {
                "index": idx,
                "source_en": source_en,
                "reference_hu": reference_hu,
                "model_translation_hu": translation_hu,
                "scores": None,
                "error": str(e),
            }

        items.append(item)

        if idx % SAVE_EVERY == 0 or idx == len(pairs):
            avg_scores = {
                metric: (score_sums[metric] / score_counts[metric]) if score_counts[metric] > 0 else None
                for metric in ["content_accuracy", "grammatical_correctness", "fluency"]
            }

            partial_result = {
                "dataset_name": dataset_name,
                "evaluated_model": model_name,
                "judge_model": judge_model,
                "processed_items": idx,
                "average_scores": avg_scores,
                "items": items,
            }

            save_json(partial_result, output_path)
            save_json(
                {
                    "dataset_name": dataset_name,
                    "evaluated_model": model_name,
                    "judge_model": judge_model,
                    "processed_items": idx,
                    "average_scores": avg_scores,
                },
                summary_path,
            )

            print(f"[{dataset_name} | {model_name}] Judged {idx}/{len(pairs)}")

    final_avg_scores = {
        metric: (score_sums[metric] / score_counts[metric]) if score_counts[metric] > 0 else None
        for metric in ["content_accuracy", "grammatical_correctness", "fluency"]
    }

    return {
        "dataset_name": dataset_name,
        "evaluated_model": model_name,
        "judge_model": judge_model,
        "num_items": len(pairs),
        "average_scores": final_avg_scores,
        "items": items,
    }


def main():
    """Run the full semantic MT evaluation pipeline: load data, translate with models, evaluate, and save outputs."""
    
    print("======================================")
    print("Semantic MT evaluation started")
    print(f"Device: {DEVICE}")
    print(f"Judge model: {OPENAI_JUDGE_MODEL}")
    print("======================================")

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    dataset_pairs_map = {}
    for dataset_name, dataset_cfg in DATASETS.items():
        print(f"\nLoading dataset: {dataset_name}")
        pairs = load_dataset_pairs(
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            limit=MAX_SENTENCES_PER_DATASET,
        )
        print(f"Loaded {len(pairs)} pairs from {dataset_name}")
        dataset_pairs_map[dataset_name] = pairs

    loaded_models = {}
    for model_name, model_path in MODELS.items():
        tokenizer, model = load_translation_model(model_path)
        loaded_models[model_name] = {
            "tokenizer": tokenizer,
            "model": model,
            "path": model_path,
        }

    global_summary = {}

    for dataset_name, pairs in dataset_pairs_map.items():
        source_texts = [p["source_en"] for p in pairs]
        global_summary[dataset_name] = {}

        for model_name, model_bundle in loaded_models.items():
            print("\n--------------------------------------")
            print(f"Dataset: {dataset_name}")
            print(f"Model:   {model_name}")
            print("--------------------------------------")

            tokenizer = model_bundle["tokenizer"]
            model = model_bundle["model"]

            translations = translate_all(
                tokenizer=tokenizer,
                model=model,
                texts=source_texts,
                batch_size=BATCH_SIZE,
            )

            translations_path = OUTPUT_DIR / f"{dataset_name}__{model_name}_translations.json"
            save_json(
                {
                    "dataset_name": dataset_name,
                    "evaluated_model": model_name,
                    "model_path": model_bundle["path"],
                    "translations": [
                        {
                            "index": i + 1,
                            "source_en": pairs[i]["source_en"],
                            "reference_hu": pairs[i]["reference_hu"],
                            "model_translation_hu": translations[i],
                        }
                        for i in range(len(translations))
                    ],
                },
                translations_path,
            )

            result = judge_translations(
                client=client,
                judge_model=OPENAI_JUDGE_MODEL,
                dataset_name=dataset_name,
                model_name=model_name,
                pairs=pairs,
                translations=translations,
            )

            global_summary[dataset_name][model_name] = {
                "judge_model": OPENAI_JUDGE_MODEL,
                "num_items": result["num_items"],
                "average_scores": result["average_scores"],
            }

            print(f"\nFinal average scores for {dataset_name} | {model_name}:")
            print(json.dumps(result["average_scores"], ensure_ascii=False, indent=2))

    summary_path = OUTPUT_DIR / "overall_summary.json"
    save_json(global_summary, summary_path)

    print("\n======================================")
    print("Finished.")
    print(f"Overall summary saved to: {summary_path.resolve()}")
    print("======================================")


if __name__ == "__main__":
    main()