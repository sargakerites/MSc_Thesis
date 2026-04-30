from __future__ import annotations
from build_prompt import build_prompt
from schema import SCHEMA_TEXT, extract_json
from token_builder import build_tokens
from pronoun_expander import expand_translations_with_gender
from Models.vllm_gemma import model_generation
import json
import os
import time
from pathlib import Path
from typing import Callable, Any



SYSTEM_PROMPT = f"""
You are an expert Hungarian → English translator.

Use definitions only as translation aids.
Do NOT output definitions.

Return ONLY a JSON object that follows this schema.

Do NOT include schema, name, or metadata fields.

Schema:
{SCHEMA_TEXT}

Output ONLY JSON.
No explanations.
No markdown.
No extra text.
""".strip()





def _build_full_prompt(prompt: str) -> str:
    """Wrap the user prompt with the system translation instructions in the expected chat-style format."""

    return f"""<system>
{SYSTEM_PROMPT}
</system>

<user>
{prompt}
</user>

<assistant>
"""




_generate_one = None
_generate_many = None
_llm_obj = None  

def _get_generators():
    """
    Lazily initialize the vLLM model once, then reuse.
    Returns:
      generate_one(prompt:str)->str
      generate_many(prompts:list[str])->list[str]
    """
    global _generate_one, _generate_many, _llm_obj

    if _generate_one is None or _generate_many is None:
        _generate_one, _generate_many, _llm_obj = model_generation()

    return _generate_one, _generate_many


def translation(annotated_sentence_dict):
    """Translate one annotated sentence entry and return the parsed structured JSON result."""

    generate_one, _ = _get_generators()

    clean_input = build_tokens(annotated_sentence_dict)
    prompt = build_prompt(clean_input)
    full_prompt = _build_full_prompt(prompt)

    last_error = None
    for attempt in range(3):
        content = (generate_one(full_prompt) or "").strip()

        try:
            result = extract_json(content)
            result = expand_translations_with_gender(annotated_sentence_dict, result)
            return result

        except Exception as e:
            last_error = e
            print("\n--- JSON extraction failed ---")
            print(content)
            print("--- Retrying ---\n")
            print(f"len(content)={len(content)}")
            print("TAIL:", content[-300:])

    raise ValueError(f"Failed after retries: {last_error}")





def _safe_dump_text(path: Path, text: str) -> None:
    """Safely write text content to a file, creating parent directories if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _safe_dump_json(path: Path, obj: Any) -> None:
    """Safely write an object as formatted JSON, creating parent directories if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def translation_batch(
    annotated_sentence_dicts: list[dict],
    *,
    batch_size: int = 32,
    max_retries: int = 3,
    on_final_fail: str = "placeholder",  
    retry_sleep_s: float = 0.0,
) -> list[dict]:
    """
    Batched translation with per-item retries.
    If an item still fails, we (optionally) fall back to single-item generation,
    then either: return a placeholder, skip it, or raise.

    Placeholder format:
      {"__error__": "...", "__raw_tail__": "..."}
    """
    _, generate_many = _get_generators()

    results: list[dict] = []

    for start in range(0, len(annotated_sentence_dicts), batch_size):
        batch = annotated_sentence_dicts[start:start + batch_size]

        full_prompts: list[str] = []
        for d in batch:
            clean_input = build_tokens(d)
            prompt = build_prompt(clean_input)
            full_prompts.append(_build_full_prompt(prompt))

        pending_idx = list(range(len(batch)))
        pending_prompts = full_prompts[:]
        pending_dicts = batch[:]
        batch_outputs: list[dict | None] = [None] * len(batch)

        last_error: Exception | None = None

        lens = [len(p) for p in pending_prompts]
        print(f"[vLLM] pending={len(pending_prompts)} prompt_chars min/avg/max="
            f"{min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")

        for attempt in range(max_retries):
            texts = [(t or "").strip() for t in generate_many(pending_prompts)]

            next_pending_idx: list[int] = []
            next_pending_prompts: list[str] = []
            next_pending_dicts: list[dict] = []

            for local_i, content in enumerate(texts):
                orig_i = pending_idx[local_i]
                d = pending_dicts[local_i]

                try:
                    parsed = extract_json(content)
                    parsed = expand_translations_with_gender(d, parsed)
                    batch_outputs[orig_i] = parsed

                except Exception as e:
                    last_error = e
                    global_i = start + orig_i

                    print("\n--- JSON extraction failed (batched) ---")
                    print(f"global_i={global_i} attempt={attempt+1}/{max_retries}")
                    print(f"error: {e!r}")
                    print(f"len(content)={len(content)}")
                    print("TAIL:", content[-300:])
                    print("--- Will retry this item ---\n")

                    next_pending_idx.append(orig_i)
                    next_pending_prompts.append(pending_prompts[local_i])
                    next_pending_dicts.append(d)

            if not next_pending_idx:
                break

            pending_idx = next_pending_idx
            pending_prompts = next_pending_prompts
            pending_dicts = next_pending_dicts

            if retry_sleep_s > 0:
                time.sleep(retry_sleep_s)

        still_bad = [i for i, o in enumerate(batch_outputs) if o is None]
        if still_bad:
            print(f"\nBatch had {len(still_bad)} stubborn items; trying single-item fallback...\n")

        for orig_i in still_bad:
            d = batch[orig_i]
            prompt = full_prompts[orig_i]

            try:
                content = (generate_many([prompt])[0] or "").strip()
                parsed = extract_json(content)
                parsed = expand_translations_with_gender(d, parsed)
                batch_outputs[orig_i] = parsed

            except Exception as e:
                last_error = e


                msg = f"FINAL JSON FAIL global_i={global_i}: {e!r}"
                print("\n" + msg)

                if on_final_fail == "raise":
                    raise ValueError(msg) from e

                if on_final_fail == "skip":
                    batch_outputs[orig_i] = {"__skipped__": True, "__error__": msg}

                else:  
                    raw_tail = (content[-300:] if "content" in locals() else "")
                    batch_outputs[orig_i] = {
                        "__error__": msg,
                        "__raw_tail__": raw_tail,
                    }


        if any(o is None for o in batch_outputs):
            raise ValueError(f"Internal error: some outputs still None. last_error={last_error!r}")

        results.extend([o for o in batch_outputs if o is not None])

    return results