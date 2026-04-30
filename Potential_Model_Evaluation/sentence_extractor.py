from pathlib import Path
import json
import re
from openai import OpenAI
from typing import List, Dict
from collections import defaultdict


client = OpenAI()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_DIR = PROJECT_ROOT / "1000_translated_sentences"

def load_all_parallel_corpora():
    """Load and merge all parallel corpus JSON files from the configured corpus directory."""

    corpora = []
    for path in CORPUS_DIR.glob("parallel_corpus_*.json"):
        with open(path, "r", encoding="utf-8") as f:
            corpora.extend(json.load(f))
    return corpora

def clean_text(text: str) -> str:
    """Normalize text by collapsing repeated whitespace and trimming leading/trailing spaces."""

    return re.sub(r"\s+", " ", text).strip()

def extract_final_translation(raw_translation):
    """Extract the final translation string from a raw translation field."""

    if isinstance(raw_translation, list):
        raw_translation = raw_translation[0] if raw_translation else ""

    lines = [l.strip() for l in raw_translation.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def load_pairs(raw_items: List[Dict]) -> List[Dict[str, str]]:
    """Build cleaned Hungarian-English sentence pairs from raw parallel corpus items."""

    pairs = []

    for idx, item in enumerate(raw_items):
        if "sentence" not in item or "translation" not in item:
            continue

        hu = clean_text(item["sentence"])
        en = extract_final_translation(item["translation"])

        if not hu or not en:
            continue

        pairs.append({
            "hu": hu,
            "en": en,
        })

    return pairs

def judge_pair(client, hu: str, en: str) -> dict:
    """Evaluate one Hungarian-English translation pair with an LLM judge and return metric scores."""

    USER_TEMPLATE = """
Evaluate the following translation.

Hungarian sentence:
{hu}

English translation:
{en}

Return ONLY the JSON score object.
"""

    SYSTEM_PROMPT = """
You are a STRICT and CONSISTENT translation judge for Hungarian → English.

Your task is to evaluate the English translation of a Hungarian sentence using
THREE INDEPENDENT criteria.

CRITICAL RULES:
- Each criterion must be judged independently.
- Do NOT let one criterion influence another.
- Use ONLY the values: -1, 0, or 1.
- If unsure between two scores, choose the LOWER one.
- Output ONLY valid JSON.
- Do NOT include explanations, reasoning, or extra text.

CRITERIA DEFINITIONS:

1) content_accuracy (content / meaning match)
- Evaluate ONLY semantic meaning.
- Ignore grammar, style, and fluency.
- 1  = all meaning preserved, no omissions or additions
- 0  = minor nuance loss, but core meaning intact
- -1 = missing, distorted, or misleading meaning
- Do NOT assume hidden beneficiaries, implied employers, or social roles
unless they are explicitly stated in the Hungarian sentence.
- A score of -1 requires a CLEAR and UNAMBIGUOUS meaning error.
If the meaning is reasonably preserved, do NOT assign -1.


2) grammatical_correctness 
- Evaluate ONLY grammar and syntax.
- Ignore wording elegance and meaning accuracy.
- 1  = grammatically correct
- 0  = minor grammatical issues that do not confuse meaning
- -1 = serious grammar errors or broken sentence structure

3) fluency (naturalness and readability)
- Evaluate ONLY naturalness and flow.
- Ignore grammar errors already penalized above.
- Ignore content accuracy.
- 1  = natural, native-like English
- 0  = awkward but understandable
- -1 = very unnatural, robotic, or hard to read

OUTPUT FORMAT:
Return EXACTLY this JSON object, with NO additional keys:

{
  "content_accuracy": -1,
  "grammatical_correctness": -1,
  "fluency": -1
}

"""

    prompt = USER_TEMPLATE.format(hu=hu, en=en)

    response = client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return json.loads(response.output_text)

def build_model_evaluation_text(pairs):
    """Format multiple translation pairs into a numbered text block for model-level evaluation."""

    blocks = []
    for i, p in enumerate(pairs, 1):
        blocks.append(
            f"{i}. HU: {p['hu']}\n   EN: {p['en']}"
        )
    return "\n\n".join(blocks)


def judge_model_translation(client, model_name: str, pairs: list[dict]) -> dict:
    """Evaluate the overall quality of one model's translations using an LLM judge."""
    
    translation_block = build_model_evaluation_text(pairs)

    SYSTEM_PROMPT = """
You are a CALIBRATED translation judge for Hungarian → English.

Your task is to evaluate the OVERALL QUALITY of a full translation
produced by ONE model.

This is NOT an error-hunting task.
This is a QUALITY DISCRIMINATION task.

Score meanings (IMPORTANT):

1  = clearly strong, close to professional human translation
0  = typical machine translation quality with noticeable issues
-1 = clearly poor, unreliable, or misleading translation

Guidelines:

- Occasional or localized errors are NORMAL for machine translation.
- Assign -1 ONLY if problems are frequent, systematic, and harmful.
- Assign 1 ONLY if quality is consistently high throughout.
- 0 is the expected score for an average MT system.

BALANCE RULES (MANDATORY):
- You may assign -1 to AT MOST TWO dimensions.
- If all dimensions seem weak, select the two worst only.
- At least one dimension should usually be 0 unless quality is extreme.

Think comparatively:
- Compare against a typical strong MT system, NOT a human translator.

Output ONLY valid JSON.
No explanation.

"""
    USER_PROMPT = """
You are a STRICT and RADICAL translation judge for Hungarian → English.

Your task is to evaluate the OVERALL QUALITY of a full translation produced
by ONE model.

CRITICAL RULES:
- You are NOT allowed to return (0, 0, 0).
- If uncertain, PENALIZE.
- Judge against a professional human translation baseline.

Scoring meanings:

tartalmi_egyezes:
  1  = meaning consistently preserved, no systematic errors
  0  = noticeable losses or distortions in some places
 -1  = frequent or severe meaning errors

nyelvhelyesseg:
  1  = grammatically correct throughout
  0  = recurring minor grammar or usage issues
 -1  = frequent or serious grammar errors

fluency:
  1  = natural, consistent, professional English
  0  = uneven, awkward, or inconsistent style
 -1  = unnatural, stilted, or hard to read

PROCEDURE (MANDATORY):
1) Actively search for recurring errors.
2) If you find repeated issues → assign -1.
3) Use 0 ONLY if strengths and weaknesses clearly coexist.
4) At least ONE dimension must be 1 or -1.

Output ONLY valid JSON.
No explanation.
"""

    response = client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    model_name=model_name,
                    translation_block=translation_block
                ),
            },
        ],
        temperature=0,
    )

    return json.loads(response.output_text)
