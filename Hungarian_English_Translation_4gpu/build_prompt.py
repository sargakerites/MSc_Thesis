def build_prompt(entry):
    """Construct a structured prompt from a sentence and its token annotations for translation guidance."""
    
    sentence = entry["sentence"]
    tokens = entry.get("tokens", [])

    token_lines = []

    for t in tokens:
        if t["pos"] == "PUNCT":
            continue

        defs_text = " | ".join(t["definitions"]) or "None"

        token_lines.append(
            f"- {t['word']} | lemma: {t['lemma']} | pos: {t['pos']} | defs: {defs_text}"
        )

    tokens_text = "\n".join(token_lines) if token_lines else "None available."

    prompt = f"""
Sentence:
{sentence}

Tokens:
{tokens_text}

Use the token information only as translation help.

Return ONLY valid JSON.
Do not output anything else.
""".strip()

    return prompt

