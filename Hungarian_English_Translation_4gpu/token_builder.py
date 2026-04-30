def build_tokens(entry):
    """Construct token-level annotations from word, lemma, POS, and definition fields, and attach them to the entry."""
    
    words = entry.get("words", [])
    lemmas = entry.get("lemmas", [])
    pos_tags = entry.get("pos_tags", [])
    defs = entry.get("definitions", {})

    tokens = []

    for i, word in enumerate(words):
        lemma = lemmas[i] if i < len(lemmas) else ""
        pos = pos_tags[i] if i < len(pos_tags) else ""

        tokens.append({
            "word": word,
            "lemma": lemma,
            "pos": pos,
            "definitions": defs.get(lemma, [])
        })

    entry["tokens"] = tokens

    entry.pop("words", None)
    entry.pop("lemmas", None)
    entry.pop("pos_tags", None)
    entry.pop("definitions", None)

    return entry
