import re
from itertools import product

GENDER_NEUTRAL_HU = {
    "ő", "őt", "neki", "nála", "vele",
    "magát", "magának", "magával"
}

PRONOUN_MAP = {
    "he": ("he", "she"),
    "she": ("he", "she"),
    "him": ("him", "her"),
    "her": ("him", "her"),
    "his": ("his", "her"),
    "hers": ("his", "hers"),
    "himself": ("himself", "herself"),
    "herself": ("himself", "herself"),
    "he'll": ("he'll", "she'll"),
    "she'll": ("he'll", "she'll"),
}

PRONOUN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in PRONOUN_MAP.keys()) + r")\b",
    re.IGNORECASE,
)

ALT_PAIR_MAP = {
    "he/she": ("he", "she"),
    "she/he": ("he", "she"),
    "him/her": ("him", "her"),
    "her/him": ("him", "her"),
    "his/her": ("his", "her"),
    "her/his": ("his", "her"),
    "himself/herself": ("himself", "herself"),
    "herself/himself": ("himself", "herself"),
    "he'll/she'll": ("he'll", "she'll"),
    "she'll/he'll": ("he'll", "she'll"),
}

ALT_PAIR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ALT_PAIR_MAP.keys()) + r")\b",
    re.IGNORECASE,
)


def contains_gender_neutral_pronoun(tokens):
    """Check whether the input token annotations contain a Hungarian gender-neutral pronoun."""

    for t in tokens:
        lemma = t.get("lemma", "")
        if lemma and lemma.lower() in GENDER_NEUTRAL_HU:
            return True
    return False


def translation_contains_gender_pronoun(translations):
    """Check whether any generated translation contains gendered English pronouns or explicit pronoun alternatives."""

    for item in translations:
        text = item.get("translation", "")
        if PRONOUN_PATTERN.search(text) or ALT_PAIR_PATTERN.search(text):
            return True
    return False


def _preserve_case(original: str, replacement: str) -> str:
    """Apply the capitalization pattern of the original word to the replacement word."""

    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement.capitalize()
    return replacement


def replace_with_gender(text: str, gender: str = "male") -> str:
    """Replace gendered pronouns in a text with either male or female pronoun variants."""

    def repl(match):
        word = match.group(0)
        low = word.lower()

        male, female = PRONOUN_MAP[low]
        replacement = male if gender == "male" else female
        return _preserve_case(word, replacement)

    return PRONOUN_PATTERN.sub(repl, text)


def expand_explicit_alternatives(text: str) -> list[str]:
    """
    Expands explicit forms like:
      he/she, him/her, his/her
    into distinct sentence variants.

    Example:
      'he/she said his/her car is here'
      ->
      [
        'he said his car is here',
        'she said her car is here'
      ]

    If multiple explicit alternative pairs are present, this creates
    all combinations.
    """
    matches = list(ALT_PAIR_PATTERN.finditer(text))
    if not matches:
        return [text]

    variants = []

    for choices in product([0, 1], repeat=len(matches)):
        pieces = []
        last = 0

        for match, choice in zip(matches, choices):
            pieces.append(text[last:match.start()])
            original = match.group(0)
            options = ALT_PAIR_MAP[original.lower()]
            replacement = _preserve_case(original, options[choice])
            pieces.append(replacement)
            last = match.end()

        pieces.append(text[last:])
        variants.append("".join(pieces))

    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            out.append(v)
            seen.add(v)

    return out


def expand_translation_text(text: str, input_tokens: list[dict]) -> list[str]:
    """
    Expansion logic:
    1) First expand explicit alternatives like he/she.
    2) Then, if Hungarian source is gender-neutral and the resulting text
       still contains gendered pronouns, create male/female variants.
    """
    explicit_variants = expand_explicit_alternatives(text)

    need_gender_expansion = contains_gender_neutral_pronoun(input_tokens)

    final_variants = []
    seen = set()

    for variant in explicit_variants:
        if need_gender_expansion and PRONOUN_PATTERN.search(variant):
            male = replace_with_gender(variant, "male")
            female = replace_with_gender(variant, "female")

            for v in (male, female):
                if v not in seen:
                    final_variants.append(v)
                    seen.add(v)
        else:
            if variant not in seen:
                final_variants.append(variant)
                seen.add(variant)

    return final_variants


def expand_translations_with_gender(input_data, translation_data):
    """Expand translation candidates with gender-specific variants when gender ambiguity is detected."""
    
    translations = translation_data.get("translations", [])
    input_tokens = input_data.get("tokens", [])

    need_expansion = (
        contains_gender_neutral_pronoun(input_tokens)
        or translation_contains_gender_pronoun(translations)
    )

    if not need_expansion:
        return translation_data

    new_translations = []
    seen = set()

    for item in translations:
        text = item.get("translation", "")
        variants = expand_translation_text(text, input_tokens)

        for variant in variants:
            if variant not in seen:
                new_translations.append({"translation": variant})
                seen.add(variant)

    translation_data["translations"] = new_translations
    return translation_data