import json

def extract_json(text):
    """Extract and parse the first valid JSON object found within a text string."""

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output.")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                json_str = text[start:i+1]
                return json.loads(json_str)

    raise ValueError("No complete JSON object found.")


"""Define the expected JSON schema for structured sentence translation outputs with reasoning and quality ranking."""

SCHEMA_TEXT = """
{
  "name": "sentence_translations_with_reasoning",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "original_sentence": {
        "type": "string",
        "description": "The original sentence to be translated."
      },
      "reasoning": {
        "type": "string",
        "description": "A brief explanation of considerations in producing the translations and their anticipated quality."
      },
      "word_by_word": {
        "type": "array",
        "description": "A sequence mapping each word (or phrase) in the original sentence to one or more literal translation options in the target language. This is shared across all translation options.",
        "items": {
          "type": "object",
          "properties": {
            "source_word": {
              "type": "string",
              "description": "The word or phrase from the original sentence."
            },
            "translated_words": {
              "type": "array",
              "description": "One or more literal translation options for the corresponding source word or phrase.",
              "minItems": 1,
              "items": {
                "type": "string"
              }
            }
          },
          "required": [
            "source_word",
            "translated_words"
          ],
          "additionalProperties": false
        }
      },
      "translations": {
        "type": "array",
        "description": "At least three alternative full sentence translations (no per-option word-by-word; see top-level word_by_word).",
        "minItems": 3,
        "items": {
          "type": "object",
          "properties": {
            "translation": {
              "type": "string",
              "description": "Full target language translation of the sentence."
            }
          },
          "required": [
            "translation"
          ],
          "additionalProperties": false
        }
      },
      "reason_about_quality": {
        "type": "string",
        "description": "This is a reasoning about the relative quality of the translations following the fluency and content matching priciples."
      },
      "quality_order": {
        "type": "array",
        "description": "Indices of the translations array, ordered from best (highest quality) to worst (lowest quality).",
        "items": {
          "type": "integer",
          "description": "Index into the translations array (starting at 0).",
          "minimum": 0
        }
      }
    },
    "required": [
      "original_sentence",
      "reasoning",
      "word_by_word",
      "translations",
      "reason_about_quality",
      "quality_order"
    ],
    "additionalProperties": false
  }
}
""".strip()


