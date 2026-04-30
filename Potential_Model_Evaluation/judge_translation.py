from openai import OpenAI
import os
from sentence_extractor import judge_pair, load_pairs, load_all_parallel_corpora, judge_model_translation
from collections import defaultdict
import json

"""Evaluate translation pairs using an LLM judge, aggregate metric scores, and print overall averages."""

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

raw_items = load_all_parallel_corpora()
pairs = load_pairs(raw_items)

model_scores = {}
JUDGE_MODEL_NAME = "gpt-5.2" 


counter = 0
score_sums = defaultdict(float)
score_counts = defaultdict(int)

for pair in pairs:
    counter += 1

    score = judge_pair(
        client, JUDGE_MODEL_NAME, pair
    )  
    print("pair")

    for k, v in score.items():
        score_sums[k] += v
        score_counts[k] += 1

overall_score = {
    k: score_sums[k] / score_counts[k]
    for k in score_sums
}

print(json.dumps(overall_score, ensure_ascii=False, indent=2))


