import json
from pathlib import Path

"""Load model evaluation scores, compute overall averages per model, rank them, and print results."""

json_path = Path(__file__).parent / "model_scores.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

model_scores = {}

for model, scores in data.items():
    overall = sum(scores.values()) / len(scores)
    model_scores[model] = overall

ranked = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

for model, score in ranked:
    print(f"{model}: {score:.3f}")


"""
google/gemma-3-27b-it: 0.55
zai-org/GLM-4.5-Air-FP8: 0.479
Qwen/Qwen3-Next-80B-A3B-Instruct: 0.475
meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo: 0.473
Qwen/Qwen3-235B-A22B-Instruct-2507-tput: 0.400
nvidia/NVIDIA-Nemotron-Nano-9B-v2: -0.267
"""