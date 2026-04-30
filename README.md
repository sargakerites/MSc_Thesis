# English–Hungarian Translation Pipeline

This repository implements a full pipeline for:
1. Hungarian sentence annotation
2. Potential models evaluation
3. Synthetic English translation generation
4. Fine-tuning a translation model
5. Final evaluation

---

## Repository Structure

- `Hungarian_Annotation/`
  - Core annotation pipeline for Hungarian text
- `Hungarian_English_Translation_4gpu/`
  - Synthetic English translation generation using vLLM
- `Fine_Tuning/Final/`
  - Dataset splitting and model fine-tuning
- `Evaluation/`
  - Quantitative evaluation scripts and results
- `Potential_Model_Evaluation/`
  - Judgment-based evaluation of potential models

---

## 1. Annotation

Purpose: enrich Hungarian sentences with linguistic metadata and dictionary definitions.

Key files:
- `Hungarian_Annotation/annotate_data.py`
  - Annotates sentences using `huspacy`
  - Builds token-level entries with `lemma`, `pos`, and dictionary `definitions`
  - Deduplicates sentences using MinHash/LSH
- `Hungarian_Annotation/eksz_definitions_only.json`
  - Dictionary definitions used for annotation
- `Hungarian_Annotation/extract_hungarian_sentences.py`
  - Extracts raw Hungarian sentences from source text files
- `Hungarian_Annotation/extract_elte_rk_sentences.py`
  - Extracts Hungarian sentences from the ELTE RK corpus
- `Hungarian_Annotation/fill_missing_annotated_sentences.py`
  - Fills gaps in the annotated dataset 

Run scripts:
- `Hungarian_Annotation/run_annotation_ppe.sh`
- `Hungarian_Annotation/extract_elte_rk_sentences.py`
- `Hungarian_Annotation/run_elte_rk_annotation.sh`
- `Hungarian_Annotation/fill_missing_annotated_sentences.py`

Outputs:
- Annotated JSON files containing:
  - `sentence`
  - `tokens` with `word`, `lemma`, `pos`, `definitions`

---

## 2. Translation

Purpose: generate synthetic English translations for annotated Hungarian sentences.

Key files:
- `Hungarian_English_Translation_4gpu/translation.py`
  - Core translation logic
  - Uses a vLLM-backed model to generate English text
- `Hungarian_English_Translation_4gpu/add_translation.py`
  - Applies translations to annotated sentence chunks
- `Hungarian_English_Translation_4gpu/parallel_corpus_generation.py`
  - Batch driver for chunk-wise translation
- `Hungarian_English_Translation_4gpu/run_translation_4gpu_array.sh`
  - SLURM script for GPU-backed parallel execution
- `Hungarian_English_Translation_4gpu/build_prompt.py`
  - Builds the vLLM prompt from a sentence and token metadata, enforcing JSON-only output
- `Hungarian_English_Translation_4gpu/schema.py`
  - Defines the expected JSON schema for translation output and extracts JSON from model text
- `Hungarian_English_Translation_4gpu/token_builder.py`
  - Converts word/lemma/pos/definition arrays into structured token objects
- `Hungarian_English_Translation_4gpu/pronoun_expander.py`
  - Expands pronoun variants and gendered alternatives in generated English translations

Outputs:
- Synthetic parallel corpus files:
  - `parallel_corpus_*.json`

---

## 3. Model Evaluation

Purpose: compare translation model candidates before generating the synthetic corpus, and later support the final evaluation phase.

Key files:
- `Potential_Model_Evaluation/judge_translation.py`
  - Judges candidate translations and computes semantic quality scores
- `Potential_Model_Evaluation/model_order.py`
  - Ranks models by their semantic evaluation results
- `Potential_Model_Evaluation/model_scores.json`
  - Stores scores and ranking data for candidate models
- `Potential_Model_Evaluation/sentence_extractor.py`
  - Extracts representative sentences for model comparison

Outputs:
- Final scores of the potential models:
  - `model_scores.json`

---

## 4. Fine-Tuning

Purpose: fine-tune the base translation model on synthetic parallel data.

Key files:
- `Fine_Tuning/Final/Load_and_Split_the_Data.py`
  - Splits the generated synthetic dataset into `train`, `val`, and `test`
- `Fine_Tuning/Final/train.py`
  - Loads model, tokenizes data, and trains with `Seq2SeqTrainer`
  - Uses `accelerate`, Hugging Face `transformers`, and evaluation metrics
- `Fine_Tuning/Final/run_split.sh`
- `Fine_Tuning/Final/run_train.sh`

Training details:
- Base model: `NYTK/translation-nllb-200-3.3B-multi12-hungarian`
- Uses:
  - `bf16`
  - gradient checkpointing
  - large effective batch size via gradient accumulation
  - periodic evaluation and checkpoint saving



---

## 5. Final Evaluation

Purpose: evaluate the fine-tuned model on both synthetic split data and benchmark data.

Key files:
- `Evaluation/evaluate.py`
  - Loads a model and dataset, generates Hungarian translations, and computes metrics like BLEU, spBLEU, chrF, chrF++, ROUGE-1, ROUGE-2, ROUGE-L
- `Evaluation/run_eval.sh`
  - SLURM job script that runs `evaluate.py` for final model evaluation 
- `Evaluation/semantic_evaluation.py`
  - Performs semantic evaluation using an OpenAI-style judge
- `Evaluation/run_semantic_eval.sh`
  - Runs the semantic evaluation pipeline
- `Evaluation/Semantic_Eval_Outputs/`
  - Stores semantic evaluation outputs and summary files

Typical evaluation flow:
- Load models:
  - base model
  - fine-tuned model
- Load dataset:
  - synthetic test split
  - FLORES+ devtest
- Generate model predictions


---

## Notes

- The repository uses SLURM job scripts for large-scale GPU / CPU workloads.
- Many scripts assume a specific HPC file layout and local caching paths.
- The actual data directories are not included in the README, but the structure is:
  - annotated Hungarian input → translated synthetic corpus → train/val/test splits → model fine-tuning → evaluation


