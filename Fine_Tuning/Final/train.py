import os
import glob
import json
import time
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback,
)

import evaluate
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Config:
    """Store training, evaluation, dataset, and generation parameters in a single configuration object."""

    model_name: str
    train_dir: str
    val_dir: str
    test_dir: str
    output_dir: str
    source_lang: str
    target_lang: str
    max_source_length: int
    max_target_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: float
    warmup_ratio: float
    save_total_limit: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    fp16: bool
    bf16: bool
    gradient_checkpointing: bool
    seed: int
    max_train_samples: Optional[int]
    max_val_samples: Optional[int]
    max_test_samples: Optional[int]
    generation_num_beams: int


def parse_args():
    """Parse command-line arguments for model training, evaluation, and generation settings."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="NYTK/translation-nllb-200-3.3B-multi12-hungarian",
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        default=str(PROJECT_ROOT / "Training_Dataset_Split" / "train"),
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=str(PROJECT_ROOT / "Training_Dataset_Split" / "val"),
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=str(PROJECT_ROOT / "Training_Dataset_Split" / "test"),
    )

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--source_lang", type=str, default="eng_Latn")
    parser.add_argument("--target_lang", type=str, default="hun_Latn")

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=6.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)

    parser.add_argument("--generation_num_beams", type=int, default=5)

    return parser.parse_args()


def build_config(args) -> Config:
    """Create a Config object from parsed command-line arguments."""

    return Config(
        model_name=args.model_name,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        generation_num_beams=args.generation_num_beams,
    )


def print0(*args, **kwargs):
    """Print messages only from the main process in distributed training."""

    if PartialState().is_main_process:
        print(*args, **kwargs)


def get_json_files(folder: str):
    """Return sorted JSON files from a folder and raise an error if none are found."""

    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {folder}")
    return files


def load_split_datasets(cfg: Config):
    """Load train, validation, and test splits from JSON files into a Hugging Face DatasetDict."""

    train_files = get_json_files(cfg.train_dir)
    val_files = get_json_files(cfg.val_dir)
    test_files = get_json_files(cfg.test_dir)

    print0(f"Found {len(train_files)} train files")
    print0(f"Found {len(val_files)} val files")
    print0(f"Found {len(test_files)} test files")

    data_files = {
        "train": train_files,
        "validation": val_files,
        "test": test_files,
    }

    datasets = load_dataset("json", data_files=data_files)
    print0(datasets)
    return datasets


def maybe_subset(dataset, max_samples: Optional[int], split_name: str):
    """Optionally select a limited number of samples from a dataset split."""

    if max_samples is None:
        return dataset

    max_samples = min(max_samples, len(dataset))
    print0(f"Using subset for {split_name}: {max_samples} samples")
    return dataset.select(range(max_samples))


def preprocess_function(examples, tokenizer, max_source_length, max_target_length):
    """Tokenize source and target texts and prepare labels for sequence-to-sequence training."""

    model_inputs = tokenizer(
        examples["source"],
        max_length=max_source_length,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["target"],
        max_length=max_target_length,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_last_checkpoint_if_any(output_dir: str) -> Optional[str]:
    """Return the latest checkpoint directory from an output folder, if one exists."""

    if not os.path.isdir(output_dir):
        return None

    checkpoint_dirs = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    if not checkpoint_dirs:
        return None

    return checkpoint_dirs[-1]


class TimeMetricsCallback(TrainerCallback):
    """Trainer callback that logs elapsed training and evaluation time to a JSONL file."""

    def __init__(self, output_dir: str):
        """Initialize callback paths and start time for time-based metric logging."""

        self.output_dir = output_dir
        self.start_time = time.time()
        self.log_path = os.path.join(output_dir, "time_metrics.jsonl")

    def _write_row(self, row: dict):
        """Append one timing or metric row to the time metrics JSONL log file."""

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Record training log events with global step, epoch, elapsed time, and logged metrics."""

        if not state.is_local_process_zero or logs is None:
            return

        row = {
            "event": "log",
            "global_step": state.global_step,
            "epoch": state.epoch,
            "elapsed_time_sec": time.time() - self.start_time,
        }
        row.update(logs)
        self._write_row(row)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Record evaluation events with global step, epoch, elapsed time, and evaluation metrics."""

        if not state.is_local_process_zero or metrics is None:
            return

        row = {
            "event": "eval",
            "global_step": state.global_step,
            "epoch": state.epoch,
            "elapsed_time_sec": time.time() - self.start_time,
        }
        row.update(metrics)
        self._write_row(row)


bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
rouge = evaluate.load("rouge")


def compute_metrics(eval_preds, tokenizer):
    """Decode predictions and labels, then compute BLEU, chrF, and ROUGE metrics."""

    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.array(preds)
    labels = np.array(labels)

    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    labels = np.where(labels == -100, pad_id, labels)

    vocab_upper_bound = int(getattr(tokenizer, "vocab_size", 256000) - 1)

    preds = np.where(preds < 0, pad_id, preds)
    preds = np.where(preds > vocab_upper_bound, pad_id, preds)

    labels = np.where(labels < 0, pad_id, labels)
    labels = np.where(labels > vocab_upper_bound, pad_id, labels)

    decoded_preds = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    bleu_result = bleu.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels],
    )

    chrf_result = chrf.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )

    rouge_result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    return {
        "bleu": bleu_result["score"],
        "chrf": chrf_result["score"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
    }


def main():
    """Run the full fine-tuning pipeline: load model and data, tokenize splits, train, evaluate, and save results."""
    args = parse_args()
    cfg = build_config(args)
    state = PartialState()

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    print0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        src_lang=cfg.source_lang,
        tgt_lang=cfg.target_lang,
        use_fast=True,
        local_files_only=True,
    )

    print0("Loading model...")
    torch_dtype = torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else None)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(cfg.target_lang)
    model.config.forced_bos_token_id = forced_bos_token_id

    print0("Loading split datasets...")
    datasets = load_split_datasets(cfg)

    train_dataset = maybe_subset(datasets["train"], cfg.max_train_samples, "train")
    val_dataset = maybe_subset(datasets["validation"], cfg.max_val_samples, "validation")
    test_dataset = maybe_subset(datasets["test"], cfg.max_test_samples, "test")

    print0(f"Train examples: {len(train_dataset)}")
    print0(f"Validation examples: {len(val_dataset)}")
    print0(f"Test examples: {len(test_dataset)}")

    print0("Tokenizing train split...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(
            x, tokenizer, cfg.max_source_length, cfg.max_target_length
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )

    print0("Tokenizing validation split...")
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(
            x, tokenizer, cfg.max_source_length, cfg.max_target_length
        ),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation",
    )

    print0("Tokenizing test split...")
    tokenized_test = test_dataset.map(
        lambda x: preprocess_function(
            x, tokenizer, cfg.max_source_length, cfg.max_target_length
        ),
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        generation_max_length=cfg.max_target_length,
        generation_num_beams=cfg.generation_num_beams,
        logging_first_step=True,
        save_safetensors=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[TimeMetricsCallback(cfg.output_dir)],
    )

    last_checkpoint = get_last_checkpoint_if_any(cfg.output_dir)
    if last_checkpoint is not None:
        print0(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print0("Starting fresh training run.")
        trainer.train()

    print0("\nFinal validation loss-based evaluation:")
    val_loss_metrics = trainer.evaluate(eval_dataset=tokenized_val, metric_key_prefix="val")
    print0(val_loss_metrics)

    print0("\nFinal test loss-based evaluation:")
    test_loss_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
    print0(test_loss_metrics)

    print0("\nRunning final generation-based evaluation...")

    gen_training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        do_train=False,
        do_eval=False,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        report_to="none",
        predict_with_generate=True,
        generation_max_length=cfg.max_target_length,
        generation_num_beams=cfg.generation_num_beams,
        remove_unused_columns=False,
    )

    gen_trainer = Seq2SeqTrainer(
        model=trainer.model,
        args=gen_training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    print0("\nFinal generation-based validation metrics...")
    val_gen_metrics = gen_trainer.predict(tokenized_val, metric_key_prefix="val")
    print0(val_gen_metrics.metrics)

    print0("\nFinal generation-based test metrics...")
    test_gen_metrics = gen_trainer.predict(tokenized_test, metric_key_prefix="test")
    print0(test_gen_metrics.metrics)

    if state.is_main_process:
        print0(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
        print0(f"Best eval loss: {trainer.state.best_metric}")
        print0("Saving final model and tokenizer...")
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)

        metrics_path = os.path.join(cfg.output_dir, "final_metrics.json")
        final_metrics = {
            "val_loss_metrics": val_loss_metrics,
            "test_loss_metrics": test_loss_metrics,
            "val_generation_metrics": val_gen_metrics.metrics,
            "test_generation_metrics": test_gen_metrics.metrics,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)

        print0(f"Saved final metrics to: {metrics_path}")

    print0("Training finished.")


if __name__ == "__main__":
    main()