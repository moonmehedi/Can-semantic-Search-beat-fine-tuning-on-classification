from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_SPECS = {
    "ELECTRA": {
        "checkpoint": "google/electra-base-discriminator",
        "artifact_dir": "electra_augmented",
        "use_fast": True,
        "padding_side": "right",
    },
    "BERT": {
        "checkpoint": "bert-base-uncased",
        "artifact_dir": "bert_augmented",
        "use_fast": True,
        "padding_side": "right",
    },
    "DistilBERT": {
        "checkpoint": "distilbert/distilbert-base-uncased",
        "artifact_dir": "distilbert_augmented",
        "use_fast": True,
        "padding_side": "right",
    },
    "RoBERTa": {
        "checkpoint": "FacebookAI/roberta-base",
        "artifact_dir": "roberta_augmented",
        "use_fast": True,
        "padding_side": "right",
    },
    "XLNet": {
        "checkpoint": "xlnet/xlnet-base-cased",
        "artifact_dir": "xlnet_augmented",
        "use_fast": True,
        "padding_side": "left",
    },
    "DeBERTa": {
        "checkpoint": "microsoft/deberta-v3-base",
        "artifact_dir": "deberta_augmented",
        "use_fast": False,
        "padding_side": "right",
    },
    "ModernBERT": {
        "checkpoint": "answerdotai/ModernBERT-base",
        "artifact_dir": "modernbert_augmented",
        "use_fast": True,
        "padding_side": "right",
    },
}

LABEL_NAMES = {0: "Normal", 1: "BPD", 2: "Inconclusive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export misclassified BPDQAC test samples model by model."
    )
    parser.add_argument(
        "--dataset-path",
        default="experiments/dataset/augmented_dataset",
        help="Path to the Hugging Face DatasetDict saved with load_from_disk().",
    )
    parser.add_argument(
        "--artifact-root",
        default="model_artifacts",
        help="Root directory containing per-model fine-tuned checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/misclassified_exports",
        help="Directory where CSV exports will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_SPECS.keys()),
        help="Subset of models to export.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max length.",
    )
    parser.add_argument(
        "--print-head",
        type=int,
        default=10,
        help="How many misclassified rows per model to print to stdout.",
    )
    return parser.parse_args()


def resolve_saved_model_dir(artifact_root: Path, model_name: str) -> Path:
    artifact_dir = artifact_root / MODEL_SPECS[model_name]["artifact_dir"]
    if (artifact_dir / "config.json").exists():
        return artifact_dir

    checkpoint_dirs = sorted(
        [path for path in artifact_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if checkpoint_dirs:
        return checkpoint_dirs[-1]

    raise FileNotFoundError(f"No saved checkpoint found for {model_name} in {artifact_dir}")


def tokenize_batch(batch: dict[str, list[str]], tokenizer, max_length: int) -> dict[str, list[int]]:
    merged_text = [
        f"question: {question} [SEP] answer: {answer}"
        for question, answer in zip(batch["question"], batch["answer"])
    ]
    return tokenizer(
        merged_text,
        truncation=True,
        max_length=max_length,
    )


def predict_labels(model_name: str, test_dataset, artifact_root: Path, max_length: int) -> np.ndarray:
    spec = MODEL_SPECS[model_name]
    saved_model_dir = resolve_saved_model_dir(artifact_root, model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        saved_model_dir if saved_model_dir.exists() else spec["checkpoint"],
        use_fast=spec["use_fast"],
    )
    tokenizer.padding_side = spec["padding_side"]

    tokenized_test = test_dataset.map(
        tokenize_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
    )
    tokenized_test = tokenized_test.remove_columns(
        [col for col in tokenized_test.column_names if col not in {"input_ids", "attention_mask", "token_type_ids", "label"}]
    )
    tokenized_test.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(saved_model_dir)
    if model_name == "XLNet":
        model.config.pad_token_id = tokenizer.pad_token_id

    eval_args = TrainingArguments(
        output_dir=str(artifact_root / "_misclassified_eval"),
        per_device_eval_batch_size=2 if model_name == "XLNet" else 8,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    try:
        pred_output = trainer.predict(tokenized_test)
        preds = pred_output.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        if preds.ndim == 1:
            return preds.astype(int)
        return np.argmax(preds, axis=-1)
    finally:
        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def export_model_errors(
    model_name: str,
    test_dataset,
    artifact_root: Path,
    output_dir: Path,
    max_length: int,
    print_head: int,
) -> dict[str, object]:
    pred_labels = predict_labels(model_name, test_dataset, artifact_root, max_length)
    actual_labels = np.array(test_dataset["label"])

    rows = []
    for idx, (actual, pred) in enumerate(zip(actual_labels, pred_labels)):
        if int(actual) == int(pred):
            continue
        sample = test_dataset[int(idx)]
        rows.append(
            {
                "row_index": int(idx),
                "question_id": sample.get("question_id"),
                "question": sample.get("question"),
                "answer": sample.get("answer"),
                "actual_label_id": int(actual),
                "predicted_label_id": int(pred),
                "actual_label": LABEL_NAMES[int(actual)],
                "predicted_label": LABEL_NAMES[int(pred)],
            }
        )

    error_df = pd.DataFrame(rows)
    output_path = output_dir / f"{model_name.lower()}_misclassified.csv"
    error_df.to_csv(output_path, index=False)

    summary = {
        "Model": model_name,
        "Total Test Samples": len(test_dataset),
        "Misclassified": int(len(error_df)),
        "Export Path": str(output_path),
    }

    print(f"\n{model_name}")
    print(json.dumps(summary, indent=2))
    if not error_df.empty and print_head > 0:
        print(error_df.head(print_head).to_string(index=False))

    return summary


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    artifact_root = Path(args.artifact_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(dataset_path))
    test_dataset = dataset["test"]

    invalid_models = [model for model in args.models if model not in MODEL_SPECS]
    if invalid_models:
        raise ValueError(f"Unknown model names: {invalid_models}")

    summaries = []
    for model_name in args.models:
        try:
            summaries.append(
                export_model_errors(
                    model_name=model_name,
                    test_dataset=test_dataset,
                    artifact_root=artifact_root,
                    output_dir=output_dir,
                    max_length=args.max_length,
                    print_head=args.print_head,
                )
            )
        except Exception as exc:
            print(f"\n{model_name}\nFailed: {exc}")

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "misclassified_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
