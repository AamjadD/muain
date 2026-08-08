# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ModelClassifier, ModelDataset, ensure_model_dependencies
from src.config import (
    MODEL_DEVICE,
    MODEL_DIR,
    MODEL_EVAL_BATCH_SIZE,
    MODEL_LEARNING_RATE,
    MODEL_MAX_LENGTH,
    MODEL_NAME,
    MODEL_NUM_EPOCHS,
    MODEL_TRAIN_BATCH_SIZE,
    MODEL_WEIGHT_DECAY,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEXT_COLUMN,
    TRAIN_DATA_PATH,
)
from src.data_loader import load_supervised_csv


def set_seed(seed: int = 42) -> None:
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_evaluation_result(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    labels = sorted(set(y_true))
    return {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }


def train_epoch(model, data_loader, optimizer, device) -> float:
    import torch

    model.train()
    total_loss = 0.0

    for batch in data_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(len(data_loader), 1)


def collect_predictions(model, data_loader, labels: list[str], device) -> tuple[list[str], list[str]]:
    import torch

    model.eval()
    true_ids: list[int] = []
    predicted_ids: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            true_ids.extend(batch["labels"].tolist())

            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            logits = model(**model_inputs).logits
            predicted_ids.extend(logits.argmax(dim=-1).cpu().tolist())

    y_true = [labels[label_id] for label_id in true_ids]
    y_pred = [labels[label_id] for label_id in predicted_ids]
    return y_true, y_pred


def resolve_device():
    import torch

    requested = MODEL_DEVICE.strip().lower()
    cuda_available = torch.cuda.is_available()

    if requested == "auto":
        return torch.device("cuda" if cuda_available else "cpu")

    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "MODEL_DEVICE=cuda was requested, but CUDA is not available. "
                "Check the NVIDIA driver/runtime and confirm torch.cuda.is_available() is True."
            )
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError("MODEL_DEVICE must be one of: auto, cuda, cpu")


def main() -> None:
    ensure_model_dependencies()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(42)

    train_df = load_supervised_csv(TRAIN_DATA_PATH, TEXT_COLUMN, TARGET_COLUMN)
    test_df = load_supervised_csv(TEST_DATA_PATH, TEXT_COLUMN, TARGET_COLUMN)

    labels = sorted(train_df[TARGET_COLUMN].unique().tolist())
    unseen_test_labels = sorted(set(test_df[TARGET_COLUMN]) - set(labels))
    if unseen_test_labels:
        raise ValueError(f"Test set contains unseen labels: {unseen_test_labels}")

    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

#AraBERT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        label2id=label_to_id,
        id2label=id_to_label,
    )

    train_dataset = ModelDataset(
        texts=train_df[TEXT_COLUMN].tolist(),
        labels=[label_to_id[label] for label in train_df[TARGET_COLUMN].tolist()],
        tokenizer=tokenizer,
        max_length=MODEL_MAX_LENGTH,
    )
    test_dataset = ModelDataset(
        texts=test_df[TEXT_COLUMN].tolist(),
        labels=[label_to_id[label] for label in test_df[TARGET_COLUMN].tolist()],
        tokenizer=tokenizer,
        max_length=MODEL_MAX_LENGTH,
    )

    train_loader = DataLoader(train_dataset, batch_size=MODEL_TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=MODEL_EVAL_BATCH_SIZE, shuffle=False)

    device = resolve_device()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MODEL_LEARNING_RATE,
        weight_decay=MODEL_WEIGHT_DECAY,
    )

    print(f"Training model: {MODEL_NAME}")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Device: {device} | Max length: {MODEL_MAX_LENGTH}")

    training_history = []
    for epoch in range(MODEL_NUM_EPOCHS):
        average_loss = train_epoch(model, train_loader, optimizer, device)
        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(float(average_loss), 4),
            }
        )
        print(f"Epoch {epoch + 1}/{MODEL_NUM_EPOCHS} - train_loss: {average_loss:.4f}")

    y_true, y_pred = collect_predictions(model, test_loader, labels, device)
    evaluation_result = build_evaluation_result(y_true, y_pred)

    trained_model = ModelClassifier(
        model=model,
        tokenizer=tokenizer,
        labels=labels,
        max_length=MODEL_MAX_LENGTH,
        base_model_name=MODEL_NAME,
        device=str(device),
    )
    trained_model.save_pretrained(MODEL_DIR)

    with open(MODEL_DIR / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_model_name": MODEL_NAME,
                "device": str(device),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "training_history": training_history,
                "held_out_evaluation": evaluation_result,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("Training complete. Saved model.")


if __name__ == "__main__":
    main()
