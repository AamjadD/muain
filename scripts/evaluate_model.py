from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVALUATION_OUTPUT_DIR, TARGET_COLUMN, TEST_DATA_PATH, TEXT_COLUMN
from src.data_loader import load_supervised_csv
from src.inference_utils import load_model


def build_evaluation_result(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

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


def main() -> None:
    model = load_model()
    test_df = load_supervised_csv(TEST_DATA_PATH, TEXT_COLUMN, TARGET_COLUMN)
    predicted_labels = model.predict(test_df[TEXT_COLUMN].tolist())
    evaluation_result = build_evaluation_result(test_df[TARGET_COLUMN].tolist(), predicted_labels)
    prediction_df = pd.DataFrame(
        {
            "case_text": test_df[TEXT_COLUMN].tolist(),
            "actual_class": test_df[TARGET_COLUMN].tolist(),
            "predicted_class": predicted_labels,
        }
    )

    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                key: value
                for key, value in evaluation_result.items()
                if key != "classification_report"
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with open(EVALUATION_OUTPUT_DIR / "classification_report.json", "w", encoding="utf-8") as handle:
        json.dump(evaluation_result["classification_report"], handle, ensure_ascii=False, indent=2)
    prediction_df.to_csv(EVALUATION_OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    print(json.dumps({k: v for k, v in evaluation_result.items() if k != "classification_report"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
