from __future__ import annotations

from functools import lru_cache

from src.config import MODEL_DIR
from src.model import ModelClassifier
from src.text_preprocessing import prepare_transformer_text


@lru_cache(maxsize=1)
def load_model() -> ModelClassifier:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            "No trained model was found. Run scripts/train_model.py first."
        )

    return ModelClassifier.from_pretrained(MODEL_DIR)


def predict_case_class(case_text: str) -> str:
    normalized_text = prepare_transformer_text(case_text)
    if not normalized_text:
        raise ValueError("Case text is required for prediction.")

    model = load_model()
    return model.predict([normalized_text])[0]
