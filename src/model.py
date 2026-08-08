from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from src.text_preprocessing import prepare_transformer_text

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    torch = None
    Dataset = object
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


def ensure_model_dependencies() -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        missing.append("transformers")
    if missing:
        raise ImportError(
            "Model support requires the following packages: "
            + ", ".join(missing)
            + ". Install requirements.txt first."
        )


class ModelDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int] | None,
        tokenizer,
        max_length: int,
    ) -> None:
        ensure_model_dependencies()
        self.texts = [prepare_transformer_text(text) for text in texts]
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, object]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class ModelClassifier:
    def __init__(
        self,
        model,
        tokenizer,
        labels: Sequence[str],
        max_length: int,
        base_model_name: str,
        device: str | None = None,
    ) -> None:
        ensure_model_dependencies()
        self.model = model
        self.tokenizer = tokenizer
        self.labels = list(labels)
        self.max_length = max_length
        self.base_model_name = base_model_name
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: Sequence[str]) -> list[str]:
        ensure_model_dependencies()
        prepared_texts = [prepare_transformer_text(text) for text in texts]
        encoded = self.tokenizer(
            prepared_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits

        predicted_ids = logits.argmax(dim=-1).tolist()
        return [self.labels[label_id] for label_id in predicted_ids]

    def encode(self, texts: Sequence[str], batch_size: int = 8) -> "numpy.ndarray":
        """Extract L2-normalised [CLS] embeddings from the base BERT encoder.

        Returns a numpy array of shape ``(len(texts), hidden_size)`` that can
        be used directly with ``sklearn.metrics.pairwise.cosine_similarity``.
        """
        import numpy as np

        ensure_model_dependencies()

        base_model = getattr(
            self.model, self.model.config.model_type, None
        ) or getattr(self.model, "base_model", self.model)

        prepared = [prepare_transformer_text(t) for t in texts]
        all_embeddings: list["numpy.ndarray"] = []

        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = base_model(**encoded)
                cls_vectors = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_vectors.cpu().numpy())

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def save_pretrained(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        metadata = {
            "labels": self.labels,
            "max_length": self.max_length,
            "base_model_name": self.base_model_name,
        }
        with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, model_dir: Path) -> "ModelClassifier":
        ensure_model_dependencies()
        model_dir = Path(model_dir)
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing model metadata file: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return cls(
            model=model,
            tokenizer=tokenizer,
            labels=metadata["labels"],
            max_length=int(metadata["max_length"]),
            base_model_name=metadata.get("base_model_name", str(model_dir)),
        )
