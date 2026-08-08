from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    EMBEDDINGS_CACHE_DIR,
    LEGAL_REFERENCE_COLUMN,
    MODEL_EMBEDDING_BATCH_SIZE,
    RAW_CASE_TEXT_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    TRAIN_DATA_PATH,
)
from src.text_preprocessing import simple_tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback: lexical (Jaccard) similarity
# ---------------------------------------------------------------------------

def lexical_similarity(left_text: str, right_text: str) -> float:
    left_tokens = set(simple_tokenizer(left_text))
    right_tokens = set(simple_tokenizer(right_text))

    if not left_tokens or not right_tokens:
        return 0.0

    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return overlap / union


# ---------------------------------------------------------------------------
# Embedding cache helpers
# ---------------------------------------------------------------------------

def _dataset_fingerprint(texts: list[str]) -> str:
    """Deterministic hash of the sorted case texts for cache invalidation."""
    joined = "\n".join(sorted(texts))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "train_embeddings.npy", cache_dir / "cache_meta.json"


def _load_cached_embeddings(
    cache_dir: Path, current_fingerprint: str
) -> np.ndarray | None:
    emb_path, meta_path = _cache_paths(cache_dir)
    if not emb_path.exists() or not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    if meta.get("fingerprint") != current_fingerprint:
        logger.info("Embedding cache fingerprint mismatch — rebuilding.")
        return None

    return np.load(emb_path)


def _save_cached_embeddings(
    cache_dir: Path, embeddings: np.ndarray, fingerprint: str
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path, meta_path = _cache_paths(cache_dir)
    np.save(emb_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"fingerprint": fingerprint, "shape": list(embeddings.shape)},
            fh,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def _get_candidate_embeddings(
    classifier, candidate_texts: list[str]
) -> np.ndarray:
    """Return cached — or freshly computed — embeddings for candidate texts."""
    fingerprint = _dataset_fingerprint(candidate_texts)
    cached = _load_cached_embeddings(EMBEDDINGS_CACHE_DIR, fingerprint)
    if cached is not None:
        logger.info("Using cached training-case embeddings (%s vectors).", cached.shape[0])
        return cached

    logger.info(
        "Computing BERT embeddings for %d training cases (batch_size=%d) …",
        len(candidate_texts),
        MODEL_EMBEDDING_BATCH_SIZE,
    )
    embeddings = classifier.encode(candidate_texts, batch_size=MODEL_EMBEDDING_BATCH_SIZE)
    _save_cached_embeddings(EMBEDDINGS_CACHE_DIR, embeddings, fingerprint)
    return embeddings


def find_best_legal_reference(query_text: str, predicted_class: str):
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    filtered_df = train_df[train_df[TARGET_COLUMN] == predicted_class].copy()

    if filtered_df.empty:
        return {
            "legal_reference": "",
            # "matched_case_text": "",
            "similarity_score": 0.0,
        }

    candidate_texts = filtered_df[TEXT_COLUMN].astype(str).tolist()

#Cosine Similarity and Jaccard
    try:
        from src.inference_utils import load_model

        classifier = load_model()
        candidate_embeddings = _get_candidate_embeddings(classifier, candidate_texts)
        query_embedding = classifier.encode(
            [query_text], batch_size=1
        )
        similarity_scores = cosine_similarity(
            query_embedding, candidate_embeddings
        ).flatten()
    except Exception as exc:
        logger.warning(
            "BERT embedding similarity failed (%s), falling back to lexical.",
            exc,
        )
        similarity_scores = np.array(
            [lexical_similarity(query_text, t) for t in candidate_texts]
        )

    best_idx = int(similarity_scores.argmax())
    best_case = filtered_df.iloc[best_idx]

    return {
        "legal_reference": str(best_case.get(LEGAL_REFERENCE_COLUMN, "")),
        # "matched_case_text": str(best_case.get(RAW_CASE_TEXT_COLUMN, best_case.get(TEXT_COLUMN, ""))),
        "similarity_score": float(similarity_scores[best_idx]),
    }
