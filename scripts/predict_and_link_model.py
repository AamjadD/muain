from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_utils import predict_case_class
from src.linking_utils import find_best_legal_reference


def predict_and_link(case_text: str):
    predicted_class = predict_case_class(case_text)
    link_result = find_best_legal_reference(case_text, predicted_class)
    legal_reference = link_result["legal_reference"]
    # matched_case_text = link_result["matched_case_text"]
    similarity_score = round(link_result["similarity_score"], 4)

    return {
        "predicted_class": predicted_class,
        "legal_reference": legal_reference,
        # "matched_case_text": matched_case_text,
        "similarity_score": similarity_score,
        # Camel-case aliases are kept for frontend/testing convenience.
        "predictedClass": predicted_class,
        "legalReference": legal_reference,
        # "matchedCaseText": matched_case_text,
        "similarityScore": similarity_score,
    }
