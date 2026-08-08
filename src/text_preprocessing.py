import re

ARABIC_STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "أن", "إن", "كان", "كانت", "يكون", "تكون",
    "هذا", "هذه", "ذلك", "تلك", "وقد", "ثم", "كما", "مع", "بعد", "قبل", "بين",
    "أو", "و", "ب", "ل", "ف", "ما", "لم", "لن", "له", "لها", "عليه", "عليها",
    "عند", "أمام", "ضمن", "حيث", "إذ", "اذا", "إذا", "كل", "بعض", "غير", "حتى",
}

def prepare_transformer_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = text.replace("ـ", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_arabic(text: str) -> str:
    text = prepare_transformer_text(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def simple_tokenizer(text: str) -> list[str]:
    text = normalize_arabic(text)
    tokens = text.split()
    return [t for t in tokens if t not in ARABIC_STOPWORDS and len(t) > 1]
