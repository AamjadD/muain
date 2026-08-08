import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "cases.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_cases.csv"
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "cases_train.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "cases_test.csv"

MODEL_OUTPUT_DIR = BASE_DIR / "model_outputs"
MODEL_DIR = MODEL_OUTPUT_DIR / "model"
EVALUATION_OUTPUT_DIR = MODEL_OUTPUT_DIR / "evaluation"
MODEL_METADATA_PATH = MODEL_DIR / "metadata.json"
EMBEDDINGS_CACHE_DIR = MODEL_OUTPUT_DIR / "embeddings_cache"

TARGET_COLUMN = "القسم"
TEXT_COLUMN = "case_text"
LEGAL_REFERENCE_COLUMN = "السند الشرعي"
RAW_CASE_TEXT_COLUMN = "الدعوى"

#AraBERT Transformers
MODEL_NAME = os.getenv("MODEL_NAME", "aubmindlab/bert-base-arabertv02")
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "auto")
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH", "256"))
MODEL_NUM_EPOCHS = int(os.getenv("MODEL_NUM_EPOCHS", "3"))
MODEL_TRAIN_BATCH_SIZE = int(os.getenv("MODEL_TRAIN_BATCH_SIZE", "4"))
MODEL_EVAL_BATCH_SIZE = int(os.getenv("MODEL_EVAL_BATCH_SIZE", "8"))
MODEL_LEARNING_RATE = float(os.getenv("MODEL_LEARNING_RATE", "2e-5"))
MODEL_WEIGHT_DECAY = float(os.getenv("MODEL_WEIGHT_DECAY", "0.01"))
MODEL_EMBEDDING_BATCH_SIZE = int(os.getenv("MODEL_EMBEDDING_BATCH_SIZE", "8"))
