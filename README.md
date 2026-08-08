Muain Legal Assistant

Muain Legal Assistant is an Arabic legal-case assistant built around two explicit components:

AraBERT for case classification.
Vector similarity for legal reference linking.

Pipeline

1. Case classification with AraBERT

The classifier is trained with the pretrained Arabic transformer checkpoint:

aubmindlab/bert-base-arabertv02

During training and inference, the project fine-tunes AraBERT as a sequence-classification model to predict the case section/category (القسم).

2. Legal reference linking with vectors

After predicting the case class, the system retrieves the most relevant legal reference from training cases in the same class by:

Encoding the query case and candidate cases into dense AraBERT embeddings.
Using the [CLS] vector from the base encoder.
L2-normalizing the vectors.
Ranking candidate cases with cosine similarity.

The best-matching case provides:

legal_reference
<!-- - `matched_case_text` --> 
similarity_score

Computed training-case embeddings are cached under model_outputs/embeddings_cache/ for faster repeated retrieval. If embedding-based retrieval fails, the system falls back to lexical Jaccard similarity.

Run Order

Create virtual enviroment:

python -m venv .venv

Activate virtual enviroment:

Linux/macOS:

source .venv/bin/activate

Windows:

.venv\Scripts\activate

Install requirements:

pip install -r requirements.txt

Prepare datasets:

Linux/macOS:

python scripts/prepare_cases_datasets.py

Windows:

python scripts\prepare_cases_datasets.py

Train the AraBERT classifier:

Linux/macOS:

python scripts/train_model.py

Windows:

python scripts\train_model.py

This saves the trained model under model_outputs/model/.

Review evaluation:

Linux/macOS:

python scripts/evaluate_model.py

Windows:

python scripts\evaluate_model.py

Run the demo API:

Linux/macOS:

python scripts/api_server.py

Windows:

python scripts\api_server.py

Then open http://127.0.0.1:5000

Configuration Notes

Default pretrained model: aubmindlab/bert-base-arabertv02
Main settings can be changed with environment variables such as MODEL_NAME, MODEL_MAX_LENGTH, MODEL_NUM_EPOCHS, and MODEL_EMBEDDING_BATCH_SIZE
To force GPU training on a GPU-enabled machine, run with MODEL_DEVICE=cuda
If MODEL_DEVICE=cuda is set and CUDA is unavailable, training fails explicitly instead of falling back silently