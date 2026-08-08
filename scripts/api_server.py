import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_and_link_model import predict_and_link

WEB_DIR = PROJECT_ROOT / "web"

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")


def normalize_lang(value: str | None) -> str:
    if isinstance(value, str) and value.lower().startswith("ar"):
        return "ar"
    return "en"

@app.route("/")
def home():
    return send_from_directory(WEB_DIR, "index.html")
#API endpoint

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    case_text = (data.get("case_text") or "").strip()
    lang = normalize_lang(data.get("lang"))

    if not case_text:
        message = "نص القضية مطلوب." if lang == "ar" else "Case text is required."
        return jsonify({"error": message}), 400
#Running the model

    try:
        result = predict_and_link(case_text)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
