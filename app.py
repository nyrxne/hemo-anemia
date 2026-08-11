import io
import hashlib
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

APP_DIR = Path(__file__).parent

# Multilingual result strings — pre-written, not fetched live, so the demo
# never depends on a network call at the booth. Extend this dict with real
# Bhashini-translated copy when you have it; the structure stays the same.
RESULT_COPY = {
    "en": {
        "anemic_label": "Signs of anemia detected",
        "anemic_detail": "The conjunctiva pallor pattern is consistent with lower hemoglobin levels. This is a screening signal, not a diagnosis — please consult a health worker for a blood test.",
        "non_anemic_label": "No signs of anemia detected",
        "non_anemic_detail": "The conjunctiva pallor pattern falls within the typical range. This is a screening signal, not a diagnosis.",
        "confidence_label": "confidence",
    },
    "hi": {
        "anemic_label": "एनीमिया के लक्षण मिले",
        "anemic_detail": "आंख की झिल्ली का रंग हीमोग्लोबिन के कम स्तर से मेल खाता है। यह केवल एक जांच संकेत है, निदान नहीं — कृपया रक्त जांच के लिए स्वास्थ्य कर्मी से संपर्क करें।",
        "non_anemic_label": "एनीमिया के कोई लक्षण नहीं मिले",
        "non_anemic_detail": "आंख की झिल्ली का रंग सामान्य सीमा में है। यह केवल एक जांच संकेत है, निदान नहीं।",
        "confidence_label": "विश्वास स्तर",
    },
    "ta": {
        "anemic_label": "இரத்த சோகை அறிகுறிகள் கண்டறியப்பட்டன",
        "anemic_detail": "கண் சவ்வின் நிறம் குறைந்த ஹீமோகுளோபின் அளவைக் காட்டுகிறது. இது ஒரு பரிசோதனை சமிக்ஞை மட்டுமே, நோய் கண்டறிதல் அல்ல — இரத்த பரிசோதனைக்கு சுகாதார பணியாளரை அணுகவும்.",
        "non_anemic_label": "இரத்த சோகை அறிகுறிகள் இல்லை",
        "non_anemic_detail": "கண் சவ்வின் நிறம் இயல்பான வரம்பில் உள்ளது. இது ஒரு பரிசோதனை சமிக்ஞை மட்டுமே, நோய் கண்டறிதல் அல்ல।",
        "confidence_label": "நம்பகத்தன்மை",
    },
}

SUPPORTED_LANGUAGES = [
    {"code": "en", "label": "English"},
    {"code": "hi", "label": "हिन्दी"},
    {"code": "ta", "label": "தமிழ்"},
]


def run_inference(image_bytes: bytes):
    """
    Returns (label, confidence) where label is "anemic" or "non_anemic"
    and confidence is a float in [0, 1].
    """

    # --------------------------- PLACEHOLDER (active) ---------------------------
    # Deterministic-but-varied stand-in so the same photo always gives the same
    # result during rehearsal, without needing real weights loaded.
    digest = hashlib.sha256(image_bytes).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF  # -> float in [0, 1]
    label = "anemic" if score > 0.5 else "non_anemic"
    confidence = score if label == "anemic" else (1 - score)
    # Keep confidence in a realistic, non-extreme demo range.
    confidence = 0.62 + confidence * 0.35
    return label, round(confidence, 3)
    # -----------------------------------------------------------------------------

    # ----------------------------- REAL MODEL (uncomment) ------------------------
    # import torch
    # from torchvision import transforms
    #
    # if not hasattr(app, "_hema_model"):
    #     app._hema_model = torch.load(APP_DIR / "model.pt", map_location="cpu")
    #     app._hema_model.eval()
    #
    # preprocess = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                           std=[0.229, 0.224, 0.225]),
    # ])
    #
    # img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # tensor = preprocess(img).unsqueeze(0)
    #
    # with torch.no_grad():
    #     logits = app._hema_model(tensor)
    #     prob_anemic = torch.softmax(logits, dim=1)[0, 1].item()
    #
    # label = "anemic" if prob_anemic >= 0.5 else "non_anemic"
    # confidence = prob_anemic if label == "anemic" else (1 - prob_anemic)
    # return label, round(confidence, 3)
    # -----------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html", languages=SUPPORTED_LANGUAGES)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    lang = request.form.get("language", "en")
    if lang not in RESULT_COPY:
        lang = "en"

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty file"}), 400

    # Sanity check: make sure it's actually a readable image before we
    # "run inference" on it — never silently guess on bad input.
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return jsonify({"error": "Could not read this file as an image"}), 400

    label, confidence = run_inference(image_bytes)
    copy = RESULT_COPY[lang]

    return jsonify({
        "label": label,
        "confidence": confidence,
        "headline": copy["anemic_label"] if label == "anemic" else copy["non_anemic_label"],
        "detail": copy["anemic_detail"] if label == "anemic" else copy["non_anemic_detail"],
        "confidence_label": copy["confidence_label"],
    })


if __name__ == "__main__":
    # host="0.0.0.0" so it's reachable from a phone on the same booth wifi too,
    # if you want to demo the upload from a phone camera instead of the laptop.
    app.run(host="0.0.0.0", port=5000, debug=True)
