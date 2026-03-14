# ============================================================
# 🌐 CAPTCHA RECOGNITION - FLASK WEB APP
# ============================================================
# Run this file to start the website!
# Command: python app.py
# Then open browser: http://localhost:5000
# ============================================================

from flask import Flask, request, jsonify, render_template
import numpy as np
import json
import os
from PIL import Image
import io
import base64

# ── Load TensorFlow (suppress noisy logs) ──
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

app = Flask(__name__)

# ── CONFIG ───────────────────────────────────────────────────
MODEL_PATH   = "captcha_model_final.keras"   # ← your saved model
MAPPING_PATH = "char_mapping.json"           # ← your saved mapping
IMG_HEIGHT   = 50
IMG_WIDTH    = 200
IMG_COLOR    = 1   # grayscale

# ── LOAD MODEL & MAPPING ON STARTUP ──────────────────────────
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

with open(MAPPING_PATH, "r") as f:
    mapping = json.load(f)

# JSON keys are strings, convert back to int keys
num_to_char    = {int(k): v for k, v in mapping["num_to_char"].items()}
captcha_length = mapping["captcha_length"]
print(f"✅ Mapping loaded! CAPTCHA length: {captcha_length}")


# ── HELPER: Preprocess image ─────────────────────────────────
def preprocess_image(image_bytes):
    """
    Take raw image bytes → resize → normalize → ready for model
    Like preparing the flashcard before showing it to the brain!
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_COLOR)
    return img_array


# ── HELPER: Predict CAPTCHA text ─────────────────────────────
def predict_captcha(img_array):
    """
    Feed image into model → get predicted text!
    """
    preds = model.predict(img_array, verbose=0)
    if not isinstance(preds, list):
        preds = [preds]
    predicted_text = "".join([num_to_char[np.argmax(p[0])] for p in preds])
    return predicted_text


# ── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def index():
    """ Serve the main page """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive uploaded image → return predicted CAPTCHA text
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()
        img_array   = preprocess_image(image_bytes)
        prediction  = predict_captcha(img_array)

        # Also return base64 image so frontend can display it
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        return jsonify({
            "success": True,
            "prediction": prediction,
            "image": img_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── START SERVER ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🌐 CAPTCHA Recognition Web App")
    print("="*50)
    print("👉 Open your browser at: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
