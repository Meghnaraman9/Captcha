import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH        = 'best_captcha_model.keras'
CHAR_MAPPING_PATH = 'char_mapping.json'

model       = None
num_to_char = None
IMG_HEIGHT  = 50
IMG_WIDTH   = 200
CAPTCHA_LEN = 6

def load_resources():
    global model, num_to_char, IMG_HEIGHT, IMG_WIDTH, CAPTCHA_LEN
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded!")
        with open(CHAR_MAPPING_PATH, 'r') as f:
            mapping = json.load(f)
        raw = mapping.get('num_to_char', mapping)
        num_to_char = {int(k): v for k, v in raw.items()}
        IMG_HEIGHT  = int(mapping.get('img_height',    IMG_HEIGHT))
        IMG_WIDTH   = int(mapping.get('img_width',     IMG_WIDTH))
        CAPTCHA_LEN = int(mapping.get('captcha_length', CAPTCHA_LEN))
        print(f"✅ Mapping loaded! {len(num_to_char)} chars | {IMG_HEIGHT}x{IMG_WIDTH} | length {CAPTCHA_LEN}")
    except Exception as e:
        print(f"⚠️  Startup error: {e}")

load_resources()

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    return img

def decode_prediction(pred):
    result     = []
    chars_info = []
    for char_pred in pred:
        probs    = np.array(char_pred[0])
        top2     = np.argsort(probs)[::-1][:2]
        best_idx = int(top2[0])
        best_ch  = num_to_char.get(best_idx, '?')
        best_conf= float(probs[best_idx])
        alt_idx  = int(top2[1])
        alt_ch   = num_to_char.get(alt_idx, '?')
        alt_conf = float(probs[alt_idx])
        result.append(best_ch)
        chars_info.append({
            'char': best_ch, 'conf': round(best_conf*100,1),
            'alt':  alt_ch,  'alt_conf': round(alt_conf*100,1),
        })
    return ''.join(result), chars_info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    allowed = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': 'Unsupported file type'}), 400
    if model is None or num_to_char is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        img         = preprocess_image(filepath)
        pred        = model.predict(img, verbose=0)
        text, chars = decode_prediction(pred)
        return jsonify({'success': True, 'text': text, 'chars': chars})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok', 'model_loaded': model is not None,
        'chars': len(num_to_char) if num_to_char else 0,
        'img_size': f'{IMG_HEIGHT}x{IMG_WIDTH}', 'captcha_len': CAPTCHA_LEN,
    })

if __name__ == '__main__':
    app.run(debug=True)
