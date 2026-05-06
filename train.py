import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# ── Config ─────────────────────────────────────────────────────
IMG_HEIGHT    = 50
IMG_WIDTH     = 200
CAPTCHA_LEN   = 6
BATCH_SIZE    = 32
EPOCHS        = 50
DATASET_PATH  = 'captcha_images/CaptchaImageDataset'
MODEL_SAVE    = 'best_captcha_model.keras'
MAPPING_SAVE  = 'char_mapping.json'

# ── Character set: A-Z + 0-9 ───────────────────────────────────
CHARACTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
NUM_CLASSES  = len(CHARACTERS)  # 36
char_to_num  = {ch: i for i, ch in enumerate(CHARACTERS)}
num_to_char  = {i: ch for i, ch in enumerate(CHARACTERS)}

print(f"✅ {NUM_CLASSES} characters | {IMG_HEIGHT}x{IMG_WIDTH} | length {CAPTCHA_LEN}")

# ── Load dataset ───────────────────────────────────────────────
def load_dataset(folder):
    images, labels = [], []
    skipped = 0
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        label = os.path.splitext(fname)[0]
        if len(label) != CAPTCHA_LEN:
            skipped += 1
            continue
        if not all(c in char_to_num for c in label):
            skipped += 1
            continue
        img_path = os.path.join(folder, fname)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.cast(img, tf.float32) / 255.0
        images.append(img.numpy())
        labels.append([char_to_num[c] for c in label])
    print(f"✅ Loaded {len(images)} images | Skipped {skipped}")
    return np.array(images), np.array(labels)

images, labels = load_dataset(DATASET_PATH)

# ── Split into train/val ───────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
print(f"✅ Train: {len(X_train)} | Val: {len(X_val)}")

# ── Prepare labels as list of 6 arrays ────────────────────────
def split_labels(y):
    return [y[:, i] for i in range(CAPTCHA_LEN)]

y_train_split = split_labels(y_train)
y_val_split   = split_labels(y_val)

# ── Build model ────────────────────────────────────────────────
def build_model():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    # 6 separate output heads, one per character
    outputs = [
        layers.Dense(NUM_CLASSES, activation='softmax', name=f'char_{i+1}')(x)
        for i in range(CAPTCHA_LEN)
    ]

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss=['sparse_categorical_crossentropy'] * CAPTCHA_LEN,
        metrics=['accuracy'] * CAPTCHA_LEN
    )
    return model

model = build_model()
model.summary()

# ── Callbacks ──────────────────────────────────────────────────
callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_SAVE, monitor='val_loss',
        save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, verbose=1
    )
]

# ── Train ──────────────────────────────────────────────────────
print("\n🚀 Starting training...\n")
history = model.fit(
    X_train, y_train_split,
    validation_data=(X_val, y_val_split),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ── Save char mapping ──────────────────────────────────────────
mapping = {
    'num_to_char':    {str(k): v for k, v in num_to_char.items()},
    'char_to_num':    {k: int(v) for k, v in char_to_num.items()},
    'img_height':     IMG_HEIGHT,
    'img_width':      IMG_WIDTH,
    'captcha_length': CAPTCHA_LEN,
    'characters':     CHARACTERS,
}
with open(MAPPING_SAVE, 'w') as f:
    json.dump(mapping, f, indent=2)

print(f"\n✅ Model saved to {MODEL_SAVE}")
print(f"✅ Mapping saved to {MAPPING_SAVE}")
print("\n🎉 Training complete!")
