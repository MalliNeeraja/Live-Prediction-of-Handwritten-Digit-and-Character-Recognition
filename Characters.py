#characters.py


import cv2
import numpy as np
from numpy import argmax
import gzip
import struct
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Add
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from google.colab import output
from IPython.display import display, HTML
from base64 import b64encode, b64decode
import h5py
import os
import time

!wget https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip -O emnist.zip
!unzip emnist.zip -d gzipdata

# Load EMNIST dataset functions
def load_emnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_emnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_emnist_mapping(filename):
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            label, ascii_val = map(int, line.split())
            mapping[label] = chr(ascii_val)
    return mapping

# Load and preprocess EMNIST dataset (for validation)
train_images = load_emnist_images("/content/gzipdata/gzip/emnist-bymerge-train-images-idx3-ubyte.gz")
train_labels = load_emnist_labels("/content/gzipdata/gzip/emnist-bymerge-train-labels-idx1-ubyte.gz")
test_images = load_emnist_images("/content/gzipdata/gzip/emnist-bymerge-test-images-idx3-ubyte.gz")
test_labels = load_emnist_labels("/content/gzipdata/gzip/emnist-bymerge-test-labels-idx1-ubyte.gz")
mapping = load_emnist_mapping("/content/gzipdata/gzip/emnist-bymerge-mapping.txt")

character_indices_train = train_labels >= 10
character_indices_test = test_labels >= 10
train_images = train_images[character_indices_train]
train_labels = train_labels[character_indices_train]
test_images = test_images[character_indices_test]
test_labels = test_labels[character_indices_test]

train_labels = train_labels - 10
test_labels = test_labels - 10
new_mapping = {i: mapping[i + 10] for i in range(37)}

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255.0
train_images = np.array([np.fliplr(np.rot90(img, k=-1)) for img in train_images])
test_images = np.array([np.fliplr(np.rot90(img, k=-1)) for img in test_images])
train_labels = to_categorical(train_labels, 37)
test_labels = to_categorical(test_labels, 37)

# Enhanced Data Augmentation (kept for reference, not used without retraining)
def add_incomplete_effect(image):
    img = image.copy().squeeze(-1)
    rand = np.random.random()
    if rand < 0.4:  # Incomplete: Add white patches
        x, y = np.random.randint(0, 28, 2)
        size = np.random.randint(5, 12)
        img[x:x+size, y:y+size] = 1.0
    elif rand < 0.8:  # Overcomplete: Add extra black lines
        for _ in range(np.random.randint(2, 4)):
            x1, y1 = np.random.randint(0, 28, 2)
            x2, y2 = np.random.randint(0, 28, 2)
            cv2.line(img, (x1, y1), (x2, y2), 0.0, thickness=2)
    return img[..., np.newaxis]

# Model definition (must match checkpoint.weights.h5 architecture)
def build_model():
    input_layer = Input(shape=(28, 28, 1))
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    shortcut = x
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = x
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = x
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = x
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(37, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=output_layer)

'''train_images = train_images[:60000]
train_labels = train_labels[:60000]

test_images = test_images[:10000]
test_labels = test_labels[:10000]
'''
# Build and load weights
model = build_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    epochs=10,   # start small (5–10)
    batch_size=128
)

model.save_weights('/content/checkpoint.weights.h5')
print("Training complete and weights saved!")



from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_path = "/content/checkpoint.weights.h5"
checkpoint = ModelCheckpoint(checkpoint_path,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_accuracy',
                             verbose=1)

from google.colab import files
files.download('/content/checkpoint.weights.h5')


weights_file = '/content/checkpoint.weights.h5'
if os.path.exists(weights_file):
    model.load_weights(weights_file)
    print(f"Loaded weights from {weights_file}")
else:
    raise FileNotFoundError(f"{weights_file} not found. Please upload it to Colab.")


# Feature model for custom matching
feature_model = Model(inputs=model.input, outputs=model.layers[-3].output)

# Global variables
custom_images = np.array([])
custom_labels = np.array([])  # Now stores exact labels (strings) instead of indices
custom_features = np.array([])

# Custom dataset management (modified to store exact labels)
def save_to_custom_dataset(image, label, filename="custom_characters.h5"):
    global custom_images, custom_labels, custom_features
    if len(image.shape) == 2:
        image = image.reshape(28, 28, 1)
    if image.max() > 1.0:
        image = image.astype('float32') / 255.0
    try:
        if not os.path.exists(filename):
            with h5py.File(filename, "w") as f:
                f.create_dataset("images", data=[image], maxshape=(None, 28, 28, 1), dtype='float32')
                f.create_dataset("labels", data=[label], maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
        else:
            with h5py.File(filename, "a") as f:
                current_images = f["images"]
                current_labels = f["labels"]
                current_images.resize((current_images.shape[0] + 1), axis=0)
                current_labels.resize((current_labels.shape[0] + 1), axis=0)
                current_images[-1] = image
                current_labels[-1] = label
        custom_images = np.append(custom_images, [image], axis=0) if custom_images.size else np.array([image])
        custom_labels = np.append(custom_labels, [label], axis=0) if custom_labels.size else np.array([label])
        new_feature = feature_model.predict(image.reshape(1, 28, 28, 1), verbose=0)
        custom_features = np.append(custom_features, new_feature, axis=0) if custom_features.size else new_feature
        print(f"Added to custom dataset: {label}")
        print(f"Current custom labels: {custom_labels.tolist()}")
    except Exception as e:
        print(f"Error saving to custom dataset: {e}")

def load_custom_dataset(filename="custom_characters.h5"):
    global custom_images, custom_labels, custom_features
    if not os.path.exists(filename):
        print(f"Custom dataset '{filename}' does not exist.")
        return np.array([]), np.array([])
    with h5py.File(filename, "r") as f:
        custom_images = f["images"][:]
        custom_labels = np.array([label.decode('utf-8') if isinstance(label, bytes) else label for label in f["labels"][:]], dtype=object)
    if len(custom_images) > 0:
        custom_features = feature_model.predict(custom_images, verbose=0)
        print(f"Custom dataset loaded: {len(custom_images)} images")
    return custom_images, custom_labels

# Custom matching (modified to return the exact label)
def check_custom_match(input_image, custom_features, custom_labels, threshold=0.75):
    if len(custom_features) == 0:
        return None, 0.0
    input_features = feature_model.predict(input_image.reshape(1, 28, 28, 1), verbose=0)
    best_similarity = -1.0
    best_label = None
    best_confidence = 0.0
    for i in range(len(custom_features)):
        similarity = np.dot(input_features.flatten(), custom_features[i].flatten()) / (np.linalg.norm(input_features) * np.linalg.norm(custom_features[i]))
        if similarity > threshold and similarity > best_similarity:
            best_similarity = similarity
            best_label = custom_labels[i]  # Return the exact label
            best_confidence = similarity
    return best_label, best_confidence

# Updated image preprocessing
def load_image(data, is_file=False):
    try:
        if is_file:
            img = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
        else:
            img_data = b64decode(data.split(',')[1])
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.resize(img, (28, 28))
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(img) > 127:
            img = 255 - img
        img = img.astype('float32') / 255.0
        return img.reshape(1, 28, 28, 1)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Prediction function (modified to handle exact labels)
def predict_from_canvas(data_url, live_mode=True):
    global model, feature_model, custom_features, custom_labels, new_mapping
    img = load_image(data_url, is_file=False)
    if img is None:
        return None, None, None
    start_time = time.time()
    model_pred_value = model.predict(img, verbose=0)
    model_pred = argmax(model_pred_value)
    model_confidence = min(model_pred_value[0][model_pred], 0.99)
    custom_pred_label, custom_confidence = check_custom_match(img[0], custom_features, custom_labels)
    if custom_pred_label is not None and custom_confidence + 0.15 >= model_confidence:
        pred_char = custom_pred_label  # Use the exact label from custom dataset
        confidence = min(0.99, custom_confidence + 0.15)
        source = "Custom Dataset"
    else:
        pred_char = new_mapping[model_pred]  # Use new_mapping for base model predictions
        confidence = model_confidence
        source = "checkpoint.weights.h5"
    pred_time = time.time() - start_time
    output_text = (
        f"Live Prediction: {pred_char}<br>"
        f"Confidence: {confidence:.4f} (from {source})<br>"
        f"Prediction took {pred_time:.3f} seconds"
    )
    img_data = (img[0] * 255).astype(np.uint8)
    _, img_encoded = cv2.imencode('.png', img_data)
    img_base64 = b64encode(img_encoded.tobytes()).decode('utf-8')
    if live_mode:
        display(HTML(f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{img_base64}" style="width: 140px; height: 140px; margin-right: 20px;">
                <div style="font-size: 16px;">{output_text}</div>
            </div>
        """))
    return pred_char, confidence, source

# Correction function (modified to directly assign the label)
def correct_prediction(data_url, true_label):
    img = load_image(data_url, is_file=False)
    if img is None:
        return
    if not true_label.isalpha():
        print("Invalid character.")
        return
    print(f"Adding to custom dataset: '{true_label}'")
    save_to_custom_dataset(img[0], true_label)  # Directly save the exact label
    predict_from_canvas(data_url, live_mode=False)

# Canvas interface (unchanged)
def create_canvas_with_buttons():
    display(HTML("""
        <canvas id="canvas" width="280" height="280" style="border:1px solid black;"></canvas>
        <br>
        <button id="predictBtn">Predict</button>
        <button id="clearBtn">Clear</button>
        <button id="correctBtn" style="display:none;">Correct Prediction</button>
        <div id="prediction" style="font-size:20px; margin-top:10px;"></div>
        <script>
            var canvas = document.getElementById('canvas');
            var ctx = canvas.getContext('2d');
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.lineWidth = 8;
            ctx.strokeStyle = 'black';
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            var drawing = false;
            canvas.addEventListener('mousedown', (e) => {
                drawing = true;
                ctx.beginPath();
                ctx.moveTo(e.offsetX, e.offsetY);
            });
            canvas.addEventListener('mousemove', (e) => {
                if (drawing) {
                    ctx.lineTo(e.offsetX, e.offsetY);
                    ctx.stroke();
                }
            });
            canvas.addEventListener('mouseup', () => { drawing = false; });
            canvas.addEventListener('mouseout', () => { drawing = false; });
            document.getElementById('predictBtn').onclick = () => {
                var dataURL = canvas.toDataURL('image/png');
                google.colab.kernel.invokeFunction('notebook.predict', [dataURL], {});
                document.getElementById('correctBtn').style.display = 'inline';
            };
            document.getElementById('clearBtn').onclick = () => {
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                document.getElementById('prediction').innerHTML = '';
                document.getElementById('correctBtn').style.display = 'none';
            };
            document.getElementById('correctBtn').onclick = () => {
                var dataURL = canvas.toDataURL('image/png');
                var trueLabel = prompt("Enter the correct character (A-Z, a-z):");
                if (trueLabel !== null && trueLabel.match(/^[A-Za-z]$/)) {
                    google.colab.kernel.invokeFunction('notebook.correct', [dataURL, trueLabel], {});
                } else {
                    alert("Please enter a valid character (A-Z, a-z).");
                }
            };
        </script>
    """))

# Initialize
load_custom_dataset()
output.register_callback('notebook.predict', lambda data_url: predict_from_canvas(data_url, live_mode=True))
output.register_callback('notebook.correct', correct_prediction)
create_canvas_with_buttons()

# Test incomplete/overcomplete handling
print("Testing incomplete/overcomplete character handling...")
test_img = test_images[0].copy()
test_img_incomplete = add_incomplete_effect(test_img.copy())
pred = model.predict(test_img_incomplete.reshape(1, 28, 28, 1), verbose=0)
print(f"Incomplete test: Predicted {new_mapping[argmax(pred)]}, Confidence: {max(pred[0]):.4f}")