
# digits_fixed.py

import cv2
import numpy as np
from numpy import argmax
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from google.colab import output
from IPython.display import display, HTML
from base64 import b64encode, b64decode
import matplotlib.pyplot as plt
import h5py
import os
import time
import shutil

# --------------------
# Load and preprocess MNIST
# --------------------
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1)).astype('float32') / 255.0
testX = testX.reshape((testX.shape[0], 28, 28, 1)).astype('float32') / 255.0
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# --------------------
# Augmentation function
# --------------------
def add_incomplete_effect(image):
    img = image.squeeze()  # fix shape (28,28,1) -> (28,28)
    rand = np.random.random()
    if rand < 0.3:
        x, y = np.random.randint(0, 28, 2)
        size = np.random.randint(3, 7)
        img[x:x+size, y:y+size] = 1.0
    elif rand < 0.6:
        for _ in range(np.random.randint(1, 3)):
            x1, y1 = np.random.randint(0, 28, 2)
            x2, y2 = np.random.randint(0, 28, 2)
            cv2.line(img, (x1, y1), (x2, y2), 0.0, thickness=1)
    return img.reshape(28, 28, 1)

# --------------------
# Data augmentation
# --------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    preprocessing_function=lambda x: add_incomplete_effect(x) if np.random.random() < 0.3 else x
)
datagen.fit(trainX)

# --------------------
# CNN Model
# --------------------
input_layer = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output_layer = Dense(10, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# --------------------
# Callbacks
# --------------------
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# --------------------
# Train or load weights
# --------------------
weights_file = 'improved1.weights.h5'
backup_weights_file = 'improved1_backup.weights.h5'

try:
    model.load_weights(weights_file)
    print(f"Loaded weights from {weights_file}")
except:
    print("Training model on MNIST...")
    batch_size = 128
    train_generator = datagen.flow(trainX, trainY, batch_size=batch_size)
    model.fit(
        train_generator,
        steps_per_epoch=len(trainX)//batch_size,
        epochs=5,
        validation_data=(testX, testY),
        callbacks=[lr_scheduler, early_stopping],
        verbose=1
    )
    model.save_weights(weights_file)
    shutil.copy(weights_file, backup_weights_file)
    print(f"Model trained and saved weights as {weights_file}")

# --------------------
# Evaluate
# --------------------
score = model.evaluate(testX, testY, verbose=0)
print(f'Test accuracy: {score[1]*100:.2f}%')
if score[1] < 0.99:
    print("Warning: Accuracy < 99%. Consider retraining.")

# --------------------
# Feature model for custom matching
# --------------------
feature_model = Model(inputs=model.input, outputs=model.layers[-3].output)

custom_images = np.array([])
custom_labels = np.array([])
custom_features = np.array([])

# --------------------
# Custom dataset functions
# --------------------
def save_to_custom_dataset(image, label, filename="custom_digits.h5"):
    global custom_images, custom_labels, custom_features
    if len(image.shape) == 2:
        image = image.reshape(28, 28, 1)
    if image.max() > 1.0:
        image = image.astype('float32') / 255.0
    try:
        if not os.path.exists(filename):
            with h5py.File(filename, "w") as f:
                f.create_dataset("images", data=[image], maxshape=(None,28,28,1), dtype='float32')
                f.create_dataset("labels", data=[label], maxshape=(None,), dtype='int32')
        else:
            with h5py.File(filename, "a") as f:
                f["images"].resize((f["images"].shape[0]+1), axis=0)
                f["labels"].resize((f["labels"].shape[0]+1), axis=0)
                f["images"][-1] = image
                f["labels"][-1] = label
        custom_images = np.append(custom_images, [image], axis=0) if custom_images.size else np.array([image])
        custom_labels = np.append(custom_labels, [label], axis=0) if custom_labels.size else np.array([label])
        new_feature = feature_model.predict(image.reshape(1,28,28,1), verbose=0)
        custom_features = np.append(custom_features, new_feature, axis=0) if custom_features.size else new_feature
        print(f"Added to custom dataset: Label {label}")
    except Exception as e:
        print(f"Error saving custom dataset: {e}")

def load_custom_dataset(filename="custom_digits.h5"):
    global custom_images, custom_labels, custom_features
    if not os.path.exists(filename):
        print(f"Custom dataset '{filename}' does not exist.")
        return np.array([]), np.array([])
    try:
        with h5py.File(filename, "r") as f:
            custom_images = f["images"][:]
            custom_labels = f["labels"][:]
        if len(custom_images) > 0:
            custom_features = feature_model.predict(custom_images, verbose=0)
            print(f"Loaded custom dataset: {len(custom_images)} images")
        return custom_images, custom_labels
    except Exception as e:
        print(f"Error loading custom dataset '{filename}': {e}")
        return np.array([]), np.array([])

def check_custom_match(input_image, custom_features, custom_labels, threshold=0.75):
    if len(custom_features) == 0:
        return None, 0.0
    input_features = feature_model.predict(input_image.reshape(1,28,28,1), verbose=0)
    best_similarity, best_label = -1, None
    best_confidence = 0.0
    for i in range(len(custom_features)):
        dot = np.dot(input_features.flatten(), custom_features[i].flatten())
        norm_input = np.linalg.norm(input_features)
        norm_custom = np.linalg.norm(custom_features[i])
        sim = dot / (norm_input*norm_custom) if norm_input*norm_custom !=0 else 0
        if sim > threshold and sim > best_similarity:
            best_similarity = sim
            best_label = custom_labels[i]
            best_confidence = sim
    return best_label, best_confidence

# --------------------
# Image processing and prediction
# --------------------
def load_image(data, is_file=False):
    try:
        if is_file:
            img = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
        else:
            img_data = b64decode(data.split(',')[1])
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        img = cv2.erode(img, np.ones((2,2), np.uint8), iterations=1)
        img = cv2.resize(img, (28,28))
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(img) > 127:
            img = 255 - img
        img = img.astype('float32')/255.0
        return img.reshape(1,28,28,1)
    except Exception as e:
        print(f"Image load error: {e}")
        return None

def predict_from_canvas(data_url, live_mode=True):
    img = load_image(data_url)
    if img is None: return None,None,None
    start = time.time()
    model_pred_value = model.predict(img, verbose=0)
    model_pred = argmax(model_pred_value)
    model_conf = min(model_pred_value[0][model_pred], 0.99)
    custom_pred, custom_conf = check_custom_match(img[0], custom_features, custom_labels, 0.75)
    if custom_pred is not None and custom_conf+0.15 >= model_conf:
        digit, confidence, source = custom_pred, min(0.99, custom_conf+0.15), "Custom Dataset"
    else:
        digit, confidence, source = model_pred, model_conf, "Improved1.weights.h5"
    pred_time = time.time()-start
    if live_mode:
        img_data = (img[0]*255).astype(np.uint8)
        _, img_enc = cv2.imencode('.png', img_data)
        img_b64 = b64encode(img_enc.tobytes()).decode('utf-8')
        display(HTML(f"""
            <div style="display:flex;align-items:center;">
            <img src="data:image/png;base64,{img_b64}" style="width:140px;height:140px;margin-right:20px;">
            <div style="font-size:16px;">
            Live Prediction: {digit}<br>
            Confidence: {confidence:.4f} (from {source})<br>
            Prediction took {pred_time:.3f} seconds
            </div></div>
        """))
    return digit, confidence, source

def correct_prediction(data_url, true_label):
    img = load_image(data_url)
    if img is None: return
    true_label = int(true_label)
    if not (0<=true_label<=9):
        print("Invalid label (0-9)")
        return
    save_to_custom_dataset(img[0], true_label)
    predict_from_canvas(data_url, live_mode=False)

# --------------------
# Canvas UI
# --------------------
def create_canvas_with_buttons():
    display(HTML("""
        <canvas id="canvas" width="280" height="280" style="border:1px solid black;"></canvas><br>
        <button id="predictBtn">Predict</button>
        <button id="clearBtn">Clear</button>
        <button id="correctBtn" style="display:none;">Correct Prediction</button>
        <div id="prediction" style="font-size:20px;margin-top:10px;"></div>
        <script>
        var canvas=document.getElementById('canvas');
        var ctx=canvas.getContext('2d');
        ctx.fillStyle='white'; ctx.fillRect(0,0,canvas.width,canvas.height);
        ctx.lineWidth=8; ctx.strokeStyle='black'; ctx.lineCap='round'; ctx.lineJoin='round';
        var drawing=false;
        canvas.addEventListener('mousedown',e=>{drawing=true;ctx.beginPath();ctx.moveTo(e.offsetX,e.offsetY);});
        canvas.addEventListener('mousemove',e=>{if(drawing){ctx.lineTo(e.offsetX,e.offsetY);ctx.stroke();}});
        canvas.addEventListener('mouseup',()=>{drawing=false;}); canvas.addEventListener('mouseout',()=>{drawing=false;});
        document.getElementById('predictBtn').onclick=()=>{var dataURL=canvas.toDataURL('image/png');
        google.colab.kernel.invokeFunction('notebook.predict',[dataURL],{}); document.getElementById('correctBtn').style.display='inline';};
        document.getElementById('clearBtn').onclick=()=>{ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);
        document.getElementById('prediction').innerHTML=''; document.getElementById('correctBtn').style.display='none';};
        document.getElementById('correctBtn').onclick=()=>{var dataURL=canvas.toDataURL('image/png');
        var trueLabel=prompt("Enter the correct digit (0-9):");
        if(trueLabel!==null && trueLabel.match(/^[0-9]$/)){google.colab.kernel.invokeFunction('notebook.correct',[dataURL,trueLabel],{});}else{alert("Enter 0-9");}};
        </script>
    """))

# --------------------
# Initialize
# --------------------
load_custom_dataset()
output.register_callback('notebook.predict', lambda data_url: predict_from_canvas(data_url, live_mode=True))
output.register_callback('notebook.correct', correct_prediction)
create_canvas_with_buttons()

# --------------------
# Test incomplete effect
# --------------------
test_img = testX[0].copy()
test_img_incomplete = add_incomplete_effect(test_img.copy())
pred = model.predict(test_img_incomplete.reshape(1,28,28,1), verbose=0)
print(f"Incomplete test: Predicted {argmax(pred)}, Confidence: {max(pred[0]):.4f}")