import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

# =========================================================
# Load model safely (handles version mismatches)
# =========================================================
MODEL_PATH = "wisebin.h5"

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("⚠️ Standard load failed, rebuilding model for current TF:", e)
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights=None, input_shape=(224,224,3)
    )
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(4, activation="softmax")
    ])
    model.load_weights(MODEL_PATH)
    print("✅ Model rebuilt and weights loaded!")

# =========================================================
# GPU Setup
# =========================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🟢 Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU found, running on CPU")

# =========================================================
# Class names and colors
# =========================================================
class_names = ["Aluminium", "Other", "Paper", "Plastic"]
colors = {
    "Plastic": (255, 100, 0),
    "Paper": (0, 255, 0),
    "Aluminium": (192, 192, 192),
    "Other": (42, 42, 165)
}

# =========================================================
# Open webcam
# =========================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("🎥 Press 'q' to quit")

# =========================================================
# Realtime prediction loop
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame grab failed")
        break

    # Resize for model input
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    cls_index = np.argmax(preds)
    cls_name = class_names[cls_index]
    conf = np.max(preds) * 100

    # Draw colored border around frame
    color = colors.get(cls_name, (255, 255, 255))
    cv2.rectangle(frame, (10, 10), (630, 470), color, 8)

    # Display overlay text
    label = f"{cls_name} ({conf:.1f}%)"
    cv2.putText(frame, label, (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

    # Show video feed
    cv2.imshow("WiseBin Trash Sorter", frame)

    # Print to terminal
    print(f"♻️ Sort into: {cls_name}  |  Confidence: {conf:.1f}%")

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()