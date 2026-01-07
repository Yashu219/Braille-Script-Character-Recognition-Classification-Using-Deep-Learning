import os
import cv2
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model

# -------------------------------
# Settings
# -------------------------------
IMG_SIZE = 128
DATASET_PATH_1 = r"E:\bml\cnn_resized"
DATASET_PATH_2 = r"E:\bml\dataset5"
MODEL_NAME = r"D:\bml\vgg16_braille_0to9.keras"


# -------------------------------
# Load Dataset
# -------------------------------
def load_dataset(paths=[DATASET_PATH_1, DATASET_PATH_2]):
    data, labels = [], []
    classes = [str(i) for i in range(10)]

    for path in paths:
        for idx, c in enumerate(classes):
            class_dir = os.path.join(path, c)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = img.astype("float32") / 255.0
                data.append(img)
                labels.append(idx)

    data = np.array(data)
    labels = to_categorical(np.array(labels), num_classes=len(classes))
    return data, labels, classes


# -------------------------------
# Build Model
# -------------------------------
def build_model(num_classes=10):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# -------------------------------
# Train Model
# -------------------------------
def train_model():
    print("[INFO] Loading dataset...")
    X, y, classes = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Building model...")
    model = build_model(num_classes=y.shape[1])

    print("[INFO] Training...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32)

    model.save(MODEL_NAME)
    print(f"[INFO] Model saved at {MODEL_NAME}")

    loss, acc = model.evaluate(X_test, y_test)
    print(f"[INFO] Test Accuracy: {acc*100:.2f}%")


# -------------------------------
# Realtime Prediction
# -------------------------------
def realtime_prediction():
    if not os.path.exists(MODEL_NAME):
        print("[ERROR] Model not found. Train first.")
        return

    model = load_model(MODEL_NAME)
    _, _, classes = load_dataset()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("[INFO] Starting realtime recognition. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------
        # Define ROI
        # -------------------
        h, w = frame.shape[:2]
        roi_size = 200
        x = w//2 - roi_size//2
        y = h//2 - roi_size//2
        roi = frame[y:y+roi_size, x:x+roi_size]

        # -------------------
        # Preprocess for model
        # -------------------
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        model_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        model_input = model_input.astype("float32") / 255.0
        model_input = np.expand_dims(model_input, axis=0)

        # -------------------
        # Predict
        # -------------------
        pred = model.predict(model_input, verbose=0)
        label_idx = np.argmax(pred)
        conf = np.max(pred)
        label = classes[label_idx]

        # -------------------
        # Display
        # -------------------
        color = (0, 255, 0) if conf > 0.6 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+roi_size, y+roi_size), color, 2)
        cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Realtime Braille Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# Predict Single Image
# -------------------------------
def predict_image(image_path):
    if not os.path.exists(MODEL_NAME):
        print("[ERROR] Model not found. Train first.")
        return
    if not os.path.exists(image_path):
        print("[ERROR] Image path not found:", image_path)
        return

    model = load_model(MODEL_NAME)
    _, _, classes = load_dataset()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    label_idx = np.argmax(pred)
    conf = np.max(pred)
    label = classes[label_idx]

    print(f"Predicted Braille Number: {label} | Confidence: {conf*100:.2f}%")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--realtime", action="store_true", help="Run realtime webcam")
    parser.add_argument("--predict", type=str, help="Predict single image")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.realtime:
        realtime_prediction()
    elif args.predict:
        predict_image(args.predict)
    else:
        print("Usage: python braille_vgg16_full.py [--train | --realtime | --predict <image_path>]")
