import os
import cv2
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
IMG_SIZE = 64
DATASET_PATH = r"/home/yashu/convo2d"
MODEL_NAME = "/home/yashu/braille_cnn_pi5.keras"


tf.config.set_soft_device_placement(True)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
def load_dataset(path=DATASET_PATH, img_size=64):
    data, labels = [], []
    classes = sorted(os.listdir(path))

    for idx, c in enumerate(classes):
        class_dir = os.path.join(path, c)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype("float32") / 255.0
            data.append(np.expand_dims(img, axis=-1))
            labels.append(idx)

    data = np.array(data)
    labels = to_categorical(np.array(labels), num_classes=len(classes))
    return data, labels, len(classes), classes

def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=5):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_and_predict_then_realtime():
    print("Loading dataset...")
    X, y, num_classes, class_names = load_dataset(DATASET_PATH, IMG_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building CNN for Pi 5...")
    model = build_cnn((IMG_SIZE, IMG_SIZE, 1), num_classes)

    print("Training started...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=10, batch_size=16, verbose=1)

    print("Saving model...")
    model.save(MODEL_NAME)

    print("Evaluating model on test set...")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = np.mean(y_pred == y_true)
    print(f"\nTest Accuracy: {acc*100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print(" Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\n All predictions done. Launching realtime webcam now...\n")
    realtime_prediction(model, class_names)


def realtime_prediction(model=None, class_names=None):
    if model is None:
        if not os.path.exists(MODEL_NAME):
            print("No trained model found. Train first.")
            return
        model = load_model(MODEL_NAME)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not detected.")
        return

    print("Realtime Braille recognition started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x, y, w, h = 100, 100, 200, 200
        roi = frame[y:y+h, x:x+w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        gray = gray.astype("float32") / 255.0
        gray = np.expand_dims(gray, axis=(0, -1))

        pred = model.predict(gray, verbose=0)
        label_idx = np.argmax(pred)
        conf = np.max(pred)
        label = class_names[label_idx] if class_names else str(label_idx + 1)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf*100:.1f}%)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Realtime Braille Recognition (Pi 5)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model then predict + realtime")
    parser.add_argument("--realtime", action="store_true", help="Run realtime webcam only")
    args = parser.parse_args()

    if args.train:
        train_and_predict_then_realtime()
    elif args.realtime:
        realtime_prediction()
    else:
        print("Usage:")
        print("  python3 braille_cnn_pi5_autorun.py --train      # Train → predict → realtime")
        print("  python3 braille_cnn_pi5_autorun.py --realtime   # Realtime only (if model exists)")
