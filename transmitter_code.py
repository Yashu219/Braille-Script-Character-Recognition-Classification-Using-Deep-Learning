import os
import cv2
import numpy as np
import serial
import time
from tensorflow.keras.models import load_model
import argparse


IMG_SIZE = 64
MODEL_NAME = "/home/yashu/braille_cnn_pi5.keras"

UART_PORT = "/dev/serial0"   
BAUD_RATE = 9600

CLASS_MAP = {
    0: "1",
    1: "2",
    2: "3",
    3: "4",
    4: "5",
}


def predict_braille(image_path):
    if not os.path.exists(MODEL_NAME):
        print("Model not found")
        return None
    if not os.path.exists(image_path):
        print("Image not found")
        return None

    print("🧠 Loading model...")
    model = load_model(MODEL_NAME)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    pred = model.predict(img, verbose=0)
    label_idx = np.argmax(pred)
    confidence = np.max(pred)

    label = CLASS_MAP.get(label_idx, str(label_idx + 1))
    print(f"🔍 Predicted Braille Digit: {label} ({confidence*100:.2f}%)")

    return label

def send_over_uart(message, port, baud):
    try:
        print(f"Connecting to {port} at {baud}...")
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  
        ser.write((message + "\n").encode())
        ser.flush()

        print(f"Sent predicted digit: {message}")
        ser.close()

    except Exception as e:
        print(f" Serial Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to Braille image")
    parser.add_argument("--port", default=UART_PORT, help="Serial/Bluetooth port")
    parser.add_argument("--baud", default=BAUD_RATE, type=int, help="Baud rate")

    args = parser.parse_args()

    predicted_digit = predict_braille(args.image)

    if predicted_digit:
        send_over_uart(predicted_digit, args.port, args.baud)
