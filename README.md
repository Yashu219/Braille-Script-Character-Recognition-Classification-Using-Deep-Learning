# Braille-Script-Character-Recognition-Classification-Using-Deep-Learning


An assistive system that helps visually impaired users recognise Braille characters 
through tactile output, combining deep learning with embedded hardware.

## Results
| Model | Accuracy |
|-------|----------|
| CNN   | 94%      |
| VGG16 | 97%      |

## How It Works
1. Camera captures Braille pattern input
2. VGG16 model (97% accuracy) classifies the Braille character
3. Servo-driven tactile actuators translate the character into physical output
4. User feels and interprets the character through touch

## Tech Stack
- Python, TensorFlow, Keras, OpenCV
- Raspberry Pi, Servo Actuators
- CNN and VGG16 deep learning models

## Files
- `cnn_trained_model.py` — CNN model training
- `transmitter_code.py` — Sends classified output to hardware
- `Recevier.py` — Hardware receiver for tactile actuators
