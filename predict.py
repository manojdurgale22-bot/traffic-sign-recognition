import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sign_names import SIGN_INFO

IMG_SIZE = 32

# Load the trained model once
model = load_model("model/traffic_sign_model.keras")


def preprocess_image(img_bgr):
    """Prepare an image in the same way we prepared it for training."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # OpenCV reads BGR, model wants RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))      # make it 32 x 32
    img = img.astype("float32") / 255.0              # pixel values 0-255 -> 0-1
    return np.expand_dims(img, axis=0)               # add a batch dimension


def predict_sign(img_bgr):
    """Return class number, sign name, meaning and confidence (in %)."""
    processed = preprocess_image(img_bgr)
    probabilities = model.predict(processed, verbose=0)[0]
    class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[class_id]) * 100
    name, meaning = SIGN_INFO[class_id]
    return class_id, name, meaning, confidence


# This part runs only when we start this file directly, as a quick test
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path_to_image")
        sys.exit()

    image = cv2.imread(sys.argv[1])
    if image is None:
        print("Could not read the image. Check the path.")
        sys.exit()

    class_id, name, meaning, confidence = predict_sign(image)
    print("Class number :", class_id)
    print("Sign name    :", name)
    print("Meaning      :", meaning)
    print("Confidence   :", round(confidence, 2), "%")