import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

IMG_SIZE = 32

model = load_model("model/traffic_sign_model.keras")

# Test.csv lists every test image and its correct class number
data = pd.read_csv("dataset/Test.csv")

images = []
true_labels = []

print("Loading test images... please wait (1 to 2 minutes)")
for _, row in data.iterrows():
    path = os.path.join("dataset", row["Path"])
    img = cv2.imread(path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    images.append(img)
    true_labels.append(int(row["ClassId"]))

X = np.array(images, dtype="float32") / 255.0
y_true = np.array(true_labels)
print("Test images loaded:", len(X))

# Predict all images
probabilities = model.predict(X, batch_size=128, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

correct = int(np.sum(y_pred == y_true))
total = len(y_true)
print("Correct predictions:", correct, "out of", total)
print("Test accuracy:", round(correct / total * 100, 2), "%")

# Show a few wrong predictions
wrong = np.where(y_pred != y_true)[0]
print("Number of wrong predictions:", len(wrong))
print("First 5 wrong ones (true class -> predicted class):")
for i in wrong[:5]:
    print(" ", data["Path"].iloc[i], ":", y_true[i], "->", y_pred[i])