import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ---------- Settings ----------
DATASET_PATH = "dataset/Train"
IMG_SIZE = 32
NUM_CLASSES = 43
EPOCHS = 10

# ---------- Step 1: Load the images ----------
images = []
labels = []

print("Loading images... please wait (1 to 3 minutes)")
for class_id in range(NUM_CLASSES):
    folder = os.path.join(DATASET_PATH, str(class_id))
    for file_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file_name))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(class_id)

X = np.array(images, dtype="float32") / 255.0
y = to_categorical(np.array(labels), NUM_CLASSES)
print("Total images loaded:", len(X))

# ---------- Step 2: Split into training and validation ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=labels
)
print("Training images:", len(X_train))
print("Validation images:", len(X_val))

# ---------- Step 3: Build the CNN ----------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------- Step 4: Train ----------
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64,
          validation_data=(X_val, y_val))

# ---------- Step 5: Check and save ----------
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print("Validation accuracy:", round(accuracy * 100, 2), "%")

os.makedirs("model", exist_ok=True)
model.save("model/traffic_sign_model.keras")
print("Model saved to model/traffic_sign_model.keras")