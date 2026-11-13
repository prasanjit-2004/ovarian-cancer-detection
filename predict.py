"""
Predict on a single image with the trained ResNet50 model.

Usage:
    python predict.py path/to/image.jpg
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

OUTPUT_DIR = Path("outputs")
IMG_SIZE = 224


def predict(image_path: str):
    model = tf.keras.models.load_model(OUTPUT_DIR / "best_model.keras")
    class_names = json.loads((OUTPUT_DIR / "class_names.json").read_text())

    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img)[np.newaxis, ...]
    arr = preprocess_input(arr)  # must match training preprocessing

    probs = model.predict(arr, verbose=0)[0]
    for name, p in sorted(zip(class_names, probs), key=lambda t: -t[1]):
        print(f"{name}: {p * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict.py path/to/image.jpg")
    predict(sys.argv[1])
