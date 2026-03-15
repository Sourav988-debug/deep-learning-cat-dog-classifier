import os
import cv2
import numpy as np

# Path to dataset
DATASET_PATH = "dataset"

# Image size
IMG_SIZE = 64

X = []
y = []

# Class labels
classes = {"cats": 0, "dogs": 1}

for category in classes:
    folder_path = os.path.join(DATASET_PATH, category)
    label = classes[category]

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)
        except:
            pass

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize images
X = X / 255.0

print("Dataset loaded successfully")
print("Total images:", len(X))
print("Image shape:", X.shape)