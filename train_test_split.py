import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os

# Load data again (same as preprocessing)
DATASET_PATH = "dataset"
IMG_SIZE = 64

X = []
y = []

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

X = np.array(X) / 255.0
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))