from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("cnn_model.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = file.filename
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            img = cv2.imread(path)
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            img = img.reshape(1, 64, 64, 3)

            result = model.predict(img)
            prediction = "Cat" if np.argmax(result) == 0 else "Dog"

    return render_template(
        "index.html",
        prediction=prediction,
        filename=filename
    )

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)