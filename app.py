from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load trained model
model = load_model("cnn_model.h5")

# Build model once (needed for GradCAM)
model(np.zeros((1, 64, 64, 3)))


# =========================
# GradCAM Function
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d"):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        # handle keras returning list
        if isinstance(predictions, list):
            predictions = predictions[0]

        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


# =========================
# Main Route
# =========================
@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    filename = None
    confidence = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filename = file.filename
            path = os.path.join(UPLOAD_FOLDER, filename)

            file.save(path)

            # preprocess image
            img = cv2.imread(path)
            img = cv2.resize(img, (64, 64))
            img = img / 255.0

            img_array = img.reshape(1, 64, 64, 3)

            # prediction
            result = model.predict(img_array)

            if result[0][0] > 0.5:
                prediction = "Dog 🐶"
                confidence = float(result[0][0]) * 100
            else:
                prediction = "Cat 🐱"
                confidence = float(1 - result[0][0]) * 100

            # =========================
            # GradCAM Heatmap
            # =========================
            try:

                heatmap = make_gradcam_heatmap(img_array, model)

                heatmap = cv2.resize(heatmap, (64, 64))
                heatmap = np.uint8(255 * heatmap)

                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                original = cv2.imread(path)

                superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

                cv2.imwrite("static/heatmap.jpg", superimposed)

            except Exception as e:
                print("GradCAM error:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        filename=filename,
        confidence=confidence,
    )


# =========================
# Serve Uploaded Image
# =========================
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)