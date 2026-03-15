# Deep Learning Cat vs Dog Classifier 🐱🐶

A web application that classifies uploaded images as **Cat or Dog** using deep learning.

This project implements a **Convolutional Neural Network (CNN)** trained on cat and dog images and deploys it through a **Flask web interface** where users can upload an image and get predictions.

---

## Features

* Upload image and classify as Cat or Dog
* CNN-based deep learning model
* Multiple machine learning models implemented for comparison
* Prediction confidence score
* Grad-CAM visualization to explain model predictions
* Flask web application interface

---

## Technologies Used

* Python
* TensorFlow / Keras
* Flask
* OpenCV
* NumPy
* Scikit-learn
* HTML / CSS

---

## Project Structure

```
deep-learning-cat-dog-classifier
│
├── app.py
├── cnn_model.py
├── data_preprocessing.py
├── ml_models.py
├── train_test_split.py
├── requirements.txt
│
├── templates
│   └── index.html
│
├── static
│   └── style.css
│
├── Procfile
└── .gitignore
```

---

## How to Run the Project

Install dependencies:

```
pip install -r requirements.txt
```

Run the Flask server:

```
python app.py
```

Open browser:

```
http://127.0.0.1:10000
```

Upload an image to get a prediction.

---

## Models Implemented

* CNN (Deep Learning)
* Support Vector Machine (SVM)
* Random Forest
* Logistic Regression

---

## Future Improvements

* Real-time webcam detection
* Support for multiple animal classes
* Improved UI with drag-and-drop upload
* Model performance dashboard
