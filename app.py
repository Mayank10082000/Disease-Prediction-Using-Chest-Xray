from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__)
model = load_model('D:\Mayank\My Projects\Covid19 Detector Using Chest XRay Website\chest_xray.h5')

# Preprocess function to resize and normalize the image
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    return image

# Function to check if the image is a chest X-ray image
def is_chest_xray_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 30, 100)

    # Perform Hough line transformation
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Check if any lines were detected
    if lines is not None:
        return True
    else:
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image file found!"

    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    is_chest_xray = is_chest_xray_image(image)  # Check if the image is a chest X-ray

    if not is_chest_xray:
        return render_template('result.html', result_text="Check your input image")

    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)

    predictions = model.predict(processed_image)
    class_labels = ['Normal', 'Covid-19', 'Pneumonia']
    predicted_label = class_labels[np.argmax(predictions)]
    confidence = round(np.max(predictions) * 100, 2)

    # Save the uploaded image
    image_file.seek(0)
    image_path = 'static/uploaded_image.jpg'
    image_file.save(image_path)

    if predicted_label == 'Normal':
        result_text = "Your X-ray is normal"
    elif predicted_label == 'Covid-19':
        result_text = "You have a possibility of Covid-19, please consult a doctor"
    else:
        result_text = "You have a possibility of pneumonia, please consult a doctor"

    return render_template('result.html', predicted_label=predicted_label, confidence=confidence, result_text=result_text)

if __name__ == '__main__':
    app.run()