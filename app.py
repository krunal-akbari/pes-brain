import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = load_model('./trained_model.h5')

# Define the class labels (replace with your actual class labels)
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Define a function to preprocess and predict
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]
    return predicted_class, prediction

# Define the home route

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Ensure uploads directory exists
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            # Save the file
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            # Make prediction
            predicted_class, prediction = predict_image(file_path)
            class_probabilities = list(zip(class_labels, (prediction[0] * 100).round(2)))
            return render_template('result.html', predicted_class=predicted_class, prediction=prediction,class_probabilities=class_probabilities)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)

