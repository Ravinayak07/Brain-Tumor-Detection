import base64
import io
import cv2 as cv
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for
from PIL import Image
from tensorflow.keras.models import load_model
from DisplayDisease import DisplayDisease

# Create the Flask application
app = Flask(__name__)
model = None
multiModel = None
dt = DisplayDisease()

# Define a function to load the pre-trained model


def initialize_model():
    global model, multiModel
    model = load_model('epoch10_sgd_acc96Point76.h5')
    multiModel = load_model('multi-model-30K-epouch20.h5')


# Load the pre-trained model
initialize_model()

# Define a function to preprocess the binary input image


def preprocess_binary_image(image):
    image = image.resize((128, 128))
    image = np.array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    return [image, image, image]

# Define a function to preprocess the multiclass input image


def preprocess_multiClass_image(image):
    image = image.resize((128, 128))
    image = np.array(image)
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    else:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    return [image, image, image]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/binary.html', methods=['GET', 'POST'])
def binary():
    # If the user uploads an image
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        # Read the image file
        image = Image.open(image_file)

        # Preprocess the image
        processed_image = preprocess_binary_image(image)

        # Predict the image contents
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        # Return the prediction result as a string
        if predicted_class == 0:
            return jsonify({'result': 'No Tumor Detected'})
        else:
            return jsonify({'result': 'Tumor Detected'})

    # If the user opens the home page
    else:
        # Return the HTML/CSS/JS code for the home page
        return render_template('binary.html')


@app.route('/multi.html', methods=['GET', 'POST'])
def multi():
    # If the user uploads an image
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        # Read the image file
        image = Image.open(image_file)

        # Preprocess the image
        processed_image = preprocess_multiClass_image(image)
        processed_image = [processed_image[i][:, :, :] for i in range(3)]

        # Predict the image contents
        prediction = multiModel.predict(processed_image)
        predicted_class = np.argmax(prediction)

        # Return the prediction result as a string
        if predicted_class == 0:
            return jsonify({'result': 'Glioma'})
        elif predicted_class == 1:
            return jsonify({'result': 'Meningioma'})
        else:
            return jsonify({'result': 'Pituitary'})

    # If the user opens the home page
    else:
        # Return the HTML/CSS/JS code for the home page
        return render_template('multi.html')


@app.route('/segment.html', methods=['GET', 'POST'])
def segment():
    img_base64 = ""
    tumor_percentage = 0
    if request.method == 'POST':
        if 'image' not in request.files:
            return "Please select an image file."
        file = request.files['image']
        if file.filename == '':
            return "Please select an image file."
        img = cv.imdecode(np.fromstring(
            file.read(), np.uint8), cv.IMREAD_UNCHANGED)
        dt.readImage(img)
        dt.removeNoise()
        dt.displayDisease()
        tumor_percentage = dt.calculateTumorPercentage()
        img_str = cv.imencode('.jpg', dt.getImage())[1].tostring()
        img_base64 = base64.b64encode(img_str).decode()
        return {'img_base64': img_base64, 'tumor_percentage': tumor_percentage}
    return render_template('segment.html')



# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
