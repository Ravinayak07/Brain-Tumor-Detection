from flask import Flask, render_template, request, redirect, url_for
import cv2 as cv
import numpy as np
import base64
import io
from DisplayTumor import DisplayTumor

# Create the Flask application
app = Flask(__name__)
dt = DisplayTumor()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/index.html', methods=['GET', 'POST'])
def process():
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
        dt.displayTumor()
        img_str = cv.imencode('.jpg', dt.getImage())[1].tostring()
        img_base64 = base64.b64encode(img_str).decode()
        return img_base64
    return render_template('index.html')


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
