from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
import pickle 

import cv2

import numpy as np
import os

from flask import Flask, render_template, request

app = Flask(__name__)

# load model and data
with open('static/dog_names.txt', 'rb') as fp:
    dog_names = pickle.load(fp)

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')

model = load_model("../dog_model")
dog_detection_model = load_model("../resnet_imagenet")

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=["GET", "POST"])

def upload_images():
    """Upload image, predict breed, construct prediction statement."""
    if request.method == 'POST':
        file = request.files['file']

        basepath = os.path.dirname(__file__)
        fpath = os.path.join(basepath, 'img_upload', file.filename)
        file.save(fpath)

        prediction = predict_dog_breed(fpath)
        dog_detection = dog_detector(fpath)
        human_detection = face_detector(fpath)

        return_statement = construct_return_statement(prediction, dog_detection, human_detection)

        return return_statement
    return None
 
def construct_return_statement(prediction, dog_detection, human_detection):
    """Construct a str return statement from predicted breed and results of dog/human detector."""
    if dog_detection:
        return 'This is a {}.'.format(prediction)
    elif human_detection:
        return 'This human looks like a {}.'.format(prediction)
    else:
        return 'No human (frontal-face) or dog detected in image.'


def face_detector(img_path):
    """Check if a face is  present in the image using cv2 library."""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(img_path):
    """Check if Resnet predicts a dog in the image."""
    img = preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(dog_detection_model.predict(img))

    return ((prediction <= 268) & (prediction >= 151)) 

def predict_dog_breed(img_path):
    """Predict dog breed using self-trained Resnet & transfer learning model."""
    # extract bottleneck features
    tensor = path_to_tensor(img_path)
    bottleneck_feature = ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def path_to_tensor(img_path):
    """Pre-process step of uploaded images"""
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)