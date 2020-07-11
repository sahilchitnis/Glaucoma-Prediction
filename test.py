from __future__ import absolute_import
import base64
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import io
import requests
from PIL import Image, ImageFile
import keras
from keras import backend
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from flask import request
from flask import jsonify
from flask import Flask, render_template
import h5py
import six
import os
import uuid
import tensorflow as tf
from keras.preprocessing import image
import os  # to import files
import numpy as np  # mathematical operations
import matplotlib.pyplot as plt
from keras.utils import np_utils
import cv2
import pickle
import tensorflow as tf
from numpy import asarray
import imutils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from keras.models import model_from_json
from keras.preprocessing import image
import h5py
from PIL import Image
import PIL
import os
import base64
import io
from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask
from keras.models import load_model
import numpy as np
import cv2
from keras_preprocessing.image import img_to_array
import imutils
from matplotlib import pyplot as plt
from keras.optimizers import Adam

# import tensorflow.contrib.keras as keras
app = Flask(__name__)


def crop_black(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    # Find contour and sort by contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ROI = image[y:y + h, x:x + w]
        break

    return ROI


def resize(img):
    width = 1024
    height = 720
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


def getROI(image):
    image_resized = resize(image)
    b, g, r = cv2.split(image_resized)
    g = cv2.GaussianBlur(g, (15, 15), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    g = ndimage.grey_opening(g, structure=kernel)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g)

    x0 = int(maxLoc[0]) - 120
    y0 = int(maxLoc[1]) - 120
    x1 = int(maxLoc[0]) + 130
    y1 = int(maxLoc[1]) + 130

    return image_resized[y0:y1, x0:x1]


def extract_bv(image):
    b, green_fundus, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)
    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    f4 = cv2.subtract(R3, contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    # vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if (shape == "circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
    blood_vessels = cv2.bitwise_not(finimage)
    blood_vessels = cv2.subtract(255, blood_vessels)
    return blood_vessels


kernel = np.ones((5, 5), np.uint8)


def image_inpaint(img):
    mask = extract_bv(img)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    img = cv2.inpaint(img, closing, 3, cv2.INPAINT_TELEA)
    return img


def res(img, width=224, height=224):
    img = imutils.resize(img, width)
    (h, w) = img.shape[:2]
    if h != w:
        img = cv2.resize(img, (224, 224))

    return img


amd_model = load_model('amd_model.hdf5')
print("amd model loaded..")
gl_model = load_model('glaucoma_model.hdf5')
print("glaucoma model loaded..")
oct_model = load_model('oct_diseases.hdf5')
print("oct model loaded..")


def prepare_amd(i):
    i = crop_black(i)
    i = image_inpaint(i)
    i = res(i)
    # cv2.imwrite(r'C:\Users\NEHA\Desktop\test2.jpg', i)
    i = i / 255
    return i.reshape(-1, 224, 224, 3)


def prepare_gl(i):
    i = image_inpaint(i)
    i = getROI(i)
    i = res(i)
    # cv2.imwrite(r'C:\Users\NEHA\Desktop\test2.jpg', i)
    i = i / 255
    return i.reshape(-1, 224, 224, 3)


def resoct(img, width=224, height=224):
    img = (np.float32(img))
    img1 = cv2.resize(img, (width, height))
    return img1


def prepare_oct(i):
    img = resoct(i)
    img = img / 255
    return img


def model_predict(img_path, model):
    '''
        Args:
            -- img_path : an URL path where a given image is stored.
            -- model : a given Keras CNN model.
    '''

    IMG = cv2.imread(img_path)
    print(type(IMG))
    img1 = np.asarray(IMG)
    test_amd = prepare_amd(img1)
    test_gl = prepare_gl(img1)
    amd = amd_model.predict(test_amd)
    print(amd)
    gl = gl_model.predict(test_gl)
    print(gl)
    if amd < 0.5 and gl < 0.5:
        print("NORMAL")
    elif amd > gl and amd > 0.5:
        print("AMD")
    else:
        if amd < gl and gl > 0.5:
            print("GLAUCOMA")

    result = [amd, gl]
    return result


def model_predict_oct(img_path, model):
    '''
        Args:
            -- img_path : an URL path where a given image is stored.
            -- model : a given Keras CNN model.
    '''

    IMG = image.load_img(img_path)
    print(type(IMG))

    new_img1 = []
    img = prepare_oct(IMG)
    new_img1.append(img)
    new_img1 = np.asarray(new_img1)

    prediction = model.predict(new_img1)
    print(prediction)

    return prediction


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        amd = []
        gl = []
        amd.append(model_predict(file_path, amd_model))
        gl.append(model_predict(file_path, gl_model))
        print("I think that is ")
        if amd[0][0] < 0.5 and gl[0][1] < 0.5:
            return "NORMAL"
        elif amd[0][0] > gl[0][1] and amd[0][0] > 0.5:
            return "AMD"
        else:
            if amd[0][0] < gl[0][1] and gl[0][1] > 0.5:
                return "GLAUCOMA"

        # print('I think that is {}.'.format(predicted_class.lower()))


@app.route('/predictoct', methods=['GET', 'POST'])
def upload_oct():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        prediction = model_predict_oct(file_path, oct_model)
        print("I think that is ")
        prediction1 = np.round(prediction, decimals=0)
        print(prediction1)
        if prediction1[0][0] == 1:  # Normal
            print("Normal")
            return "Normal"
        elif prediction1[0][1] == 1:  # Drusen
            print("Drusen")
            return "Drusen"
        elif prediction1[0][2] == 1:  # AMD
            print("AMD")
            return "AMD"
        elif prediction1[0][3] == 1:  # CNV
            print("CNV")
            return "CNV"
        else:
            print("wrong value")
            return "wrong value"


@app.route('/')
def index():
    # Main Page
    return render_template('index.html')


@app.route('/index')
def indexh():
    # Main Page
    return render_template('index.html')


@app.route('/about_us')
def about_us():
    # Main Page
    return render_template('about_us.html')


@app.route('/documentation')
def documentation():
    # Main Page
    return render_template('documentation.html')


@app.route('/funpredict', methods=['GET', 'POST'])
def funpredict():
    # Main Page
    return render_template('funpredict.html')


@app.route('/help')
def help():
    # Main Page
    return render_template('help.html')


@app.route('/octpredict', methods=['GET'])
def octpredict():
    # Main Page
    return render_template('octpredict.html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=False, threaded=False)
