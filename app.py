import os
import sqlite3
from flask import Flask, flash, redirect, render_template, request, session, url_for, jsonify
from flask_session import Session
import onnxruntime as rt
import onnx
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from datetime import datetime
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD'] = os.path.join('static', 'uploads')
app.config['MODEL'] = os.path.join('static', 'models')
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

onnx_model = onnx.load(os.path.join(app.config['MODEL'], 'plantnew.onnx'))
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(os.path.join(app.config['MODEL'], 'plantnew.onnx'), providers=providers)
output_names = [n.name for n in onnx_model.graph.output]
input_name = m.get_inputs()[0].name

Session(app)

@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

mapper = {
    0: 'compostable material',
    1: 'recyclable material',
    2: 'disposable material (trash)',
}

@app.route("/")

def index():
    return render_template("index.html")

@app.route("/pic", methods = ['GET', 'POST'])

def pic():
    if request.method == 'POST':
        data = request.get_json()  # Get the JSON data (image in Base64 format)
        image_data_url = data.get('image')
        image_data = image_data_url.split(',')[1]
        # image_binary = base64.b64decode(image_data)
        # image = Image.open(BytesIO(image_binary))
        # image.save('./static/uploads/hi.png')

        # Store the image in session
        session['image'] = image_data

        # Redirect to the show_image route to display the image
        return redirect(url_for('classify'))
    return render_template('pic.html')

@app.route("/classify")

def classify():
    string = 'data:image/png;base64, ' + session.get('image')
    base64_string = string.split(',')[1]
    decoded_image = base64.b64decode(base64_string)
    img_tensor = tf.io.decode_image(decoded_image)

    img_tensor = tf.image.resize(img_tensor, [384, 384])
    img_tensor = tf.keras.applications.resnet.preprocess_input(img_tensor)
    img_tensor = img_tensor.numpy()
    img_tensor = np.expand_dims(img_tensor, 0)
    x = img_tensor
    onnx_pred = m.run(output_names, {input_name: x})
    y_pred = tf.nn.softmax(onnx_pred)
    y_pred = y_pred.numpy()
    classer = np.argmax(y_pred[0][0])
    session['prediction'] = classer
    classification = "The item is a " + mapper[classer] + "."
    return render_template("classify.html", img = string, prediction = classification)

@app.route("/upload_image", methods = ['GET', 'POST'])

def upload_image():
    correct = request.form.get('correct')
    image_data = session.get('image')
    image_binary = base64.b64decode(image_data)
    if correct == 'true':  # If the image is correct
        conn = sqlite3.connect("recycle.db")
        db = conn.cursor()
        db.executemany("INSERT INTO NewTrain (image, label) VALUES (?, ?)", [(image_binary, str(session.get('prediction')))])
        conn.commit()
        conn.close()
        return jsonify({'status': 'Image uploaded successfully!'})

    else:  # If the image is incorrect, get the label
        label = request.form.get('label')
        session['prediction'] = label
        conn = sqlite3.connect("recycle.db")
        db = conn.cursor()
        db.executemany("INSERT INTO NewTrain (image, label) VALUES (?, ?)", [(image_binary, str(session.get('prediction')))])
        conn.commit()
        conn.close()
        return jsonify({'status': 'Image uploaded successfully!'})


@app.route("/picmult", methods = ['POST', 'GET'])

def picmult():
    if request.method == 'POST':
        data = request.get_json()  # Get the JSON data (image in Base64 format)
        image_data_url = data.get('imagr')
        session['img_mult'] = image_data_url
        print("SHOW UP: " + str(len(image_data_url)))
        return redirect('/classmult')
    return render_template('picmult.html')

@app.route("/classmult", methods = ['POST', 'GET'])

def classmult():
    images = []
    image_data = session.get('img_mult')
    for image in image_data:
        print(image[:30])
        base64_string = image.split(',')[1]
        decoded_image = base64.b64decode(base64_string)
        img_tensor = tf.io.decode_image(decoded_image)

        img_tensor = tf.image.resize(img_tensor, [384, 384])
        img_tensor = tf.keras.applications.resnet.preprocess_input(img_tensor)
        img_tensor = img_tensor.numpy()
        img_tensor = np.expand_dims(img_tensor, 0)
        x = img_tensor
        onnx_pred = m.run(output_names, {input_name: x})
        y_pred = tf.nn.softmax(onnx_pred)
        y_pred = y_pred.numpy()
        classer = np.argmax(y_pred[0][0])
        session['prediction'] = classer
        classification = "The item is a " + mapper[classer] + "."
        images.append({'img_data': image, 'pred': classification})
    return render_template('classmult.html', images = images)

@app.route("/upload", methods = ['POST', 'GET'])
def upload():
    if request.method == 'POST':
        try:
            data = request.get_json()

            if 'imaginger' not in data:
                return jsonify({'message': 'No images uploaded'}), 400

            images_base64 = data['imaginger']

            # Check if more than 5 images are uploaded
            if len(images_base64) > 5:
                return jsonify({'message': 'You can only upload up to 5 images.'}), 400

            session['img_mult'] = images_base64

            # Redirect the user to another page after successful upload
            return jsonify({'message': 'Images uploaded', 'redirect_url': url_for('classmult')}), 200

        except Exception as e:
            # Log the exception and return a JSON error response
            print(f"Error: {e}")
            return jsonify({'message': 'An error occurred while processing images.'}), 500
    return render_template('upload.html')

@app.route('/showcase', methods = ['GET', 'POST'])
def showcase():
    return render_template('showcase.html')

@app.route('/reccenters', methods = ['GET', 'POST'])
def reccenters():
    return render_template('loc.html')
