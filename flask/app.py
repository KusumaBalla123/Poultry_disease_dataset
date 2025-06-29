import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("poultry_disease_model.h5")


@app.route("/")
def home():
    return render_template("page.html")

@app.route("/pred.html")
def pred_page():
    return render_template("pred.html")
@app.route("/about.html")
def about():
    return render_template("about.html")
      

@app.route("/team_info.html")
def contact():
    return render_template("team_info.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "pc_image" not in request.files:
        return "No file uploaded", 400

    f = request.files["pc_image"]
    if f.filename == "":
        return "No file selected", 400

    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True) 
    img_path = os.path.join(upload_folder, f.filename)
    f.save(img_path)

    # Preprocess the image
    img = load_img(img_path, target_size=(224, 224))
    image_array = img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction_index = np.argmax(model.predict(image_array), axis=1)[0]
    labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
    prediction = labels[prediction_index]


    return render_template("contact.html", predict=prediction, image_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
