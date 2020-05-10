from flask import Flask, render_template, url_for, request,redirect
from flask_bootstrap import Bootstrap
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
Bootstrap(app)



@app.route('/',methods=['GET','POST'])
def index():
    global model
    if request.method == 'POST':
        className = ['NORMAL','PNEUMONIA']
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join('static',uploaded_file.filename)
            uploaded_file.save(image_path)
            img = load_img(image_path,target_size=(299,299))
            img = img_to_array(img)
            img = np.expand_dims(img,axis=0)
            prediction = model.predict(img)
            class_name = className[np.argmax(prediction)]
            print("CLASS : ",class_name)
            result = {
                'class_name' : class_name,
                'image_path' : image_path,
            }
            return render_template('index.html', result=result)
    return render_template('index.html',result={'class_name' : '','image_path' : ''})

if __name__== "__main__":
    with open('model 1.2 json/model.json', 'r') as f:
        modelJ = f.read()
    model = tf.keras.models.model_from_json(modelJ)
    model.load_weights('model 1.2 json/weights.h5')
    app.run(debug=True)
