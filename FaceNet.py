# import necessary libraries 
import os
import tensorflow as tf 
import PIL 
import logging
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img 
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static') 

# model file path 
model_path = 'models/FacialEmotionNet01.h5' 

# load the Facial Emotion detection model 
model = tf.keras.models.load_model(model_path)
Emotion_class = ['angry', 'happy', 'sad']
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

logging.basicConfig(filename="FaceNet.log", level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s') 


@app.route('/')
def home(): 
    return render_template('FaceNet.html') 

@app.route('/upload', methods = ['POST']) 
def upload(): 
    if request.method == 'POST': 
        # get image from form file
        data = request.files['Face_image'] 
        # remove unsafe characters and normalize filename
        file_name = secure_filename(data.filename) 
        # create image path
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name).replace('\\', '/') 
        data.save(image_path) # save image path
        
        # Convert the image to tensors 
        image = load_img(image_path, color_mode = 'grayscale', target_size = (48, 48))
        input_img = img_to_array(image) 
        input_arr = np.array([input_img])
        

        # Detection Facial Emotion 
        pred_probabilities = model.predict(input_arr)
        
        # Facial Emotion Detection class
        pred_class = pred_probabilities.argmax(axis = 1) 
        flat_probabilities = pred_class[0]
        pred_emotion = Emotion_class[flat_probabilities]
        
        # log inormation
        app.logger.info(f'The prediction probabilities {pred_probabilities} {image_path} gives a/an {pred_emotion} emotion')
        # return variables needed for final output 
        image_url = url_for('static', filename = image_path)
        return redirect(url_for('result', up_image_path = file_name, up_emotion = pred_emotion, img_url = image_url))

#@app.route('/uploads/<path:file_name') 
#def serve_upload(file_name): 
 #   return send_from_directory(app.config['UPLOAD_FOLDER'], file_name) 

@app.route('/result') 
def result(): 
    file_name = request.args.get('up_image_path')
    nemotion = request.args.get('up_emotion') 
    im_url = request.args.get('img_url') 

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    app.logger.info(f"The file path '{file_name}'")
    return render_template('FaceRecognition.html', path = im_url, emotion = nemotion, im_path = image_path, filepath = file_name) 
if __name__ == '__main__': 
    app.run(debug=True)
