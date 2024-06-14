from flask import Flask, request, render_template, redirect, url_for
import mysql.connector
from werkzeug.utils import secure_filename
import base64
import tensorflow as tf
import numpy as np
import cv2
import os
import json
import tensorflow_hub as hub

app = Flask(__name__)

db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='dog_records'
)

def load_model(model_path):
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model

model=load_model("20240514-23401715730025-fill_model_mobilenetv2-Adam.h5")
BATCH_SIZE=32
IMG_SIZE=224
with open("labels.json", "r") as f:
    label_dict = json.loads(f.read())

#create a function for preprocessing
def process_img(image_path):

  #Read an image file
  image=tf.io.read_file(image_path)
  #Turn the jpeg image into Tensor  with 3 color channels
  image=tf.image.decode_jpeg(image,channels=3)
  #Convert the color channel images from 0-255 to 0-1 values
  image=tf.image.convert_image_dtype(image,tf.float32)
  #Resize the Image to (244,244)
  image=tf.image.resize(image,size=[IMG_SIZE,IMG_SIZE])

  return image

def get_image_label(image_path,label):
  image=process_img(image_path)
  return image,label

def create_batches(x,y=None,baatch_size=BATCH_SIZE,valid_data=False,test_data=False):
  #If the data is test data we dont have labels
  if test_data:
    print("Creating test data batches...")
    data=tf.data.Dataset.from_tensor_slices(tf.constant(x))
    data_batch=data.map(process_img).batch(BATCH_SIZE)
    return data_batch

  #If the data is valid dataset,we dont need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data=tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
    data_batch=data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    print("Creating a Training data batches...")
    data=tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
    data=data.shuffle(buffer_size=len(x))
    data_batch=data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch
  
def get_pred_label(prediction_probabilities):
  print(str(np.argmax(prediction_probabilities)))
  return label_dict[str(np.argmax(prediction_probabilities))]

def classify_image(image_path):
    custom_image_paths = [image_path]
    custom_data=create_batches(custom_image_paths,test_data=True)
    custom__preds=model.predict(custom_data)
    custom_preds_labels=[get_pred_label(custom__preds[i]) for i in range(len(custom__preds))]
    top_3_indices = np.argsort(custom__preds[0])[::-1][:3]
    top_3_label = [label_dict[str(index)] for index in top_3_indices]
    print("top_3_label", top_3_label)
    return top_3_label

def get_dog_records():
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM dog")
    records = cursor.fetchall()
    cursor.close()
    return records

def query_dogs_by_breed(breed):
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM dog WHERE breed LIKE %s", (f"%{breed}%",))
    records = cursor.fetchall()
    cursor.close()
    return records

@app.route('/')
def index():
    dog_records = get_dog_records()
    for dog in dog_records:
        dog["photo"] = base64.b64encode(dog["photo"]).decode('utf-8')
    return render_template('index2.html', dogs=dog_records)


@app.route('/photo-upload', methods=['GET', 'POST'])
def photo_upload():
    if request.method == 'POST':
        breed_prediction = request.form['type']           
        color = request.form['color']
        address = request.form['address']
        phone = request.form['phone']
        photo = request.files['photo']

        if photo.filename == '':
            return 'No selected file'
        if photo:
            filename = secure_filename(photo.filename)
            photo_path = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            photo.save(photo_path)
            if breed_prediction == "":
                print("selam")
                breed_prediction = classify_image(photo_path)
            with open(photo_path, 'rb') as file:
                photo_data = file.read()

            cursor = db.cursor()
            query = """
            INSERT INTO dog (breed, color, photo, address, phone_number)
            VALUES (%s, %s, %s, %s, %s)
            """

            # Remove the saved file after prediction
            os.remove(photo_path)
            if type(breed_prediction) == list:               
                cursor.execute(query, (breed_prediction[0], color, photo_data, address, phone))
                db.commit()
                cursor.close()
                return redirect(url_for('search', search_term=breed_prediction[0], pred_1=breed_prediction[1], pred_2=breed_prediction[2]))
            
            cursor.execute(query, (breed_prediction, color, photo_data, address, phone))
            db.commit()
            cursor.close()    
            return redirect(url_for('search', search_term=breed_prediction))

        return 'Error saving photo'

    return render_template('photo_upload.html')


@app.route('/search')
def search():
    search_term = request.args.get('search_term')
    pred_1 = request.args.get('pred_1')
    pred_2 = request.args.get('pred_2')
    dogs = query_dogs_by_breed(search_term)
    for dog in dogs:
        dog["photo"] = base64.b64encode(dog["photo"]).decode('utf-8')
    return render_template('index2.html', search_term=search_term, dogs=dogs, pred_1=pred_1, pred_2=pred_2)


if __name__ == '__main__':
    app.run()
