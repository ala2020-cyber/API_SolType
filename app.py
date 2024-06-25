from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pymongo import MongoClient
from datetime import datetime
import gdown
from dotenv import load_dotenv

app = Flask(__name__)


# Charger les variables d'environnement à partir du fichier .env
load_dotenv()


UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


file_id = '1jtvL6rC0ogPw4FgcnXVYGnuHAviy4IeP'
url = f'https://drive.google.com/uc?id={file_id}'
output= 'soil_recognition_model.h5'

if not os.path.exists(output):
    print("téléchargement du modèle depuis Google Drive ...")
    gdown.download(url,output,quiet=False)
    print("Téléchargement terminé.")

# charger le modèle
model = load_model(output)  # Charger le modèle à partir du fichier .h5

# Connect to MongoDB
MongoURI=os.getenv('MONGO_URI')
client = MongoClient(MongoURI)
db = client['SysArrosage']
collection = db['predictions']


# Function to predict soil type
def predict_soil_type(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    class_indices = {0: "Alluvial", 1: "Black", 2: "Clay", 3: "Red"}
    predicted_class = class_indices[np.argmax(predictions)]
    return predicted_class

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        soil_type = predict_soil_type(filepath)

        # Store the response in MongoDB
        prediction_record = {
            "filename": file.filename,
            "filepath": filepath,
            "soil_type": soil_type,
            "timestamp": datetime.utcnow()
        }
        collection.insert_one(prediction_record)

        return jsonify({"message": "File uploaded successfully", "filepath": filepath, "soil_type": soil_type}), 200

@app.route('/latest_prediction', methods=['GET'])
def latest_prediction():
    # Find the most recent prediction
    latest_record = collection.find_one(sort=[("timestamp", -1)])
    if latest_record:
        return jsonify({
            "filename": latest_record['filename'],
            "filepath": latest_record['filepath'],
            "soil_type": latest_record['soil_type'],
            "timestamp": latest_record['timestamp']
        }), 200
    else:
        return jsonify({"error": "No predictions found"}), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
