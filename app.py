from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from pymongo import MongoClient
from datetime import datetime
import gdown
from dotenv import load_dotenv

app = Flask(__name__)


# Charger les variables d'environnement à partir du fichier .env
load_dotenv()


# Connect to MongoDB
MongoURI=os.getenv('MONGO_URI')
client = MongoClient(MongoURI)
db = client['SysArrosage']
collection = db['predictions']




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
