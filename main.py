# Python libraries
import configparser, requests, os
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging below ERROR level


# Deep learning libraries
import cv2
import tensorflow as tf
import numpy as np

from pathlib import Path


# Flask
from flask import Flask, Response, request, render_template, send_file #, jsonify, redirect, url_for, 


from app.model import draw_bounding_boxes

# Création d'un objet ConfigParser
config = configparser.ConfigParser()
# Lecture du fichier de configuration
config.read('config.ini')

# Chargement du modèle pré-entraîné
# model = tf.saved_model.load('data/saved_model')
# model = tf.saved_model.load('chemin_vers_votre_modèle')
# Récupération du chemin du modèle à partir du fichier de configuration

DETECTION_THRESHOLD = float(config['DEFAULT']['DetectionThreshold'])

def sendToWebhook(image):
    # Encodage de l'image traitée au format JPEG pour l'envoi via le flux vidéo
    _, buffer = cv2.imencode('.jpg', image)

    # Préparation pour envoyer sur le monitoring serveur
    imageFrame = buffer.tobytes()

    # URL du webhook Symfony
    # webhook_url = 'https://vp.edounze.com/detection_endpoint'  # Assurez-vous de mettre la bonne route de votre API Symfony
    webhook_url = config['DEFAULT']['WebHook']


    # Création du formulaire multipart/form-data
    # Les clés du dictionnaire correspondent aux noms des champs attendus par votre API Symfony
    latitude = 0
    longitude = 0
    files = {
        'image': ('image.jpg', imageFrame, 'image/jpeg'),
        'latitude': (None, str(latitude)),
        'longitude': (None, str(longitude)),
    }

    # Préparation des headers HTTP pour indiquer qu'il s'agit d'un fichier
    # headers = {'Content-Type': 'application/octet-stream'}

    # Envoi de la requête POST avec l'image en binaire
    # response = requests.post(webhook_url, data=imageFrame, headers=headers)

    # Envoi de la requête POST avec le formulaire multipart incluant l'image et les métadonnées
    response = requests.post(webhook_url, files=files)

    print(webhook_url,"Webhook response status code:", response.status_code)
    print("Webhook response:", response.text)

app = Flask(__name__, static_url_path='/static')

# Model loading
model_path = 'train/bin/greengardians.tflite'

# Ensure the model file exists
if not Path(model_path).is_file():
    raise FileNotFoundError(f"Model file not found at {model_path}")

interpreter = tf.lite.Interpreter(model_path=model_path)
try:
    print('Allocate Tensors')
    interpreter.allocate_tensors()
except Exception as e:
    raise RuntimeError("Failed to allocate tensors in the interpreter") from e

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()


@app.route('/')
def index():
    # Page d'accueil
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "Aucun fichier 'image' n'a été trouvé dans la requête", 400

    image_file = request.files['image']
    if image_file.filename == '':
        return "Aucun fichier sélectionné pour l'upload", 400

    # return "This is a test response", 200
    try:
        # Sauvegarde du fichier un sous dossier tmp du dossier courant
        # temp_file = os.path.join('tmp', str(uuid.uuid4()) + os.path.splitext(image_file.filename)[1])
        TEMP_FILE = os.path.join('tmp', image_file.filename)
        image_file.save(TEMP_FILE)

        print (f"Processing image: {TEMP_FILE}")

        # # Now you can use the temp_file.name in your draw_bounding_boxes function
        # detection_result_image = draw_bounding_boxes(TEMP_FILE, threshold=DETECTION_THRESHOLD)

        detection_result_image = draw_bounding_boxes(
            TEMP_FILE,
            interpreter,
            threshold=DETECTION_THRESHOLD
        )

        # Envoie vers Symfony (si nécessaire)
        # sendToWebhook(detection_result_image)

        # Encoder l'image traitée au format JPEG
        _, img_encoded = cv2.imencode('.jpg', detection_result_image)
        
        # # Retourner l'image en tant que réponse HTTP de type 'image/jpeg'
        response = Response(img_encoded.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        print(f"Error during object detection: {e}")
        traceback.print_exc()  # Provides a traceback that helps in debugging
        return "Error processing the image", 500
    finally:
        # Suppression du fichier temporaire
        print(f"Deleting temporary file: {TEMP_FILE}")
        os.remove(TEMP_FILE)

    return response


# Lancement de l'application
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0')
