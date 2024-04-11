# Python libraries
import configparser

# Deep learning libraries
import cv2
import tensorflow as tf
import numpy as np

# Flask
from flask import Flask, Response, request, render_template, jsonify, redirect, url_for, send_file
from aiortc import RTCPeerConnection, RTCSessionDescription

import io
import json
import uuid
import asyncio
import logging
import time
import base64
import requests



# Création d'un objet ConfigParser
config = configparser.ConfigParser()
# Lecture du fichier de configuration
config.read('config.ini')

# Chargement du modèle pré-entraîné
# model = tf.saved_model.load('data/saved_model')
# model = tf.saved_model.load('chemin_vers_votre_modèle')
# Récupération du chemin du modèle à partir du fichier de configuration
model_path = config['DEFAULT']['ModelPath']
model = tf.saved_model.load(model_path)

infer = model.signatures['serving_default']

# Set to keep track of RTCPeerConnection instances
pcs = set()

def draw_bounding_boxes(image):

    # Conversion de la frame de BGR (format par défaut d'OpenCV) en RGB pour le traitement par TensorFlow
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Conversion de l'image RGB en un tenseur TensorFlow, ajout d'une dimension supplémentaire pour le batch
    input_tensor = tf.convert_to_tensor([image_rgb], dtype=tf.uint8)

    # Utilisation du modèle pour effectuer la détection d'objets sur l'image, stockage des résultats dans output_dict
    output_dict = infer(tf.constant(input_tensor))

    # Extraction des boîtes englobantes, des scores et des classes des objets détectés depuis output_dict
    boxes = output_dict['detection_boxes'].numpy()[0]
    scores = output_dict['detection_scores'].numpy()[0]
    classes = output_dict['detection_classes'].numpy()[0].astype(np.int32)

    # Obtention des dimensions de l'image pour le calcul des coordonnées réelles des boîtes englobantes
    height, width, _ = image.shape

    # Itération sur chaque objet détecté
    for i in range(boxes.shape[0]):
        # Si le score de détection est supérieur à 0.5 et que la classe détectée est 44 (bouteille), dessiner un rectangle
        if scores[i] > 0.5 and classes[i] == 44:  # Classe "bouteille"
            print('Bouteille detectée')
            # Extraction des coordonnées de la boîte englobante de l'objet
            box = boxes[i]
            y_min, x_min, y_max, x_max = box
            # Dessin du rectangle autour de l'objet sur l'image
            cv2.rectangle(image, (int(x_min * width), int(y_min * height)),
                            (int(x_max * width), int(y_max * height)), (0, 255, 0), 2)
            # Envoie vers Symfony
            sendToWebhook(image)

    return image

# Génère et traite les frames de la vidéo
def generate_frames():
    # Initialisation de la capture vidéo à partir de la webcam (0 représente la première webcam disponible)
    # Pour utiliser une caméra IP, il faut le lien RTSP (ex : 'rtsp://[USER]:[PASS]@[IP_ADDRESS]:[RTSP PORT]/media/video[STREAM TYPE]')
    camera = cv2.VideoCapture(0)

    # Boucle infinie pour lire les frames de la vidéo en continu
    while True:
        start_time = time.time()
        # Lecture d'une frame de la vidéo. 'success' est un booléen indiquant si la lecture a réussi, 'image' est la frame lue
        success, image = camera.read()

        # Si la lecture de la frame échoue, sortir de la boucle
        if not success:
            break

        # Conversion de la frame de BGR (format par défaut d'OpenCV) en RGB pour le traitement par TensorFlow
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Conversion de l'image RGB en un tenseur TensorFlow, ajout d'une dimension supplémentaire pour le batch
        input_tensor = tf.convert_to_tensor([image_rgb], dtype=tf.uint8)

        # Utilisation du modèle pour effectuer la détection d'objets sur l'image, stockage des résultats dans output_dict
        output_dict = infer(tf.constant(input_tensor))

        # Extraction des boîtes englobantes, des scores et des classes des objets détectés depuis output_dict
        boxes = output_dict['detection_boxes'].numpy()[0]
        scores = output_dict['detection_scores'].numpy()[0]
        classes = output_dict['detection_classes'].numpy()[0].astype(np.int32)

        # Obtention des dimensions de l'image pour le calcul des coordonnées réelles des boîtes englobantes
        height, width, _ = image.shape

        # Itération sur chaque objet détecté
        for i in range(boxes.shape[0]):
            # Si le score de détection est supérieur à 0.5 et que la classe détectée est 44 (bouteille), dessiner un rectangle
            if scores[i] > 0.5 and classes[i] == 44:  # Classe "bouteille"
                # Extraction des coordonnées de la boîte englobante de l'objet
                box = boxes[i]
                y_min, x_min, y_max, x_max = box
                # Dessin du rectangle autour de l'objet sur l'image
                cv2.rectangle(image, (int(x_min * width), int(y_min * height)),
                              (int(x_max * width), int(y_max * height)), (0, 255, 0), 2)
                
                # Envoie vers Symfony
                sendToWebhook(image)

        # Encodage de l'image traitée au format JPEG pour l'envoi via le flux vidéo
        _, buffer = cv2.imencode('.jpg', image)

        # Préparation pour envoyer sur le monitoring serveur
        frame = buffer.tobytes()

        elapsed_time = time.time() - start_time
        logging.debug(f"Frame generation time: {elapsed_time} seconds")

        # Génération du flux vidéo avec le frame encodé, prêt à être envoyé au client
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


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


@app.route('/')
def index():
    # Page d'accueil
    # return redirect(url_for('video_feed'))
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Page utilisée pour le flux vidéo (voir fichier templates/index.html)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/process', methods=['POST'])
def process_image():
    # Le début est le même que dans l'exemple précédent
    image_file = request.files['image']
    npimg = np.fromfile(image_file, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Detection de bouteilles + ajout de rectangles autour de l'objet detecté
    image = draw_bounding_boxes(image)

    # Encoder l'image traitée au format JPEG
    _, img_encoded  = cv2.imencode('.jpg', image)
    # Convertir le tableau d'octets encodé en JPEG en un objet de type bytes
    # image_bytes = encoded_image.tobytes()

    # # Créer un objet BytesIO à partir des bytes de l'image
    # byte_io = io.BytesIO(image_bytes)

    # Retourner l'image en tant que réponse HTTP de type 'image/jpeg'
    return Response(img_encoded.tobytes(), mimetype='image/jpeg')


# Lancement de l'application
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0')
