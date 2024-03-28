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
            # Extraction des coordonnées de la boîte englobante de l'objet
            box = boxes[i]
            y_min, x_min, y_max, x_max = box
            # Dessin du rectangle autour de l'objet sur l'image
            cv2.rectangle(image, (int(x_min * width), int(y_min * height)),
                            (int(x_max * width), int(y_max * height)), (0, 255, 0), 2)

    return image

# Définition de la fonction generate_frames qui sera utilisée pour générer et traiter les frames de la vidéo
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

        # Encodage de l'image traitée au format JPEG pour l'envoi via le flux vidéo
        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        elapsed_time = time.time() - start_time
        logging.debug(f"Frame generation time: {elapsed_time} seconds")

        # Génération du flux vidéo avec le frame encodé, prêt à être envoyé au client
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


app = Flask(__name__, static_url_path='/static')

# Asynchronous function to handle offer exchange
async def offer_async():
    params = await request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create an RTCPeerConnection instance
    pc = RTCPeerConnection()

    # Generate a unique ID for the RTCPeerConnection
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pc_id = pc_id[:8]

    # Create a data channel named "chat"
    # pc.createDataChannel("chat")

    # Create and set the local description
    await pc.createOffer(offer)
    await pc.setLocalDescription(offer)

    # Prepare the response data with local SDP and type
    response_data = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    return jsonify(response_data)

# Wrapper function for running the asynchronous offer function
def offer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    future = asyncio.run_coroutine_threadsafe(offer_async(), loop)
    return future.result()

# Route to handle the offer request
@app.route('/offer', methods=['POST'])
def offer_route():
    return offer()


@app.route('/')
def index():
    # Page d'accueil
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Page utilisée pour le flux vidéo (voir fichier templates/index.html)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/process', methods=['POST'])
def process_image():
    # Le début est le même que dans l'exemple précédent
    image_file = request.files['image']
    image_stream = image_file.stream
    image_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Detection de bouteilles + ajout de rectangles autour de l'objet detecté
    image = draw_bounding_boxes(image)

    # Encoder l'image traitée au format JPEG
    _, encoded_image = cv2.imencode('.jpg', image)
    # Convertir le tableau d'octets encodé en JPEG en un objet de type bytes
    # image_bytes = encoded_image.tobytes()

    # # Créer un objet BytesIO à partir des bytes de l'image
    # byte_io = io.BytesIO(image_bytes)

    # Retourner l'image en tant que réponse HTTP de type 'image/jpeg'
    return Response(encoded_image.tobytes(), mimetype='image/jpeg')


# Lancement de l'application
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0')
