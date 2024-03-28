# Python libraries
import configparser

# Deep learning libraries
import cv2
import tensorflow as tf
import numpy as np

# Flask
from flask import Flask, Response, render_template

app = Flask(__name__)


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

# Définition de la fonction generate_frames qui sera utilisée pour générer et traiter les frames de la vidéo
def generate_frames():
    # Initialisation de la capture vidéo à partir de la webcam (0 représente la première webcam disponible)
    cap = cv2.VideoCapture(0)

    # Boucle infinie pour lire les frames de la vidéo en continu
    while True:
        # Lecture d'une frame de la vidéo. 'success' est un booléen indiquant si la lecture a réussi, 'image' est la frame lue
        success, image = cap.read()

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

        # Génération du flux vidéo avec le frame encodé, prêt à être envoyé au client
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    # Page d'accueil
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # Page utilisé pour le flux vidéo (voir fichier templates/index.html)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
