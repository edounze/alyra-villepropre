from flask import Flask, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Charger le modèle TensorFlow
model = tf.saved_model.load(r'C:\Users\charl\Documents\Projet\Villepropre\data\saved_model')
infer = model.signatures['serving_default']

def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la webcam.")
    else:
        print("Webcam ouverte avec succès.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire la vidéo.")
            break

        # Convertir l'image pour TensorFlow
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(img, dtype=tf.uint8)
        img = tf.expand_dims(img, 0)  # Ajouter une dimension batch

        # Effectuer la détection
        output = infer(img)

        # Traitement des sorties, exemple simple
        boxes = output['detection_boxes'].numpy()[0]  # Exemple de boîtes de délimitation
        scores = output['detection_scores'].numpy()[0]
        classes = output['detection_classes'].numpy()[0]

        height, width, _ = frame.shape
        for i in range(boxes.shape[0]):
            if scores[i] > 0.5:  # Seuil de confiance
                y_min, x_min, y_max, x_max = boxes[i]
                (left, right, top, bottom) = (x_min * width, x_max * width, 
                                              y_min * height, y_max * height)
                left, right, top, bottom = int(left), int(right), int(top), int(bottom)

                # Dessiner un rectangle autour de l'objet détecté
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Convertir l'image pour Flask Response
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
