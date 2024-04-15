# Python libraries
import configparser, os
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging below ERROR level


# Deep learning libraries
import cv2
import tensorflow as tf

from pathlib import Path


# Flask
from flask import Flask, Response, request, render_template # send_file, jsonify, redirect, url_for, 


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

app = Flask(__name__, static_url_path='/static')

# Model loading
model_path = 'train/bin/greengardians.tflite'

# Ensure the model file exists
if not Path(model_path).is_file():
    raise FileNotFoundError(f"Model file not found at {model_path}")

try:
    print('Allocate Tensors')
   # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
except Exception as e:
    raise RuntimeError("Failed to allocate tensors in the interpreter") from e


@app.route('/')
def index():
    # Page d'accueil
    return render_template('index.html')    
    # array = np.arange(0, 737280, 1, np.uint8)     
    # # check type of array 
    # print(type(array)) 
    
    # # our array will be of width  
    # # 737280 pixels That means it  
    # # will be a long dark line 
    # print(array.shape) 
    
    # # Reshape the array into a  
    # # familiar resoluition 
    # image_Array = np.reshape(array, (1024, 720)) 
    
    # # show the shape of the array 
    # print(image_Array.shape) 

    # # show the array 
    # print(image_Array) 
    # sendToWebhook(image_Array)

    # return Response('Hello world')

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
        
        try:
            print('Detecting image')
            detection_result_image = draw_bounding_boxes(
                TEMP_FILE,
                interpreter,
                threshold=DETECTION_THRESHOLD
            )
        except Exception as e:
            raise RuntimeError("Failed to detect image") from e

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
