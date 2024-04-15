import numpy as np
import cv2
from .preprocess import preprocess_image

import configparser, requests


# Load the labels into a list
classe_list = ['plastique', 'canette']

# Define a list of colors for visualization
# COLORS = np.random.randint(0, 255, size=(len(classe_list), 3), dtype=np.uint8)
COLORS = {
    'plastique': (0, 255, 0),  # Green
    'canette': (0, 0, 255)  # Red
}


def sendToWebhook(image, category = 'plastique'):    
  print("Sending to webhook")
  # Création d'un objet ConfigParser
  config = configparser.ConfigParser()
  # Lecture du fichier de configuration
  config.read('config.ini')

  # Encodage de l'image traitée au format JPEG pour l'envoi via le flux vidéo
  _, buffer = cv2.imencode('.png', image)

  # Préparation pour envoyer sur le monitoring serveur
  imageFrame = buffer.tobytes()

  # URL du webhook Symfony
  # webhook_url = 'https://vp.edounze.com/detection_endpoint'
  webhook_url = config['DEFAULT']['WebHook']

  # Création du formulaire multipart/form-data
  # Les clés du dictionnaire correspondent aux noms des champs attendus par l'API Symfony
  latitude = 48
  longitude = 23
  files = {
      'image': ('image.jpg', imageFrame, 'image/jpeg'),
      'latitude': (None, str(latitude)),
      'longitude': (None, str(longitude)),
      'category' : (None, category)
  }

  # Préparation des headers HTTP pour indiquer qu'il s'agit d'un fichier
  # headers = {'Content-Type': 'application/octet-stream'}

  # Envoi de la requête POST avec l'image en binaire
  # response = requests.post(webhook_url, data=imageFrame, headers=headers)

  # Envoi de la requête POST avec le formulaire multipart incluant l'image et les métadonnées
  response = requests.post(webhook_url, files=files)

  print(webhook_url,"Webhook response status code:", response.status_code)
  print("Webhook response:", response.text)

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  signature_fn = interpreter.get_signature_runner()

  image = np.float32(image)

  # Feed the input image to the model
  output = signature_fn(images=image)

  # Get all outputs from the model
  count = int(np.squeeze(output['output_0']))
  scores = np.squeeze(output['output_1'])
  classes = np.squeeze(output['output_2'])
  boxes = np.squeeze(output['output_3'])

  results = []
  print('threshold:::', threshold)  
  for i in range(count):
    # print('score:::', i,  scores[i])
    if scores[i] >= threshold:
      print('scores:::', scores[i])
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  
  print("results", results)
  return results

def draw_bounding_boxes(image_path, interpreter, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path,
      (input_height, input_width)
  )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)

  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])
    class_name = classe_list[class_id]

    # Draw the bounding box and label on the image
    # color = [int(c) for c in COLORS[class_id]]
    color = COLORS[class_name]  # Get color based on class name
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(class_name, obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Send to webhook
    sendToWebhook(original_image_np, category=class_name)

  # Return the final image
  # original_float32 = original_image_np.astype(np.float32)
  # return original_float32
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8



# from PIL import Image
# DETECTION_THRESHOLD = 0.3
# IMG_FILE = str(PROJECT_FOLDER) + '/train/datasets/dataset.voc/test/IMG_20231025_095509_jpg.rf.c83cb3628dda974ef28a37ce404b9ab7.jpg'
# detection_result_image = draw_bounding_boxes(
#     IMG_FILE,
#     interpreter,
#     threshold=DETECTION_THRESHOLD
# )

# # Show the detection result
# Image.fromarray(detection_result_image)