import tensorflow as tf
import numpy as np
import cv2

from pathlib import Path

from .preprocess import preprocess_image

PROJECT_FOLDER = Path(Path.cwd())

MODEL_PATH = str(PROJECT_FOLDER) + '/train/bin/greengardians.tflite'

# Ensure the model file exists
if not Path(MODEL_PATH).is_file():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
try:
    print('Allocate Tensors')
    interpreter.allocate_tensors()
except Exception as e:
    raise RuntimeError("Failed to allocate tensors in the interpreter") from e

# Load the labels into a list
classes = ['plastique', 'canette']

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def detect_objects(image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  signature_fn = interpreter.get_signature_runner()

  # Feed the input image to the model
  output = signature_fn(images=image)

  results = {
        'bounding_box': 0,
        'class_id': 0,
        'score': 0
  }
  return results

def draw_bounding_boxes(image_path, threshold=0.5):
  """Run object detection on the input image and draw the detection results"""

  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path,
      (input_height, input_width)
  )

  # Run object detection on the input image
  results = detect_objects(preprocessed_image, threshold=threshold)

  return True


  # print('results', results)

  # # Plot the detection results on the input image
  # original_image_np = original_image.numpy().astype(np.uint8)
  # for obj in results:
  #   # Convert the object bounding box from relative coordinates to absolute
  #   # coordinates based on the original image resolution
  #   ymin, xmin, ymax, xmax = obj['bounding_box']
  #   xmin = int(xmin * original_image_np.shape[1])
  #   xmax = int(xmax * original_image_np.shape[1])
  #   ymin = int(ymin * original_image_np.shape[0])
  #   ymax = int(ymax * original_image_np.shape[0])

  #   # Find the class index of the current object
  #   class_id = int(obj['class_id'])

  #   # Draw the bounding box and label on the image
  #   color = [int(c) for c in COLORS[class_id]]
  #   cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
  #   # Make adjustments to make the label visible for all objects
  #   y = ymin - 15 if ymin - 15 > 15 else ymin + 15
  #   label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
  #   cv2.putText(original_image_np, label, (xmin, y),
  #       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # # Return the final image
  # original_uint8 = original_image_np.astype(np.uint8)
  # return original_uint8



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