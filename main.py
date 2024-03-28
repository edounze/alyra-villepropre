import cv2
import numpy as np
import tensorflow as tf

# Chargement du modèle pré-entraîné SSD MobileNet V2
# Choix des modèles : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md#tensorflow-2-detection-model-zoo
# model = tf.saved_model.load('data/saved_model')

model = tf.saved_model.load(r'C:\Users\charl\Documents\Projet\Villepropre\data\saved_model')


# Extraction du modèle en format TensorFlow Lite
# Fonction d'inférence à partir des signatures du modèle
infer = model.signatures['serving_default']

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if not ret:
        break
    
    # Convertion de l'image en RGB (OpenCV utilise BGR par défaut)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Préparer le tenseur d'entrée sans convertir en float32 cette fois
    input_tensor = tf.convert_to_tensor([image_rgb], dtype=tf.uint8)

    # Utiliser la signature d'inférence avec les entrées nommées correctement
    output_dict = infer(tf.constant(input_tensor))
    
    # Extraction des boîtes de détection, des scores, et classes
    boxes = output_dict['detection_boxes'].numpy()[0]
    scores = output_dict['detection_scores'].numpy()[0]
    classes = output_dict['detection_classes'].numpy()[0].astype(np.int32)

    height, width, _ = image.shape

    # Boucler sur chaque détection
    for i in range(boxes.shape[0]):
        if scores[i] > 0.5:  # Seuil de confiance
            # La classe 44 correspond à "bouteille" dans le modèle COCO
            if classes[i] == 44:
                box = boxes[i]
                y_min, x_min, y_max, x_max = box
                x_min, x_max, y_min, y_max = (x_min * width, x_max * width,
                                              y_min * height, y_max * height)
                x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

                # Dessine un rectangle autour de la bouteille
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Afficher l'image
    cv2.imshow('Detection de bouteilles en plastique', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



