import cv2
import numpy as np
import tensorflow as tf

# Chemin vers le modèle Keras sauvegardé
model_path = 'train/garbage_detection_model.keras'

# Chargement du modèle Keras
model = tf.keras.models.load_model(model_path)

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if not ret:
        break

    # Convertion de l'image en RGB (OpenCV utilise BGR par défaut)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Redimensionner l'image à la taille attendue par le modèle (exemple : 180x180)
    # Remplacer (180, 180) par la taille d'entrée attendue par votre modèle
    resized_image = cv2.resize(image_rgb, (180, 180))
    
    # Normaliser l'image si nécessaire (exemple : les valeurs de pixels de 0-255 à 0-1)
    normalized_image = resized_image / 255.0

    # Ajouter une dimension batch à l'image
    image_batch = np.expand_dims(normalized_image, axis=0)

    # Faire une prédiction avec le modèle
    predictions = model.predict(image_batch)

    # Ici, vous devrez déterminer la meilleure façon de traiter les prédictions de votre modèle
    # Cela dépend de la sortie de votre modèle spécifique (par exemple, classification, détection, etc.)

    # Pour l'affichage, convertissez l'image RGB en BGR pour l'affichage avec OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow('Detection dechet', image_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
