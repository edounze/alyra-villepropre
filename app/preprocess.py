import tensorflow as tf

def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    # Lire l'image à partir du chemin du fichier
    img = tf.io.read_file(image_path)
    # Décoder l'image en utilisant le format d'image spécifié avec 3 canaux de couleur (Rouge, Vert, Bleu)
    img = tf.io.decode_image(img, channels=3)
    # Convertir les valeurs des pixels de l'image en des nombres entiers non signés de 8 bits (valeurs comprises entre 0 et 255)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    # Conserver une copie de l'image d'origine pour une utilisation ultérieure
    original_image = img
    # Redimensionner l'image à la taille spécifiée pour correspondre à la taille d'entrée requise par le modèle TFLite
    resized_img = tf.image.resize(img, input_size)
    # Ajouter une dimension supplémentaire à l'image pour indiquer le nombre de canaux de couleur (1 pour une image en niveaux de gris ou 3 pour une image couleur)
    resized_img = resized_img[tf.newaxis, :]
    # Convertir l'image redimensionnée en des nombres entiers non signés de 8 bits (valeurs comprises entre 0 et 255)
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    # Renvoyer l'image redimensionnée et l'image d'origine
    return resized_img, original_image