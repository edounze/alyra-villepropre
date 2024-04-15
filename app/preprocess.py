import tensorflow as tf

def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    # Lecture de l'image à partir du chemin du fichier
    img = tf.io.read_file(image_path)
    
    # Décodage l'image en utilisant le format d'image spécifié avec 3 canaux de couleur (Rouge, Vert, Bleu)
    img = tf.io.decode_image(img, channels=3)

    # Convertion des valeurs des pixels de l'image en nombres entiers non signés de 8 bits (valeurs comprises entre 0 et 255)
    img = tf.image.convert_image_dtype(img, tf.uint8)

    # On conserve une copie de l'image d'origine pour une utilisation ultérieure
    original_image = img

    # Redimensionnement de l'image à la taille spécifiée pour correspondre à la taille d'entrée requise par le modèle TFLite
    resized_img = tf.image.resize(img, input_size)

    # Ajout d'une dimension supplémentaire à l'image pour indiquer le nombre de canaux de couleur (1 pour une image en niveaux de gris ou 3 pour une image couleur)
    resized_img = resized_img[tf.newaxis, :]

    # Convertion de l'image redimensionnée en nombres entiers non signés de 8 bits (valeurs comprises entre 0 et 255)
    resized_img = tf.cast(resized_img, dtype=tf.uint8)

    # On renvoie l'image redimensionnée et l'image d'origine
    return resized_img, original_image