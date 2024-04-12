import tensorflow as tf

# Load the labels into a list
# classes = ['???'] * model.model_spec.config.num_classes
# label_map = model.model_spec.config.label_map
# for label_id, label_name in label_map.as_dict().items():
#   classes[label_id-1] = label_name

# # Define a list of colors for visualization
# COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    # img = image.read()
    img = tf.io.read_file(image_path)
    # Décodage de l'image (remplace tf.io.read_file et tf.io.decode_image)
    img = tf.io.decode_image(img, channels=3)
    # Convertir les types de données de l'image
    img = tf.image.convert_image_dtype(img, tf.float32)  # Utilisation de float32 si le modèle s'attend à cela
    original_image = img
    # Redimensionnement de l'image pour correspondre à l'entrée du modèle
    resized_img = tf.image.resize(img, input_size)
    # Ajout d'une dimension de batch
    resized_img = resized_img[tf.newaxis, :]
    # Convertir à nouveau en uint8 si le modèle s'attend à cela
    resized_img = tf.cast(resized_img, dtype=tf.uint8)

    return resized_img, original_image

