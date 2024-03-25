# Entrainement par transfert du model COCO SSD MobileNet v1 pour détecter différents types de déchets
# Documentation : https://www.tensorflow.org/tutorials/images/transfer_learning?hl=fr
# - Charger le modèle pré-entraîné SSD MobileNet v1
# - Convertir le modèle en format TensorFlow Lite
# - Entraîner le modèle sur les images de déchets
# - Sauvegarder le modèle entraîné en format TensorFlow Lite

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds