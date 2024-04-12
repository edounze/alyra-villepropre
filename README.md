# Projet Ville Propre

## Définitions utiles

### Normalisation

- Moyenne nulle (0)
- Ecart type de 1
  La normalisation c’est le fait de modifier les données d’entrée d’un réseau de neurones de telle sorte que la moyenne soit nulle et l’écart-type égale à un,
  Cela permet d’avoir des données plus stables car appartenant à une même échelle. Par conséquent le réseau de neurones a plus de facilité à s’entraîner.
  La normalisation c’est donc le fait de formater ses données d’entrée pour faciliter le processus de Machine Learning.

```python
import numpy as np

data_list = [60, 9, 37, 14, 23, 4]

print(np.mean(data_list)) # Moyenne : 24.5
print(np.std(data_list)) # Ecart type : 19.102792117035317

data_list_norm = [(element - np.mean(data_list)) / np.std(data_list) for element in data_list]
print(data_list_norm) # [1.86, -0.81, 0.65, -0.55, -0.08, -1.07]

np.mean(l_norm), np.std(l_norm) # (0.0, 1.0)

```

### Batch Normalisation

Normaliser la sortie d’une couche dans le contexte du Deep Learning.
La normalisation après chaque couche (donc à l’entrée de chaque nouvelle couche) permet un meilleur apprentissage du réseau.

## Cycle de vie d'un modèle d'IA

### Deep et machine learning

1. Nettoyage et préparation de données :Etapes permettant de se préparer à l'entrainement
2. Sélection de l'algorithme et du modèle : Etape déterminant en fonction du type de problème et de la nature l'algo ou le modèle à utiliser
3. Entrainement du modèle : Etape centrale, elle permet d'entrainer le modèle, impliquant d'ajuster les paramètres pour de bonnes prédictions
4. Evaluation du modèle : Etape post entrainement, il est crucial d'évaluer les performances du modèles sur un ensemble de données de test
5. Déploiement du modèle : Etape finale, permettant d'exposer le modèle à des données de productions et plus seulement au jeu de test
6. Surveillance et maintenance : après le déploiement, il est crucial de surveiller en continu les performances du modèle et de le mettre à jour si nécessaire en fonction des feedback utilisateurs et de l'évolution des données
