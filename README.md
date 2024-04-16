# Projet Ville Propre

## Définitions utiles

### Accuracy

`L’accuracy répond à la question : combien d’individus ont été correctement prédits`
L’accuracy mesure l’efficacité d’un modèle à prédire correctement à la fois les individus positifs et négatifs.
L’accuracy est une métrique pour évaluer la performance des modèles de classification à 2 classes ou plus. L’accuracy peut être traduite par “précision” en français mais on risque alors de la confondre avec la métrique Precision.

L’accuracy permet de décrire la performance du modèle sur les individus positifs et négatifs de façon symétrique

`Elle mesure le taux de prédictions correctes sur l’ensemble des individus`

### CNN (Réseaux neuronaux convolutif)

Les CNN peuvent être considérés comme des extracteurs automatiques de caractéristiques au sein d’une image. Si NN signifie Neural Network (réseau de neurones), le C signifie convolutif.

La convolution est une technique qui utilise un filtre pour transformer une image en une autre, à l’aide d’opérations mathématiques.

Un CNN est comme un détective qui regarde attentivement une image, trouvant des motifs simples, puis des motifs plus complexes, pour finalement décider ce qu'il y a dans cette image. C'est un outil très puissant pour la reconnaissance d'objets dans les images.

<details>
Un réseau de neurones convolutionnel (CNN) est un type spécial de réseau de neurones inspiré par la manière dont le cerveau humain traite l'information visuelle.

Pense à une image comme une grille de petits carrés appelés pixels. Chaque pixel a des valeurs qui représentent la couleur ou l'intensité de la lumière à cet endroit de l'image. Maintenant, imagine que tu veux que l'ordinateur comprenne ce qui est dans cette image.

Eh bien, un CNN fait cela en passant par différentes étapes. D'abord, il prend de petits morceaux de l'image, appelés "filtres" ou "noyaux de convolution", et les déplace à travers toute l'image. Ces filtres regardent des motifs simples, comme des lignes ou des courbes, à différents endroits de l'image.

Ensuite, à chaque endroit où le filtre passe, il calcule une nouvelle valeur en combinant les valeurs des pixels qu'il couvre. Cela crée une nouvelle image appelée "carte de caractéristiques" qui montre où ces motifs simples ont été trouvés dans l'image.

Ensuite, le réseau passe cette carte de caractéristiques à travers d'autres couches de neurones qui regardent des motifs de plus en plus complexes. Par exemple, une couche pourrait regarder des combinaisons de lignes pour voir des formes plus complexes comme des cercles ou des carrés.

Finalement, après avoir traversé plusieurs de ces couches, le réseau arrive à une dernière couche qui décide ce qui est dans l'image en se basant sur tous ces motifs qu'il a trouvés. Cela pourrait être quelque chose comme "chat", "voiture", ou "ballon de football", en fonction de ce qu'il a appris pendant son entraînement.

</details>

### Deep learning et Machine learning

L’Intelligence artificielle est une science visant à permettre aux machines de penser et d’agir comme des humains.

Le Machine Learning a été défini par son pionnier Arthur Samuel en 1959 comme le « champ d’étude qui donne aux ordinateurs la capacité d’apprendre sans être explicitement programmés à apprendre ».

Le Deep Learning, sous-catégorie du Machine Learning, est une méthode d’apprentissage automatique qui s’inspire du fonctionnement du système nerveux des êtres vivants.

Le Deep Learning requiert de plus larges volumes de données d’entraînement, mais apprend de son propre environnement et de ses erreurs.

Au contraire, le Machine Learning permet l’entraînement sur des jeux de données moins vastes, mais requiert davantage d’intervention humaine pour apprendre et corriger ses erreurs.

Les algorithmes de Machine Learning vont généralement traiter des données quantitatives et structurées (des valeurs numériques), lorsque ceux de Deep Learning traiteront des données non-structurées, comme le son, le texte, l’image.

### F1-score (F-mesure)

Le F1-score est une métrique pour évaluer la performance des modèles de classification à 2 classes ou plus. Il est particulièrement utilisé pour les problèmes utilisant des données déséquilibrées comme la détection de fraudes ou la prédiction d’incidents graves.

L'idée de la F-mesure est de s'assurer qu'un classificateur fait de bonnes prédictions de la classe pertinente (bonne précision) en suffisamment grand nombre (bon rappel) sur un jeu de données cible.

### [Normalisation](https://inside-machinelearning.com/pourquoi-et-comment-normaliser-ces-donnees-pytorch-une-etape-essentielles-du-deep-learning-partie-1/)

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

#### [Batch Normalisation](https://inside-machinelearning.com/batch-normalization-la-meilleure-technique-pour-ameliorer-son-deep-learning/)

Normaliser la sortie d’une couche dans le contexte du Deep Learning.
La normalisation après chaque couche (donc à l’entrée de chaque nouvelle couche) permet un meilleur apprentissage du réseau.

### Precision, Recall et Precision

- La precision répond à la question : combien d’individus sélectionnés sont pertinents ? Elle mesure la capacité du modèle à ne pas faire d’erreur lors d’une prédiction positive.
- Le recall répond à la question : combien d’individus pertinents sont sélectionnés ? Le recall est également appelé `sensitivity` (sensibilité), `true positive rate` ou encore `hit rate` (taux de détection). Il correspond au taux d’individus positifs détectés par le modèle.

### [Quantization](https://inside-machinelearning.com/quantization-tensorflow/)

La quantization en Deep Learning est un processus permettant de réduire la taille d’un modèle dans le but d’optimiser sa vitesse de prédiction.

### SSD (Single Shot MultiBox Detector)

Framework de détection d'objets en temps réel qui combine la vitesse de la détection en une seule fois et la précision de la détection en plusieurs étapes.

## Cycle de vie d'un modèle d'IA

### Deep et machine learning

1. Nettoyage et préparation de données :Etapes permettant de se préparer à l'entrainement
2. Sélection de l'algorithme et du modèle : Etape déterminant en fonction du type de problème et de la nature l'algo ou le modèle à utiliser
3. Entrainement du modèle : Etape centrale, elle permet d'entrainer le modèle, impliquant d'ajuster les paramètres pour de bonnes prédictions
4. Evaluation du modèle : Etape post entrainement, il est crucial d'évaluer les performances du modèles sur un ensemble de données de test
5. Déploiement du modèle : Etape finale, permettant d'exposer le modèle à des données de productions et plus seulement au jeu de test
6. Surveillance et maintenance : après le déploiement, il est crucial de surveiller en continu les performances du modèle et de le mettre à jour si nécessaire en fonction des feedback utilisateurs et de l'évolution des données

## Veille technologique et sélection

- Adéquation avec le problème à résoudre : adapter à la problématique
- Performance et précision : évaluer les métriques du modèles
- Scalabilité et efficacité : gestion de volumes et traitement des données rapidement en post-déploiement
- Documentation et support : essentielle pour résoudre les problèmes à venir
- Compatibilité et intégration : compatible avec la stack technologique actuelle
