# Entraînement du Modèle ViT pour la Reconnaissance d'Éléphants

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Structure des données](#structure-des-données)
- [Utilisation](#utilisation)
  - [Entraînement du modèle](#entraînement-du-modèle)
  - [Extraction d'embeddings](#extraction-dembeddings)
  - [Utilisation du modèle entraîné](#utilisation-du-modèle-entraîné)
- [Architecture du modèle](#architecture-du-modèle)
- [Méthode d'apprentissage](#méthode-dapprentissage)
- [Exemples](#exemples)
- [Dépannage](#dépannage)

## Vue d'ensemble

Le fichier [_vit_train.py_](./vit_train.py) implémente l'entraînement d'un modèle Vision Transformer (ViT) pour la reconnaissance d'éléphants individuels basée sur les caractéristiques de leurs oreilles. Ce modèle est entraîné à l'aide d'une approche par triplets (triplet loss) pour apprendre à distinguer les différents individus.

Une fois entraîné, le modèle génère des embeddings vectoriels pour chaque image d'oreille, permettant une recherche efficace par similarité. Ces embeddings sont ensuite utilisés par le pipeline d'identification (`pipeline.py`) pour reconnaître les éléphants.

## Fonctionnalités

- Entraînement d'un modèle ViT avec triplet loss
- Chargement et prétraitement automatique des données d'entraînement
- Extraction d'embeddings à partir d'images d'oreilles
- Création d'index FAISS pour la recherche par similarité
- Sauvegarde et chargement de modèles pré-entraînés

## Prérequis

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- FAISS
- PIL, NumPy, tqdm

## Installation

Toutes les dépendances sont incluses dans le fichier `requirements.txt` du projet principal.

```bash
pip install -r requirements.txt
```

## Structure des données

Le script attend une structure de données spécifique pour l'entraînement :

```
dataset/
├── elephant_1_left_ear_1.jpg
├── elephant_1_left_ear_2.jpg
├── elephant_2_left_ear_1.jpg
├── elephant_2_right_ear_1.jpg
└── ...
```

Les noms de fichiers doivent suivre un format permettant d'identifier l'individu (par exemple, un préfixe ou un identifiant dans le nom).

## Utilisation

### Entraînement du modèle

```python
from vit_train import build_labels_dict, TripletEarDataset, ViTEmbedder, train_model

# Construire le dictionnaire d'étiquettes
labels_dict = build_labels_dict("path/to/dataset")

# Créer le dataset
dataset = TripletEarDataset("path/to/dataset", labels_dict)

# Initialiser le modèle
model = ViTEmbedder()

# Entraîner le modèle
train_model(dataset, model, save_dir="checkpoints", epochs=600, batch_size=1, lr=1e-5, margin=1.0)
```

### Extraction d'embeddings

```python
from vit_train import extract_embeddings, ViTEmbedder

# Charger un modèle pré-entraîné
model = ViTEmbedder()
model.load_state_dict(torch.load("path/to/model.pth"))

# Extraire les embeddings et créer un index FAISS
extract_embeddings(
    model, 
    "path/to/images", 
    labels_dict,
    output_faiss="faiss.index",
    output_npy="embeddings.npy"
)
```

### Utilisation du modèle entraîné

Une fois entraîné, le modèle peut être utilisé avec le pipeline d'identification :

```python
from pipeline import EarRecognizerSystem

# Initialiser le système avec le modèle entraîné
system = EarRecognizerSystem(vit_model_path="path/to/trained_model.pth")

# Utiliser le système pour l'identification
results = system.run("path/to/elephant_image.jpg")
```

## Architecture du modèle

La classe `ViTEmbedder` encapsule un modèle Vision Transformer pré-entraîné de Hugging Face et ajoute une couche de projection pour générer des embeddings de dimension fixe (768 par défaut). L'architecture utilise :

- Un modèle ViT de base (google/vit-base-patch16-224-in21k)
- Un processeur d'images ViT pour le prétraitement
- Une couche de projection linéaire pour l'adaptation des embeddings

## Méthode d'apprentissage

Le modèle est entraîné avec une fonction de perte par triplets (triplet loss) qui :
1. Rapproche les embeddings d'images du même éléphant (ancre et positif)
2. Éloigne les embeddings d'images d'éléphants différents (ancre et négatif)

Cette approche permet au modèle d'apprendre à distinguer les caractéristiques uniques des oreilles de chaque éléphant.

## Exemples

Exemple complet d'entraînement et d'extraction d'embeddings :
```shell
  python3 ./vit_train.py
```


```python
from vit_train import main

# Exécuter le processus complet (entraînement + extraction)
main()
```

## Dépannage

- **Erreur de mémoire GPU** : Réduisez la taille du batch ou utilisez un modèle ViT plus petit.
- **Convergence lente** : Ajustez le taux d'apprentissage ou augmentez la marge de la triplet loss.
- **Faible précision** : Augmentez le nombre d'exemples d'entraînement ou améliorez la qualité des images.
- **Erreur de chargement de modèle** : Vérifiez que la version de transformers est compatible avec le modèle sauvegardé.