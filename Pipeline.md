# Pipeline d'Identification des Éléphants par Reconnaissance d'Oreilles

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
  - [Extraction des oreilles](#extraction-des-oreilles)
  - [Recherche d'éléphants similaires](#recherche-déléphants-similaires)
  - [Ajout de nouveaux éléphants à la base de données](#ajout-de-nouveaux-éléphants-à-la-base-de-données)
- [Architecture du système](#architecture-du-système)
- [Dépannage](#dépannage)

## Vue d'ensemble

Le fichier [_pipeline.py_](./pipeline.py) implémente un système complet de reconnaissance d'éléphants basé sur l'identification des oreilles. Ce système utilise une approche en deux étapes :

1. **Détection et segmentation des oreilles** : Utilisation d'un modèle YOLO pour localiser et extraire les oreilles gauche et droite des éléphants dans les images.
2. **Identification par similarité** : Utilisation d'un modèle Vision Transformer (ViT) pour générer des embeddings des oreilles et rechercher des correspondances dans une base de données FAISS.

Ce pipeline permet d'identifier des éléphants individuels à partir de photos, facilitant ainsi le suivi des populations et les études écologiques.

## Fonctionnalités

- Détection automatique des oreilles gauche et droite
- Extraction des régions d'intérêt (ROI) des oreilles
- Génération d'embeddings via un modèle ViT pré-entraîné
- Recherche rapide par similarité avec FAISS
- Ajout de nouveaux éléphants à la base de données
- Visualisation des résultats avec scores de confiance

## Prérequis

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- FAISS
- Transformers (Hugging Face)
- PIL, Matplotlib, NumPy

## Installation

Toutes les dépendances sont incluses dans le fichier `requirements.txt` du projet principal.

```bash
pip install -r requirements.txt
```

Voici la section **Utilisation** du README adaptée pour refléter l'appel du script principal tel que montré dans ton code, en utilisant les arguments CLI :

## Utilisation

### Commandes principales

```bash
python pipeline.py --path chemin/vers/image.jpg
```

#### Options disponibles

- `--path PATH` : Chemin vers l'image à traiter OU dossier contenant les images à traiter **(obligatoire sauf si "--reset")**.
- `--add` : Ajoute l'image à la base de données après la recherche (extraction et indexation des oreilles).
- `--extract` : Sauvegarde les oreilles découpées dans le dossier d’archive `db/images/`.
- `--reset` : Réinitialise l’index FAISS et les fichiers NPY associés (aucun traitement d’image n’est alors résumé).

#### Exemples d'utilisation

- **Rechercher les éléphants similaires à partir d'une image** :
  ```bash
  python pipeline.py --path path/to/elephant_image.jpg
  ```

- **Traiter tout un dossier d’images** :
  ```bash
  python pipeline.py --path path/to/folder/
  ```

- **Ajouter une nouvelle image dans l'index après traitement** :
  ```bash
  python pipeline.py --path path/to/new_elephant.jpg --add
  ```

- **Extraire et enregistrer uniquement les oreilles dans le dossier `db/images/`** :
  ```bash
  python pipeline.py --path path/to/elephant_image.jpg --extract
  ```

- **Réinitialiser entièrement la base de données (FAISS et NPY)** :
  ```bash
  python pipeline.py --reset
  ```

#### Remarques

- L’argument `--path` accepte un fichier (image unique) ou un dossier (toutes les images seront traitées).
- Si vous souhaitez à la fois extraire et indexer des oreilles, combinez `--add` et `--extract`.
- Si vous souhaitez repartir à zéro, utilisez `--reset`.  
  **Attention :** cette commande efface l’index et la base d’embeddings.


## Architecture du système

La classe principale `EarRecognizerSystem` gère l'ensemble du pipeline et contient les composants suivants :

- **Modèle YOLO** : Détecte et segmente les oreilles dans les images
- **Modèle ViT** : Génère des embeddings à partir des images d'oreilles
- **Index FAISS** : Stocke et recherche efficacement les embeddings
- **Base de données de chemins** : Associe les embeddings aux fichiers sources

Le système maintient des index séparés pour les oreilles gauche et droite, améliorant ainsi la précision de l'identification.

## Dépannage

- **Erreur de chargement de modèle** : Vérifiez que les chemins vers les modèles YOLO et ViT sont corrects.
- **Aucune oreille détectée** : Utilisez une image où les oreilles sont plus visibles.
- **Faible précision d'identification** : Entraînez le modèle ViT sur davantage d'exemples ou ajustez les paramètres de recherche.