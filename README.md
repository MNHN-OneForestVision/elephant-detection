# Système de Reconnaissance d'Éléphants par Analyse d'Oreilles

## Vue d'ensemble

Ce projet implémente un système complet pour la détection, la segmentation et l'identification d'éléphants individuels basé sur l'analyse des caractéristiques uniques de leurs oreilles. Le système combine des techniques avancées de vision par ordinateur et d'apprentissage profond pour permettre un suivi non invasif des populations d'éléphants.

## Composants principaux

Le projet est organisé en trois composants principaux, chacun documenté en détail :

1. [**Interface YOLO**](./YOLO%20Interface.md) - Interface graphique pour la détection et la segmentation d'éléphants dans des images et vidéos.
2. [**Pipeline d'identification**](./Pipeline.md) - Système de reconnaissance d'éléphants basé sur l'analyse des oreilles.
3. [**Entraînement du modèle ViT**](./ViT%20Train.md) - Outils pour l'entraînement du modèle d'identification.

## Fonctionnalités clés

- Détection et segmentation d'éléphants dans des images et vidéos
- Extraction automatique des régions d'oreilles (gauche et droite)
- Génération d'embeddings vectoriels pour chaque oreille
- Recherche rapide par similarité pour identifier des individus connus
- Interface graphique conviviale pour l'analyse visuelle
- Outils d'entraînement pour améliorer la précision du modèle

## Architecture du système

Le système fonctionne en trois étapes principales :

1. **Détection** : Localisation des éléphants et segmentation des oreilles à l'aide de modèles YOLO.
2. **Extraction de caractéristiques** : Génération d'embeddings vectoriels à partir des images d'oreilles via un modèle Vision Transformer (ViT).
3. **Identification** : Recherche par similarité dans une base de données d'éléphants connus à l'aide de FAISS.

## Installation

Le projet peut être installé et exécuté de plusieurs façons, selon votre système d'exploitation :

- **Linux** : Installation via Docker (recommandé)
- **macOS/Windows** : Installation via environnement virtuel Python

Pour des instructions détaillées, consultez la [documentation de l'interface YOLO](./YOLO%20Interface.md#installation).

## Utilisation rapide

### Interface graphique

```bash
# Lancer l'interface graphique
python ui_yolo.py
```

### Pipeline d'identification

```python
from pipeline import EarRecognizerSystem

# Initialiser le système
system = EarRecognizerSystem()

# Identifier un éléphant à partir d'une image
results = system.run("path/to/elephant_image.jpg")
```

### Entraînement du modèle

```python
from vit_train import main

# Lancer l'entraînement avec les paramètres par défaut
main()
```

## Structure du projet

```
elephant-detection/
├── README.md                 # Documentation générale
├── YOLO Interface.md         # Documentation de l'interface
├── Pipeline.md            # Documentation du pipeline d'identification
├── ViT Train.md           # Documentation de l'entraînement
├── ui_yolo.py                # Interface graphique YOLO
├── pipeline.py               # Système d'identification
├── vit_train.py              # Entraînement du modèle ViT
├── lib.py                    # Fonctions utilitaires
├── requirements.txt          # Dépendances Python
├── models/                   # Modèles pré-entraînés
│   ├── detection/            # Modèles de détection
│   ├── segmentation/         # Modèles de segmentation
│   └── identification/       # Modèles d'identification
├── db/                       # Base de données d'embeddings
└── docker-compose.yml        # Configuration Docker
```

## Licence

Ce projet est distribué sous licence Attribution-NonCommercial-ShareAlike 4.0 International. Voir le fichier [LICENSE](./LICENSE) pour plus de détails.
