# YOLO Interface - Interface de Détection et Segmentation

# [File: ui_yolo.py](./ui_yolo.py)

Une interface graphique moderne pour utiliser les modèles YOLO sur images et vidéos avec support Docker pour Linux.
_Fonction pour tout type de modèle YOLO (pas uniquement pour les éléphants)._

## Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
  - [Linux (Docker recommandé)](#linux-docker-recommandé)
  - [macOS (Environnement virtuel)](#macos-environnement-virtuel)
  - [Windows (Environnement virtuel)](#windows-environnement-virtuel)
- [Utilisation de l'interface](#utilisation-de-linterface)
- [Structure du projet](#structure-du-projet)
- [Dépannage](#dépannage)

## Fonctionnalités

- Interface graphique intuitive avec Tkinter
- Support des modèles YOLO (.pt)
- Traitement d'images (JPG, PNG, BMP, TIFF)
- Traitement de vidéos (MP4, AVI, MOV, MKV, WMV, FLV)
- Prévisualisation des médias intégrée
- Configuration du seuil de confiance
- Support du tracking pour les vidéos
- Sauvegarde personnalisée des résultats
- Logs en temps réel
- Conteneurisation Docker (Linux uniquement)

## Prérequis

### Système requis
- **Linux** : Docker et Docker Compose
- **macOS/Windows** : Python 3.11+
- 4GB RAM minimum
- Carte graphique Nvidia recommandée pour de meilleures performances

## Installation

### Linux (Docker recommandé)

Docker simplifie l'installation en encapsulant toutes les dépendances.

#### Prérequis Docker
```bash
# Installer Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose-plugin

# Ajouter votre utilisateur au groupe docker
sudo usermod -aG docker $USER
newgrp docker
```

#### Installation et lancement

1. **Cloner le repository**
   ```bash
   git clone git@github.com:MNHN-OneForestVision/elephant-detection.git
   cd elephant-detection
   ```

2. **Configurer l'affichage X11**
   ```bash
   xhost +local:docker
   ```

3. **Adapter le volume de données (optionnel)**
   Modifier la ligne dans `docker-compose.yml` selon vos besoins :
   ```yaml
   volumes:
     - /votre/dossier/de/données:/data  # Changez ce chemin mais laissez le :/data
   ```

4. **Lancer avec Docker Compose**
   ```bash
   docker-compose up --build
   ```

5. **L'interface s'ouvre automatiquement** 🎉

#### Arrêter l'application
```bash
# Ctrl+C dans le terminal ou
docker-compose down
```

### macOS (Environnement virtuel)

#### Installation

1. **Installer Homebrew (si pas déjà installé)**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Installer Python et les dépendances système**
   ```bash
   brew install python@3.11 ffmpeg
   ```

3. **Cloner le repository**
   ```bash
   git clone git@github.com:MNHN-OneForestVision/elephant-detection.git
   cd elephant-detection
   ```

4. **Créer et activer l'environnement virtuel**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Installer les dépendances Python**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. **Lancer l'application**
   ```bash
   python ui_yolo.py
   ```

#### Utilisation ultérieure
```bash
cd elephant-detection
source venv/bin/activate  # Activer l'environnement
python ui_yolo.py         # Lancer l'app
deactivate               # Désactiver l'environnement (quand terminé)
```

### Windows (Environnement virtuel)

#### Installation

1. **Installer Python 3.11+**
   - Télécharger depuis [python.org](https://www.python.org/downloads/)
   - ⚠️ **IMPORTANT** : Cocher "Add Python to PATH" lors de l'installation

2. **Cloner le repository**
   ```cmd
   git clone git@github.com:MNHN-OneForestVision/elephant-detection.git
   cd yolo-interface
   ```

3. **Créer et activer l'environnement virtuel**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Installer les dépendances Python**
   ```cmd
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Lancer l'application**
   ```cmd
   python ui_yolo.py
   ```

#### Utilisation ultérieure
```cmd
cd elephant-detection
venv\Scripts\activate     # Activer l'environnement
python ui_yolo.py         # Lancer l'app
deactivate               # Désactiver l'environnement (quand terminé)
```

## Utilisation de l'interface

### 1. Configuration du modèle
- Cliquer sur "Parcourir" pour sélectionner votre modèle YOLO (.pt)
- Ajuster le seuil de confiance avec le curseur ou en écrivant (0.0 - 1.0)

### 2. Sélection des fichiers
**Le modèle analyse soit des images soit des vidéos mais pas les 2 en même temps**
- **Images** : Cliquer sur "Sélectionner Images" (JPG, PNG, BMP, TIFF)
- **Vidéos** : Cliquer sur "Sélectionner Vidéos" (MP4, AVI, MOV, MKV, WMV, FLV)
- **Prévisualisation** : Double-cliquer sur un fichier ou utiliser "Visualiser"
- **Nettoyage** : "Effacer" pour vider la liste

### 3. Options de traitement
- ☑️ **Tracking** : Active le suivi d'objets pour les vidéos
- ☑️ **Afficher les résultats** : Montre les résultats en temps réel
- ☑️ **Sauvegarder les résultats** : 
  - Choisir le dossier de destination
  - Nommer le sous-dossier (optionnel)


### 4. Visualisation
- **Images** : Aperçu redimensionné automatiquement
- **Vidéos** : Lecteur intégré avec contrôles (Play/Pause/Stop)

## Structure du projet

```
yolo-interface/
├── ui_yolo.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Dépannage

### Erreurs communes
**"Impossible d'ouvrir la vidéo"**
- Vérifier que FFmpeg est correctement installé
- Tester avec un format vidéo différent (MP4 recommandé)
- Vérifier que le fichier n'est pas corrompu

**"Modèle YOLO ne se charge pas"**
- Vérifier que le fichier .pt est valide
- Vérifier les permissions du fichier

### Problèmes spécifiques

**Linux Docker : "Cannot connect to display"**
```bash
xhost +local:docker
export DISPLAY=:0
```

### Performance et optimisation

**Pour de meilleures performances** :
- Utiliser un GPU Nvidia compatible CUDA (Linux/Windows)
- Ajuster le seuil de confiance selon vos besoins
- Fermer d'autres applications gourmandes en ressources

