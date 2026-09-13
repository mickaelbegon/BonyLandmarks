# BonyLandmarks — App Étudiant

Outil interactif 3D pour placer des repères anatomiques sur son propre scan corporel BodyLoop.

[![CI](https://github.com/mickaelbegon/BonyLandmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/mickaelbegon/BonyLandmarks/actions/workflows/ci.yml)

---

## Table des matières

1. [Téléchargement et installation (étudiants)](#1-téléchargement-et-installation-étudiants)
2. [Utilisation pas à pas](#2-utilisation-pas-à-pas)
3. [Repères anatomiques (24 total)](#3-repères-anatomiques-24-total)
4. [Soumettre sur Moodle](#4-soumettre-sur-moodle)
5. [Installation pour le développement](#5-installation-pour-le-développement)
6. [Architecture technique](#6-architecture-technique)

---

## 1. Téléchargement et installation (étudiants)

### Windows

1. Télécharger `BonyLandmarks-Windows.exe` depuis la page [Releases](https://github.com/mickaelbegon/BonyLandmarks/releases).
2. Double-cliquer pour lancer. Si Windows affiche « Application inconnue », cliquer **Informations complémentaires → Exécuter quand même**.
3. Aucune installation Python requise.

### macOS

1. Télécharger `BonyLandmarks-macOS` depuis la page [Releases](https://github.com/mickaelbegon/BonyLandmarks/releases).
2. Dans le Terminal, rendre le fichier exécutable :
   ```bash
   chmod +x ~/Downloads/BonyLandmarks-macOS
   ```
3. Lancer :
   ```bash
   ~/Downloads/BonyLandmarks-macOS
   ```
   Si macOS bloque l'application : **Préférences système → Confidentialité et sécurité → Ouvrir quand même**.

### Via pip (développeurs)

```bash
pip install -e ".[dev]"
bonylandmarks
```

---

## 2. Utilisation pas à pas

### Étape 1 — Connexion

Au démarrage, une fenêtre de connexion s'affiche :

| Champ | Valeur |
|---|---|
| **Matricule** | Votre matricule UdeM (ex. `20111111`) |
| **Date de naissance** | Votre date de naissance |
| **URL du serveur** | Fournie par l'enseignant (ex. `http://192.168.1.42:8765`) |

Cliquer **Se connecter** : l'app télécharge votre scan chiffré et le déchiffre localement. Aucune donnée personnelle ne transite en clair.

### Étape 2 — Exploration du scan

Votre corps 3D s'affiche avec :

- **Sphères vertes** : les 24 repères anatomiques que vous devez identifier (positions de référence BodyLoop)
- **Interface de navigation** :
  - Clic gauche + glisser → rotation
  - Clic droit + glisser (ou molette) → zoom
  - Clic molette + glisser → déplacement latéral

### Étape 3 — Placement des repères

Pour chaque repère (24 au total) :

1. Le panneau de droite affiche le **nom du repère** et un **indice de palpation**.
2. Cliquer sur la surface du corps à l'endroit correspondant au repère.
3. Une **sphère jaune** indique votre sélection.
4. Cliquer **Confirmer** pour valider :
   - L'erreur en mm s'affiche en **vert** (≤ 20 mm), **orange** (≤ 40 mm) ou **rouge** (> 40 mm).
   - La sphère devient **bleue** (repère confirmé).
5. Cliquer **Recommencer** pour annuler et reprendre le placement si nécessaire.

> Le bouton de langue (FR / EN) dans le coin supérieur droit bascule l'interface en anglais.

### Étape 4 — Fin de session

Après les 24 repères, un récapitulatif s'affiche.
Cliquer **Exporter JSON** pour sauvegarder vos résultats dans un fichier `.json`.

---

## 3. Repères anatomiques (24 total)

| Code | Français | English | Côté |
|---|---|---|---|
| `ASIS_left/right` | Épine iliaque antéro-supérieure | Anterior superior iliac spine | G/D |
| `greater_trochanter_left/right` | Grand trochanter | Greater trochanter | G/D |
| `acromion_left/right` | Acromion | Acromion | G/D |
| `lateral_epicondyle_left/right` | Épicondyle latéral (coude) | Lateral epicondyle | G/D |
| `medial_epicondyle_left/right` | Épicondyle médial (coude) | Medial epicondyle | G/D |
| `ulnar_styloid_left/right` | Styloïde ulnaire | Ulnar styloid | G/D |
| `radial_styloid_left/right` | Styloïde radiale | Radial styloid | G/D |
| `lateral_knee_left/right` | Condyle latéral (genou) | Lateral knee condyle | G/D |
| `medial_knee_left/right` | Condyle médial (genou) | Medial knee condyle | G/D |
| `lateral_malleolus_left/right` | Malléole latérale | Lateral malleolus | G/D |
| `medial_malleolus_left/right` | Malléole médiale | Medial malleolus | G/D |
| `heel_left/right` | Talon | Heel | G/D |

---

## 4. Soumettre sur Moodle

1. À la fin de la session, cliquer **Exporter JSON**.
2. Choisir un dossier de sauvegarde. Le fichier s'appellera `{matricule}_{timestamp}.json`.
3. Se connecter à Moodle, naviguer vers l'activité de remise, et déposer ce fichier JSON.

---

## 5. Installation pour le développement

### Prérequis

- Python 3.11 ou 3.12
- pip 23+

### Installer

```bash
git clone https://github.com/mickaelbegon/BonyLandmarks.git
cd BonyLandmarks
pip install -e ".[dev]"
```

### Lancer l'application

```bash
bonylandmarks
```

### Lancer les tests

```bash
pytest
```

### Build exécutable

```bash
# Windows
pyinstaller --onefile --windowed --name BonyLandmarks `
    --add-data "src/bonylandmarks;bonylandmarks" `
    src/bonylandmarks/main.py

# macOS / Linux
pyinstaller --onefile --windowed --name BonyLandmarks \
    --add-data "src/bonylandmarks:bonylandmarks" \
    src/bonylandmarks/main.py
```

Les exécutables Windows et macOS sont produits automatiquement par GitHub Actions à chaque push sur `main`.

---

## 6. Architecture technique

### Stack

| Composant | Bibliothèque |
|---|---|
| Interface graphique | PySide6 6.7+ |
| Vue 3D | PyVista 0.44+ + pyvistaqt |
| Lecture GLB | trimesh 4.3+ + pygltflib 1.16+ |
| Chiffrement | cryptography 42+ (AES-256-GCM) |
| Requêtes HTTP | httpx 0.27+ |

### Modules

```
src/bonylandmarks/
├── main.py          # Point d'entrée : fenêtre de login → viewer → export
├── login_dialog.py  # Dialogue de connexion (matricule + DDN + URL serveur)
├── mesh_loader.py   # Chargement GLB : corps 3D + AutoMarkers BodyLoop
├── viewer.py        # Widget 3D PySide6/PyVista : picking de surface
├── landmarks.py     # 24 Landmark (code, nom FR/EN, indice FR/EN)
├── scoring.py       # Calcul erreur euclidienne + feedback couleur
├── export.py        # Export JSON (format Moodle)
├── client.py        # Client HTTP : téléchargement + déchiffrement du scan
├── crypto.py        # AES-256-GCM déchiffrement
└── i18n.py          # Traductions FR/EN
```

### Chiffrement

La clé de déchiffrement est calculée localement à partir du matricule et de la date de naissance :

```
clé = SHA-256( matricule.lower() + ":" + YYYYMMDD )
```

Aucune clé n'est transmise au serveur. Le serveur distribue uniquement des fichiers chiffrés ; il ne peut pas lui-même lire les scans.

### Format GLB BodyLoop (avatar_3d)

Les scans BodyLoop `avatar_3d` contiennent :

- Un maillage 3D du corps avec texture PBR photographique
- Un nœud `AutoMarkers` : 83 repères nommés positionnés automatiquement par BodyLoop
- Des nœuds squelette avec les positions des articulations

L'app extrait les 24 repères correspondant aux landmarks anatomiques ciblés et les affiche comme sphères vertes de référence.
