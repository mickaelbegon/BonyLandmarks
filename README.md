# BonyLandmarks — Plateforme d'apprentissage des repères anatomiques 3D

Outil interactif 3D pour apprendre à identifier et placer des repères anatomiques osseux. Plusieurs exercices pédagogiques sont disponibles : palpation sur son propre scan BodyLoop, quiz sur os BodyParts3D, et exercice collaboratif sur scan partagé.

[![CI](https://github.com/mickaelbegon/BonyLandmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/mickaelbegon/BonyLandmarks/actions/workflows/ci.yml)

---

## Table des matières

1. [Téléchargement et installation (étudiants)](#1-téléchargement-et-installation-étudiants)
2. [Exercices disponibles](#2-exercices-disponibles)
   - [Exercice 1 — Mon scan 3D](#exercice-1--mon-scan-3d)
   - [Exercice 2 — Quiz osseux](#exercice-2--quiz-osseux)
   - [Exercice 3 — Scan partagé](#exercice-3--scan-partagé)
3. [Repères anatomiques](#3-repères-anatomiques)
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

## 2. Exercices disponibles

Au démarrage, l'écran d'accueil propose trois exercices. Cliquer sur la carte correspondante pour sélectionner l'exercice, puis cliquer **Commencer**.

### Exercice 1 — Mon scan 3D

**Connexion requise.** L'application télécharge votre scan corporel 3D chiffré depuis le serveur de l'enseignant et le déchiffre localement. Vous placez 24 repères osseux sur votre propre corps numérique et recevez un score en millimètres comparé aux positions de référence BodyLoop.

#### Connexion

Au démarrage, saisir vos identifiants :

| Champ | Valeur |
|---|---|
| **Nom** | Sélectionner votre nom dans la liste déroulante |
| **Date de naissance** | Votre date de naissance (utilisée comme clé de déchiffrement) |

Cliquer **Commencer** : l'app télécharge votre scan chiffré et le déchiffre localement. Aucune donnée personnelle ne transite en clair.

#### Exploration et placement

Votre corps 3D s'affiche. Pour chaque repère (24 au total) :

1. Le panneau de droite affiche le **nom du repère** et un **indice de palpation**.
2. Cliquer sur la surface du corps à l'endroit correspondant au repère.
3. Une **sphère jaune** indique votre sélection.
4. Cliquer **Confirmer** pour valider :
   - L'erreur en mm s'affiche en **vert** (≤ 20 mm), **orange** (≤ 40 mm) ou **rouge** (> 40 mm).
   - La sphère devient **bleue** (repère confirmé).
5. Cliquer **Recommencer** pour annuler et reprendre le placement si nécessaire.

#### Notation

| Note | Erreur moyenne |
|---|---|
| A | < 30 mm |
| B | < 80 mm |
| C | < 150 mm |
| D | ≥ 150 mm |

Les repères notés D sont remis en queue pour une deuxième tentative.

#### Export

Après les 24 repères, un récapitulatif s'affiche. Cliquer **Exporter JSON** pour sauvegarder vos résultats dans un fichier `.json` à déposer sur Moodle.

> Le bouton de langue (FR / EN) dans le coin supérieur droit bascule l'interface en anglais.

---

### Exercice 2 — Quiz osseux

**Pas de connexion requise.** Cet exercice utilise les meshes osseux BodyParts3D disponibles localement. Il se déroule en deux phases :

1. **Phase individuelle** : pour chaque os sélectionné, identifier et placer des repères anatomiques spécifiques sur le mesh 3D de l'os isolé. Un score en millimètres est calculé par rapport aux positions de référence algorithmeques.

2. **Phase squelette complet** : replacer les mêmes repères sur une vue du squelette entier pour renforcer la contextualisation anatomique.

Ce quiz est adapté à l'entraînement autonome, sans scan personnel, à partir des meshes BodyParts3D.

---

### Exercice 3 — Scan partagé

**Scan commun, login optionnel.** L'enseignant distribue un fichier GLB commun à tous les étudiants. Chaque étudiant place librement des repères sur ce scan partagé, puis les résultats de l'ensemble du groupe sont agrégés :

- La **position centroïde** de chaque repère (moyenne des placements) est calculée.
- L'**écart-type** indique la dispersion des placements inter-étudiants.

Cet exercice est non-évaluatif : il favorise la discussion collective sur la variabilité de palpation et la reproductibilité inter-observateurs.

Au démarrage de cet exercice, une fenêtre de sélection de fichier s'ouvre pour choisir le fichier GLB partagé. Les fichiers JSON de soumissions des autres étudiants doivent se trouver dans le même dossier.

---

## 3. Repères anatomiques

L'application utilise deux référentiels de repères distincts.

### 3a. Repères de palpation virtuelle (exercice 1 — scan BodyLoop)

24 repères osseux détectés automatiquement par BodyLoop dans le scan `avatar_3d`. Ils servent de **vérité terrain** pour évaluer l'erreur de placement de l'étudiant.

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

### 3b. Référentiel anatomique complet (`data/landmarks.json`)

Le fichier [`src/bonylandmarks/data/landmarks.json`](src/bonylandmarks/data/landmarks.json) contient **182 repères osseux** bilingues (FR/EN) utilisés pour les exercices ISB, anthropométriques et le quiz osseux. Chaque repère inclut :

| Champ | Type | Description |
|---|---|---|
| `code` | `string` | Identifiant unique snake_case (ex. `ASIS_left`) |
| `category` | `string` | `"BONE"`, `"EMG"`, `"SKINFOLD"` ou `"ANTHRO"` |
| `name_fr` / `name_en` | `string` | Nom anatomique bilingue |
| `hint_fr` / `hint_en` | `string` | Description précise de palpation (2–5 phrases) |
| `body_side` | `"left"` \| `"right"` \| `"midline"` | Latéralité du repère |
| `theme` | `string` | Thème clinique / domaine d'application |
| `application_fr` / `application_en` | `string` | Rôle clinique et applications biomécaniques (optionnel) |

**Thèmes disponibles :**

| Thème | Repères | Domaine |
|---|---|---|
| `anatomy` | 81 | Repères généraux de palpation |
| `shoulder` | 26 | Épaule et ceinture scapulaire |
| `gait` | 20 | Analyse de la marche |
| `posture` | 19 | Analyse posturale |
| `upper_limb` | 12 | Membre supérieur |
| `core` | 11 | Tronc et rachis |
| `knee_rehab` | 10 | Genou et réhabilitation |
| `lower_limb` | 2 | Membre inférieur |
| `cpr` | 1 | Réanimation cardio-pulmonaire |

Pour **ajouter ou modifier un repère**, éditer directement ce fichier JSON — aucun redémarrage Python n'est nécessaire, le fichier est chargé à l'exécution.

---

## 4. Soumettre sur Moodle

1. À la fin de la session (exercice 1), cliquer **Exporter JSON**.
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
├── main.py                    # Point d'entrée : écran d'accueil → exercices → export
├── splash.py                  # Écran d'accueil : sélection d'exercice + login
├── viewer.py                  # Widget 3D (exercice 1) : picking, workflow repère
├── session.py                 # Machine à états de session (sans Qt), scoring
├── landmarks.py               # 24 repères BodyLoop (exercice 1)
├── landmarks_extended.py      # 182 repères complets (tous exercices)
├── bone_quiz_exercise.py      # Exercice 2 : quiz os BodyParts3D (deux phases)
├── shared_scan_exercise.py    # Exercice 3 : scan partagé, agrégation inter-étudiants
├── tutorial.py                # Mode tutoriel guidé (exercice 1)
├── isb_exercise.py            # Exercice ISB (Wu et al. 2002/2005)
├── isb_recipes.py             # Recettes déclaratives repères ISB
├── isb_step_engine.py         # Moteur wizard ISB
├── frame_template.py          # Templates FrameTemplate → steps (source de vérité)
├── anthro_recipes.py          # 20 recettes anthropométriques
├── anthro_step_engine.py      # Moteur guidé anthropo
├── anthro_measures_exercise.py # 20 mesures anthropométriques structurées
├── bone_map.py                # LANDMARK_BONE, BONE_JOINTS, BONE_LABEL_FR
├── muscle_map.py              # BONE_MUSCLES (FMA IDs)
├── mesh_loader.py             # Chargement GLB → meshes PyVista (mm)
├── scoring.py                 # Calcul erreur euclidienne + feedback couleur
├── export.py                  # Export JSON (format Moodle)
├── client.py                  # Client HTTP : téléchargement + déchiffrement scan
├── crypto.py                  # AES-256-GCM déchiffrement
├── annotator.py               # Outil annotation expert (enseignant)
├── i18n.py                    # Traductions FR/EN
└── data/
    ├── landmarks.json         # 182 repères osseux bilingues (9 thèmes)
    └── bones/
        └── landmark_positions.json  # code → [x, y, z] (mm, BodyParts3D)
```

### Chiffrement (exercice 1)

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

---

## Crédits

### BodyParts3D

Les meshes 3D osseux de référence sont issus de la base de données **BodyParts3D**.

> BodyParts3D, © The Database Center for Life Science licensed under CC Attribution-Share Alike 2.1 Japan

https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html

### Claude Code

Outil d'assistance au développement IA utilisé pour construire ce projet.

https://claude.ai/claude-code

### Bibliothèques open source

- **PyVista** : visualisation 3D
- **PySide6** : interface graphique
- **pyvistaqt** : intégration PyVista/Qt
- **trimesh** : traitement des maillages
- **pygltflib** : lecture des fichiers GLB
- **cryptography** : chiffrement AES-256-GCM
- **httpx** : requêtes HTTP

### Auteur

**Mickael Begon**  
Université de Montréal  
mickael.begon@umontreal.ca
