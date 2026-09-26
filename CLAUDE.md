# CLAUDE.md — Guide de contexte pour agents IA

Ce fichier est chargé automatiquement par Claude Code au démarrage de chaque session.
Il donne le contexte dense nécessaire pour travailler efficacement sur BonyLandmarks.

---

## 1. Vue d'ensemble du projet

**BonyLandmarks** est un outil pédagogique d'annotation de repères anatomiques sur scan corporel 3D BodyLoop (GLB).

Deux applications distinctes :
- **App étudiant** (`main.py` → `viewer.py`) : viewer interactif guidé, l'étudiant place 24 repères sur son propre scan, reçoit un score en mm vs. la vérité terrain BodyLoop.
- **App enseignant / annotateur** (`annotator.py`) : outil expert indépendant, ouvre un GLB "vierge" (sans AutoMarkers), permet de cliquer et d'enregistrer les 182 repères anatomiques, produit un JSON de positions de référence.

Stack : Python 3.11+, PySide6 6.7+, PyVista 0.44+ / pyvistaqt, trimesh 4.3+, pygltflib 1.16+, cryptography 42+ (AES-256-GCM), httpx 0.27+, NumPy.

Point d'entrée CLI : `bonylandmarks` (app étudiant) / `python -m bonylandmarks.annotator` (annotateur).

---

## 2. Map des fichiers clés

### Racine du projet

| Fichier | Rôle |
|---|---|
| `compute_landmarks.py` | Script standalone : calcule algorithmiquement les positions de référence des repères sur les meshes BodyParts3D, écrit `landmark_positions.json` et des PNG de validation |

### `src/bonylandmarks/`

| Fichier | Rôle |
|---|---|
| `main.py` | Point d'entrée app étudiant : fenêtre login → viewer → export |
| `annotator.py` | Outil annotation expert (enseignant) : viewer 3D libre + dock "Os de référence" (édition position, squelette/adjacents, courbure, muscles BP3D) |
| `viewer.py` | Widget 3D principal (app étudiant) : PySide6/PyVista, picking VTK, workflow par repère (pick → jaune → confirmer → bleu), débrief, retry queue |
| `session.py` | Machine à états de session pure Python (sans Qt) : progression, queue de retry, scoring — testable sans GUI |
| `landmarks.py` | 24 Landmark hardcodés (app étudiant, exercice 1) : codes BodyLoop, noms FR/EN, indices |
| `landmarks_extended.py` | Référentiel complet 182 repères (BONE, EMG, SKINFOLD, ANTHRO) — source de vérité pour l'annotateur et les exercices ISB/anthropo ; exporte `LANDMARKS` |
| `bone_map.py` | Trois dicts : `LANDMARK_BONE` (code → bone_stem), `BONE_JOINTS` (bone_stem → adjacents), `BONE_LABEL_FR` (bone_stem → nom français) |
| `muscle_map.py` | `BONE_MUSCLES` : bone_stem → liste de FMA concept IDs muscles BodyParts3D à afficher dans le dock annotateur |
| `mesh_loader.py` | Chargement GLB BodyLoop → meshes PyVista + extraction AutoMarkers (avatar_3d) ou mesh seul (mesh_3d) ; tout en **millimètres** |
| `scoring.py` | Calcul erreur euclidienne mm, `LandmarkResult`, `SessionScore`, feedback couleur vert/orange/rouge |
| `export.py` | Export JSON session étudiant (format Moodle) |
| `client.py` | Client HTTP httpx : téléchargement scan chiffré depuis serveur enseignant |
| `crypto.py` | AES-256-GCM déchiffrement ; dérivation clé : SHA-256(matricule + ":" + YYYYMMDD) ; format fichier : 12B nonce \| 16B tag \| ciphertext |
| `manifest.py` | Chargement manifest étudiant (URL serveur, liste étudiants) |
| `login_dialog.py` | Dialogue connexion : dropdown nom étudiant, matricule, date de naissance |
| `splash.py` | Splash screen unifié : onboarding + login en un seul dialog |
| `welcome.py` | Dialog de bienvenue affiché avant la session landmark |
| `tutorial.py` | Mode tutoriel guidé (exploration libre avant session notée) : mode Reconnaissance (position visible) et mode Quiz (caché) |
| `dialogs.py` | Builders de QDialog standalone pour la session (debrief, transition catégorie, etc.) |
| `i18n.py` | Table de traductions bilingues FR/EN minimaliste pour l'UI étudiant |
| `validation.py` | Validation présence des marqueurs GLB vs. codes requis |
| `isb_exercise.py` | Exercice ISB autonome (Wu et al. 2002/2005) : `compute_isb_lcs`, métadonnées segments, mode guidé, écarts angulaires |
| `isb_recipes.py` | Recettes de construction guidée des 7 repères locaux ISB, dérivées des FrameTemplate |
| `isb_step_engine.py` | Moteur wizard ISB : séquence étapes, feedback pédagogique sur choix erronés |
| `frame_template.py` | Templates déclaratifs FrameTemplate → Steps ISB (source de vérité) |
| `anthro_recipes.py` | 20 recettes anthropométriques interactives (exercice 4) |
| `anthro_step_engine.py` | Moteur guidé anthropo : distances, angles, asymétries |
| `anthro_measures_exercise.py` | 20 mesures anthropométriques structurées (circumférences, diamètres) ; chaque AnthroMeasure implémente `compute(ground_truth)` |
| `biomechanics.py` | Mesures biomécaniques calculables depuis positions landmarks 3D mm (ISB Wu 2002/2005) |
| `face_blur.py` | Flou visage/tête sur vertex colors (matrice adjacence sparse pour perf) |
| `sticker_removal.py` | Inpainting vertex-colour : suppression stickers photogrammétriques verts ~20mm par IDW |

### `src/bonylandmarks/data/`

| Chemin | Contenu |
|---|---|
| `landmarks.json` | 182 repères osseux bilingues (9 thèmes : anatomy, shoulder, gait, posture, upper_limb, core, knee_rehab, lower_limb, cpr) |
| `bones/landmark_positions.json` | `code → [x, y, z]` — positions de référence corrigées par l'expert (généré par `compute_landmarks.py`, affiné par l'annotateur) |
| `bones/*.stl` ou `*.obj` | Meshes osseux BodyParts3D (**gitignorés**) |
| `bones/validation/*.png` | Images de validation des algorithmes géométriques |

---

## 3. Système de coordonnées BodyParts3D (CRITIQUE pour tous les agents d'algo)

```
X : GAUCHE = +X,  DROITE = -X   (opposé du standard RAS)
Y : POSTÉRIEUR = +Y, ANTÉRIEUR = -Y   → ANT = -1 dans le code
Z : SUPÉRIEUR = +Z, INFÉRIEUR = -Z   (0 aux pieds, ~1750 au sommet du crâne)

LAT_RIGHT = -1   (right = -X)
LAT_LEFT  = +1   (left  = +X)
SUP       = +1
ANT       = -1

def lat(side: str) -> int:
    return LAT_RIGHT if side == "right" else LAT_LEFT
```

Confirmé empiriquement : glabella est à min Y, crâne à max Z, fémur droit centroïde X < 0.

Les coordonnées BodyLoop (scan étudiant) sont aussi en **millimètres** mais dans un référentiel différent (pas BodyParts3D). Ne pas mélanger les deux.

---

## 4. Algorithmes géométriques (`compute_landmarks.py`)

### Helpers principaux

```python
pct_mask(pts, axis, lo, hi) -> np.ndarray
# Masque booléen : vertices entre lo-ième et hi-ième percentile sur l'axe

extreme(pts, axis, sign, mask=None) -> int
# Index du vertex maximisant (sign=+1) ou minimisant (sign=-1) sur l'axe

score_extreme(pts, [(axis, weight), ...], mask=None) -> int
# Index du vertex maximisant la somme pondérée sign*pts[:,ax]
# ATTENTION : raw values, pas normalisées — Z domine si range >> X ou Y
# Quand les axes ont des amplitudes très différentes, normaliser manuellement

get_pts(bone_stem) -> np.ndarray  # charge mesh OBJ/STL depuis data/bones/
```

### Structure RULES

```python
RULES: dict[str, tuple[str, callable]] = {
    "landmark_code": ("bone_stem", lambda pts: [x, y, z]),
    ...
}
```

Chaque entrée `code → (bone_stem, fn(pts))` où `pts` est le tableau (N, 3) des vertices du mesh.

---

## 5. Structures de données importantes

### `bone_map.py`

```python
LANDMARK_BONE: dict[str, str | None]  # code → bone_stem (None = pas de mesh dispo)
BONE_JOINTS:   dict[str, list[str]]   # bone_stem → [bones adjacents pour mode "Adjacents"]
BONE_LABEL_FR: dict[str, str]         # bone_stem → nom français affiché dans le dock
```

### `landmarks_extended.py`

```python
LANDMARKS: list[Landmark]  # 182 repères
# Chaque Landmark : code, name_fr, name_en, hint_fr, hint_en, body_side, theme, category
```

### `data/bones/landmark_positions.json`

```json
{
  "landmark_code": [x, y, z],
  ...
}
```

Positions en mm dans le référentiel BodyParts3D. Générées algorithmiquement par `compute_landmarks.py`, puis corrigées manuellement via l'annotateur.

---

## 6. Workflow de correction des algorithmes

1. L'expert ouvre l'annotateur → corrige la position d'un repère en cliquant sur le mesh → `Enregistrer` → `landmark_positions.json` mis à jour.
2. Lancer `compute_landmarks.py --dry` pour recalculer les positions algo sans écraser le JSON.
3. Comparer : `dX = algo_x - corrigé_x`, `dY = algo_y - corrigé_y`, `dZ = algo_z - corrigé_z`.
4. Interpréter le biais : ex. dZ > 0 → algo trop supérieur, ajuster la règle RULES.
5. Modifier `compute_landmarks.py` → relancer → itérer.

Script de comparaison disponible dans le scratchpad de session (non versionné).

### Seuils d'erreur

- **Objectif** : toutes les erreurs < 10 mm.
- **EAM** (external_acoustic_meatus_left/right) : erreur résiduelle ~21–28 mm, irréductible — l'arcade zygomatique est englobée dans le mesh crânien, le méat acoustique externe réel n'est pas accessible en surface.

---

## 7. Base de données BodyParts3D

### Meshes osseux (gitignorés)

- Chemin local : `src/bonylandmarks/data/bones/`
- Format : `*.stl` ou `*.obj` (stem = bone_stem de `LANDMARK_BONE`)
- **Non versionnés** (`.gitignore` inclut `*.stl`, `*.obj`)

### Meshes musculaires

- Dossier configurable localement (ex. `C:\Users\micka\Downloads\isa_BP3D_4.0_obj_99\...`)
- Nommés `FJ{concept_id}.obj` (ex. `FJ13039.obj`)
- En-tête de chaque fichier OBJ contient :
  ```
  # English name : Pectoralis major
  # Concept ID : FMA13039
  ```
- `isa_parts_list_e.txt` : mapping concept ID ↔ nom anglais
- `BONE_MUSCLES` dans `muscle_map.py` : `bone_stem → [FMA IDs]`

### Attribution obligatoire

> BodyParts3D, © The Database Center for Life Science licensed under CC Attribution-Share Alike 2.1 Japan

https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html

---

## 8. Annotateur — Dock "Os de référence"

L'annotateur (`annotator.py`) intègre un dock latéral avec :

- **Édition de position** : affichage X/Y/Z en mm, bouton Enregistrer → écrit dans `landmark_positions.json`
- **Vue squelette / adjacents** : bascule entre affichage os seul et os + adjacents (via `BONE_JOINTS`)
- **Courbure** : `mesh.curvature(curv_type="mean")` → ndarray PyVista ; slider percentile P50–P99 pour filtrer les valeurs extrêmes
- **Muscles BP3D** : chargement des OBJ muscles depuis dossier local selon `BONE_MUSCLES[bone_stem]`
- **Picking** : observers VTK (pas callbacks PyVista) pour éviter les conflits de double-clic

---

## 9. Chiffrement des scans

```
clé = SHA-256(matricule.lower().strip() + ":" + YYYYMMDD)
Format fichier chiffré : 12B nonce | 16B GCM tag | ciphertext (AES-256-GCM)
```

Aucune clé ne transite vers le serveur. Déchiffrement 100% local.

---

## 10. Format GLB BodyLoop

Deux variantes :

| Type GLB | Contenu |
|---|---|
| `avatar_3d` | Mesh corps + texture PBR + nœud `AutoMarkers` (83 repères nommés) + nœuds squelette + `Analysis_Spine` |
| `mesh_3d` | Mesh corps + texture PBR uniquement (pas de markers) — utilisé par l'annotateur |

Unité glTF : **mètres** → `mesh_loader.py` convertit automatiquement en **millimètres** (×1000).

---

## 11. Conventions git

- Branche principale : `main`
- Messages de commit : **français**, préfixe conventionnel (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- Attribution agents IA : ajouter en fin de message de commit :
  ```
  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
  ```
- Fichiers gitignorés importants : `*.stl`, `*.obj`, `manifest_private*.json`, `dev_landmarks.json`, `data/bones/validation/`

---

## 12. Tests et CI

```bash
pytest                  # lance tous les tests
pip install -e ".[dev]" # installation développement
```

CI GitHub Actions : tests + build exécutables Windows/macOS à chaque push sur `main`.

---

## 13. Points d'attention pour les agents

1. **Ne jamais mélanger** les coordonnées BodyParts3D (référence os) et BodyLoop (scan étudiant) — systèmes distincts.
2. **`score_extreme` n'est pas normalisé** : si Z a une plage ~1750 mm et X ~80 mm, Z dominera complètement. Normaliser si besoin ou utiliser `pct_mask` pour restreindre.
3. **Les fichiers STL/OBJ sont gitignorés** — ne pas tenter de les versionner.
4. **L'annotateur est indépendant du viewer étudiant** — ne pas importer `viewer.py` depuis `annotator.py` ni vice-versa.
5. **`LANDMARKS` dans `landmarks_extended.py`** est la source de vérité des 182 repères — `landmarks.py` ne contient que les 24 de l'exercice 1 (app étudiant).
6. **Picking VTK** dans l'annotateur : utiliser des observers VTK (`AddObserver`) et non des callbacks PyVista (`on_left_click`) pour éviter les conflits.
