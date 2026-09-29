# Meshes de référence osseuse

Ce dossier contient les meshes 3D (STL ou OBJ) des os affichés dans le panneau
de référence de l'outil d'annotation.

## Source recommandée

**BodyParts3D** (DBCLS, Japon) — licence Creative Commons BY-SA 2.1 JP  
<https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html>

Télécharger les fichiers individuels (format OBJ ou X3D → convertir en STL avec
MeshLab ou Blender si nécessaire).

## Fichiers attendus

Placer ici les fichiers STL **ou** OBJ avec les noms exacts suivants :

| Fichier                   | Os                          |
|---------------------------|-----------------------------|
| `skull.stl`               | Crâne                       |
| `sternum.stl`             | Sternum                     |
| `clavicle_left.stl`       | Clavicule gauche            |
| `clavicle_right.stl`      | Clavicule droite            |
| `scapula_left.stl`        | Scapula gauche              |
| `scapula_right.stl`       | Scapula droite              |
| `humerus_left.stl`        | Humérus gauche              |
| `humerus_right.stl`       | Humérus droit               |
| `radius_left.stl`         | Radius gauche               |
| `radius_right.stl`        | Radius droit                |
| `ulna_left.stl`           | Ulna gauche                 |
| `ulna_right.stl`          | Ulna droit                  |
| `pelvis.stl`              | Pelvis (deux os coxaux)     |
| `sacrum.stl`              | Sacrum                      |
| `femur_left.stl`          | Fémur gauche                |
| `femur_right.stl`         | Fémur droit                 |
| `patella_left.stl`        | Patella gauche              |
| `patella_right.stl`       | Patella droite              |
| `tibia_left.stl`          | Tibia gauche                |
| `tibia_right.stl`         | Tibia droit                 |
| `fibula_left.stl`         | Fibula gauche               |
| `fibula_right.stl`        | Fibula droite               |
| `calcaneus_left.stl`      | Calcanéus gauche            |
| `calcaneus_right.stl`     | Calcanéus droit             |
| `foot_left.stl`           | Pied gauche (métatarses…)   |
| `foot_right.stl`          | Pied droit                  |
| `cervical_vertebrae.stl`  | Vertèbres cervicales C1-C7  |
| `thoracic_vertebrae.stl`  | Vertèbres thoraciques T1-T12|
| `lumbar_vertebrae.stl`    | Vertèbres lombaires L1-L5   |
| `rib_left.stl`            | Côtes gauches               |
| `rib_right.stl`           | Côtes droites               |

> Les fichiers manquants n'affichent simplement pas de référence — l'outil
> fonctionne normalement avec un sous-ensemble des os.
>
> Les formats `.obj` sont aussi acceptés (même nom, extension `.obj`).

## Exercice « Anatomie 3D — os et muscles »

Cet exercice n'utilise **pas** ce dossier mais `../bones_full/` (un os par
fichier) et `../muscles/` (un muscle par fichier), générés à partir de
l'archive BodyParts3D 4.0 (OBJ) par :

```bash
python scripts/import_bp3d.py --kind all          # os + muscles
python scripts/import_bp3d.py --kind bone --dry-run
```

Le script lit `data/anatomy_bones.json` / `data/anatomy_muscles.json`,
retrouve les fichiers `FJ*.obj` via l'en-tête `# Concept ID` (le numéro dans
le nom du fichier n'est pas le numéro FMA) et écrit les meshes avec leurs
coordonnées BodyParts3D d'origine.  Voir `--help` pour `--src`, `--force`,
`--target-faces`.
