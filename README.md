# BonyLandmarks — Student App

Interactive 3D tool for students to place anatomical landmarks on their own BodyLoop body scan.

## Features

- Loads an encrypted GLB body scan from the teacher server (authenticated with student ID + date of birth)
- Displays 24 BodyLoop anatomical landmarks as green reference spheres
- Guides the student to click each landmark on the 3D surface
- Computes Euclidean error (mm) between student pick and BodyLoop ground truth
- Color-coded feedback (green ≤ 20 mm / orange ≤ 40 mm / red > 40 mm)
- Exports results as JSON → upload to Moodle
- Bilingual interface (FR / EN)

## Anatomical landmarks (24 total)

| Code | French | English |
|------|--------|---------|
| `ASIS_left/right` | Épine iliaque antéro-supérieure | Anterior superior iliac spine |
| `greater_trochanter_left/right` | Grand trochanter | Greater trochanter |
| `acromion_left/right` | Acromion | Acromion |
| `lateral/medial_epicondyle_left/right` | Épicondyle latéral/médial | Lateral/medial epicondyle |
| `ulnar/radial_styloid_left/right` | Styloïde ulnaire/radiale | Ulnar/radial styloid |
| `lateral/medial_knee_left/right` | Condyle latéral/médial genou | Lateral/medial knee condyle |
| `lateral/medial_malleolus_left/right` | Malléole latérale/médiale | Lateral/medial malleolus |
| `heel_left/right` | Talon | Heel |

## Installation (development)

```bash
pip install -e ".[dev]"
```

## Run

```bash
bonylandmarks
```

## Tests

```bash
pytest
```

## Build executable

```bash
pyinstaller --onefile --windowed --name BonyLandmarks src/bonylandmarks/main.py
```

Executables for Windows and macOS are automatically built by GitHub Actions on every push to `main`.
