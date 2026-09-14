"""Minimal bilingual string table for the student GUI."""

from __future__ import annotations

from typing import Literal

Language = Literal["fr", "en"]

_STRINGS: dict[str, dict[Language, str]] = {
    "app_title": {
        "fr": "BonyLandmarks — Repères anatomiques",
        "en": "BonyLandmarks — Anatomical Landmarks",
    },
    "login_title": {
        "fr": "Connexion",
        "en": "Login",
    },
    "matricule_label": {
        "fr": "Matricule étudiant",
        "en": "Student ID",
    },
    "birthdate_label": {
        "fr": "Date de naissance (AAAA-MM-JJ)",
        "en": "Date of birth (YYYY-MM-DD)",
    },
    "server_url_label": {
        "fr": "URL du serveur enseignant",
        "en": "Teacher server URL",
    },
    "connect_button": {
        "fr": "Se connecter",
        "en": "Connect",
    },
    "error_credentials": {
        "fr": "Matricule ou date de naissance incorrects.",
        "en": "Wrong student ID or date of birth.",
    },
    "error_network": {
        "fr": "Impossible de joindre le serveur : {detail}",
        "en": "Cannot reach the server: {detail}",
    },
    "instructions": {
        "fr": "Cliquez sur la surface du scan pour placer le repère.",
        "en": "Click on the scan surface to place the landmark.",
    },
    "landmark_label": {
        "fr": "Repère à placer ({index}/{total})",
        "en": "Landmark to place ({index}/{total})",
    },
    "confirm_button": {
        "fr": "Confirmer ce point",
        "en": "Confirm this point",
    },
    "redo_button": {
        "fr": "Recommencer ce repère",
        "en": "Redo this landmark",
    },
    "error_mm_label": {
        "fr": "Erreur : {value:.1f} mm",
        "en": "Error: {value:.1f} mm",
    },
    "session_complete": {
        "fr": "Session terminée ! Erreur moyenne : {mean:.1f} mm",
        "en": "Session complete! Mean error: {mean:.1f} mm",
    },
    "save_button": {
        "fr": "Sauvegarder les résultats (JSON)",
        "en": "Save results (JSON)",
    },
    "green_marker_tooltip": {
        "fr": "Marqueur BodyLoop (référence)",
        "en": "BodyLoop marker (reference)",
    },
    "lang_toggle": {
        "fr": "English",
        "en": "Français",
    },
    "select_name_label": {
        "fr": "Sélectionnez votre nom",
        "en": "Select your name",
    },
    "dob_label": {
        "fr": "Date de naissance (JJ/MM/AAAA)",
        "en": "Date of birth (DD/MM/YYYY)",
    },
    "error_server_config": {
        "fr": "Fichier server.json introuvable. Contactez l'enseignant.",
        "en": "server.json not found. Contact your instructor.",
    },
    "error_invalid_matricule": {
        "fr": "Le matricule doit contenir 8 chiffres.",
        "en": "Student ID must be 8 digits.",
    },
    "error_invalid_dob": {
        "fr": "Date de naissance invalide. Format : JJ/MM/AAAA",
        "en": "Invalid date of birth. Format: DD/MM/YYYY",
    },
}


def tr(key: str, lang: Language = "fr", **kwargs: object) -> str:
    """Translate *key* to *lang*, formatting with **kwargs."""
    row = _STRINGS.get(key)
    if row is None:
        return key
    text = row.get(lang, row.get("fr", key))
    return text.format(**kwargs) if kwargs else text
