"""Export student session results to JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .scoring import SessionScore


def export_session(
    score: SessionScore,
    matricule: str,
    output_path: Path,
) -> None:
    """Write session results as a JSON file."""
    payload = {
        "app_version": "0.1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "matricule": matricule,
        "score": score.to_dict(),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
