"""Validation of GLB marker presence against required landmark codes.

Usage
-----
>>> from bonylandmarks.validation import validate_markers, report_validation
>>> result = validate_markers(landmark_markers, required_codes)
>>> report_validation(result)
"""

from __future__ import annotations

import numpy as np


def validate_markers(
    landmark_markers: dict[str, np.ndarray],
    required_codes: list[str],
) -> dict[str, list[str]]:
    """Compare markers found in a GLB file with a required list of codes.

    Parameters
    ----------
    landmark_markers:
        Mapping of {code: 3D-position} as returned by ``load_avatar_glb``.
        Only the keys (codes) are used — positions are ignored.
    required_codes:
        Codes that must be present for a valid evaluation session.

    Returns
    -------
    dict with three keys:
        ``"present"``  — codes that are both required and present in the GLB.
        ``"missing"``  — required codes that are absent from the GLB.
        ``"extra"``    — codes present in the GLB but not in *required_codes*.
    """
    glb_codes: set[str] = set(landmark_markers.keys())
    required: set[str] = set(required_codes)

    return {
        "present": sorted(required & glb_codes),
        "missing": sorted(required - glb_codes),
        "extra":   sorted(glb_codes - required),
    }


def report_validation(result: dict[str, list[str]]) -> None:
    """Print a colour-coded console report of a validation result.

    Uses ANSI escape codes (visible in any modern terminal).
    The three sections are:

    * GREEN  — present markers (required & found)
    * RED    — missing markers (required but absent from the GLB)
    * YELLOW — extra markers (in the GLB but not required)
    """
    _GREEN  = "\033[92m"
    _RED    = "\033[91m"
    _YELLOW = "\033[93m"
    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"

    present = result.get("present", [])
    missing = result.get("missing", [])
    extra   = result.get("extra",   [])

    print(f"\n{_BOLD}=== Validation des marqueurs GLB ==={_RESET}")

    print(
        f"{_GREEN}{_BOLD}Présents  ({len(present):>3}){_RESET}"
        + (f" : {', '.join(present)}" if present else " : —")
    )

    if missing:
        print(
            f"{_RED}{_BOLD}Manquants ({len(missing):>3}){_RESET}"
            f" : {', '.join(missing)}"
        )
    else:
        print(f"{_GREEN}{_BOLD}Manquants (  0){_RESET} : —")

    if extra:
        print(
            f"{_YELLOW}{_BOLD}Supplémentaires ({len(extra):>3}){_RESET}"
            f" : {', '.join(extra)}"
        )
    else:
        print(f"Supplémentaires (  0) : —")

    print()
