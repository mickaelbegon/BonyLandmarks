"""Shared UI style constants for BonyLandmarks 3D exercises."""

BG = "#1a1a2e"
CARD = "#252540"
TEXT = "#e8e8f0"
MUTED = "#a0a0c0"
BLUE = "#7cb9ff"

BASE_STYLE = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI", Arial, sans-serif;
}}
"""

BTN_PRIMARY = (
    f"QPushButton {{ background: #3a5faa; color: {TEXT}; border: none; "
    f"border-radius: 6px; padding: 8px 14px; font-size: 13px; }}"
    f"QPushButton:hover {{ background: #4a70cc; }}"
    f"QPushButton:disabled {{ background: #2a2a44; color: #606080; }}"
)

NAV_BTN = (
    "QPushButton { background-color: rgba(26,26,46,180); color: #e0e0e0; "
    "border: 1px solid rgba(255,255,255,0.15); border-radius: 5px; "
    "font-size: 11px; padding: 3px 7px; }"
    "QPushButton:hover { background-color: rgba(60,80,140,210); "
    "border-color: rgba(100,150,255,0.7); }"
)

NAV_BTN_CHECK = NAV_BTN + (
    "QPushButton:checked { background-color: rgba(120,60,20,210); "
    "border-color: rgba(255,160,60,0.8); color: #ffcc88; }"
)

COLOR_BONE_PRIMARY = "#e8d5b0"
COLOR_BONE_REF = "#555577"
COLOR_SPHERE_CANDIDATE = "#ffdd00"
COLOR_SPHERE_REF = "#ff2222"
COLOR_SPHERE_OK = "#00cc44"
