"""
fec_layout.py — pure-data layout for the Fe–Fe₃C diagram (no plotting imports).

Shared by the interactive Plotly view (`fec_ui.py`) and the static Matplotlib
figure used in reports (`fec_report.py`), so both label the diagram identically.
"""

from __future__ import annotations

from .ironcarbon import C_C, C_CEM, C_E, C_P, C_S

# Each view: axis window, in-field text labels (x, y, text, size) and
# tag labels that point at fields too thin to hold text: (tip_x, tip_y, text, dx_px, dy_px)
VIEWS = {
    "full": {
        "title": "Full diagram (0 – 6.67 wt% C)",
        "x": (0.0, 6.9),
        "y": (0.0, 1600.0),
        "labels": [
            (4.50, 1420, "Liquid (L)", 13),
            (3.05, 1218, "L + γ", 11),
            (5.80, 1168, "L + Fe₃C", 9),
            (1.00, 1000, "γ (austenite)", 12),
            (4.40, 930, "γ + Fe₃C", 12),
            (3.50, 420, "α + Fe₃C", 12),
            (0.38, 815, "α+γ", 9),
        ],
        "tags": [
            (0.006, 380, "α (ferrite)", 60, -26),
            (0.03, 1470, "δ", 46, -6),
        ],
        "letters": ["A", "B", "N", "G", "E", "C", "F", "D", "S", "K", "P"],
    },
    "steel": {
        "title": "Steel region (0 – 2.5 wt% C)",
        "x": (0.0, 2.5),
        "y": (300.0, 1600.0),
        "labels": [
            (1.60, 1520, "Liquid (L)", 13),
            (1.30, 1360, "L + γ", 12),
            (1.00, 1050, "γ (austenite)", 12),
            (1.95, 900, "γ + Fe₃C", 12),
            (0.19, 820, "α + γ", 10),
            (1.50, 520, "α + Fe₃C", 12),
        ],
        "tags": [
            (0.006, 420, "α (ferrite)", 60, -26),
            (0.02, 1468, "δ", 62, 14),
            (0.20, 1518, "L + δ", 62, -30),
            (0.10, 1448, "δ + γ", 70, 34),
        ],
        "letters": ["A", "B", "H", "J", "N", "G", "E", "S", "P"],
    },
    "peritectic": {
        "title": "Peritectic corner (0 – 0.7 wt% C)",
        "x": (0.0, 0.7),
        "y": (1340.0, 1560.0),
        "labels": [
            (0.45, 1535, "Liquid (L)", 13),
            (0.20, 1512, "L + δ", 11),
            (0.098, 1466, "δ + γ", 9),
            (0.62, 1440, "L + γ", 12),
            (0.30, 1395, "γ (austenite)", 12),
        ],
        "tags": [
            (0.025, 1468, "δ", 46, 8),
        ],
        "letters": ["A", "B", "H", "J", "N"],
    },
}

LETTER_POSITION = {
    "A": "top right", "B": "top right", "H": "top left", "J": "bottom center",
    "N": "bottom right", "G": "bottom right", "E": "top right", "C": "top center",
    "F": "bottom left", "D": "top left", "P": "bottom center", "S": "bottom center",
    "K": "bottom left", "Q": "top right",
}

# Alloy-class bands drawn along the top of the diagram: (from, to, label)
CLASS_BANDS = [
    (C_P, C_S, "Hypo-<br>eutectoid<br>steel"),
    (C_S, C_E, "Hyper-<br>eutectoid<br>steel"),
    (C_E, C_C, "Hypoeutectic<br>cast iron"),
    (C_C, C_CEM, "Hypereutectic<br>cast iron"),
]

# Invariant-line colours
INVARIANT_COLORS = {
    "Peritectic 1495 °C": "#B8860B",
    "Eutectic 1147 °C": "#8E44AD",
    "Eutectoid 727 °C (A₁)": "#1F7A5A",
}
