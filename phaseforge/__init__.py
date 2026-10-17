"""
PhaseForge — Computational Materials Science Toolkit
======================================================

A Python toolkit for analyzing binary alloy phase diagrams and phase
transformations: automated phase-diagram construction, tie-line and
lever-rule analysis, solidification-path tracing, and cooling-curve
simulation — plus a complete iron–carbon (Fe–Fe₃C) module with
microstructure evolution.

Developed by Mohib Mehdi(MME 24-28).
"""

from .core import EutecticPhaseDiagram, PhaseRegion, TieLine
from .simulation import SolidificationPath, CoolingCurveResult, simulate_cooling_curve
from .ironcarbon import IronCarbonDiagram, FeCRegion, AlloyClass, PathEvent, MicroState
from .cuni import CuNiDiagram, CuNiRegion
from .monotectic import MonotecticDiagram, MonotecticRegion, PRESETS as MONOTECTIC_PRESETS
from .ordering import OrderingDiagram, PRESETS as ORDERING_PRESETS

__all__ = [
    "EutecticPhaseDiagram",
    "PhaseRegion",
    "TieLine",
    "SolidificationPath",
    "CoolingCurveResult",
    "simulate_cooling_curve",
    "IronCarbonDiagram",
    "FeCRegion",
    "AlloyClass",
    "PathEvent",
    "MicroState",
    "CuNiDiagram",
    "CuNiRegion",
    "MonotecticDiagram",
    "MonotecticRegion",
    "MONOTECTIC_PRESETS",
    "OrderingDiagram",
    "ORDERING_PRESETS",
]

__version__ = "1.5.0"
__author__ = "Mohib Mehdi(MME 24-28)"
