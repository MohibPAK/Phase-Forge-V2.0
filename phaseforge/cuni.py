"""
cuni.py — Copper–Nickel (Cu–Ni) isomorphous phase diagram engine.

Cu–Ni is the textbook isomorphous binary: Cu and Ni are completely soluble in
each other in both the liquid and the solid (single FCC solid solution, α),
so there is no eutectic, peritectic or eutectoid reaction at all — just a
"lens" between the two liquidus/solidus curves.

Data (melting points and a handful of liquidus/solidus tie-line points) are
the standard classroom values traced from the Cu–Ni diagram of
Callister & Rethwisch, *Materials Science and Engineering* (adapted from
*Phase Diagrams of Binary Nickel Alloys*, P. Nash, ed., ASM International,
1991):

    Cu melts at 1085 °C, Ni melts at 1453 °C.
    liquidus (wt% Ni, °C): (0,1085) (24,1220) (32,1250) (35,1260) (100,1453)
    solidus  (wt% Ni, °C): (0,1085) (35,1220) (43,1250) (46,1260) (100,1453)

Both curves are fit with a shape-preserving (PCHIP) spline so they stay
monotonic between the two melting points, and the solidus is nudged so it
never crosses the liquidus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from .ironcarbon import MicroState  # generic (C0, T, region, recipe, ...) container

T_ROOM = 25.0
T_CU = 1085.0
T_NI = 1453.0

_LIQ_PTS = np.array([(0.0, T_CU), (24.0, 1220.0), (32.0, 1250.0), (35.0, 1260.0), (100.0, T_NI)])
_SOL_PTS = np.array([(0.0, T_CU), (35.0, 1220.0), (43.0, 1250.0), (46.0, 1260.0), (100.0, T_NI)])

_liquidus_spline = PchipInterpolator(_LIQ_PTS[:, 0], _LIQ_PTS[:, 1])
_solidus_spline = PchipInterpolator(_SOL_PTS[:, 0], _SOL_PTS[:, 1])


class CuNiRegion(str, Enum):
    LIQUID = "Liquid (L)"
    L_ALPHA = "L + α"
    ALPHA = "α (FCC solid solution)"


class CuNiError(ValueError):
    pass


@dataclass
class CuNiTieLine:
    temperature: float
    c_liquid: float
    c_solid: float

    @property
    def points(self) -> List[Tuple[str, float]]:
        return [("L", self.c_liquid), ("α", self.c_solid)]

    @property
    def labels(self) -> List[str]:
        return [f"{p} ({c:.1f}% Ni)" for p, c in self.points]


@dataclass
class PathEvent:
    temperature: float
    kind: str
    title: str
    reaction: str
    phases_after: str
    microstructure: str
    t_sample: float = 0.0

    @property
    def temp_text(self) -> str:
        return f"{self.temperature:.0f}"


class CuNiDiagram:
    """Equilibrium Cu–Ni isomorphous diagram, tie lines, lever rule and
    the (very short) equilibrium-cooling transformation path."""

    name = "Cu–Ni"
    component_A = "Cu"
    component_B = "Ni"

    # ------------------------------------------------------------------ curves
    @staticmethod
    def liquidus_T(C) -> np.ndarray:
        return _liquidus_spline(np.clip(C, 0.0, 100.0))

    @staticmethod
    def solidus_T(C) -> np.ndarray:
        return _solidus_spline(np.clip(C, 0.0, 100.0))

    @staticmethod
    def _check_C(C: float):
        if not (0.0 <= C <= 100.0):
            raise CuNiError("Composition must lie between 0 and 100 wt% Ni.")

    # ------------------------------------------------------------------ phase lookup
    def phase_at(self, C: float, T: float) -> CuNiRegion:
        self._check_C(C)
        t_liq, t_sol = float(self.liquidus_T(C)), float(self.solidus_T(C))
        if T > t_liq + 1e-9:
            return CuNiRegion.LIQUID
        if T < t_sol - 1e-9:
            return CuNiRegion.ALPHA
        return CuNiRegion.L_ALPHA

    def _c_on_liquidus(self, T: float) -> float:
        return float(brentq(lambda c: float(self.liquidus_T(c)) - T, 0.0, 100.0))

    def _c_on_solidus(self, T: float) -> float:
        return float(brentq(lambda c: float(self.solidus_T(c)) - T, 0.0, 100.0))

    def tie_line(self, C: float, T: float) -> CuNiTieLine:
        if self.phase_at(C, T) != CuNiRegion.L_ALPHA:
            raise CuNiError("Point is single-phase — no tie line.")
        return CuNiTieLine(temperature=T, c_liquid=self._c_on_liquidus(T), c_solid=self._c_on_solidus(T))

    @staticmethod
    def lever_rule(C0: float, c_liquid: float, c_solid: float) -> Tuple[float, float]:
        """(weight fraction liquid, weight fraction α)."""
        if c_solid == c_liquid:
            return 0.5, 0.5
        f_alpha = (C0 - c_liquid) / (c_solid - c_liquid)
        f_alpha = min(max(f_alpha, 0.0), 1.0)
        return 1.0 - f_alpha, f_alpha

    def phase_fractions(self, C0: float, T: float) -> Dict[str, float]:
        region = self.phase_at(C0, T)
        if region == CuNiRegion.LIQUID:
            return {"L": 1.0}
        if region == CuNiRegion.ALPHA:
            return {"α": 1.0}
        tl = self.tie_line(C0, T)
        f_l, f_a = self.lever_rule(C0, tl.c_liquid, tl.c_solid)
        return {"L": f_l, "α": f_a}

    # ------------------------------------------------------------------ classification
    @staticmethod
    def common_name(C0: float) -> str:
        if C0 <= 5:
            return "commercially pure copper"
        if C0 <= 45:
            return "copper-nickel alloy (e.g. cupronickel piping / condenser-tube range)"
        if C0 < 95:
            return "nickel-copper alloy (e.g. Monel-type range)"
        return "commercially pure nickel"

    def critical_temperatures(self, C0: float) -> Dict[str, float]:
        self._check_C(C0)
        return {"Liquidus": float(self.liquidus_T(C0)), "Solidus": float(self.solidus_T(C0))}

    # ------------------------------------------------------------------ transformation path
    def transformation_path(self, C0: float) -> List[PathEvent]:
        self._check_C(C0)
        t_liq, t_sol = float(self.liquidus_T(C0)), float(self.solidus_T(C0))
        ev = [
            PathEvent(
                t_liq, "liquidus", "Liquidus — solidification begins", "L → L + α", "L + α",
                "α dendrites (richer in Ni, the higher-melting element) nucleate and grow into the melt.",
                t_sample=t_liq - 0.35 * (t_liq - t_sol),
            ),
            PathEvent(
                t_sol, "solidus", "Solidification complete", "L + α → α", "α",
                "Fully solid single-phase solid solution: equiaxed grains, uniform composition C₀ "
                "under true equilibrium cooling.",
                t_sample=t_sol - 0.15 * (t_liq - t_sol),
            ),
            PathEvent(
                T_ROOM, "room", "Room temperature", "—", "α",
                "No further transformation on cooling — Cu and Ni stay fully soluble in the solid state "
                "all the way to room temperature, so the grains formed at the solidus persist unchanged. "
                "(Real castings cool too fast for full diffusion and end up 'cored' — Ni-rich grain "
                "cores surrounded by more Cu-rich rims — but that is a non-equilibrium effect this "
                "equilibrium model does not include.)",
                t_sample=T_ROOM,
            ),
        ]
        return ev

    def evolution_stages(self, C0: float) -> List[Tuple[str, float, str]]:
        t_liq = float(self.liquidus_T(C0))
        events = self.transformation_path(C0)
        stages = [("Molten alloy", min(t_liq + 40.0, 1600.0), "L")]
        stages += [(e.title, e.t_sample, e.phases_after) for e in events]
        return stages

    # ------------------------------------------------------------------ microstructure state
    def micro_state(self, C0: float, T: float) -> MicroState:
        self._check_C(C0)
        region = self.phase_at(C0, T)
        ms = MicroState(C0=C0, T=T, region=region.value, recipe="grains", fractions={})
        if region == CuNiRegion.LIQUID:
            ms.recipe = "liquid"
            ms.fractions = {"L": 1.0}
            ms.constituents = {"Liquid": 1.0}
        elif region == CuNiRegion.ALPHA:
            ms.recipe, ms.phase = "grains", "α"
            ms.fractions = {"α": 1.0}
            ms.constituents = {"α solid solution grains": 1.0}
        else:
            fr = self.phase_fractions(C0, T)
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "dendrites", "L", "α", fr["α"]
            ms.fractions = fr
            ms.constituents = {"Liquid": fr["L"], "α dendrites": fr["α"]}
        return ms

    # ------------------------------------------------------------------ plotting geometry
    def boundaries(self) -> Dict[str, List[Tuple[float, float]]]:
        cs = np.linspace(0.0, 100.0, 60)
        return {
            "Liquidus": list(zip(cs.tolist(), self.liquidus_T(cs).tolist())),
            "Solidus": list(zip(cs.tolist(), self.solidus_T(cs).tolist())),
        }

    def field_polygons(self) -> Dict[CuNiRegion, List[Tuple[float, float]]]:
        cs = np.linspace(0.0, 100.0, 60)
        liq = list(zip(cs.tolist(), self.liquidus_T(cs).tolist()))
        sol = list(zip(cs.tolist(), self.solidus_T(cs).tolist()))
        top = 1600.0
        bottom = 950.0
        return {
            CuNiRegion.LIQUID: [(0.0, top)] + liq + [(100.0, top)],
            CuNiRegion.L_ALPHA: liq + list(reversed(sol)),
            CuNiRegion.ALPHA: [(0.0, bottom)] + sol + [(100.0, bottom)],
        }


FIELD_COLORS: Dict[CuNiRegion, str] = {
    CuNiRegion.LIQUID: "#FCE9D2",
    CuNiRegion.L_ALPHA: "#F3E6B8",
    CuNiRegion.ALPHA: "#E3F1EE",
}
