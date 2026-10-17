"""
ordering.py — generic engine for isomorphous binary systems with a solid-state
order-disorder transformation: Cu-Au and Cu-Pd.

Above the solidus these behave like Cu-Ni (`cuni.py`): complete liquid and solid
solubility, one disordered FCC solid solution (A1). The difference is well
below the solidus: at stoichiometric compositions the randomly occupied lattice
orders into a superlattice (L1₂ / L1₀ for Cu-Au, L1₂ / B2 for Cu-Pd) below a
critical temperature, forming a dome in composition-temperature space that
peaks at the stoichiometric composition.

Simplification: the order/disorder boundary is a single curve (ordered below,
disordered above) rather than the narrow ordered+disordered two-phase sliver of
research-grade assessments — the same simplification most introductory texts
make for Cu₃Au. Each dome is an inverted parabola with its apex at
(stoichiometric composition, critical temperature); the dome WIDTH is
schematic.

Data
----
Cu-Au: liquidus/solidus from the Callister & Rethwisch problem table (8 points,
with the congruent minimum at 80 wt% Au, 911 degC). Ordering temperatures:
Cu3Au 390 degC (663 K), CuAu ~410 degC (683 K), CuAu3 ~197 degC (~470 K), from
the Cu-Au literature (compositions 25/50/75 at% Au converted to wt%).

Cu-Pd: pure-metal melting points are exact; the liquidus/solidus between them
is an engineering estimate (narrow lens) — no tabulated points were found.
Ordering: CuPd (B2) at 50 at% Pd stable above ~600 degC per Subramanian &
Laughlin's assessed diagram; Cu3Pd (L1₂) at 25 at% Pd ~500 degC (this region is
still being refined experimentally, e.g. Popov et al. 2018).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from .ironcarbon import MicroState

T_ROOM = 25.0


@dataclass
class OrderDome:
    label: str        # "Cu₃Au"
    key: str          # colour key in microstructure.COLOR, e.g. "Cu3Au"
    structure: str    # "L1₂"
    center: float     # wt% of component B at the stoichiometric composition
    Tc: float         # order -> disorder temperature at the apex, degC
    halfwidth: float  # composition half-width at 0 degC (schematic)

    def T_at(self, C: float) -> float:
        x = (C - self.center) / self.halfwidth
        if abs(x) >= 1.0:
            return -1e9
        return self.Tc * (1.0 - x * x)

    def contains(self, C: float) -> bool:
        return abs(C - self.center) < self.halfwidth


class OrderingError(ValueError):
    pass


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


@dataclass
class OrderTieLine:
    temperature: float
    points: List[Tuple[str, float]]

    @property
    def labels(self) -> List[str]:
        return [f"{p} ({c:.2f}%)" for p, c in self.points]


DISORDERED = "α (disordered FCC solid solution)"


class OrderingDiagram:
    def __init__(self, *, name, component_A, component_B, T_A, T_B,
                 liquidus_points, solidus_points, domes: List[OrderDome], data_note=""):
        self.name, self.component_A, self.component_B = name, component_A, component_B
        self.T_A, self.T_B = float(T_A), float(T_B)
        self.domes = domes
        self.data_note = data_note
        lp = sorted(liquidus_points, key=lambda p: p[0])
        sp = sorted(solidus_points, key=lambda p: p[0])
        self._liq = PchipInterpolator([p[0] for p in lp], [p[1] for p in lp])
        self._sol = PchipInterpolator([p[0] for p in sp], [p[1] for p in sp])
        # a congruent MINIMUM (Cu-Au) splits the diagram into two lenses that touch at c_min
        grid = np.linspace(0.0, 100.0, 4001)
        lv = self._liq(grid)
        i = int(np.argmin(lv))
        self.c_min: Optional[float] = None
        if 5.0 < grid[i] < 95.0 and lv[i] < min(lv[0], lv[-1]) - 1.0:
            self.c_min = float(grid[i])

    # ------------------------------------------------------------------ curves
    def liquidus_T(self, C):
        return self._liq(np.clip(C, 0.0, 100.0))

    def solidus_T(self, C):
        return self._sol(np.clip(C, 0.0, 100.0))

    @staticmethod
    def _check_C(C: float):
        if not (0.0 <= C <= 100.0):
            raise OrderingError("Composition must lie between 0 and 100 wt%.")

    def dome_at(self, C: float) -> Optional[OrderDome]:
        for d in self.domes:
            if d.contains(C):
                return d
        return None

    # ------------------------------------------------------------------ phase lookup
    def phase_at(self, C: float, T: float) -> str:
        self._check_C(C)
        t_liq, t_sol = float(self.liquidus_T(C)), float(self.solidus_T(C))
        if T > t_liq:
            return "Liquid (L)"
        if T > t_sol:
            return "L + α"
        d = self.dome_at(C)
        if d is not None and T <= d.T_at(C):
            return f"{d.label} (ordered, {d.structure})"
        return DISORDERED

    def _branch(self, C: float) -> Tuple[float, float]:
        """Composition bracket of the lens this alloy belongs to."""
        if self.c_min is None:
            return 0.0, 100.0
        return (0.0, self.c_min) if C < self.c_min else (self.c_min, 100.0)

    def _c_on(self, curve, T: float, lo: float, hi: float) -> float:
        return float(brentq(lambda c: float(curve(c)) - T, lo, hi))

    def tie_line(self, C: float, T: float) -> OrderTieLine:
        if self.phase_at(C, T) != "L + α":
            raise OrderingError("Point is single-phase — no tie line.")
        lo, hi = self._branch(C)
        return OrderTieLine(T, [("L", self._c_on(self.liquidus_T, T, lo, hi)),
                                ("α", self._c_on(self.solidus_T, T, lo, hi))])

    @staticmethod
    def lever_rule(C0: float, c_left: float, c_right: float) -> Tuple[float, float]:
        if c_right == c_left:
            return 0.5, 0.5
        f = min(max((C0 - c_left) / (c_right - c_left), 0.0), 1.0)
        return 1.0 - f, f

    def phase_fractions(self, C0: float, T: float) -> Dict[str, float]:
        region = self.phase_at(C0, T)
        if region == "Liquid (L)":
            return {"L": 1.0}
        if region == "L + α":
            tl = self.tie_line(C0, T)
            f_l, f_a = self.lever_rule(C0, tl.points[0][1], tl.points[1][1])
            return {"L": f_l, "α": f_a}
        return {region: 1.0}

    # ------------------------------------------------------------------ classification
    def common_name(self, C0: float) -> str:
        d = self.dome_at(C0)
        if d is not None and abs(C0 - d.center) < 0.6:
            return f"{d.label} stoichiometric alloy"
        return f"{self.component_A}-{self.component_B} alloy ({C0:.1f} wt% {self.component_B})"

    def critical_temperatures(self, C0: float) -> Dict[str, float]:
        out = {"Liquidus": float(self.liquidus_T(C0)), "Solidus": float(self.solidus_T(C0))}
        d = self.dome_at(C0)
        if d is not None and d.T_at(C0) > T_ROOM:
            out[f"{d.label} ordering ({d.structure})"] = d.T_at(C0)
        return out

    # ------------------------------------------------------------------ transformation path
    def transformation_path(self, C0: float) -> List[PathEvent]:
        self._check_C(C0)
        t_liq, t_sol = float(self.liquidus_T(C0)), float(self.solidus_T(C0))
        events = [
            PathEvent(t_liq, "liquidus", "Liquidus — solidification begins", "L → L + α", "L + α",
                      "α dendrites (disordered FCC solid solution) nucleate and grow into the melt.",
                      t_sample=t_liq - 0.3 * (t_liq - t_sol)),
            PathEvent(t_sol, "solidus", "Solidification complete", "L + α → α", "α (disordered)",
                      "Fully solid, disordered α: equiaxed grains of uniform composition C₀ "
                      "(equilibrium cooling).",
                      t_sample=t_sol - 0.05 * max(t_sol - T_ROOM, 1.0)),
        ]
        d = self.dome_at(C0)
        if d is not None:
            t_ord = d.T_at(C0)
            if T_ROOM < t_ord < t_sol:
                events.append(PathEvent(
                    t_ord, "ordering", f"Order-disorder transition ({d.label})",
                    f"α (random) → {d.label} ({d.structure}, ordered)", f"{d.label} (ordered)",
                    f"{self.component_A} and {self.component_B} atoms move from random lattice sites into a "
                    f"regular alternating pattern — the {d.structure} superlattice. Grains and overall "
                    "composition are unchanged; only the atom arrangement inside them changes. "
                    "Antiphase domain boundaries form where ordering started independently in neighbouring "
                    "regions.",
                    t_sample=t_ord - 0.2 * (t_ord - T_ROOM)))
        events.append(PathEvent(T_ROOM, "room", "Room temperature", "—", self.phase_at(C0, T_ROOM),
                                 self._final_text(C0), t_sample=T_ROOM))
        events.sort(key=lambda e: -e.temperature)
        return events

    def _final_text(self, C0: float) -> str:
        d = self.dome_at(C0)
        if d is not None and d.T_at(C0) > T_ROOM:
            return f"Ordered {d.label} ({d.structure}) superlattice with antiphase domain boundaries."
        return "Disordered α (FCC) grains — this composition does not order above room temperature."

    def evolution_stages(self, C0: float) -> List[Tuple[str, float, str]]:
        events = self.transformation_path(C0)
        T_top = min(events[0].temperature + 40.0, max(self.T_A, self.T_B) + 60.0)
        stages = [("Molten alloy", T_top, "L")]
        stages += [(e.title, e.t_sample, e.phases_after) for e in events]
        return stages

    # ------------------------------------------------------------------ microstructure state
    def micro_state(self, C0: float, T: float) -> MicroState:
        region = self.phase_at(C0, T)
        ms = MicroState(C0=C0, T=T, region=region, recipe="grains", fractions={})
        if region == "Liquid (L)":
            ms.recipe = "liquid"
            ms.fractions = {"L": 1.0}
            ms.constituents = {"Liquid": 1.0}
        elif region == "L + α":
            fr = self.phase_fractions(C0, T)
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "dendrites", "L", "α", fr.get("α", 0.0)
            ms.fractions = fr
            ms.constituents = {"Liquid": fr.get("L", 0.0), "α dendrites": fr.get("α", 0.0)}
        elif region == DISORDERED:
            ms.recipe, ms.phase = "grains", "α"
            ms.fractions = {"α": 1.0}
            ms.constituents = {"Disordered α grains": 1.0}
        else:
            d = self.dome_at(C0)
            ms.recipe, ms.phase = "ordered_domains", d.key
            ms.fractions = {region: 1.0}
            ms.constituents = {f"{d.label} ({d.structure}) ordered domains": 1.0}
        return ms

    # ------------------------------------------------------------------ plotting geometry
    def boundaries(self) -> Dict[str, List[Tuple[float, float]]]:
        cs = np.linspace(0.0, 100.0, 80)
        out = {
            "Liquidus": list(zip(cs.tolist(), self.liquidus_T(cs).tolist())),
            "Solidus": list(zip(cs.tolist(), self.solidus_T(cs).tolist())),
        }
        for d in self.domes:
            cc = np.linspace(d.center - d.halfwidth * 0.999, d.center + d.halfwidth * 0.999, 60)
            tt = np.array([d.T_at(c) for c in cc])
            keep = tt >= T_ROOM
            out[f"{d.label} order/disorder"] = list(zip(cc[keep].tolist(), tt[keep].tolist()))
        return out

    def field_polygons(self) -> Dict[str, List[Tuple[float, float]]]:
        cs = np.linspace(0.0, 100.0, 80)
        liq = list(zip(cs.tolist(), self.liquidus_T(cs).tolist()))
        sol = list(zip(cs.tolist(), self.solidus_T(cs).tolist()))
        top = max(self.T_A, self.T_B) + 60.0
        polys = {
            "Liquid (L)": [(0.0, top)] + liq + [(100.0, top)],
            "L + α": liq + list(reversed(sol)),
            DISORDERED: [(0.0, T_ROOM)] + sol + [(100.0, T_ROOM)],
        }
        for d in self.domes:
            cc = np.linspace(d.center - d.halfwidth * 0.999, d.center + d.halfwidth * 0.999, 60)
            tt = np.array([d.T_at(c) for c in cc])
            keep = tt >= T_ROOM
            pts = list(zip(cc[keep].tolist(), tt[keep].tolist()))
            polys[f"{d.label} (ordered, {d.structure})"] = (
                [(pts[0][0], T_ROOM)] + pts + [(pts[-1][0], T_ROOM)])
        return polys


FIELD_COLORS: Dict[str, str] = {
    "Liquid (L)": "#FCE9D2",
    "L + α": "#F3E6B8",
    DISORDERED: "#E3F1EE",
}
DOME_COLORS = ["#F6D98A", "#EFC46B", "#F8E7A8"]

PRESETS = {
    "Cu-Au": dict(
        name="Cu-Au", component_A="Cu", component_B="Au", T_A=1085.0, T_B=1064.0,
        liquidus_points=[(0, 1085), (20, 1042), (40, 996), (60, 946), (80, 911), (90, 942), (95, 984), (100, 1064)],
        solidus_points=[(0, 1085), (20, 1019), (40, 972), (60, 934), (80, 911), (90, 928), (95, 974), (100, 1064)],
        domes=[
            OrderDome("Cu₃Au", "Cu3Au", "L1₂", 50.83, 390.0, 8.0),
            OrderDome("CuAu", "CuAu", "L1₀", 75.62, 410.0, 8.0),
            OrderDome("CuAu₃", "CuAu3", "L1₂", 90.30, 197.0, 5.0),
        ],
        data_note=("Liquidus/solidus: Callister & Rethwisch problem table (congruent minimum 911 °C at 80 wt% Au). "
                   "Ordering temperatures: Cu₃Au 390 °C, CuAu ≈410 °C, CuAu₃ ≈197 °C from the Cu-Au literature; "
                   "compositions (25/50/75 at% Au) converted to wt%. Dome widths are schematic."),
    ),
    "Cu-Pd": dict(
        name="Cu-Pd", component_A="Cu", component_B="Pd", T_A=1085.0, T_B=1555.0,
        liquidus_points=[(0, 1085), (25, 1195), (50, 1325), (75, 1440), (100, 1555)],
        solidus_points=[(0, 1085), (25, 1175), (50, 1300), (75, 1425), (100, 1555)],
        domes=[
            OrderDome("Cu₃Pd", "Cu3Pd", "L1₂", 35.8, 500.0, 6.0),
            OrderDome("CuPd", "CuPd", "B2", 62.6, 600.0, 7.0),
        ],
        data_note=("Melting points are exact; the liquidus/solidus between them is an ENGINEERING ESTIMATE "
                   "(narrow lens) because no tabulated points were found — check an ASM assessment for "
                   "research use. Ordering: CuPd (B2) above ~600 °C per Subramanian & Laughlin; Cu₃Pd (L1₂) "
                   "~500 °C, a region still being refined experimentally (Popov et al., 2018). "
                   "Compositions converted from 25/50 at% Pd; dome widths are schematic."),
    ),
}
