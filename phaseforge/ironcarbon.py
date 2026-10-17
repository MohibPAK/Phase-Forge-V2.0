"""
ironcarbon.py — Iron–iron-carbide (Fe–Fe₃C) metastable phase diagram engine.

Composition axis is wt% C from 0 to 6.67 (6.67 wt% C = cementite, Fe₃C).
Every boundary is a straight segment between the standard textbook
critical points (letters A, B, H, J, N, G, E, C, F, D, P, S, K, Q — the
conventional Fe–C labelling), so tie lines and the lever rule are exact and
need no root finding.

Three invariant reactions are modelled:

    peritectic  1495 °C   L (0.53) + δ (0.09)  →  γ (0.17)
    eutectic    1147 °C   L (4.30)             →  γ (2.14) + Fe₃C (6.67)   [ledeburite]
    eutectoid    727 °C   γ (0.76)             →  α (0.022) + Fe₃C (6.67)  [pearlite]

On top of the diagram geometry this module provides:

    * phase_at / tie_line / lever_rule / phase_fractions  — equilibrium analysis
    * transformation_path(C0)  — ordered list of every event met while cooling
      an alloy of composition C0 from the melt to room temperature
    * micro_state(C0, T)       — what the microstructure looks like at (C0, T);
      consumed by `phaseforge.microstructure` to draw a schematic micrograph
    * constituents / critical_temperatures / alloy_class — steel vs cast-iron
      bookkeeping (proeutectoid phase, pearlite, ledeburite, ...)

All values are textbook-level approximations for teaching (Callister / ASM
style straight-line boundaries), not CALPHAD-grade data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants — critical points (wt% C, °C)
# ---------------------------------------------------------------------------
CEM = "Fe₃C"
C_CEM = 6.67            # composition of cementite
T_ROOM = 25.0

T_A = 1538.0            # melting point of pure Fe
T_N = 1394.0            # δ ⇌ γ in pure Fe
T_G = 912.0             # γ ⇌ α in pure Fe

T_PER = 1495.0          # peritectic isotherm
C_H, C_J, C_B = 0.09, 0.17, 0.53      # δ, γ, L compositions at peritectic

T_EUT = 1147.0          # eutectic isotherm
C_E, C_C, C_F = 2.14, 4.30, C_CEM     # γ, L, Fe₃C compositions at eutectic
T_D = 1227.0            # (approx.) melting point of Fe₃C

T_EUD = 727.0           # eutectoid isotherm (A1)
C_P, C_S, C_K = 0.022, 0.76, C_CEM    # α, γ, Fe₃C compositions at eutectoid
C_Q = 0.005             # α solubility drawn at room temperature (illustrative)

T_CURIE_CEM = 210.0     # A0: magnetic transition of Fe₃C
T_CURIE_FE = 770.0      # A2: magnetic transition of α-Fe

TOL_C = 0.005           # composition tolerance for "eutectoid"/"eutectic"/"peritectic" alloys
_EPS_T = 1e-6

POINTS: Dict[str, Tuple[float, float]] = {
    "A": (0.0, T_A),
    "B": (C_B, T_PER),
    "H": (C_H, T_PER),
    "J": (C_J, T_PER),
    "N": (0.0, T_N),
    "G": (0.0, T_G),
    "E": (C_E, T_EUT),
    "C": (C_C, T_EUT),
    "F": (C_F, T_EUT),
    "D": (C_CEM, T_D),
    "P": (C_P, T_EUD),
    "S": (C_S, T_EUD),
    "K": (C_K, T_EUD),
    "Q": (C_Q, T_ROOM),
}

POINT_NOTES: Dict[str, str] = {
    "A": "Melting point of pure iron",
    "B": "Liquid composition at the peritectic",
    "H": "Max. carbon in δ-ferrite (peritectic)",
    "J": "Peritectic point — γ formed",
    "N": "δ ⇌ γ transformation of pure iron (A₄)",
    "G": "γ ⇌ α transformation of pure iron (A₃)",
    "E": "Max. carbon in austenite (2.14 wt%)",
    "C": "Eutectic point (ledeburite)",
    "F": "Cementite at the eutectic temperature",
    "D": "Melting point of cementite (approx.)",
    "P": "Max. carbon in α-ferrite (0.022 wt%)",
    "S": "Eutectoid point (pearlite)",
    "K": "Cementite at the eutectoid temperature",
    "Q": "Carbon in α at room temperature (drawn value)",
}


# ---------------------------------------------------------------------------
# Enums / dataclasses
# ---------------------------------------------------------------------------
class FeCRegion(str, Enum):
    LIQUID = "Liquid (L)"
    DELTA = "δ-ferrite"
    GAMMA = "γ-austenite"
    ALPHA = "α-ferrite"
    L_DELTA = "L + δ"
    DELTA_GAMMA = "δ + γ"
    L_GAMMA = "L + γ"
    L_CEM = "L + Fe₃C"
    ALPHA_GAMMA = "α + γ"
    GAMMA_CEM = "γ + Fe₃C"
    ALPHA_CEM = "α + Fe₃C"
    PERITECTIC = "L + δ + γ  (peritectic isotherm)"
    EUTECTIC = "L + γ + Fe₃C  (eutectic isotherm)"
    EUTECTOID = "α + γ + Fe₃C  (eutectoid isotherm)"


_SINGLE_PHASE = {
    FeCRegion.LIQUID: "L",
    FeCRegion.DELTA: "δ",
    FeCRegion.GAMMA: "γ",
    FeCRegion.ALPHA: "α",
}


class AlloyClass(str, Enum):
    FERRITIC = "Ferritic iron (< 0.022 wt% C)"
    HYPOEUTECTOID = "Hypoeutectoid steel"
    EUTECTOID = "Eutectoid steel"
    HYPEREUTECTOID = "Hypereutectoid steel"
    HYPOEUTECTIC_CI = "Hypoeutectic white cast iron"
    EUTECTIC_CI = "Eutectic white cast iron"
    HYPEREUTECTIC_CI = "Hypereutectic white cast iron"


class FeCError(ValueError):
    """Raised for out-of-range compositions or single-phase tie-line requests."""


def fmt_c(c: float) -> str:
    return f"{c:.3f}" if c < 0.1 else f"{c:.2f}"


@dataclass
class FeCTieLine:
    temperature: float
    region: FeCRegion
    points: List[Tuple[str, float]]        # [(phase, wt% C)] ordered left → right

    @property
    def c_left(self) -> float:
        return self.points[0][1]

    @property
    def c_right(self) -> float:
        return self.points[-1][1]

    @property
    def is_invariant(self) -> bool:
        return len(self.points) == 3

    @property
    def label_left(self) -> str:
        return f"{self.points[0][0]} ({fmt_c(self.points[0][1])}% C)"

    @property
    def label_right(self) -> str:
        return f"{self.points[-1][0]} ({fmt_c(self.points[-1][1])}% C)"

    @property
    def labels(self) -> List[str]:
        return [f"{p} ({fmt_c(c)}% C)" for p, c in self.points]


@dataclass
class PathEvent:
    temperature: float
    kind: str                 # liquidus, solidus, peritectic, eutectic, eutectoid, A3, Acm, ...
    title: str
    reaction: str
    phases_after: str
    microstructure: str
    t_sample: float = 0.0     # temperature used to draw the "after this event" micrograph
    t_label: Optional[str] = None

    @property
    def temp_text(self) -> str:
        return self.t_label if self.t_label else f"{self.temperature:.0f}"


@dataclass
class MicroState:
    C0: float
    T: float
    region: str
    recipe: str               # liquid | dendrites | plates_in_liquid | grains | rim |
                              # pearlite_rim | pearlite | ledeburite
    phase: str = ""           # main / parent phase
    rim_phase: str = ""       # secondary phase (rim / dendrite / precipitate)
    f: float = 0.0            # area fraction of the secondary phase
    extras: Dict[str, object] = field(default_factory=dict)
    fractions: Dict[str, float] = field(default_factory=dict)        # phase wt fractions
    constituents: Dict[str, float] = field(default_factory=dict)     # what is drawn


# ---------------------------------------------------------------------------
# The diagram
# ---------------------------------------------------------------------------
def _seg(x, x0, x1, y0, y1):
    """Linear interpolation of y between (x0,y0) and (x1,y1), clipped to the segment."""
    x = np.asarray(x, dtype=float)
    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
    t = (np.clip(x, lo, hi) - x0) / (x1 - x0)
    return y0 + (y1 - y0) * t


class IronCarbonDiagram:
    """Equilibrium Fe–Fe₃C diagram with analysis and microstructure helpers."""

    name = "Fe–Fe₃C"
    component_A = "Fe"
    component_B = "C"

    # ------------------------------------------------------------------ boundaries (C as f(T))
    @staticmethod
    def c_liquidus_AB(T):   return _seg(T, T_PER, T_A, C_B, 0.0)
    @staticmethod
    def c_liquidus_BC(T):   return _seg(T, T_EUT, T_PER, C_C, C_B)
    @staticmethod
    def c_liquidus_CD(T):   return _seg(T, T_EUT, T_D, C_C, C_CEM)
    @staticmethod
    def c_delta_solidus(T): return _seg(T, T_PER, T_A, C_H, 0.0)
    @staticmethod
    def c_delta_solvus(T):  return _seg(T, T_N, T_PER, 0.0, C_H)
    @staticmethod
    def c_gamma_delta(T):   return _seg(T, T_N, T_PER, 0.0, C_J)
    @staticmethod
    def c_gamma_solidus(T): return _seg(T, T_EUT, T_PER, C_E, C_J)
    @staticmethod
    def c_A3(T):            return _seg(T, T_EUD, T_G, C_S, 0.0)
    @staticmethod
    def c_alpha_gamma(T):   return _seg(T, T_EUD, T_G, C_P, 0.0)
    @staticmethod
    def c_Acm(T):           return _seg(T, T_EUD, T_EUT, C_S, C_E)
    @staticmethod
    def c_alpha_solvus(T):  return _seg(T, T_ROOM, T_EUD, C_Q, C_P)

    @staticmethod
    def liquidus_T(C):
        return np.interp(C, [0.0, C_B, C_C, C_CEM], [T_A, T_PER, T_EUT, T_D])

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _check_C(C: float, open_interval: bool = False):
        lo_ok = C > 0 if open_interval else C >= 0
        if not (lo_ok and C <= C_CEM + 1e-9):
            raise FeCError(f"Carbon content must lie between 0 and {C_CEM} wt% C.")

    @staticmethod
    def _at(T: float, T_inv: float) -> bool:
        return abs(T - T_inv) < _EPS_T

    # ------------------------------------------------------------------ phase region lookup
    def phase_at(self, C: float, T: float) -> FeCRegion:
        R = FeCRegion
        self._check_C(C)

        if self._at(T, T_PER) and C_H < C < C_B:
            return R.PERITECTIC
        if self._at(T, T_EUT) and C_E < C < C_CEM:
            return R.EUTECTIC
        if self._at(T, T_EUD) and C_P < C < C_CEM:
            return R.EUTECTOID

        # --- above the peritectic isotherm ---------------------------------
        if T >= T_PER:
            if T >= float(self.liquidus_T(C)):
                return R.LIQUID
            return R.DELTA if C <= float(self.c_delta_solidus(T)) else R.L_DELTA

        # --- between the eutectic and peritectic isotherms ------------------
        if T >= T_EUT:
            if C > C_C:
                return R.LIQUID if T >= float(self.liquidus_T(C)) else R.L_CEM
            if T >= float(self.liquidus_T(C)):
                return R.LIQUID
            if C >= float(self.c_gamma_solidus(T)):
                return R.L_GAMMA
            if T > T_N and C <= float(self.c_delta_solvus(T)):
                return R.DELTA
            if T > T_N and C <= float(self.c_gamma_delta(T)):
                return R.DELTA_GAMMA
            return R.GAMMA

        # --- between the eutectoid and eutectic isotherms --------------------
        if T > T_EUD:
            c_cm = float(self.c_Acm(T))
            if T > T_G:
                return R.GAMMA if C <= c_cm else R.GAMMA_CEM
            if C <= float(self.c_alpha_gamma(T)):
                return R.ALPHA
            if C <= float(self.c_A3(T)):
                return R.ALPHA_GAMMA
            return R.GAMMA if C <= c_cm else R.GAMMA_CEM

        # --- below the eutectoid isotherm -------------------------------------
        return R.ALPHA if C <= float(self.c_alpha_solvus(T)) else R.ALPHA_CEM

    # ------------------------------------------------------------------ tie lines / lever rule
    def tie_line(self, C: float, T: float) -> FeCTieLine:
        R = FeCRegion
        region = self.phase_at(C, T)
        if region in _SINGLE_PHASE:
            raise FeCError(f"Point is single-phase {region.value} — no tie line.")

        f = lambda fn: float(fn(T))
        if region == R.PERITECTIC:
            pts = [("δ", C_H), ("γ", C_J), ("L", C_B)]
        elif region == R.EUTECTIC:
            pts = [("γ", C_E), ("L", C_C), (CEM, C_CEM)]
        elif region == R.EUTECTOID:
            pts = [("α", C_P), ("γ", C_S), (CEM, C_CEM)]
        elif region == R.L_DELTA:
            pts = [("δ", f(self.c_delta_solidus)), ("L", f(self.c_liquidus_AB))]
        elif region == R.DELTA_GAMMA:
            pts = [("δ", f(self.c_delta_solvus)), ("γ", f(self.c_gamma_delta))]
        elif region == R.L_GAMMA:
            pts = [("γ", f(self.c_gamma_solidus)), ("L", f(self.c_liquidus_BC))]
        elif region == R.L_CEM:
            pts = [("L", f(self.c_liquidus_CD)), (CEM, C_CEM)]
        elif region == R.ALPHA_GAMMA:
            pts = [("α", f(self.c_alpha_gamma)), ("γ", f(self.c_A3))]
        elif region == R.GAMMA_CEM:
            pts = [("γ", f(self.c_Acm)), (CEM, C_CEM)]
        else:  # ALPHA_CEM
            pts = [("α", f(self.c_alpha_solvus)), (CEM, C_CEM)]
        return FeCTieLine(temperature=T, region=region, points=pts)

    @staticmethod
    def lever_rule(C0: float, c_left: float, c_right: float) -> Tuple[float, float]:
        """(fraction of left phase, fraction of right phase) for overall composition C0."""
        if not (c_left - 1e-9 <= C0 <= c_right + 1e-9):
            raise FeCError(
                f"C₀ = {C0:.3f} wt% lies outside the tie-line span [{c_left:.3f}, {c_right:.3f}]."
            )
        if c_right == c_left:
            return 0.5, 0.5
        f_right = (C0 - c_left) / (c_right - c_left)
        return 1.0 - f_right, f_right

    def _shift(self, T: float, side: str) -> float:
        for T_inv in (T_PER, T_EUT, T_EUD):
            if self._at(T, T_inv):
                return T_inv + (1e-3 if side == "above" else -1e-3)
        return T

    def phase_fractions(self, C0: float, T: float, side: str = "above") -> Dict[str, float]:
        """Weight fraction of every phase present at (C0, T).

        Exactly on an invariant isotherm the amounts depend on how far the reaction
        has run, so `side` picks the state 'above' (before) or 'below' (after) it.
        """
        T_eff = self._shift(T, side)
        region = self.phase_at(C0, T_eff)
        if region in _SINGLE_PHASE:
            return {_SINGLE_PHASE[region]: 1.0}
        tl = self.tie_line(C0, T_eff)
        f_l, f_r = self.lever_rule(C0, tl.c_left, tl.c_right)
        return {tl.points[0][0]: f_l, tl.points[-1][0]: f_r}

    # ------------------------------------------------------------------ classification
    @staticmethod
    def alloy_class(C0: float) -> AlloyClass:
        if C0 < C_P:
            return AlloyClass.FERRITIC
        if abs(C0 - C_S) <= TOL_C:
            return AlloyClass.EUTECTOID
        if C0 < C_S:
            return AlloyClass.HYPOEUTECTOID
        if C0 <= C_E:
            return AlloyClass.HYPEREUTECTOID
        if abs(C0 - C_C) <= TOL_C:
            return AlloyClass.EUTECTIC_CI
        if C0 < C_C:
            return AlloyClass.HYPOEUTECTIC_CI
        return AlloyClass.HYPEREUTECTIC_CI

    @staticmethod
    def common_name(C0: float) -> str:
        if C0 < C_P:
            return "ingot / electrical iron"
        if C0 <= 0.30:
            return "low-carbon (mild) steel"
        if C0 <= 0.60:
            return "medium-carbon steel"
        if C0 <= C_E:
            return "high-carbon steel"
        return "white cast iron"

    # ------------------------------------------------------------------ critical temperatures
    def solidus_T(self, C0: float) -> float:
        if C0 > C_E:
            return T_EUT
        if C0 < C_H:
            return T_A - (T_A - T_PER) * C0 / C_H
        if C0 <= C_J + TOL_C:
            return T_PER
        return T_PER - (C0 - C_J) * (T_PER - T_EUT) / (C_E - C_J)

    def critical_temperatures(self, C0: float) -> Dict[str, float]:
        """Liquidus, solidus, A₃ / A_cm and A₁ for an alloy of composition C0 (°C)."""
        self._check_C(C0, open_interval=True)
        out: Dict[str, float] = {
            "Liquidus": float(self.liquidus_T(C0)),
            "Solidus": self.solidus_T(C0),
        }
        if C0 < C_S - TOL_C:
            out["A₃ (γ → γ + α)"] = T_G - (T_G - T_EUD) * C0 / C_S
        elif C0 > C_S + TOL_C and C0 <= C_E:
            out["A_cm (γ → γ + Fe₃C)"] = T_EUD + (T_EUT - T_EUD) * (C0 - C_S) / (C_E - C_S)
        if C0 > C_P:
            out["A₁ (eutectoid)"] = T_EUD
        return out

    # ------------------------------------------------------------------ transformation path
    def transformation_path(self, C0: float) -> List[PathEvent]:
        """Every event met by an alloy of composition C0 on equilibrium cooling,
        from just above the liquidus down to room temperature."""
        self._check_C(C0, open_interval=True)
        if C0 >= C_CEM - 0.02:
            raise FeCError("Use a composition below 6.65 wt% C (6.67 wt% C is pure Fe₃C).")

        ev: List[PathEvent] = []
        starts = set()          # kinds whose "after" picture is best drawn part-way to the next event

        def add(T, kind, title, reaction, phases, micro, start=False, t_label=None):
            ev.append(PathEvent(T, kind, title, reaction, phases, micro, t_label=t_label))
            if start:
                starts.add(len(ev) - 1)

        T_liq = float(self.liquidus_T(C0))

        # ============================ solidification ==========================
        if C0 < C_H:
            add(T_liq, "liquidus", "Liquidus — solidification begins", "L → L + δ", "L + δ",
                "δ-ferrite (BCC) dendrites nucleate and grow in the melt.", start=True)
            add(T_A - (T_A - T_PER) * C0 / C_H, "solidus", "Solidification complete",
                "L + δ → δ", "δ",
                "Fully solid δ-ferrite: equiaxed polygonal grains.")
            add(T_N + (T_PER - T_N) * C0 / C_H, "delta_gamma", "δ → γ transformation begins",
                "δ → δ + γ", "δ + γ",
                "Austenite (FCC) nucleates on the δ grain boundaries and grows inward.", start=True)
            add(T_N + (T_PER - T_N) * C0 / C_J, "delta_gamma_end", "δ → γ transformation complete",
                "δ + γ → γ", "γ",
                "Single-phase austenite: equiaxed grains — this sets the prior-austenite grain size.")
        elif C0 < C_J - TOL_C:
            add(T_liq, "liquidus", "Liquidus — solidification begins", "L → L + δ", "L + δ",
                "δ-ferrite dendrites grow in the melt.", start=True)
            add(T_PER, "peritectic", "Peritectic reaction (1495 °C)", "L (0.53) + δ (0.09) → γ (0.17)",
                "δ + γ  (all liquid used up)",
                "Austenite forms as a shell around the δ dendrites and consumes the liquid; "
                "δ is left over because C₀ is on the δ side of the peritectic point.")
            add(T_N + (T_PER - T_N) * C0 / C_J, "delta_gamma_end", "Remaining δ transforms to γ",
                "δ + γ → γ", "γ",
                "Single-phase austenite grains.")
        elif abs(C0 - C_J) <= TOL_C:
            add(T_liq, "liquidus", "Liquidus — solidification begins", "L → L + δ", "L + δ",
                "δ-ferrite dendrites grow in the melt.", start=True)
            add(T_PER, "peritectic", "Peritectic reaction (1495 °C)", "L (0.53) + δ (0.09) → γ (0.17)",
                "γ  (L and δ consumed exactly)",
                "At the peritectic composition the reaction uses up both liquid and δ — "
                "the alloy becomes fully austenitic in one step.")
        elif C0 < C_B:
            add(T_liq, "liquidus", "Liquidus — solidification begins", "L → L + δ", "L + δ",
                "δ-ferrite dendrites grow in the melt.", start=True)
            add(T_PER, "peritectic", "Peritectic reaction (1495 °C)", "L (0.53) + δ (0.09) → γ (0.17)",
                "L + γ  (all δ used up)",
                "Austenite shells replace the δ dendrites; liquid is left over because C₀ is on "
                "the liquid side of the peritectic point.")
            add(self.solidus_T(C0), "solidus", "Solidification complete", "L + γ → γ", "γ",
                "The last liquid freezes as austenite; polygonal austenite grains.")
        elif C0 <= C_E:
            add(T_liq, "liquidus", "Liquidus — solidification begins", "L → L + γ", "L + γ",
                "Austenite (γ) dendrites grow directly from the melt (no δ, no peritectic).", start=True)
            add(self.solidus_T(C0), "solidus", "Solidification complete", "L + γ → γ", "γ",
                "The last liquid freezes as austenite; polygonal austenite grains.")
        elif C0 < C_C - TOL_C:
            add(T_liq, "liquidus", "Liquidus — solidification begins", "L → L + γ", "L + γ",
                "Primary austenite dendrites grow in the melt, which is enriched in carbon as it freezes.",
                start=True)
            add(T_EUT, "eutectic", "Eutectic reaction (1147 °C)", "L (4.30) → γ (2.14) + Fe₃C (6.67)",
                "γ + Fe₃C  (γ dendrites + ledeburite)",
                "The remaining liquid (4.30 wt% C) freezes as ledeburite — Fe₃C matrix with "
                "austenite rods/particles — around the primary γ dendrites.")
        elif abs(C0 - C_C) <= TOL_C:
            add(T_EUT, "eutectic", "Liquidus = eutectic (1147 °C)", "L (4.30) → γ (2.14) + Fe₃C (6.67)",
                "γ + Fe₃C  (100 % ledeburite)",
                "The whole melt freezes at one temperature as ledeburite: Fe₃C matrix with austenite particles.")
        else:
            add(T_liq, "liquidus", "Liquidus — solidification begins", "L → L + Fe₃C", "L + Fe₃C",
                "Primary cementite grows as long white plates in the melt.", start=True)
            add(T_EUT, "eutectic", "Eutectic reaction (1147 °C)", "L (4.30) → γ (2.14) + Fe₃C (6.67)",
                "Fe₃C (primary) + γ + Fe₃C (ledeburite)",
                "The melt has been depleted to 4.30 wt% C and freezes as ledeburite around the primary Fe₃C plates.")

        # ============================ solid-state ===============================
        if C0 <= C_E:
            if C0 < C_P:
                T3 = T_G - (T_G - T_EUD) * C0 / C_S
                Ta = T_G - (T_G - T_EUD) * C0 / C_P
                add(T3, "A3", "A₃ — ferrite starts to form", "γ → γ + α", "γ + α",
                    "Ferrite nucleates on the austenite grain boundaries.", start=True)
                add(Ta, "alpha_complete", "Transformation to ferrite complete", "γ + α → α", "α",
                    "Fully ferritic: equiaxed α grains (very soft, ductile).")
                if C0 > C_Q:
                    Ts = T_ROOM + (T_EUD - T_ROOM) * (C0 - C_Q) / (C_P - C_Q)
                    add(Ts, "solvus", "Solvus crossed — tertiary cementite", "α → α + Fe₃C", "α + Fe₃C",
                        "A trace of Fe₃C precipitates at the ferrite grain boundaries "
                        "(far too little to see at normal magnification).", start=True)
            elif C0 < C_S - TOL_C:
                T3 = T_G - (T_G - T_EUD) * C0 / C_S
                add(T3, "A3", "A₃ — proeutectoid ferrite starts to form", "γ → γ + α", "γ + α",
                    "Ferrite nucleates on the austenite grain boundaries and thickens into a network.",
                    start=True)
                add(T_EUD, "eutectoid", "Eutectoid reaction (727 °C)", "γ (0.76) → α (0.022) + Fe₃C (6.67)",
                    "α + Fe₃C  (proeutectoid α + pearlite)",
                    "The austenite left in the grain interiors (now 0.76 wt% C) turns into pearlite — "
                    "alternating lamellae of ferrite and cementite.")
            elif abs(C0 - C_S) <= TOL_C:
                add(T_EUD, "eutectoid", "Eutectoid reaction (727 °C)", "γ (0.76) → α (0.022) + Fe₃C (6.67)",
                    "α + Fe₃C  (100 % pearlite)",
                    "All the austenite transforms to pearlite: colonies of alternating ferrite / cementite lamellae.")
            else:
                Tcm = T_EUD + (T_EUT - T_EUD) * (C0 - C_S) / (C_E - C_S)
                add(Tcm, "Acm", "A_cm — proeutectoid cementite starts to form", "γ → γ + Fe₃C", "γ + Fe₃C",
                    "Cementite precipitates on the austenite grain boundaries as a continuous network.",
                    start=True)
                add(T_EUD, "eutectoid", "Eutectoid reaction (727 °C)", "γ (0.76) → α (0.022) + Fe₃C (6.67)",
                    "α + Fe₃C  (proeutectoid Fe₃C + pearlite)",
                    "The austenite (now 0.76 wt% C) turns into pearlite inside the cementite network.")
        else:
            add(T_EUT - 0.5, "Acm_ci", "Secondary cementite (1147 → 727 °C)", "γ → γ + Fe₃C",
                "γ + Fe₃C",
                "As the alloy cools, austenite sheds carbon along the A_cm line (2.14 → 0.76 wt% C) and "
                "secondary Fe₃C deposits on the existing cementite and around the γ dendrites.",
                start=True, t_label="1147 → 727")
            add(T_EUD, "eutectoid", "Eutectoid reaction (727 °C)", "γ (0.76) → α (0.022) + Fe₃C (6.67)",
                "α + Fe₃C  (pearlite + Fe₃C + transformed ledeburite)"
                if C0 < C_C - TOL_C else "α + Fe₃C",
                "All austenite (dendrites and the γ inside ledeburite) becomes pearlite. Ledeburite is now "
                "'transformed ledeburite': cementite matrix with pearlite islands.")

        add(T_ROOM, "room", "Room temperature (equilibrium)", "—", "α + Fe₃C",
            self._final_text(C0))

        # ---- sort (descending T) and assign sample temperatures ---------------
        order = sorted(range(len(ev)), key=lambda i: (-ev[i].temperature, i))
        ev = [ev[i] for i in order]
        starts = {order.index(i) for i in starts}
        for i, e in enumerate(ev):
            if e.kind == "Acm_ci":
                e.t_sample = 0.5 * (T_EUT + T_EUD)
            elif e.kind == "room":
                e.t_sample = T_ROOM
            elif i in starts:
                T_next = ev[i + 1].temperature if i + 1 < len(ev) else T_ROOM
                e.t_sample = e.temperature - 0.4 * (e.temperature - T_next)
            else:
                e.t_sample = e.temperature - 0.2
        return ev

    def evolution_stages(self, C0: float) -> List[Tuple[str, float, str]]:
        """[(title, sample temperature, phases)] — the molten alloy plus one frame per event,
        in cooling order. Used for the micrograph 'filmstrip'."""
        events = self.transformation_path(C0)
        T_liq = float(self.liquidus_T(C0))
        stages = [("Molten alloy", min(T_liq + 40.0, 1600.0), "L")]
        stages += [(e.title, e.t_sample, e.phases_after) for e in events]
        return stages

    def _final_text(self, C0: float) -> str:
        cls = self.alloy_class(C0)
        return {
            AlloyClass.FERRITIC: "Ferrite grains (with a trace of tertiary Fe₃C at boundaries).",
            AlloyClass.HYPOEUTECTOID: "Proeutectoid ferrite (white) around pearlite colonies (dark, lamellar).",
            AlloyClass.EUTECTOID: "100 % pearlite — ferrite / cementite lamellae in colonies.",
            AlloyClass.HYPEREUTECTOID: "Proeutectoid cementite network (white) around pearlite colonies.",
            AlloyClass.HYPOEUTECTIC_CI: "Pearlite dendrites (with secondary Fe₃C rims) in a matrix of transformed ledeburite.",
            AlloyClass.EUTECTIC_CI: "100 % transformed ledeburite: Fe₃C matrix with pearlite islands.",
            AlloyClass.HYPEREUTECTIC_CI: "Primary cementite plates in a matrix of transformed ledeburite.",
        }[cls]

    # ------------------------------------------------------------------ microstructure state
    def micro_state(self, C0: float, T: float) -> MicroState:
        """Describe the equilibrium microstructure at (C0, T) for the micrograph renderer."""
        R = FeCRegion
        self._check_C(C0, open_interval=True)
        T_eff = self._shift(T, "above")
        region = self.phase_at(C0, T_eff)
        fr = self.phase_fractions(C0, T_eff)
        ms = MicroState(C0=C0, T=T, region=region.value, recipe="grains", fractions=fr)

        if region == R.LIQUID:
            ms.recipe = "liquid"
            ms.constituents = {"Liquid": 1.0}
        elif region == R.L_DELTA:
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "dendrites", "L", "δ", fr["δ"]
            ms.constituents = {"Liquid": fr["L"], "δ-ferrite dendrites": fr["δ"]}
        elif region == R.L_GAMMA:
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "dendrites", "L", "γ", fr["γ"]
            ms.constituents = {"Liquid": fr["L"], "Austenite dendrites": fr["γ"]}
        elif region == R.L_CEM:
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "plates_in_liquid", "L", CEM, fr[CEM]
            ms.constituents = {"Liquid": fr["L"], "Primary Fe₃C plates": fr[CEM]}
        elif region in (R.DELTA, R.GAMMA):
            ph = _SINGLE_PHASE[region]
            ms.recipe, ms.phase = "grains", ph
            ms.constituents = {("δ-ferrite grains" if ph == "δ" else "Austenite grains"): 1.0}
        elif region == R.DELTA_GAMMA:
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "rim", "δ", "γ", fr["γ"]
            ms.constituents = {"δ-ferrite": fr["δ"], "Austenite (new)": fr["γ"]}
        elif region == R.ALPHA_GAMMA:
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "rim", "γ", "α", fr["α"]
            ms.constituents = {"Austenite": fr["γ"], "Proeutectoid ferrite": fr["α"]}
        elif region == R.ALPHA:
            ms.recipe, ms.phase = "grains", "α"
            ms.constituents = {"Ferrite grains": 1.0}
        elif region == R.GAMMA_CEM:
            if C0 <= C_E:
                ms.recipe, ms.phase, ms.rim_phase, ms.f = "rim", "γ", CEM, fr[CEM]
                ms.constituents = {"Austenite": fr["γ"], "Proeutectoid Fe₃C": fr[CEM]}
            else:
                self._fill_ledeburite(ms, C0, T_eff, transformed=False)
        elif region == R.ALPHA_CEM:
            if C0 < C_P:
                ms.recipe, ms.phase, ms.rim_phase, ms.f = "grains", "α", CEM, fr[CEM]
                ms.constituents = {"Ferrite": fr["α"], "Tertiary Fe₃C": fr[CEM]}
            elif C0 < C_S - TOL_C:
                f_a = (C_S - C0) / (C_S - C_P)
                ms.recipe, ms.rim_phase, ms.f = "pearlite_rim", "α", f_a
                ms.constituents = {"Proeutectoid ferrite": f_a, "Pearlite": 1.0 - f_a}
            elif abs(C0 - C_S) <= TOL_C:
                ms.recipe = "pearlite"
                ms.constituents = {"Pearlite": 1.0}
            elif C0 <= C_E:
                f_c = (C0 - C_S) / (C_CEM - C_S)
                ms.recipe, ms.rim_phase, ms.f = "pearlite_rim", CEM, f_c
                ms.constituents = {"Proeutectoid Fe₃C (network)": f_c, "Pearlite": 1.0 - f_c}
            else:
                self._fill_ledeburite(ms, C0, T_eff, transformed=True)
        return ms

    def _fill_ledeburite(self, ms: MicroState, C0: float, T: float, transformed: bool):
        ms.recipe = "ledeburite"
        led_particle = "pearlite" if transformed else "γ"
        f_gamma_in_led = (C_CEM - C_C) / (C_CEM - C_E)            # ≈ 0.523 (wt fraction of γ in ledeburite)
        if C0 < C_C - TOL_C:
            f_prim = (C_C - C0) / (C_C - C_E)
            c_g = C_S if transformed else float(self.c_Acm(T))
            f_sec = (C_E - c_g) / (C_CEM - c_g)                    # Fe₃C shed by the primary γ
            ms.extras.update(primary="dendrites", f_primary=f_prim, f_sec=f_sec)
            if transformed:
                ms.constituents = {
                    "Pearlite (from primary γ)": f_prim * (1 - f_sec),
                    "Secondary Fe₃C": f_prim * f_sec,
                    "Transformed ledeburite": 1.0 - f_prim,
                }
            else:
                ms.constituents = {
                    "Primary austenite dendrites": f_prim * (1 - f_sec),
                    "Secondary Fe₃C": f_prim * f_sec,
                    "Ledeburite (γ + Fe₃C)": 1.0 - f_prim,
                }
        elif abs(C0 - C_C) <= TOL_C:
            ms.extras.update(primary="none", f_primary=0.0, f_sec=0.0)
            ms.constituents = {("Transformed ledeburite" if transformed else "Ledeburite (γ + Fe₃C)"): 1.0}
        else:
            f_pl = (C0 - C_C) / (C_CEM - C_C)
            ms.extras.update(primary="plates", f_primary=f_pl, f_sec=0.0)
            ms.constituents = {
                "Primary Fe₃C plates": f_pl,
                ("Transformed ledeburite" if transformed else "Ledeburite (γ + Fe₃C)"): 1.0 - f_pl,
            }
        ms.extras.update(particle=led_particle, f_particle=f_gamma_in_led)

    # ------------------------------------------------------------------ plotting geometry
    def boundaries(self) -> Dict[str, List[Tuple[float, float]]]:
        """Named polylines [(C, T), ...] for every line of the diagram."""
        P = POINTS
        return {
            "Liquidus": [P["A"], P["B"], P["C"], P["D"]],
            "δ solidus": [P["A"], P["H"]],
            "δ solvus": [P["H"], P["N"]],
            "γ / (δ+γ)": [P["J"], P["N"]],
            "γ solidus": [P["J"], P["E"]],
            "A₃": [P["G"], P["S"]],
            "α / (α+γ)": [P["G"], P["P"]],
            "A_cm": [P["E"], P["S"]],
            "α solvus": [P["P"], P["Q"]],
            "Fe₃C line": [(C_CEM, T_ROOM), (C_CEM, T_D)],
        }

    def invariant_lines(self) -> Dict[str, Tuple[float, float, float]]:
        """name -> (C_left, C_right, T)."""
        return {
            "Peritectic 1495 °C": (C_H, C_B, T_PER),
            "Eutectic 1147 °C": (C_E, C_CEM, T_EUT),
            "Eutectoid 727 °C (A₁)": (C_P, C_CEM, T_EUD),
        }

    def magnetic_lines(self) -> Dict[str, Tuple[float, float, float]]:
        return {
            "A₂ (α Curie, 770 °C)": (0.0, float(self.c_alpha_gamma(T_CURIE_FE)), T_CURIE_FE),
            "A₀ (Fe₃C Curie, 210 °C)": (float(self.c_alpha_solvus(T_CURIE_CEM)), C_CEM, T_CURIE_CEM),
        }

    def field_polygons(self) -> Dict[FeCRegion, List[Tuple[float, float]]]:
        """Closed polygons (C, T) for every single- and two-phase field."""
        P, R = POINTS, FeCRegion
        top = 1600.0
        return {
            R.LIQUID: [P["A"], P["B"], P["C"], P["D"], (C_CEM, top), (0.0, top)],
            R.L_DELTA: [P["A"], P["B"], P["H"]],
            R.DELTA: [P["A"], P["H"], P["N"]],
            R.DELTA_GAMMA: [P["H"], P["J"], P["N"]],
            R.GAMMA: [P["N"], P["J"], P["E"], P["S"], P["G"]],
            R.L_GAMMA: [P["J"], P["B"], P["C"], P["E"]],
            R.L_CEM: [P["C"], P["D"], P["F"]],
            R.GAMMA_CEM: [P["E"], P["F"], P["K"], P["S"]],
            R.ALPHA_GAMMA: [P["G"], P["S"], P["P"]],
            R.ALPHA: [P["G"], P["P"], P["Q"], (0.0, T_ROOM)],
            R.ALPHA_CEM: [P["P"], P["K"], (C_CEM, T_ROOM), P["Q"]],
        }


# Shared colours for the diagram fields (light tints; used by the UI and the report)
FIELD_COLORS: Dict[FeCRegion, str] = {
    FeCRegion.LIQUID: "#FCE9D2",
    FeCRegion.L_DELTA: "#F6D6B0",
    FeCRegion.DELTA: "#E4DDF2",
    FeCRegion.DELTA_GAMMA: "#D6CCEA",
    FeCRegion.GAMMA: "#F3E6B8",
    FeCRegion.L_GAMMA: "#F8D9BD",
    FeCRegion.L_CEM: "#F4C9B8",
    FeCRegion.GAMMA_CEM: "#E9D3D3",
    FeCRegion.ALPHA_GAMMA: "#D6E8D0",
    FeCRegion.ALPHA: "#E3F1EE",
    FeCRegion.ALPHA_CEM: "#D3E1F1",
}
