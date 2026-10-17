"""
core.py — Binary eutectic phase-diagram model.

Implements a parametric binary eutectic system (component A, component B,
composition axis = wt% B) using closed-form power-law liquidus/solidus
curves, plus linear solvus lines below the eutectic temperature. This is
the standard textbook topology (e.g. Pb-Sn, Cu-Ag, Al-Si): two terminal
solid-solution phases (alpha, beta), a eutectic invariant reaction, and
the six classic regions:

    Liquid | alpha+Liquid | beta+Liquid | alpha | beta | alpha+beta

All curves have closed-form forward (T -> C) *and* inverse (C -> T)
expressions, which keeps tie-line construction (fixed T -> two boundary
compositions) exact and fast — no root-finding needed for the diagram
geometry itself. SciPy is used downstream in `simulation.py` for the
cooling-curve ODE integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class PhaseRegion(str, Enum):
    LIQUID = "Liquid"
    ALPHA = "α (solid solution)"
    BETA = "β (solid solution)"
    ALPHA_LIQUID = "α + Liquid"
    BETA_LIQUID = "β + Liquid"
    ALPHA_BETA = "α + β (eutectic mixture)"
    EUTECTIC_INVARIANT = "α + L + β (eutectic invariant)"


@dataclass
class TieLine:
    temperature: float
    region: PhaseRegion
    c_left: float          # left-hand boundary composition (wt% B)
    c_right: float         # right-hand boundary composition (wt% B)
    label_left: str
    label_right: str


class PhaseDiagramError(ValueError):
    """Raised for physically inconsistent / out-of-range input parameters."""


class EutecticPhaseDiagram:
    """A parametric binary eutectic phase diagram.

    Parameters
    ----------
    T_A, T_B : melting points of pure A and pure B (°C)
    T_E      : eutectic temperature (°C)
    C_E      : eutectic composition (wt% B)
    C_alpha_E: max solubility of B in alpha at the eutectic temperature (wt% B)
    C_beta_E : composition of beta at the eutectic temperature (wt% B) — i.e.
               max solubility of A in beta, expressed on the wt%B axis
    C_alpha_RT, C_beta_RT : terminal solid solubility limits at room temperature,
               used to draw solvus lines (optional; if omitted, solvus is
               assumed flat, i.e. solubility does not change below T_E)
    T_room   : reference low temperature for the solvus lines (°C)
    curvature_liquidus, curvature_solidus : shape exponents (0 < p <= 1) controlling
               how concave the liquidus/solidus curves are. p=1 gives straight
               lines; smaller p bows the curve toward the eutectic point, which
               is qualitatively how real liquidus/solidus curves look.
    component_A, component_B : element/compound names, for labeling
    name : display name for the system (e.g. "Pb-Sn")
    """

    def __init__(
        self,
        T_A: float,
        T_B: float,
        T_E: float,
        C_E: float,
        C_alpha_E: float,
        C_beta_E: float,
        C_alpha_RT: Optional[float] = None,
        C_beta_RT: Optional[float] = None,
        T_room: float = 25.0,
        curvature_liquidus: float = 0.62,
        curvature_solidus: float = 0.85,
        component_A: str = "A",
        component_B: str = "B",
        name: str = "Custom Binary System",
    ):
        self.T_A = float(T_A)
        self.T_B = float(T_B)
        self.T_E = float(T_E)
        self.C_E = float(C_E)
        self.C_alpha_E = float(C_alpha_E)
        self.C_beta_E = float(C_beta_E)
        self.T_room = float(T_room)
        self.C_alpha_RT = float(C_alpha_RT) if C_alpha_RT is not None else self.C_alpha_E
        self.C_beta_RT = float(C_beta_RT) if C_beta_RT is not None else self.C_beta_E
        self.p_l = float(curvature_liquidus)
        self.p_s = float(curvature_solidus)
        self.component_A = component_A
        self.component_B = component_B
        self.name = name
        self._validate()

    # ------------------------------------------------------------------
    def _validate(self):
        errs = []
        if not (self.T_E < self.T_A and self.T_E < self.T_B):
            errs.append("Eutectic temperature must be lower than both melting points.")
        if not (0 <= self.C_alpha_E < self.C_E < self.C_beta_E <= 100):
            errs.append("Require 0 <= C_alpha_E < C_E < C_beta_E <= 100.")
        if not (0 <= self.C_alpha_RT <= self.C_alpha_E):
            errs.append("C_alpha_RT must be between 0 and C_alpha_E.")
        if not (self.C_beta_E <= self.C_beta_RT <= 100):
            errs.append("C_beta_RT must be between C_beta_E and 100.")
        if not (0 < self.p_l <= 1.2 and 0 < self.p_s <= 1.2):
            errs.append("Curvature exponents should be in (0, 1.2].")
        if self.T_room >= self.T_E:
            errs.append("T_room must be lower than T_E.")
        if errs:
            raise PhaseDiagramError(" ".join(errs))

    # -- liquidus ---------------------------------------------------------
    def liquidus_T(self, C: np.ndarray) -> np.ndarray:
        """Forward liquidus: composition (wt% B) -> temperature (°C)."""
        C = np.asarray(C, dtype=float)
        T = np.where(
            C <= self.C_E,
            self.T_A - (self.T_A - self.T_E) * np.power(np.clip(C / self.C_E, 0, 1), self.p_l),
            self.T_B - (self.T_B - self.T_E) * np.power(np.clip((100 - C) / (100 - self.C_E), 0, 1), self.p_l),
        )
        return T

    def liquidus_C_left(self, T: np.ndarray) -> np.ndarray:
        """Inverse of the A-rich liquidus branch: T -> C, valid for T_E<=T<=T_A."""
        T = np.asarray(T, dtype=float)
        frac = np.clip((self.T_A - T) / (self.T_A - self.T_E), 0, 1)
        return self.C_E * np.power(frac, 1.0 / self.p_l)

    def liquidus_C_right(self, T: np.ndarray) -> np.ndarray:
        """Inverse of the B-rich liquidus branch: T -> C, valid for T_E<=T<=T_B."""
        T = np.asarray(T, dtype=float)
        frac = np.clip((self.T_B - T) / (self.T_B - self.T_E), 0, 1)
        return 100 - (100 - self.C_E) * np.power(frac, 1.0 / self.p_l)

    # -- solidus ------------------------------------------------------------
    def solidus_alpha_T(self, C: np.ndarray) -> np.ndarray:
        """Forward alpha solidus: C in [0, C_alpha_E] -> T."""
        C = np.asarray(C, dtype=float)
        frac = np.clip(C / self.C_alpha_E, 0, 1)
        return self.T_A - (self.T_A - self.T_E) * np.power(frac, self.p_s)

    def solidus_beta_T(self, C: np.ndarray) -> np.ndarray:
        """Forward beta solidus: C in [C_beta_E, 100] -> T."""
        C = np.asarray(C, dtype=float)
        frac = np.clip((100 - C) / (100 - self.C_beta_E), 0, 1)
        return self.T_B - (self.T_B - self.T_E) * np.power(frac, self.p_s)

    def solidus_alpha_C(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        frac = np.clip((self.T_A - T) / (self.T_A - self.T_E), 0, 1)
        return self.C_alpha_E * np.power(frac, 1.0 / self.p_s)

    def solidus_beta_C(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        frac = np.clip((self.T_B - T) / (self.T_B - self.T_E), 0, 1)
        return 100 - (100 - self.C_beta_E) * np.power(frac, 1.0 / self.p_s)

    # -- solvus (sub-eutectic solubility limits) -----------------------------
    def solvus_alpha_C(self, T: np.ndarray) -> np.ndarray:
        """Linear solvus: alpha solubility limit for T in [T_room, T_E]."""
        T = np.asarray(T, dtype=float)
        T_c = np.clip(T, self.T_room, self.T_E)
        frac = (T_c - self.T_room) / (self.T_E - self.T_room)
        return self.C_alpha_RT + (self.C_alpha_E - self.C_alpha_RT) * frac

    def solvus_beta_C(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        T_c = np.clip(T, self.T_room, self.T_E)
        frac = (T_c - self.T_room) / (self.T_E - self.T_room)
        return self.C_beta_RT + (self.C_beta_E - self.C_beta_RT) * frac

    # -- phase region lookup --------------------------------------------------
    def phase_at(self, C: float, T: float) -> PhaseRegion:
        """Identify which phase field the point (composition, temperature) falls in."""
        if not (0 <= C <= 100):
            raise PhaseDiagramError("Composition must be between 0 and 100 wt%.")

        T_liq = float(self.liquidus_T(np.array([C]))[0])
        if T > T_liq:
            return PhaseRegion.LIQUID

        if abs(T - self.T_E) < 1e-9 and self.C_alpha_E < C < self.C_beta_E:
            return PhaseRegion.EUTECTIC_INVARIANT

        if C <= self.C_alpha_E:
            T_sol = float(self.solidus_alpha_T(np.array([C]))[0])
            if T > T_sol:
                return PhaseRegion.ALPHA_LIQUID
            C_solv = float(self.solvus_alpha_C(np.array([T]))[0])
            return PhaseRegion.ALPHA if C <= C_solv else PhaseRegion.ALPHA_BETA

        if C >= self.C_beta_E:
            T_sol = float(self.solidus_beta_T(np.array([C]))[0])
            if T > T_sol:
                return PhaseRegion.BETA_LIQUID
            C_solv = float(self.solvus_beta_C(np.array([T]))[0])
            return PhaseRegion.BETA if C >= C_solv else PhaseRegion.ALPHA_BETA

        # C_alpha_E < C < C_beta_E
        if T > self.T_E:
            return PhaseRegion.ALPHA_LIQUID if C < self.C_E else PhaseRegion.BETA_LIQUID
        return PhaseRegion.ALPHA_BETA

    # -- tie line construction -----------------------------------------------
    def tie_line(self, C: float, T: float) -> TieLine:
        """Construct the horizontal tie line through (C, T).

        Raises PhaseDiagramError if (C, T) falls in a single-phase field
        (no tie line — the field itself gives the composition).
        """
        region = self.phase_at(C, T)

        if region == PhaseRegion.LIQUID:
            raise PhaseDiagramError("Point is single-phase Liquid — no tie line.")
        if region in (PhaseRegion.ALPHA, PhaseRegion.BETA):
            raise PhaseDiagramError(f"Point is single-phase {region.value} — no tie line.")

        if region == PhaseRegion.EUTECTIC_INVARIANT:
            return TieLine(
                temperature=T, region=region,
                c_left=self.C_alpha_E, c_right=self.C_beta_E,
                label_left=f"α ({self.C_alpha_E:.2f}% {self.component_B})",
                label_right=f"β ({self.C_beta_E:.2f}% {self.component_B})",
            )

        if region == PhaseRegion.ALPHA_LIQUID:
            c_alpha = float(self.solidus_alpha_C(np.array([T]))[0])
            c_liq = float(self.liquidus_C_left(np.array([T]))[0])
            return TieLine(
                temperature=T, region=region,
                c_left=c_alpha, c_right=c_liq,
                label_left=f"α ({c_alpha:.2f}% {self.component_B})",
                label_right=f"L ({c_liq:.2f}% {self.component_B})",
            )

        if region == PhaseRegion.BETA_LIQUID:
            c_liq = float(self.liquidus_C_right(np.array([T]))[0])
            c_beta = float(self.solidus_beta_C(np.array([T]))[0])
            return TieLine(
                temperature=T, region=region,
                c_left=c_liq, c_right=c_beta,
                label_left=f"L ({c_liq:.2f}% {self.component_B})",
                label_right=f"β ({c_beta:.2f}% {self.component_B})",
            )

        # ALPHA_BETA (sub-eutectoid two-phase solid)
        c_alpha = float(self.solvus_alpha_C(np.array([T]))[0])
        c_beta = float(self.solvus_beta_C(np.array([T]))[0])
        return TieLine(
            temperature=T, region=region,
            c_left=c_alpha, c_right=c_beta,
            label_left=f"α ({c_alpha:.2f}% {self.component_B})",
            label_right=f"β ({c_beta:.2f}% {self.component_B})",
        )

    # -- lever rule -----------------------------------------------------------
    @staticmethod
    def lever_rule(C0: float, c_left: float, c_right: float) -> Tuple[float, float]:
        """Return (fraction_of_left_phase, fraction_of_right_phase) for overall
        composition C0 between c_left and c_right (order-independent)."""
        lo, hi = min(c_left, c_right), max(c_left, c_right)
        if not (lo - 1e-9 <= C0 <= hi + 1e-9):
            raise PhaseDiagramError(
                f"Overall composition {C0:.2f}% is outside the tie-line span "
                f"[{lo:.2f}%, {hi:.2f}%]."
            )
        if hi == lo:
            return 0.5, 0.5
        f_right_of_span = (C0 - lo) / (hi - lo)
        # map back to the (possibly reversed) left/right ordering supplied
        if c_left <= c_right:
            f_left = 1 - f_right_of_span
            f_right = f_right_of_span
        else:
            f_left = f_right_of_span
            f_right = 1 - f_right_of_span
        return f_left, f_right

    def phase_fractions(self, C0: float, T: float) -> dict:
        """Convenience: tie line + lever-rule phase fractions at (C0, T)."""
        tl = self.tie_line(C0, T)
        f_left, f_right = self.lever_rule(C0, tl.c_left, tl.c_right)
        return {
            "region": tl.region.value,
            "temperature": T,
            "tie_line": tl,
            tl.label_left: f_left,
            tl.label_right: f_right,
        }

    # -- vectorized region classification (for fast diagram shading) ------------
    def region_grid(self, C: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Vectorized phase-region classification over arrays of the same shape.

        Returns an integer code array: 0=Liquid, 1=alpha, 2=beta,
        3=alpha+Liquid, 4=beta+Liquid, 5=alpha+beta.
        """
        C = np.asarray(C, dtype=float)
        T = np.asarray(T, dtype=float)
        T_liq = self.liquidus_T(C)

        code = np.zeros(C.shape, dtype=int)  # default: Liquid
        below_liq = T <= T_liq

        left = C <= self.C_alpha_E
        right = C >= self.C_beta_E
        mid = ~left & ~right

        # left region (alpha-side)
        T_sol_a = self.solidus_alpha_T(np.clip(C, 0, self.C_alpha_E))
        in_al_liq = below_liq & left & (T > T_sol_a)
        code[in_al_liq] = 3
        below_sol_left = below_liq & left & (T <= T_sol_a)
        C_solv_a = self.solvus_alpha_C(T)
        is_alpha = below_sol_left & (C <= C_solv_a)
        is_ab_left = below_sol_left & ~is_alpha
        code[is_alpha] = 1
        code[is_ab_left] = 5

        # right region (beta-side)
        T_sol_b = self.solidus_beta_T(np.clip(C, self.C_beta_E, 100))
        in_be_liq = below_liq & right & (T > T_sol_b)
        code[in_be_liq] = 4
        below_sol_right = below_liq & right & (T <= T_sol_b)
        C_solv_b = self.solvus_beta_C(T)
        is_beta = below_sol_right & (C >= C_solv_b)
        is_ab_right = below_sol_right & ~is_beta
        code[is_beta] = 2
        code[is_ab_right] = 5

        # middle region (between C_alpha_E and C_beta_E)
        above_TE = T > self.T_E
        mid_al_liq = below_liq & mid & above_TE & (C < self.C_E)
        mid_be_liq = below_liq & mid & above_TE & (C >= self.C_E)
        mid_ab = below_liq & mid & ~above_TE
        code[mid_al_liq] = 3
        code[mid_be_liq] = 4
        code[mid_ab] = 5

        return code

    # -- diagram sampling for plotting -----------------------------------------
    def sample_curves(self, n: int = 250) -> dict:
        """Return dense (C, T) polylines for every boundary curve, for plotting."""
        T_top = max(self.T_A, self.T_B) + 0.05 * (max(self.T_A, self.T_B) - self.T_E)

        T_liq_left = np.linspace(self.T_E, self.T_A, n)
        C_liq_left = self.liquidus_C_left(T_liq_left)

        T_liq_right = np.linspace(self.T_E, self.T_B, n)
        C_liq_right = self.liquidus_C_right(T_liq_right)

        T_sol_alpha = np.linspace(self.T_E, self.T_A, n)
        C_sol_alpha = self.solidus_alpha_C(T_sol_alpha)

        T_sol_beta = np.linspace(self.T_E, self.T_B, n)
        C_sol_beta = self.solidus_beta_C(T_sol_beta)

        T_solvus_alpha = np.linspace(self.T_room, self.T_E, n)
        C_solvus_alpha = self.solvus_alpha_C(T_solvus_alpha)

        T_solvus_beta = np.linspace(self.T_room, self.T_E, n)
        C_solvus_beta = self.solvus_beta_C(T_solvus_beta)

        return {
            "liquidus_left": (C_liq_left, T_liq_left),
            "liquidus_right": (C_liq_right, T_liq_right),
            "solidus_alpha": (C_sol_alpha, T_sol_alpha),
            "solidus_beta": (C_sol_beta, T_sol_beta),
            "solvus_alpha": (C_solvus_alpha, T_solvus_alpha),
            "solvus_beta": (C_solvus_beta, T_solvus_beta),
            "eutectic_line": (np.array([self.C_alpha_E, self.C_beta_E]), np.array([self.T_E, self.T_E])),
            "T_top": T_top,
        }
