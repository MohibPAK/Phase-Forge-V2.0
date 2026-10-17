"""
monotectic.py — generic engine for binary monotectic systems: Cu-Pb and Al-Pb.

A monotectic system has liquid-liquid immiscibility: above the monotectic
temperature T_mon, alloys in a composition band separate into two liquids,
L1 (A-rich) and L2 (B-rich), coexisting in a dome-shaped miscibility gap that
closes at the critical (consolute) temperature T_c. On cooling through T_mon,
the monotectic reaction

    L1 -> alpha + L2      (alpha = A-rich terminal solid solution)

replaces L1 with more solid alpha plus a second liquid L2. L2 then continues
to reject alpha as it cools (an ordinary liquidus/solidus process) until it
reaches a (here strongly degenerate, since Cu/Al and Pb barely dissolve in
each other) eutectic reaction very close to pure B:

    L2 -> alpha + beta     (beta = B-rich terminal solid solution)

One class (`MonotecticDiagram`) is parametrized per system; `PRESETS` holds
the two systems this module ships.

Data sources
------------
Cu-Pb: monotectic at 954 degC (L1 36 wt% Pb, L2 87 wt% Pb), critical closure
at 990 degC, eutectic at 326 degC / 99.94 wt% Pb, and three extra points on
the L2 liquidus (2 wt% Cu at 700 degC, 7.5 wt% Cu at 900 degC) are from
Carpenter & Kullerud, *Am. Mineralogist* 53 (1968) and F.C. Campbell (ed.),
*Phase Diagrams velmo Understanding the Basics*, ASM International (2012), ch. 7.
The critical *composition* (61.5 wt% Pb) is not given in either source and is
an engineering estimate (roughly midway across the gap at T_mon).

Al-Pb: monotectic at 659 degC (L1 1.19 wt% Pb), critical point 1422 degC at
79.3 wt% Pb, from Ratke et al., *IOP Conf. Series: Mater. Sci. Eng.* 27
(2011) 012005 (their Fig. 1, after Yu). The L2 composition at the monotectic
and the (near-pure-Pb) eutectic point are not given there and are engineering
estimates consistent with the reported "gap spans almost the whole
composition range" description.

Both systems: solid solubility of Cu/Al in Pb and of Pb in Cu/Al is
essentially zero (per-mille level or less) and is represented here by small
but non-zero placeholder limits so the lever rule stays well posed; this
matches the ASM-quoted room-temperature solubilities (Pb in Cu ~0.002-0.005%,
Cu in Pb <0.007%) closely enough for teaching purposes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

from .ironcarbon import MicroState

T_ROOM = 25.0
_EPS_T = 1e-6


class MonotecticRegion(str, Enum):
    LIQUID = "Liquid (L)"
    TWO_LIQUID = "L₁ + L₂ (liquid immiscibility)"
    L_ALPHA = "L + α"
    ALPHA = "α"
    ALPHA_L2 = "α + L₂"
    L_BETA = "L + β"
    BETA = "β"
    ALPHA_BETA = "α + β"
    MONOTECTIC = "L₁ + α + L₂ (monotectic isotherm)"
    EUTECTIC = "L₂ + α + β (eutectic isotherm)"


class MonotecticError(ValueError):
    pass


@dataclass
class MonoTieLine:
    temperature: float
    points: List[Tuple[str, float]]

    @property
    def labels(self) -> List[str]:
        return [f"{p} ({c:.2f}%)" for p, c in self.points]


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


class MonotecticDiagram:
    def __init__(self, *, name, component_A, component_B, T_A, T_B,
                 T_mon, C_L1, C_L2, T_c, C_crit, T_eu, C_eu,
                 C_alpha_max, C_beta_min, liq2_points, p_A=0.7, p_B=0.7):
        self.name, self.component_A, self.component_B = name, component_A, component_B
        self.T_A, self.T_B = float(T_A), float(T_B)
        self.T_mon, self.C_L1, self.C_L2 = float(T_mon), float(C_L1), float(C_L2)
        self.T_c, self.C_crit = float(T_c), float(C_crit)
        self.T_eu, self.C_eu = float(T_eu), float(C_eu)
        self.C_alpha_max, self.C_beta_min = float(C_alpha_max), float(C_beta_min)
        self.p_A, self.p_B = float(p_A), float(p_B)

        pts = sorted(liq2_points, key=lambda p: p[0])
        self._liq2 = PchipInterpolator([p[0] for p in pts], [p[1] for p in pts])
        self._T_liq2_lo, self._T_liq2_hi = pts[0][0], pts[-1][0]

        if not (0 < self.C_alpha_max < self.C_L1 < self.C_crit < self.C_L2 < self.C_eu < self.C_beta_min < 100):
            raise MonotecticError("Critical points must satisfy 0 < C_alpha < C_L1 < C_crit < C_L2 < C_eu < C_beta < 100")

    # ------------------------------------------------------------------ boundary curves
    def liquidus_A_T(self, C):
        return self.T_A - (self.T_A - self.T_mon) * (np.clip(C, 0, self.C_L1) / self.C_L1) ** self.p_A

    def _liquidus_A_inv(self, T: float) -> float:
        frac = (self.T_A - T) / (self.T_A - self.T_mon)
        return self.C_L1 * max(frac, 0.0) ** (1.0 / self.p_A)

    def liquidus_Bnear_T(self, C):
        frac = (100.0 - np.clip(C, self.C_eu, 100.0)) / (100.0 - self.C_eu)
        return self.T_B - (self.T_B - self.T_eu) * frac ** self.p_B

    def _liquidus_B_inv(self, T: float) -> float:
        frac = (self.T_B - T) / (self.T_B - self.T_eu)
        return 100.0 - (100.0 - self.C_eu) * max(frac, 0.0) ** (1.0 / self.p_B)

    def dome_T(self, C):
        C = np.asarray(C, dtype=float)
        left = self.T_mon + (self.T_c - self.T_mon) * np.clip((C - self.C_L1) / (self.C_crit - self.C_L1), 0, 1) ** 2
        right = self.T_mon + (self.T_c - self.T_mon) * np.clip((self.C_L2 - C) / (self.C_L2 - self.C_crit), 0, 1) ** 2
        return np.where(C <= self.C_crit, left, right)

    def _dome_inv(self, T: float) -> Tuple[float, float]:
        frac = math.sqrt(max((T - self.T_mon) / (self.T_c - self.T_mon), 0.0))
        return self.C_L1 + (self.C_crit - self.C_L1) * frac, self.C_L2 - (self.C_L2 - self.C_crit) * frac

    def liq2_T(self, T: float) -> float:
        return float(self._liq2(np.clip(T, self._T_liq2_lo, self._T_liq2_hi)))

    # ------------------------------------------------------------------ classification
    def _shift(self, T: float, side: str) -> float:
        for Ti in (self.T_mon, self.T_eu):
            if abs(T - Ti) < _EPS_T:
                return Ti + (1e-3 if side == "above" else -1e-3)
        return T

    def phase_at(self, C: float, T: float) -> MonotecticRegion:
        R = MonotecticRegion
        if not (0.0 <= C <= 100.0):
            raise MonotecticError("Composition must lie between 0 and 100 wt%.")

        if abs(T - self.T_mon) < _EPS_T and self.C_L1 < C < self.C_L2:
            return R.MONOTECTIC
        if abs(T - self.T_eu) < _EPS_T and self.C_alpha_max < C < self.C_beta_min:
            return R.EUTECTIC

        if T >= self.T_c:
            return R.LIQUID

        if self.C_L1 <= C <= self.C_L2:
            dtop = float(self.dome_T(C))
            if T > dtop:
                return R.LIQUID
            if T > self.T_mon:
                return R.TWO_LIQUID
            # else fall through to the post-monotectic classification below

        if T > self.T_mon:
            if C < self.C_L1:
                liq = float(self.liquidus_A_T(C))
                if T >= liq:
                    return R.LIQUID
                return R.ALPHA if C <= self.C_alpha_max else R.L_ALPHA
            return R.LIQUID  # C > C_L2, above T_mon: a single (B-rich) liquid

        # T <= T_mon
        if T > self.T_eu:
            if C <= self.C_alpha_max:
                return R.ALPHA
            if C >= self.C_eu:
                liqB = float(self.liquidus_Bnear_T(C))
                if T >= liqB:
                    return R.LIQUID
                return R.BETA if C > self.C_beta_min else R.L_BETA
            c_l2 = self.liq2_T(T)
            return R.ALPHA_L2 if C < c_l2 else R.LIQUID

        # T <= T_eu
        if C <= self.C_alpha_max:
            return R.ALPHA
        if C >= self.C_beta_min:
            return R.BETA
        return R.ALPHA_BETA

    # ------------------------------------------------------------------ tie lines / lever rule
    def tie_line(self, C: float, T: float) -> MonoTieLine:
        R = MonotecticRegion
        region = self.phase_at(C, T)
        if region == R.MONOTECTIC:
            return MonoTieLine(T, [("α", self.C_alpha_max), ("L₁", self.C_L1), ("L₂", self.C_L2)])
        if region == R.EUTECTIC:
            return MonoTieLine(T, [("α", self.C_alpha_max), ("L₂", self.C_eu), ("β", self.C_beta_min)])
        if region == R.TWO_LIQUID:
            cl, cr = self._dome_inv(T)
            return MonoTieLine(T, [("L₁", cl), ("L₂", cr)])
        if region == R.L_ALPHA:
            return MonoTieLine(T, [("α", self.C_alpha_max), ("L", self._liquidus_A_inv(T))])
        if region == R.ALPHA_L2:
            return MonoTieLine(T, [("α", self.C_alpha_max), ("L₂", self.liq2_T(T))])
        if region == R.L_BETA:
            return MonoTieLine(T, [("L", self._liquidus_B_inv(T)), ("β", self.C_beta_min)])
        if region == R.ALPHA_BETA:
            return MonoTieLine(T, [("α", self.C_alpha_max), ("β", self.C_beta_min)])
        raise MonotecticError(f"Point is single-phase ({region.value}) — no tie line.")

    @staticmethod
    def lever_rule(C0: float, c_left: float, c_right: float) -> Tuple[float, float]:
        if c_right == c_left:
            return 0.5, 0.5
        f_right = (C0 - c_left) / (c_right - c_left)
        f_right = min(max(f_right, 0.0), 1.0)
        return 1.0 - f_right, f_right

    def phase_fractions(self, C0: float, T: float, side: str = "above") -> Dict[str, float]:
        T_eff = self._shift(T, side)
        region = self.phase_at(C0, T_eff)
        if region in (MonotecticRegion.LIQUID,):
            return {"L": 1.0}
        if region == MonotecticRegion.ALPHA:
            return {"α": 1.0}
        if region == MonotecticRegion.BETA:
            return {"β": 1.0}
        tl = self.tie_line(C0, T_eff)
        if len(tl.points) == 2:
            (p1, c1), (p2, c2) = tl.points
            f1, f2 = self.lever_rule(C0, c1, c2)
            return {p1: f1, p2: f2}
        # 3-phase invariant: split proportionally between the two flanking 2-phase lever rules
        (p1, c1), (p2, c2), (p3, c3) = tl.points
        if c1 <= C0 <= c2:
            f1, f2 = self.lever_rule(C0, c1, c2)
            return {p1: f1, p2: f2, p3: 0.0}
        f2, f3 = self.lever_rule(C0, c2, c3)
        return {p1: 0.0, p2: f2, p3: f3}

    # ------------------------------------------------------------------ classification helpers
    def common_name(self, C0: float) -> str:
        A, B = self.component_A, self.component_B
        if C0 <= self.C_alpha_max:
            return f"commercially pure {A}"
        if C0 < self.C_L1:
            return f"hypomonotectic {A}-{B} alloy (primary α, then monotectic + eutectic constituent)"
        if C0 < self.C_L2:
            return f"in-gap {A}-{B} alloy (passes through the L₁+L₂ miscibility gap on cooling)"
        if C0 < self.C_beta_min:
            return f"{B}-rich {A}-{B} alloy (primary β, then eutectic constituent)"
        return f"commercially pure {B}"

    def critical_temperatures(self, C0: float) -> Dict[str, float]:
        out = {}
        if C0 < self.C_L1:
            out["Liquidus"] = float(self.liquidus_A_T(C0))
        elif C0 <= self.C_L2:
            out["Enters miscibility gap"] = float(self.dome_T(C0)) if C0 >= self.C_L1 else self.T_mon
        else:
            out["Liquidus"] = float(self.liquidus_Bnear_T(C0)) if C0 >= self.C_eu else self.T_mon
        out["Monotectic"] = self.T_mon
        out["Eutectic"] = self.T_eu
        return out

    # ------------------------------------------------------------------ transformation path (auto-scanned)
    def _event_text(self, prev, nxt, T_evt):
        R = MonotecticRegion
        near_mon = abs(T_evt - self.T_mon) < 1.0
        near_eu = abs(T_evt - self.T_eu) < 1.0

        if near_mon and nxt == R.ALPHA_L2 and prev in (R.TWO_LIQUID, R.L_ALPHA, R.LIQUID):
            reaction = f"L₁ ({self.C_L1:.2f}%) → α ({self.C_alpha_max:.2f}%) + L₂ ({self.C_L2:.2f}%)"
            if prev == R.TWO_LIQUID:
                micro = ("The A-rich liquid (L₁) reacts completely to more solid α plus additional L₂; "
                         "the L₂ already present simply carries on. What remains is α dendrites/droplets "
                         "in a matrix of L₂.")
            else:
                micro = ("The last of the primary liquid (now at the monotectic composition) reacts "
                         "completely to more solid α plus a new liquid, L₂ — the α dendrites keep growing, "
                         "now surrounded by L₂.")
            return "monotectic", "Monotectic reaction", reaction, micro

        if near_eu and nxt == R.ALPHA_BETA:
            reaction = f"L₂ ({self.C_eu:.2f}%) → α ({self.C_alpha_max:.2f}%) + β ({self.C_beta_min:.2f}%)"
            if prev == R.ALPHA_L2:
                micro = ("The last L₂ (now enriched to the eutectic composition, overwhelmingly B) reacts "
                         "to a fine, 'divorced' eutectic mixture — small α particles in a continuous β "
                         "matrix — around the existing α dendrites.")
            elif prev == R.L_BETA:
                micro = ("The last liquid (now at the eutectic composition) reacts to a fine, divorced "
                         "eutectic mixture around the existing primary β grains.")
            else:
                micro = "The (near-pure-B) liquid reacts directly to a fine, divorced α+β eutectic mixture."
            return "eutectic", "Eutectic reaction", reaction, micro

        generic = {
            (R.LIQUID, R.TWO_LIQUID): ("Enters the liquid miscibility gap", "L → L₁ + L₂",
                "The single liquid separates into two immiscible liquids — like oil and water — with the "
                "minority one appearing as droplets dispersed in the majority liquid."),
            (R.LIQUID, R.L_ALPHA): ("Liquidus — solidification begins", "L → L + α",
                "α dendrites (the A-rich terminal solid solution) nucleate and grow into the melt."),
            (R.LIQUID, R.L_BETA): ("Liquidus — solidification begins", "L → L + β",
                "β dendrites (the B-rich terminal solid solution) nucleate and grow into the melt."),
            (R.LIQUID, R.ALPHA_L2): ("Liquidus — primary α begins to form", "L₂ → L₂ + α",
                "α forms directly from this (already B-rich) liquid as primary dendrites, well before the "
                "eutectic reaction — a quirk of alloys that sit between the monotectic and eutectic points."),
            (R.LIQUID, R.ALPHA): ("Freezes to α (negligible mushy range)", "L → α",
                "Solidifies over a razor-thin range, like a near-pure metal; equiaxed α grains form almost "
                "immediately."),
            (R.LIQUID, R.BETA): ("Freezes to β (negligible mushy range)", "L → β",
                "Solidifies over a razor-thin range, like a near-pure metal; equiaxed β grains form almost "
                "immediately."),
            (R.L_ALPHA, R.ALPHA): ("Solidification complete", "L + α → α", "Fully solid α: equiaxed grains."),
            (R.L_BETA, R.BETA): ("Solidification complete", "L + β → β", "Fully solid β: equiaxed grains."),
        }
        if (prev, nxt) in generic:
            title, reaction, micro = generic[(prev, nxt)]
        else:
            title, reaction, micro = f"{prev.value} → {nxt.value}", "—", ""
        return "boundary", title, reaction, micro

    def transformation_path(self, C0: float) -> List[PathEvent]:
        if not (0.0 < C0 < 100.0):
            raise MonotecticError("Composition must lie strictly between 0 and 100 wt%.")
        T_hi = min(max(self.T_A, self.T_c) + 30.0, 1650.0)
        Ts = np.linspace(T_hi, T_ROOM, 6000)
        events: List[PathEvent] = []
        prev = self.phase_at(C0, Ts[0])
        for i in range(1, len(Ts)):
            cur = self.phase_at(C0, Ts[i])
            if cur != prev:
                lo, hi = Ts[i - 1], Ts[i]
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if self.phase_at(C0, mid) == prev:
                        lo = mid
                    else:
                        hi = mid
                T_evt = hi
                kind, title, reaction, micro = self._event_text(prev, cur, T_evt)
                events.append(PathEvent(T_evt, kind, title, reaction,
                                         self.phase_at(C0, T_evt - 0.05).value, micro))
                prev = cur
        events.append(PathEvent(T_ROOM, "room", "Room temperature", "—",
                                 self.phase_at(C0, T_ROOM).value, self._final_text(C0)))
        for i, e in enumerate(events):
            nxt_T = events[i + 1].temperature if i + 1 < len(events) else T_ROOM
            e.t_sample = e.temperature - 0.3 * (e.temperature - nxt_T)
        events[-1].t_sample = T_ROOM
        return events

    def _final_text(self, C0: float) -> str:
        R = MonotecticRegion
        r = self.phase_at(C0, T_ROOM)
        if r == R.ALPHA:
            return f"Single-phase α grains ({self.common_name(C0)})."
        if r == R.BETA:
            return f"Single-phase β grains ({self.common_name(C0)})."
        return ("α grains/dendrites (accumulated from primary solidification and the monotectic reaction) "
                "sitting in a matrix of fine, divorced α+β eutectic — small α particles in continuous β, "
                "since the eutectic composition is so close to pure B.")

    def evolution_stages(self, C0: float) -> List[Tuple[str, float, str]]:
        events = self.transformation_path(C0)
        T_start = min(max(self.T_A, self.T_c) - 10.0, events[0].temperature + 40.0)
        stages = [("Molten alloy", T_start, "L")]
        stages += [(e.title, e.t_sample, e.phases_after) for e in events]
        return stages

    # ------------------------------------------------------------------ microstructure state
    def micro_state(self, C0: float, T: float) -> MicroState:
        R = MonotecticRegion
        T_eff = self._shift(T, "above")
        region = self.phase_at(C0, T_eff)
        fr = self.phase_fractions(C0, T_eff) if region not in (R.LIQUID, R.ALPHA, R.BETA) else \
            {"L": 1.0} if region == R.LIQUID else {"α": 1.0} if region == R.ALPHA else {"β": 1.0}
        ms = MicroState(C0=C0, T=T, region=region.value, recipe="grains", fractions=fr)

        if region == R.LIQUID:
            ms.recipe = "liquid"
            ms.constituents = {"Liquid": 1.0}
        elif region == R.TWO_LIQUID:
            ms.recipe = "droplets"
            f1, f2 = fr.get("L₁", 0.0), fr.get("L₂", 0.0)
            if f1 <= f2:
                ms.phase, ms.rim_phase, ms.f = "L₂", "L₁", f1
                ms.constituents = {"L₂ (matrix)": f2, "L₁ droplets": f1}
            else:
                ms.phase, ms.rim_phase, ms.f = "L₁", "L₂", f2
                ms.constituents = {"L₁ (matrix)": f1, "L₂ droplets": f2}
        elif region in (R.L_ALPHA, R.ALPHA_L2):
            liq_name = "L" if region == R.L_ALPHA else "L₂"
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "dendrites", liq_name, "α", fr.get("α", 0.0)
            ms.constituents = {liq_name: fr.get(liq_name, 0.0), "α dendrites": fr.get("α", 0.0)}
        elif region == R.L_BETA:
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "dendrites", "L", "β", fr.get("β", 0.0)
            ms.constituents = {"L": fr.get("L", 0.0), "β dendrites": fr.get("β", 0.0)}
        elif region == R.ALPHA:
            ms.recipe, ms.phase = "grains", "α"
            ms.constituents = {"α grains": 1.0}
        elif region == R.BETA:
            ms.recipe, ms.phase = "grains", "β"
            ms.constituents = {"β grains": 1.0}
        elif region == R.ALPHA_BETA:
            # accumulated primary/monotectic alpha (dendrite skeleton) sitting in a divorced eutectic
            # matrix of beta with small alpha particles — same shape as Fe-C's hypoeutectic ledeburite case
            f_prim = min(max((self.C_beta_min - C0) / (self.C_beta_min - self.C_alpha_max), 0.0), 1.0) \
                if C0 <= self.C_eu else 0.0
            # crude but serviceable estimate of how much alpha had already formed before the eutectic
            f_prim = 1.0 - self.phase_fractions(C0, self.T_eu, "above").get("L₂", 1.0) \
                if C0 < self.C_eu else (1.0 if C0 > self.C_eu else 0.0)
            ms.recipe = "ledeburite"
            ms.extras = dict(primary="dendrites" if f_prim > 0.02 else "none", f_primary=f_prim, f_sec=0.0,
                              particle="divorced", f_particle=0.14,
                              matrix_color="β", primary_color="α", particle_color="α")
            ms.constituents = {"α (accumulated dendrites)": f_prim, "Divorced eutectic (β + fine α)": 1.0 - f_prim}
        return ms

    # ------------------------------------------------------------------ plotting geometry
    def boundaries(self) -> Dict[str, List[Tuple[float, float]]]:
        c1 = np.linspace(0.0, self.C_L1, 30)
        c2 = np.linspace(self.C_L1, self.C_crit, 20)
        c3 = np.linspace(self.C_crit, self.C_L2, 20)
        t_liq2 = np.linspace(self._T_liq2_lo, self._T_liq2_hi, 30)
        c4 = np.linspace(self.C_eu, 100.0, 20)
        return {
            "Liquidus (A side)": list(zip(c1.tolist(), self.liquidus_A_T(c1).tolist())),
            "Miscibility gap (left)": list(zip(c2.tolist(), self.dome_T(c2).tolist())),
            "Miscibility gap (right)": list(zip(c3.tolist(), self.dome_T(c3).tolist())),
            "Liquidus (L₂ branch)": list(zip(self._liq2(t_liq2).tolist(), t_liq2.tolist())),
            "Liquidus (B side)": list(zip(c4.tolist(), self.liquidus_Bnear_T(c4).tolist())),
        }

    def field_polygons(self) -> Dict[MonotecticRegion, List[Tuple[float, float]]]:
        R = MonotecticRegion
        top = max(self.T_A, self.T_c) + 40.0
        room = T_ROOM - 5.0
        c1 = np.linspace(0.0, self.C_L1, 20)
        c2 = np.linspace(self.C_L1, self.C_crit, 15)
        c3 = np.linspace(self.C_crit, self.C_L2, 15)
        t_liq2 = np.linspace(self._T_liq2_hi, self._T_liq2_lo, 20)  # T_mon -> T_eu
        c_liq2 = self._liq2(t_liq2)
        c4 = np.linspace(self.C_eu, 100.0, 15)

        liqA = list(zip(c1.tolist(), self.liquidus_A_T(c1).tolist()))
        domeL = list(zip(c2.tolist(), self.dome_T(c2).tolist()))
        domeR = list(zip(c3.tolist(), self.dome_T(c3).tolist()))
        liqBnear = list(zip(c4.tolist(), self.liquidus_Bnear_T(c4).tolist()))
        liq2curve = list(zip(c_liq2.tolist(), t_liq2.tolist()))  # from (C_L2,T_mon) to (C_eu,T_eu)

        return {
            R.LIQUID: [(0.0, top)] + liqA + domeL + domeR + liq2curve + liqBnear + [(100.0, top)],
            R.TWO_LIQUID: domeL + domeR,
            R.L_ALPHA: [(self.C_alpha_max, self.T_A)] + liqA[1:] + [(self.C_alpha_max, self.T_mon)],
            R.ALPHA: [(0.0, room), (self.C_alpha_max, room), (self.C_alpha_max, self.T_mon), (0.0, self.T_A)],
            R.ALPHA_L2: [(self.C_alpha_max, self.T_mon)] + liq2curve + [(self.C_alpha_max, self._T_liq2_lo)],
            R.L_BETA: [(self.C_eu, self.T_eu)] + liqBnear + [(self.C_beta_min, self.T_eu)],
            R.BETA: [(self.C_beta_min, room), (100.0, room), (100.0, self.T_B), (self.C_beta_min, self.T_mon)],
            R.ALPHA_BETA: [(self.C_alpha_max, room), (self.C_beta_min, room),
                            (self.C_beta_min, self.T_eu), (self.C_alpha_max, self.T_eu)],
        }


FIELD_COLORS: Dict[MonotecticRegion, str] = {
    MonotecticRegion.LIQUID: "#FCE9D2",
    MonotecticRegion.TWO_LIQUID: "#F6C6D9",
    MonotecticRegion.L_ALPHA: "#F3E6B8",
    MonotecticRegion.ALPHA: "#E3F1EE",
    MonotecticRegion.ALPHA_L2: "#D6CCEA",
    MonotecticRegion.L_BETA: "#F8D9BD",
    MonotecticRegion.BETA: "#D3E1F1",
    MonotecticRegion.ALPHA_BETA: "#E9D3D3",
}

PRESETS = {
    "Cu-Pb": dict(
        name="Cu-Pb", component_A="Cu", component_B="Pb",
        T_A=1085.0, T_B=327.5,
        T_mon=954.0, C_L1=36.0, C_L2=87.0,
        T_c=990.0, C_crit=61.5,
        T_eu=326.0, C_eu=99.94,
        C_alpha_max=0.05, C_beta_min=99.95,
        liq2_points=[(954.0, 87.0), (900.0, 92.5), (700.0, 98.0), (326.0, 99.94)],
        p_A=0.7, p_B=0.7,
    ),
    "Al-Pb": dict(
        name="Al-Pb", component_A="Al", component_B="Pb",
        T_A=660.3, T_B=327.5,
        T_mon=659.0, C_L1=1.19, C_L2=99.3,
        T_c=1422.0, C_crit=79.3,
        T_eu=327.0, C_eu=99.8,
        C_alpha_max=0.02, C_beta_min=99.85,
        liq2_points=[(659.0, 99.3), (327.0, 99.8)],
        p_A=0.6, p_B=0.6,
    ),
}
