"""
eutectic_micro.py — microstructure evolution for the standard binary eutectic
model in `core.py`. Works for ANY `EutecticPhaseDiagram` instance — Pb-Sn,
Cu-Ag, Al-Si, or a Custom system the user builds — since it only calls that
class's existing public API (liquidus/solidus/solvus curves, phase_at,
tie_line, lever_rule).

Cooling an alloy of overall composition C0 from the melt follows one of five
patterns, mirroring the classic hypoeutectic / eutectic / hypereutectic
teaching sequence:

    C0 <= C_alpha_E              : freezes to single-phase alpha (plus a
                                    trace of secondary beta if C0 > C_alpha_RT)
    C_alpha_E < C0 < C_E         : primary alpha, then the eutectic reaction
                                    on the remaining liquid
    C0 == C_E (within tolerance) : freezes entirely at T_E — 100% eutectic
    C_E < C0 < C_beta_E          : primary beta, then the eutectic reaction
    C0 >= C_beta_E                : freezes to single-phase beta (plus a
                                    trace of secondary alpha)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .core import EutecticPhaseDiagram, PhaseDiagramError, PhaseRegion
from .ironcarbon import MicroState

TOL_C = 0.02


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


def _add(events, T, kind, title, reaction, phases, micro):
    events.append(PathEvent(T, kind, title, reaction, phases, micro))


def transformation_path(d: EutecticPhaseDiagram, C0: float) -> List[PathEvent]:
    if not (0.0 < C0 < 100.0):
        raise PhaseDiagramError("Composition must lie strictly between 0 and 100 wt%.")

    B = d.component_B
    events: List[PathEvent] = []

    if C0 <= d.C_alpha_E:
        T_liq = float(d.liquidus_T(C0))
        T_sol = float(d.solidus_alpha_T(C0))
        _add(events, T_liq, "liquidus", "Liquidus — solidification begins", "L → L + α",
             "L + α", "α dendrites nucleate and grow into the melt.")
        _add(events, T_sol, "solidus", "Solidification complete", "L + α → α", "α",
             "Fully solid α: equiaxed grains.")
        if C0 > d.C_alpha_RT + TOL_C:
            T_sv = float(d.solidus_alpha_T(C0)) - 0.35 * (T_sol - d.T_room)
            _add(events, T_sv, "solvus", "Solvus crossed — secondary β",
                 "α → α + β", "α + β",
                 f"A small amount of secondary β precipitates as {C0:.2f} wt% {B} exceeds the "
                 "falling room-temperature solubility limit of α.")

    elif C0 < d.C_E - TOL_C:
        T_liq = float(d.liquidus_T(C0))
        _add(events, T_liq, "liquidus", "Liquidus — solidification begins", "L → L + α",
             "L + α", "Primary α dendrites nucleate and grow, rejecting solute into the shrinking liquid.")
        _add(events, d.T_E, "eutectic", "Eutectic reaction", f"L ({d.C_E:.2f}) → α ({d.C_alpha_E:.2f}) + β ({d.C_beta_E:.2f})",
             "α + β (eutectic)",
             "The remaining liquid (now at the eutectic composition) transforms into the eutectic "
             "mixture — alternating α/β lamellae — filling the space between the primary α dendrites.")
        if C0 > d.C_alpha_RT + TOL_C:
            T_sv = d.T_room + 0.55 * (d.T_E - d.T_room)
            _add(events, T_sv, "solvus", "Solvus crossed — secondary β in the primary grains",
                 "α → α + β", "α + β",
                 "A small amount of secondary β precipitates within the primary α dendrites as their "
                 "solubility limit falls with temperature.")

    elif C0 <= d.C_E + TOL_C:
        _add(events, d.T_E, "eutectic", "Eutectic reaction (liquidus = eutectic)",
             f"L ({d.C_E:.2f}) → α ({d.C_alpha_E:.2f}) + β ({d.C_beta_E:.2f})", "α + β (100% eutectic)",
             "The whole melt freezes at one temperature into the eutectic mixture: fine, alternating "
             "α/β lamellae, with no primary phase at all.")

    elif C0 < d.C_beta_E - TOL_C:
        T_liq = float(d.liquidus_T(C0))
        _add(events, T_liq, "liquidus", "Liquidus — solidification begins", "L → L + β",
             "L + β", "Primary β dendrites nucleate and grow, rejecting solute into the shrinking liquid.")
        _add(events, d.T_E, "eutectic", "Eutectic reaction", f"L ({d.C_E:.2f}) → α ({d.C_alpha_E:.2f}) + β ({d.C_beta_E:.2f})",
             "α + β (eutectic)",
             "The remaining liquid (now at the eutectic composition) transforms into the eutectic "
             "mixture, filling the space between the primary β dendrites.")
        if C0 < d.C_beta_RT - TOL_C:
            T_sv = d.T_room + 0.55 * (d.T_E - d.T_room)
            _add(events, T_sv, "solvus", "Solvus crossed — secondary α in the primary grains",
                 "β → β + α", "α + β",
                 "A small amount of secondary α precipitates within the primary β dendrites as their "
                 "solubility limit falls with temperature.")

    else:
        T_liq = float(d.liquidus_T(C0))
        T_sol = float(d.solidus_beta_T(C0))
        _add(events, T_liq, "liquidus", "Liquidus — solidification begins", "L → L + β",
             "L + β", "β dendrites nucleate and grow into the melt.")
        _add(events, T_sol, "solidus", "Solidification complete", "L + β → β", "β",
             "Fully solid β: equiaxed grains.")
        if C0 < d.C_beta_RT - TOL_C:
            T_sv = float(d.solidus_beta_T(C0)) - 0.35 * (T_sol - d.T_room)
            _add(events, T_sv, "solvus", "Solvus crossed — secondary α",
                 "β → β + α", "α + β",
                 f"A small amount of secondary α precipitates as {B} solubility in β falls with temperature.")

    events.append(PathEvent(d.T_room, "room", "Room temperature", "—",
                             _region_label(d, C0, d.T_room), _final_text(d, C0)))

    events.sort(key=lambda e: -e.temperature)
    for i, e in enumerate(events):
        T_next = events[i + 1].temperature if i + 1 < len(events) else d.T_room
        e.t_sample = e.temperature - 0.3 * (e.temperature - T_next) if e.kind != "room" else d.T_room
    return events


def _region_label(d: EutecticPhaseDiagram, C0: float, T: float) -> str:
    try:
        return d.phase_at(C0, T).value
    except PhaseDiagramError:
        return "?"


def _final_text(d: EutecticPhaseDiagram, C0: float) -> str:
    if C0 <= d.C_alpha_E:
        return "Single-phase α grains (with a trace of secondary β if the alloy is near C_alpha_E)."
    if C0 >= d.C_beta_E:
        return "Single-phase β grains (with a trace of secondary α if the alloy is near C_beta_E)."
    if C0 < d.C_E - TOL_C:
        return "Primary α dendrites in a matrix of eutectic (α+β lamellae)."
    if C0 > d.C_E + TOL_C:
        return "Primary β dendrites in a matrix of eutectic (α+β lamellae)."
    return "100% eutectic mixture: fine α/β lamellae, no primary phase."


def evolution_stages(d: EutecticPhaseDiagram, C0: float) -> List[Tuple[str, float, str]]:
    events = transformation_path(d, C0)
    T_top = min(events[0].temperature + 40.0, max(d.T_A, d.T_B) + 60.0)
    stages = [("Molten alloy", T_top, "L")]
    stages += [(e.title, e.t_sample, e.phases_after) for e in events]
    return stages


def critical_temperatures(d: EutecticPhaseDiagram, C0: float) -> dict:
    out = {}
    if C0 <= d.C_alpha_E:
        out["Liquidus"] = float(d.liquidus_T(C0))
        out["Solidus"] = float(d.solidus_alpha_T(C0))
    elif C0 >= d.C_beta_E:
        out["Liquidus"] = float(d.liquidus_T(C0))
        out["Solidus"] = float(d.solidus_beta_T(C0))
    elif abs(C0 - d.C_E) > TOL_C:
        out["Liquidus"] = float(d.liquidus_T(C0))
    out["Eutectic"] = d.T_E
    return out


def common_name(d: EutecticPhaseDiagram, C0: float) -> str:
    A, B = d.component_A, d.component_B
    if C0 <= d.C_alpha_E:
        return f"{A}-rich {A}-{B} alloy (single-phase α)"
    if C0 < d.C_E - TOL_C:
        return f"hypoeutectic {A}-{B} alloy (primary α + eutectic)"
    if C0 <= d.C_E + TOL_C:
        return f"eutectic {A}-{B} alloy"
    if C0 < d.C_beta_E - TOL_C:
        return f"hypereutectic {A}-{B} alloy (primary β + eutectic)"
    return f"{B}-rich {A}-{B} alloy (single-phase β)"


# ---------------------------------------------------------------------------
# microstructure state
# ---------------------------------------------------------------------------
def micro_state(d: EutecticPhaseDiagram, C0: float, T: float) -> MicroState:
    R = PhaseRegion
    T_eff = T
    for Ti in (d.T_E,):
        if abs(T - Ti) < 1e-6:
            T_eff = Ti + 1e-3  # just above the isotherm: "before the reaction" state, for a clean render
    region = d.phase_at(C0, T_eff)
    ms = MicroState(C0=C0, T=T, region=region.value, recipe="grains", fractions={})

    if region == R.LIQUID:
        ms.recipe = "liquid"
        ms.fractions = {"L": 1.0}
        ms.constituents = {"Liquid": 1.0}
    elif region == R.ALPHA:
        ms.recipe, ms.phase = "grains", "α"
        ms.fractions = {"α": 1.0}
        ms.constituents = {"α grains": 1.0}
    elif region == R.BETA:
        ms.recipe, ms.phase = "grains", "β"
        ms.fractions = {"β": 1.0}
        ms.constituents = {"β grains": 1.0}
    elif region in (R.ALPHA_LIQUID, R.BETA_LIQUID):
        fr_dict = d.phase_fractions(C0, T_eff)
        tl = fr_dict["tie_line"]
        f_left, f_right = d.lever_rule(C0, tl.c_left, tl.c_right)
        if region == R.ALPHA_LIQUID:
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "dendrites", "L", "α", f_left
            ms.constituents = {"Liquid": f_right, "α dendrites": f_left}
        else:
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "dendrites", "L", "β", f_right
            ms.constituents = {"Liquid": f_left, "β dendrites": f_right}
    elif region == R.ALPHA_BETA:
        if C0 < d.C_alpha_E:
            f_beta = max(0.0, (C0 - d.solvus_alpha_C(T_eff)) / (100.0 - d.solvus_alpha_C(T_eff)))
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "rim", "α", "β", min(f_beta, 0.35)
            ms.constituents = {"α grains": 1.0 - ms.f, "Secondary β": ms.f}
        elif C0 > d.C_beta_E:
            f_alpha = max(0.0, (d.solvus_beta_C(T_eff) - C0) / d.solvus_beta_C(T_eff))
            ms.recipe, ms.phase, ms.rim_phase, ms.f = "rim", "β", "α", min(f_alpha, 0.35)
            ms.constituents = {"β grains": 1.0 - ms.f, "Secondary α": ms.f}
        elif C0 < d.C_E - TOL_C:
            f_prim = (d.C_E - C0) / (d.C_E - d.C_alpha_E)
            ms.recipe = "eutectic_dendrites"
            ms.extras = dict(primary="dendrites", f_primary=f_prim, primary_phase="α")
            ms.constituents = {"Primary α dendrites": f_prim, "Eutectic (α+β)": 1.0 - f_prim}
        elif C0 > d.C_E + TOL_C:
            f_prim = (C0 - d.C_E) / (d.C_beta_E - d.C_E)
            ms.recipe = "eutectic_dendrites"
            ms.extras = dict(primary="dendrites", f_primary=f_prim, primary_phase="β")
            ms.constituents = {"Primary β dendrites": f_prim, "Eutectic (α+β)": 1.0 - f_prim}
        else:
            ms.recipe = "pearlite"
            ms.constituents = {"Eutectic (α+β)": 1.0}
    return ms
