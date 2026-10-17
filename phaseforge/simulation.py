"""
simulation.py — Solidification-path tracing and cooling-curve simulation.

Solidification path (equilibrium / lever-rule cooling):
    For an overall composition C0, step temperature down from the liquidus
    to the solidus (and, for alloys between C_alpha_E and C_beta_E, through
    the eutectic invariant), recording the instantaneous liquid and solid
    compositions and phase fractions at each step via `EutecticPhaseDiagram`.

Cooling-curve simulation:
    A lumped-parameter thermal model integrated with SciPy's `solve_ivp`.
    Above the liquidus and below full solidification, ordinary Newtonian
    cooling applies: dT/dt = -k(T - T_ambient). While solidification is in
    progress, releasing latent heat of fusion slows the cooling rate; this
    is modeled with an effective-heat-capacity term driven by the
    solidification path's df_solid/dT (a standard simplified approach used
    in teaching cooling-curve analysis). Passage through the eutectic
    invariant (if applicable) is modeled as an explicit thermal arrest
    (isothermal hold) whose duration is set by the latent heat released by
    the fraction of liquid still present when the alloy reaches T_E.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .core import EutecticPhaseDiagram, PhaseRegion, PhaseDiagramError


# ---------------------------------------------------------------------------
# Solidification path
# ---------------------------------------------------------------------------

@dataclass
class SolidificationStep:
    temperature: float
    liquid_comp: Optional[float]
    solid_comp: Optional[float]
    f_liquid: float
    f_solid_total: float
    region: str


@dataclass
class SolidificationPath:
    C0: float
    steps: List[SolidificationStep] = field(default_factory=list)
    T_liquidus: float = 0.0
    T_solidus_end: float = 0.0
    reaches_eutectic: bool = False
    proeutectic_phase: Optional[str] = None
    final_phase_fractions: dict = field(default_factory=dict)

    def as_arrays(self):
        T = np.array([s.temperature for s in self.steps])
        fL = np.array([s.f_liquid for s in self.steps])
        fS = np.array([s.f_solid_total for s in self.steps])
        return T, fL, fS


def trace_solidification_path(pd: EutecticPhaseDiagram, C0: float, n_steps: int = 120) -> SolidificationPath:
    """Trace the equilibrium (lever-rule) solidification path for composition C0."""
    if not (0 < C0 < 100):
        raise PhaseDiagramError("Composition must be strictly between 0 and 100 wt%.")

    T_liq = float(pd.liquidus_T(np.array([C0]))[0])
    path = SolidificationPath(C0=C0, T_liquidus=T_liq)

    hypo = C0 < pd.C_E  # proeutectic alpha vs beta
    path.proeutectic_phase = ("α" if hypo else "β") if pd.C_alpha_E < C0 < pd.C_beta_E else None

    if C0 <= pd.C_alpha_E:
        # Terminal alpha solid solution: solidifies entirely between liquidus and solidus,
        # never reaches the eutectic.
        T_end = float(pd.solidus_alpha_T(np.array([C0]))[0])
        path.reaches_eutectic = False
        T_grid = np.linspace(T_liq, T_end, n_steps)
        for T in T_grid:
            if T >= T_liq - 1e-9:
                path.steps.append(SolidificationStep(T, C0, None, 1.0, 0.0, PhaseRegion.LIQUID.value))
                continue
            if T <= T_end + 1e-9:
                path.steps.append(SolidificationStep(T, None, C0, 0.0, 1.0, PhaseRegion.ALPHA.value))
                continue
            tl = pd.tie_line(C0, T)  # c_left = alpha (solid), c_right = liquid
            f_solid, f_liquid = pd.lever_rule(C0, tl.c_left, tl.c_right)
            path.steps.append(SolidificationStep(T, tl.c_right, tl.c_left, f_liquid, f_solid, tl.region.value))
        path.T_solidus_end = T_end
        path.final_phase_fractions = {"α": 1.0}

    elif C0 >= pd.C_beta_E:
        T_end = float(pd.solidus_beta_T(np.array([C0]))[0])
        path.reaches_eutectic = False
        T_grid = np.linspace(T_liq, T_end, n_steps)
        for T in T_grid:
            if T >= T_liq - 1e-9:
                path.steps.append(SolidificationStep(T, C0, None, 1.0, 0.0, PhaseRegion.LIQUID.value))
                continue
            if T <= T_end + 1e-9:
                path.steps.append(SolidificationStep(T, None, C0, 0.0, 1.0, PhaseRegion.BETA.value))
                continue
            tl = pd.tie_line(C0, T)  # c_left = liquid, c_right = beta (solid)
            f_liquid, f_solid = pd.lever_rule(C0, tl.c_left, tl.c_right)
            path.steps.append(SolidificationStep(T, tl.c_left, tl.c_right, f_liquid, f_solid, tl.region.value))
        path.T_solidus_end = T_end
        path.final_phase_fractions = {"β": 1.0}

    else:
        # Off-eutectic alloy: proeutectic phase forms down to T_E, then eutectic reaction.
        path.reaches_eutectic = True
        T_grid = np.linspace(T_liq, pd.T_E, n_steps)
        f_liq_at_TE = None
        for T in T_grid:
            if T >= T_liq - 1e-9:
                path.steps.append(SolidificationStep(T, C0, None, 1.0, 0.0, PhaseRegion.LIQUID.value))
                continue

            if abs(T - pd.T_E) < 1e-6:
                # right at T_E: proeutectic solid + liquid at the eutectic composition
                tl_region = PhaseRegion.ALPHA_LIQUID if hypo else PhaseRegion.BETA_LIQUID
                c_solid = float((pd.solidus_alpha_C if hypo else pd.solidus_beta_C)(np.array([pd.T_E]))[0])
                c_liq = pd.C_E
                f_solid, f_liquid = pd.lever_rule(C0, c_solid, c_liq)
                f_liq_at_TE = f_liquid
                path.steps.append(SolidificationStep(T, c_liq, c_solid, f_liquid, f_solid, tl_region.value))
                continue

            tl = pd.tie_line(C0, T)
            # For hypoeutectic: c_left=proeutectic-alpha(solid), c_right=liquid.
            # For hypereutectic: c_left=liquid, c_right=proeutectic-beta(solid).
            solid_c = tl.c_left if hypo else tl.c_right
            liq_c = tl.c_right if hypo else tl.c_left
            f_solid, f_liquid = pd.lever_rule(C0, solid_c, liq_c)
            path.steps.append(SolidificationStep(T, liq_c, solid_c, f_liquid, f_solid, tl.region.value))

        if f_liq_at_TE is None:
            # composition never triggered the per-step branch above (e.g. C0 == C_E exactly,
            # so T_liquidus == T_E and there is no proeutectic region at all) -> the alloy
            # remains fully liquid until the eutectic reaction.
            f_liq_at_TE = 1.0

        # Eutectic reaction: remaining liquid (composition C_E) -> alpha(C_alpha_E) + beta(C_beta_E)
        f_alpha_in_eutectic, f_beta_in_eutectic = pd.lever_rule(pd.C_E, pd.C_alpha_E, pd.C_beta_E)
        f_proeutectic = 1 - f_liq_at_TE
        f_alpha_total = (f_proeutectic if hypo else 0.0) + f_liq_at_TE * f_alpha_in_eutectic
        f_beta_total = (f_proeutectic if not hypo else 0.0) + f_liq_at_TE * f_beta_in_eutectic

        path.steps.append(
            SolidificationStep(pd.T_E - 1e-6, None, None, 0.0, 1.0, PhaseRegion.ALPHA_BETA.value)
        )
        path.T_solidus_end = pd.T_E
        path.final_phase_fractions = {"α": f_alpha_total, "β": f_beta_total}

    return path


# ---------------------------------------------------------------------------
# Cooling curve simulation
# ---------------------------------------------------------------------------

@dataclass
class CoolingCurveResult:
    t: np.ndarray
    T: np.ndarray
    T_liquidus: float
    T_arrest_end: float          # end of solidification range (solidus, or T_E if eutectic)
    has_eutectic_plateau: bool
    t_liquidus_reached: float
    t_solidus_reached: float
    t_eutectic_plateau_end: Optional[float]
    notes: List[str] = field(default_factory=list)


def simulate_cooling_curve(
    pd: EutecticPhaseDiagram,
    C0: float,
    T_pour: Optional[float] = None,
    T_ambient: float = 25.0,
    k: float = 0.008,
    latent_heat_ratio: float = 9.0,
    t_max: float = 1200.0,
    n_points: int = 2000,
) -> CoolingCurveResult:
    """Simulate a cooling curve T(t) for an alloy of composition C0.

    Parameters
    ----------
    T_pour : starting (pour/melt) temperature; defaults to 8% above the liquidus.
    T_ambient : ambient/mold temperature (°C), the Newtonian-cooling sink.
    k : Newtonian cooling constant (1/s-ish, arbitrary time units).
    latent_heat_ratio : L/Cp ratio (dimensionless) — how many degrees' worth
        of sensible heat the latent heat of fusion is equivalent to; larger
        values give a more pronounced slope change / longer eutectic arrest.
    t_max : maximum simulated time.
    """
    path = trace_solidification_path(pd, C0)
    T_liq = path.T_liquidus
    T_end = path.T_solidus_end
    if T_pour is None:
        T_pour = T_liq + 0.08 * (T_liq - T_ambient)

    notes = []

    # --- df_solid/dT interpolant from the traced solidification path -------
    T_arr, fL_arr, fS_arr = path.as_arrays()
    # ensure strictly increasing T for interpolation (flip since T_arr is descending)
    order = np.argsort(T_arr)
    T_sorted = T_arr[order]
    fS_sorted = fS_arr[order]
    # de-duplicate
    T_u, idx_u = np.unique(T_sorted, return_index=True)
    fS_u = fS_sorted[idx_u]
    if len(T_u) >= 2:
        fS_interp = interp1d(T_u, fS_u, bounds_error=False, fill_value=(fS_u[0], fS_u[-1]))
    else:
        fS_interp = lambda T: np.zeros_like(np.atleast_1d(T))

    dT_span = max(T_liq - T_end, 1e-6)

    def dfs_dT_mag(T):
        """|d f_solid / dT|, estimated by finite difference on the interpolant."""
        h = max(dT_span * 1e-3, 1e-4)
        f_hi = float(fS_interp(min(T + h, T_liq)))
        f_lo = float(fS_interp(max(T - h, T_end)))
        return abs(f_hi - f_lo) / (2 * h)

    # --- Phase 1: liquid cooling, T_pour -> T_liq (closed-form Newtonian) ----
    def newton_T(t, T0):
        return T_ambient + (T0 - T_ambient) * np.exp(-k * t)

    if T_pour > T_liq:
        t_liq = -np.log((T_liq - T_ambient) / (T_pour - T_ambient)) / k
    else:
        t_liq = 0.0
        T_pour = T_liq

    # --- Phase 2: solidification range, T_liq -> T_end, with latent-heat drag
    def rhs(t, y):
        T = y[0]
        drag = 1.0 + latent_heat_ratio * dfs_dT_mag(T)
        dTdt = -k * (T - T_ambient) / drag
        return [dTdt]

    t_span_guess = t_max - t_liq
    sol = solve_ivp(
        rhs, [0, t_span_guess], [T_liq],
        method="RK45", dense_output=True, max_step=t_span_guess / 200,
        events=None,
    )
    # find when T crosses T_end
    t_eval2 = np.linspace(0, sol.t[-1], 800)
    T_eval2 = sol.sol(t_eval2)[0]
    below = np.where(T_eval2 <= T_end)[0]
    if len(below) > 0:
        t2_end = t_eval2[below[0]]
    else:
        t2_end = sol.t[-1]
        notes.append("Solidification range did not fully complete within t_max; consider increasing t_max.")
    t2 = t_eval2[t_eval2 <= t2_end]
    T2 = sol.sol(t2)[0]
    if len(t2) == 0 or t2[-1] < t2_end:
        t2 = np.append(t2, t2_end)
        T2 = np.append(T2, T_end)

    t_solidus_reached = t_liq + t2_end

    # --- Phase 3: eutectic isothermal arrest (if applicable) -----------------
    has_plateau = path.reaches_eutectic
    t_plateau = 0.0
    if has_plateau:
        f_liq_remaining = path.steps[-2].f_liquid if len(path.steps) >= 2 else 0.2
        # Time to dissipate the eutectic latent heat at the local (Newtonian) heat-extraction
        # rate: latent_heat_ratio (in "equivalent degrees" per unit liquid fraction) times the
        # liquid fraction still present, divided by the cooling-rate constant k. This keeps the
        # arrest duration on the same natural timescale (~1/k) as the rest of the simulation.
        t_plateau = (latent_heat_ratio * f_liq_remaining) / k
        t_plateau = float(np.clip(t_plateau, 0, t_max * 0.6))

    t_eutectic_end = t_solidus_reached + t_plateau if has_plateau else None

    # --- Phase 4: solid-state cooling from T_end (or T_E) down towards ambient
    t_remaining = max(t_max - (t_solidus_reached + t_plateau), 1.0)
    t4 = np.linspace(0, t_remaining, 400)
    T4 = newton_T(t4, T_end)
    t4_abs = t4 + t_solidus_reached + t_plateau

    # --- assemble full timeline ----------------------------------------------
    t1 = np.linspace(0, t_liq, 200) if t_liq > 0 else np.array([0.0])
    T1 = newton_T(t1, T_pour)

    t2_abs = t2 + t_liq

    if has_plateau:
        t_plat = np.linspace(t_solidus_reached, t_solidus_reached + t_plateau, 50)
        T_plat = np.full_like(t_plat, T_end)
        t_all = np.concatenate([t1, t2_abs, t_plat, t4_abs])
        T_all = np.concatenate([T1, T2, T_plat, T4])
    else:
        t_all = np.concatenate([t1, t2_abs, t4_abs])
        T_all = np.concatenate([T1, T2, T4])

    # sort/clean and resample to n_points for a smooth, evenly spaced curve
    order = np.argsort(t_all)
    t_all, T_all = t_all[order], T_all[order]
    t_all, uniq_idx = np.unique(t_all, return_index=True)
    T_all = T_all[uniq_idx]
    t_final = np.linspace(t_all[0], t_all[-1], n_points)
    T_final = np.interp(t_final, t_all, T_all)

    return CoolingCurveResult(
        t=t_final,
        T=T_final,
        T_liquidus=T_liq,
        T_arrest_end=T_end,
        has_eutectic_plateau=has_plateau,
        t_liquidus_reached=t_liq,
        t_solidus_reached=t_solidus_reached,
        t_eutectic_plateau_end=t_eutectic_end,
        notes=notes,
    )
