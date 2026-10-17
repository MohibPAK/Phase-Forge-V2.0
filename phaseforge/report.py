"""
report.py — Matplotlib figure generation + automated HTML report builder.

These are the "publication style" static plots (used in the downloadable
report and PNG exports); the live Streamlit UI uses Plotly for interactive
on-screen charts, but every number and curve comes from the same
`phaseforge.core` / `phaseforge.simulation` engine either way.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import EutecticPhaseDiagram, TieLine
from .simulation import SolidificationPath, CoolingCurveResult
from . import eutectic_micro as _em
from .microstructure import render_micrograph as _render_micrograph

PALETTE = {
    "liquid": "#2E86AB",
    "alpha": "#3FA34D",
    "beta": "#E07A5F",
    "boundary": "#1B1B1E",
    "grid": "#D9D9D9",
    "eutectic": "#8E44AD",
    "tie": "#C0392B",
    "text": "#1B1B1E",
}


def _style_axes(ax, pd: EutecticPhaseDiagram):
    ax.set_xlabel(f"Composition (wt% {pd.component_B})", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_phase_diagram(
    pd: EutecticPhaseDiagram,
    tie_line: Optional[TieLine] = None,
    marker: Optional[tuple] = None,
    solidification_path: Optional[SolidificationPath] = None,
    figsize=(9, 6.5),
    dpi=130,
):
    curves = pd.sample_curves(300)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.plot(*curves["liquidus_left"], color=PALETTE["boundary"], lw=2)
    ax.plot(*curves["liquidus_right"], color=PALETTE["boundary"], lw=2, label="Liquidus")
    ax.plot(*curves["solidus_alpha"], color=PALETTE["boundary"], lw=1.6, ls="--")
    ax.plot(*curves["solidus_beta"], color=PALETTE["boundary"], lw=1.6, ls="--", label="Solidus")
    ax.plot(*curves["solvus_alpha"], color=PALETTE["boundary"], lw=1.2, ls=":")
    ax.plot(*curves["solvus_beta"], color=PALETTE["boundary"], lw=1.2, ls=":", label="Solvus")
    ax.plot(*curves["eutectic_line"], color=PALETTE["eutectic"], lw=2)

    # shade phase fields
    T_top = curves["T_top"]
    Cg = np.linspace(0, 100, 220)
    Tg = np.linspace(pd.T_room, T_top, 220)
    CC, TT = np.meshgrid(Cg, Tg)
    Z = pd.region_grid(CC, TT)
    cmap = matplotlib.colors.ListedColormap(
        ["#EAF4FB", "#E7F5EA", "#FBEDE8", "#CFE9F7", "#F6D9CE", "#EFE3F5"]
    )
    ax.pcolormesh(CC, TT, Z, cmap=cmap, vmin=0, vmax=5, shading="auto", zorder=0)

    ax.text(pd.C_E / 2, T_top * 0.93, "α + L", ha="center", fontsize=10, color=PALETTE["alpha"])
    ax.text((pd.C_E + 100) / 2, T_top * 0.93, "β + L", ha="center", fontsize=10, color=PALETTE["beta"])
    ax.text(pd.C_alpha_E / 2, (pd.T_room + pd.T_E) / 2, "α", ha="center", fontsize=11, fontweight="bold")
    ax.text((pd.C_beta_E + 100) / 2, (pd.T_room + pd.T_E) / 2, "β", ha="center", fontsize=11, fontweight="bold")
    ax.text(pd.C_E, (pd.T_room + pd.T_E) / 2, "α + β", ha="center", fontsize=10)
    ax.text(50, T_top * 0.99, "Liquid", ha="center", fontsize=11, color=PALETTE["liquid"], fontweight="bold")

    ax.plot(pd.C_E, pd.T_E, "o", color=PALETTE["eutectic"], ms=6, zorder=5)
    ax.annotate(
        f"Eutectic\n{pd.C_E:.1f}% {pd.component_B}, {pd.T_E:.0f}°C",
        (pd.C_E, pd.T_E), xytext=(12, -28), textcoords="offset points", fontsize=8.5,
        color=PALETTE["eutectic"],
    )

    if tie_line is not None:
        ax.plot(
            [tie_line.c_left, tie_line.c_right], [tie_line.temperature, tie_line.temperature],
            color=PALETTE["tie"], lw=2.2, marker="o", ms=5, zorder=6,
        )
        ax.annotate(tie_line.label_left, (tie_line.c_left, tie_line.temperature),
                    xytext=(-8, 8), textcoords="offset points", fontsize=8.5, color=PALETTE["tie"], ha="right")
        ax.annotate(tie_line.label_right, (tie_line.c_right, tie_line.temperature),
                    xytext=(8, 8), textcoords="offset points", fontsize=8.5, color=PALETTE["tie"])

    if marker is not None:
        C0, T0 = marker
        ax.plot(C0, T0, "*", color="black", ms=15, zorder=7, markeredgecolor="white", markeredgewidth=0.8)

    if solidification_path is not None:
        Ts = [s.temperature for s in solidification_path.steps if s.liquid_comp is not None]
        Cs_liq = [s.liquid_comp for s in solidification_path.steps if s.liquid_comp is not None]
        Cs_sol = [s.solid_comp for s in solidification_path.steps if s.solid_comp is not None]
        Ts_sol = [s.temperature for s in solidification_path.steps if s.solid_comp is not None]
        ax.plot(Cs_liq, Ts, color=PALETTE["liquid"], lw=2.4, zorder=8, label="Liquid composition path")
        ax.plot(Cs_sol, Ts_sol, color="#B8860B", lw=2.4, zorder=8, label="Solid composition path")
        ax.axvline(solidification_path.C0, color="black", lw=1, ls=":", alpha=0.6)

    _style_axes(ax, pd)
    ax.set_ylim(pd.T_room, T_top)
    ax.set_title(f"{pd.name} Binary Phase Diagram", fontsize=13, fontweight="bold")
    if tie_line is not None or solidification_path is not None:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_solidification_fractions(path: SolidificationPath, figsize=(8, 5), dpi=130):
    T, fL, fS = path.as_arrays()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(T, fL, color=PALETTE["liquid"], lw=2.2, label="Liquid fraction")
    ax.plot(T, fS, color="#B8860B", lw=2.2, label="Solid fraction")
    ax.axvline(path.T_liquidus, color="gray", ls="--", lw=1, label=f"Liquidus ({path.T_liquidus:.1f}°C)")
    ax.axvline(path.T_solidus_end, color="gray", ls=":", lw=1, label=f"Solidus/eutectic end ({path.T_solidus_end:.1f}°C)")
    ax.invert_xaxis()
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Phase fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Solidification Path — C₀ = {path.C0:.1f} wt%", fontsize=12, fontweight="bold")
    ax.grid(True, color=PALETTE["grid"], alpha=0.7)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_cooling_curve(cc: CoolingCurveResult, figsize=(8, 5), dpi=130):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(cc.t, cc.T, color=PALETTE["boundary"], lw=2)
    ax.axhline(cc.T_liquidus, color=PALETTE["liquid"], ls="--", lw=1, label=f"Liquidus ({cc.T_liquidus:.1f}°C)")
    ax.axhline(cc.T_arrest_end, color=PALETTE["eutectic"], ls="--", lw=1,
               label=f"{'Eutectic' if cc.has_eutectic_plateau else 'Solidus'} ({cc.T_arrest_end:.1f}°C)")
    ax.axvline(cc.t_liquidus_reached, color=PALETTE["liquid"], ls=":", lw=0.8, alpha=0.6)
    if cc.has_eutectic_plateau and cc.t_eutectic_plateau_end:
        ax.axvspan(cc.t_solidus_reached, cc.t_eutectic_plateau_end, color=PALETTE["eutectic"], alpha=0.12,
                   label="Eutectic thermal arrest")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Simulated Cooling Curve", fontsize=12, fontweight="bold")
    ax.grid(True, color=PALETTE["grid"], alpha=0.7)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    return fig


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PhaseForge Report — {system_name}</title>
<style>
  :root {{
    --accent: #2E5EAA;
    --bg: #F7F9FC;
    --card: #FFFFFF;
    --text: #1B1B1E;
    --muted: #6B7280;
  }}
  body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text); margin:0; padding:0; }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 32px 24px 64px; }}
  header {{ border-bottom: 3px solid var(--accent); padding-bottom: 16px; margin-bottom: 28px; }}
  header h1 {{ margin: 0; font-size: 28px; color: var(--accent); }}
  header .sub {{ color: var(--muted); font-size: 14px; margin-top: 4px; }}
  .card {{ background: var(--card); border-radius: 10px; padding: 20px 24px; margin-bottom: 22px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .card h2 {{ margin-top: 0; font-size: 18px; color: var(--accent); border-left: 4px solid var(--accent); padding-left: 10px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  td, th {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #EEE; }}
  th {{ color: var(--muted); font-weight: 600; width: 45%; }}
  img {{ max-width: 100%; border-radius: 6px; border: 1px solid #EEE; margin-top: 10px; }}
  .strip {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:14px; margin-top:10px; }}
  .strip figure {{ margin:0; }}
  .strip img {{ width:100%; margin-top:0; }}
  .strip figcaption {{ font-size:12px; margin-top:4px; }}
  .strip figcaption small {{ color:var(--muted); }}
  .pill {{ display:inline-block; background:#EAF1FC; color:var(--accent); padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-right:6px;}}
  footer {{ text-align:center; color: var(--muted); font-size: 12px; margin-top: 40px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>PhaseForge — Computational Report</h1>
    <div class="sub">System: <b>{system_name}</b> &nbsp;|&nbsp; Generated: {timestamp} &nbsp;|&nbsp; Developed by <b>Mohib Mehdi(MME 24-28)</b></div>
  </header>

  <div class="card">
    <h2>System Parameters</h2>
    <table>{params_rows}</table>
  </div>

  <div class="card">
    <h2>Phase Diagram</h2>
    <img src="data:image/png;base64,{diagram_b64}">
  </div>

  {analysis_section}

  {solidification_section}

  {cooling_section}

  {microstructure_section}

  <footer>
    Generated automatically by <b>PhaseForge</b> — Computational Materials Science Toolkit.<br>
    Developed by Mohib Mehdi(MME 24-28) &middot; Python · NumPy · SciPy · Matplotlib
  </footer>
</div>
</body>
</html>
"""


def _rows(d: dict) -> str:
    return "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in d.items())


def _img_b64(arr, quality: int = 86) -> str:
    import numpy as _np
    from PIL import Image as _Image
    buf = io.BytesIO()
    _Image.fromarray(_np.asarray(arr)).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_html_report(
    pd: EutecticPhaseDiagram,
    analysis: Optional[dict] = None,          # {"C0":.., "T":.., "tie_line":TieLine, "fractions": {...}}
    solidification: Optional[SolidificationPath] = None,
    cooling: Optional[CoolingCurveResult] = None,
    microstructure_C0: Optional[float] = None,
    seed: int = 7,
) -> str:
    params = {
        "Component A / B": f"{pd.component_A} / {pd.component_B}",
        "Melting point A (T_A)": f"{pd.T_A:.1f} °C",
        "Melting point B (T_B)": f"{pd.T_B:.1f} °C",
        "Eutectic temperature (T_E)": f"{pd.T_E:.1f} °C",
        "Eutectic composition (C_E)": f"{pd.C_E:.2f} wt% {pd.component_B}",
        "Max solubility of B in α (at T_E)": f"{pd.C_alpha_E:.2f} wt% {pd.component_B}",
        "Max solubility of A in β (at T_E)": f"{pd.C_beta_E:.2f} wt% {pd.component_B}",
    }

    marker = (analysis["C0"], analysis["T"]) if analysis else None
    tie_line = analysis["tie_line"] if analysis else None
    diagram_fig = plot_phase_diagram(pd, tie_line=tie_line, marker=marker, solidification_path=solidification)
    diagram_b64 = fig_to_base64(diagram_fig)

    analysis_section = ""
    if analysis:
        rows = {
            "Overall composition (C₀)": f"{analysis['C0']:.2f} wt% {pd.component_B}",
            "Temperature": f"{analysis['T']:.1f} °C",
            "Phase region": analysis["region"],
        }
        for label, frac in analysis["fractions"].items():
            rows[f"Fraction — {label}"] = f"{frac*100:.1f} %"
        analysis_section = f"""<div class="card"><h2>Tie-Line &amp; Lever-Rule Analysis</h2><table>{_rows(rows)}</table></div>"""

    solidification_section = ""
    if solidification:
        frac_fig = plot_solidification_fractions(solidification)
        frac_b64 = fig_to_base64(frac_fig)
        rows = {
            "Overall composition (C₀)": f"{solidification.C0:.2f} wt% {pd.component_B}",
            "Liquidus temperature": f"{solidification.T_liquidus:.1f} °C",
            "Solidification end temperature": f"{solidification.T_solidus_end:.1f} °C",
            "Passes through eutectic invariant": "Yes" if solidification.reaches_eutectic else "No",
        }
        for label, frac in solidification.final_phase_fractions.items():
            rows[f"Final fraction — {label}"] = f"{frac*100:.1f} %"
        solidification_section = f"""<div class="card"><h2>Solidification Path</h2><table>{_rows(rows)}</table>
        <img src="data:image/png;base64,{frac_b64}"></div>"""

    cooling_section = ""
    if cooling:
        cool_fig = plot_cooling_curve(cooling)
        cool_b64 = fig_to_base64(cool_fig)
        rows = {
            "Liquidus arrest temperature": f"{cooling.T_liquidus:.1f} °C",
            f"{'Eutectic' if cooling.has_eutectic_plateau else 'Solidus'} temperature": f"{cooling.T_arrest_end:.1f} °C",
            "Time to reach liquidus": f"{cooling.t_liquidus_reached:.1f}",
            "Time to reach solidus/eutectic": f"{cooling.t_solidus_reached:.1f}",
        }
        if cooling.has_eutectic_plateau and cooling.t_eutectic_plateau_end:
            rows["Eutectic arrest duration"] = f"{cooling.t_eutectic_plateau_end - cooling.t_solidus_reached:.1f}"
        cooling_section = f"""<div class="card"><h2>Cooling Curve Simulation</h2><table>{_rows(rows)}</table>
        <img src="data:image/png;base64,{cool_b64}"></div>"""

    microstructure_section = ""
    if microstructure_C0 is not None:
        events = _em.transformation_path(pd, microstructure_C0)
        stages = _em.evolution_stages(pd, microstructure_C0)
        figs = []
        for title, t_s, phases in stages:
            ms = _em.micro_state(pd, microstructure_C0, t_s)
            img = _render_micrograph(ms, size=300, seed=seed)
            figs.append(
                f'<figure><img src="data:image/jpeg;base64,{_img_b64(img)}">'
                f"<figcaption><b>{title}</b><br><small>{t_s:.0f} °C · {phases}</small></figcaption></figure>"
            )
        rt = _em.micro_state(pd, microstructure_C0, pd.T_room)
        rows = {
            "Overall composition (C₀)": f"{microstructure_C0:.2f} wt% {pd.component_B}",
            "Classification": _em.common_name(pd, microstructure_C0),
        }
        for name, frac in rt.constituents.items():
            rows[f"Room-temperature — {name}"] = f"{frac*100:.1f} %"
        seq_rows = "".join(
            f"<tr><th>{i}. {e.title} ({e.temp_text} °C)</th><td>{e.reaction}<br>{e.microstructure}</td></tr>"
            for i, e in enumerate(events, 1)
        )
        microstructure_section = f"""<div class="card"><h2>Microstructure Evolution</h2><table>{_rows(rows)}</table>
        <table>{seq_rows}</table>
        <div class="strip">{''.join(figs)}</div></div>"""

    return _HTML_TEMPLATE.format(
        system_name=pd.name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        params_rows=_rows(params),
        diagram_b64=diagram_b64,
        analysis_section=analysis_section,
        solidification_section=solidification_section,
        cooling_section=cooling_section,
        microstructure_section=microstructure_section,
    )
