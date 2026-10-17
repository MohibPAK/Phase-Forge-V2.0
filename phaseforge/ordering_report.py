"""
ordering_report.py — static figure + self-contained HTML report for the
isomorphous-with-ordering systems (Cu-Au, Cu-Pd). Reuses the monotectic
report's HTML template so all PhaseForge reports look alike.
"""

from __future__ import annotations

import html
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .microstructure import render_micrograph
from .monotectic_report import _TEMPLATE as _MONO_TEMPLATE, _fig_b64, _img_b64, _rows
from .ordering import DOME_COLORS, FIELD_COLORS, OrderingDiagram

_TEMPLATE = _MONO_TEMPLATE.replace("(Monotectic) Report", "Report")


def region_color(d: OrderingDiagram, name: str) -> str:
    if name in FIELD_COLORS:
        return FIELD_COLORS[name]
    for i, dome in enumerate(d.domes):
        if name.startswith(dome.label):
            return DOME_COLORS[i % len(DOME_COLORS)]
    return "#EEEEEE"


def plot_ordering_diagram(d: OrderingDiagram, C0=None, T=None, events=None, figsize=(9.6, 6.4), dpi=130):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    top = max(d.T_A, d.T_B) + 60.0
    for name, verts in d.field_polygons().items():
        ax.add_patch(Polygon(verts, closed=True, facecolor=region_color(d, name), edgecolor="none", zorder=0))
    for name, pts in d.boundaries().items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="#1B1B1E", lw=1.8 if "order" not in name else 1.4, zorder=3,
                ls="-" if "order" not in name else "--")
    for dome in d.domes:
        ax.annotate(f"{dome.label}\n{dome.Tc:.0f} °C", (dome.center, dome.Tc), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    ax.text(50, top - 30, "Liquid (L)", ha="center", fontsize=10)
    ax.text(12, d.T_A - 25, "L+α", ha="center", fontsize=8)
    ax.text(50, 120, "α  (disordered FCC solid solution) — with ordered domes below", ha="center", fontsize=9)
    for c, t, lbl in ((0, d.T_A, f"{d.component_A}\n{d.T_A:.0f} °C"), (100, d.T_B, f"{d.component_B}\n{d.T_B:.0f} °C")):
        ax.plot(c, t, "o", ms=4, color="#1B1B1E", zorder=5)
        ax.annotate(lbl, (c, t), xytext=(8 if c == 0 else -8, 6), textcoords="offset points",
                    ha="left" if c == 0 else "right", fontsize=8.5, fontweight="bold")
    if C0 is not None:
        ax.axvline(C0, color="#C0392B", lw=1.4, ls="--", zorder=7)
        for i, e in enumerate(events or [], 1):
            if T_LO <= e.temperature <= top:
                ax.plot(C0, e.temperature, "o", ms=10, color="#C0392B", zorder=8)
                ax.text(C0, e.temperature, str(i), ha="center", va="center", fontsize=6.2,
                        color="white", fontweight="bold", zorder=9, clip_on=True)
        if T is not None:
            ax.plot(C0, T, "*", ms=15, color="black", markeredgecolor="white", markeredgewidth=0.8, zorder=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(25, top)
    ax.set_xlabel(f"Composition (wt% {d.component_B})", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_title(f"{d.component_A}–{d.component_B} Phase Diagram (isomorphous with ordering)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, color="#DDDDDD", lw=0.5, alpha=0.6, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


T_LO = 25.0


def build_ordering_html_report(d: OrderingDiagram, C0: float, T: float, seed: int = 7) -> str:
    events = d.transformation_path(C0)
    crit = d.critical_temperatures(C0)
    summary = [("Composition (C₀)", f"{C0:.2f} wt% {d.component_B}"), ("Classification", d.common_name(C0))]
    summary += [(k, f"{v:.0f} °C") for k, v in crit.items()]

    ms = d.micro_state(C0, T)
    analysis = [("Temperature", f"{T:.0f} °C"), ("Phase region", ms.region)]
    try:
        for lab in d.tie_line(C0, T).labels:
            analysis.append(("Tie-line end", lab))
    except Exception:
        pass
    for ph, fr in ms.fractions.items():
        analysis.append((f"Fraction — {ph}", f"{fr*100:.1f} %"))

    seq = "".join(
        f"<tr><td>{i}</td><td>{html.escape(e.temp_text)}</td>"
        f"<td><b>{html.escape(e.title)}</b><br>{html.escape(e.reaction)}</td>"
        f"<td>{html.escape(e.phases_after)}</td><td>{html.escape(e.microstructure)}</td></tr>"
        for i, e in enumerate(events, 1))
    figs = []
    for title, t_s, phases in d.evolution_stages(C0):
        img = render_micrograph(d.micro_state(C0, t_s), size=300, seed=seed)
        figs.append(f'<figure><img src="data:image/jpeg;base64,{_img_b64(img)}"><figcaption><b>{html.escape(title)}'
                    f"</b><br><small>{t_s:.0f} °C · {html.escape(phases)}</small></figcaption></figure>")
    rt = d.micro_state(C0, 25.0)
    cons = [(n, f"{v*100:.1f} %") for n, v in rt.constituents.items()]
    note = (f"{d.component_A}-{d.component_B} is isomorphous — one disordered FCC solid solution above the ordering "
            "temperatures — with solid-state order-disorder domes below. Equilibrium (infinitely slow) cooling is "
            "assumed; the order/disorder boundary is drawn as a single curve (real diagrams have a narrow "
            "two-phase sliver). " + d.data_note)
    return _TEMPLATE.format(
        sysname=d.name, compB=d.component_B, c0=C0, T=T, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        summary_rows=_rows(summary), diagram_b64=_fig_b64(plot_ordering_diagram(d, C0=C0, T=T, events=events)),
        analysis_rows=_rows(analysis), sequence_rows=seq, strip="".join(figs), constituent_rows=_rows(cons),
        model_note=html.escape(note))
