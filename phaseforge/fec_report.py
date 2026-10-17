"""
fec_report.py — static Matplotlib figure + self-contained HTML report for Fe–Fe₃C.

Mirrors `report.py` (the eutectic report): one downloadable HTML file bundling the
phase diagram, the analysis at a chosen (C₀, T), the full cooling sequence, a
micrograph filmstrip and the room-temperature constituents.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from PIL import Image

from .fec_layout import CLASS_BANDS, INVARIANT_COLORS, VIEWS
from .ironcarbon import (
    FIELD_COLORS, POINTS, POINT_NOTES, C_CEM, FeCError, IronCarbonDiagram, fmt_c,
)
from .microstructure import legend_items, render_micrograph


def _marker_temperature(e) -> float:
    return e.t_sample if e.kind == "Acm_ci" else e.temperature


def plot_fec_diagram(
    d: IronCarbonDiagram,
    C0: Optional[float] = None,
    T: Optional[float] = None,
    events=None,
    figsize=(9.6, 6.6),
    dpi=130,
):
    view = VIEWS["full"]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for region, verts in d.field_polygons().items():
        ax.add_patch(Polygon(verts, closed=True, facecolor=FIELD_COLORS[region], edgecolor="none", zorder=0))
    for pts in d.boundaries().values():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="#1B1B1E", lw=1.7, zorder=3)
    for name, (c0, c1, t) in d.invariant_lines().items():
        ax.plot([c0, c1], [t, t], color=INVARIANT_COLORS[name], lw=2.2, zorder=4)
        ax.text(c1 + 0.06, t, name.split(" ")[0], va="center", fontsize=8, color=INVARIANT_COLORS[name])
    for c0, c1, t in d.magnetic_lines().values():
        ax.plot([c0, c1], [t, t], color="#888888", lw=1.0, ls=(0, (4, 3)), zorder=2)

    for x, y, txt, size in view["labels"]:
        ax.text(x, y, txt, ha="center", va="center", fontsize=size * 0.72, color="#1B1B1E", zorder=5)
    for tx, ty, txt, dx, dy in view["tags"]:
        ax.annotate(txt, (tx, ty), xytext=(dx * 0.75, -dy * 0.75), textcoords="offset points",
                    fontsize=8.5, ha="left", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999", lw=0.6))
    for letter in view["letters"]:
        cx, ty = POINTS[letter]
        ax.plot(cx, ty, "o", ms=3.2, color="#1B1B1E", zorder=6)
        ax.annotate(letter, (cx, ty), xytext=(4, 4), textcoords="offset points", fontsize=8,
                    color="#333333", zorder=6, fontweight="bold")

    for c_lo, c_hi, label in CLASS_BANDS:
        ax.axvline(c_lo, color="#777777", lw=0.6, ls=":", zorder=1)
        ax.text((c_lo + c_hi) / 2, 1.008, label.replace("<br>", "\n"), ha="center", va="bottom",
                fontsize=6.8, color="#555555", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.axvline(C_CEM, color="#1B1B1E", lw=1.7, zorder=3)

    if C0 is not None:
        ax.axvline(C0, color="#C0392B", lw=1.4, ls="--", zorder=7)
        if events:
            for i, e in enumerate(events, 1):
                ax.plot(C0, _marker_temperature(e), "o", ms=11, color="#C0392B", zorder=8)
                ax.text(C0, _marker_temperature(e), str(i), ha="center", va="center",
                        fontsize=6.6, color="white", fontweight="bold", zorder=9)
        if T is not None:
            ax.plot(C0, T, "*", ms=15, color="black", markeredgecolor="white", markeredgewidth=0.8, zorder=10)

    ax.set_xlim(*view["x"])
    ax.set_ylim(*view["y"])
    ax.set_xlabel("Carbon content (wt% C)", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_title("Iron–Iron Carbide (Fe–Fe₃C) Phase Diagram", fontsize=13, fontweight="bold", pad=42)
    ax.grid(True, color="#DDDDDD", lw=0.5, alpha=0.6, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_b64(arr: np.ndarray, quality: int = 86) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _rows(items) -> str:
    return "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in items)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>PhaseForge Report — Fe–Fe₃C, {c0:.2f} wt% C</title>
<style>
  :root {{ --accent:#2E5EAA; --bg:#F7F9FC; --card:#fff; --text:#1B1B1E; --muted:#6B7280; }}
  body {{ font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif; background:var(--bg); color:var(--text); margin:0; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:32px 24px 64px; }}
  header {{ border-bottom:3px solid var(--accent); padding-bottom:16px; margin-bottom:28px; }}
  header h1 {{ margin:0; font-size:28px; color:var(--accent); }}
  header .sub {{ color:var(--muted); font-size:14px; margin-top:4px; }}
  .card {{ background:var(--card); border-radius:10px; padding:20px 24px; margin-bottom:22px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  .card h2 {{ margin-top:0; font-size:18px; color:var(--accent); border-left:4px solid var(--accent); padding-left:10px; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  td,th {{ padding:6px 10px; text-align:left; border-bottom:1px solid #EEE; vertical-align:top; }}
  th {{ color:var(--muted); font-weight:600; }}
  table.kv th {{ width:38%; }}
  img.wide {{ max-width:100%; border-radius:6px; border:1px solid #EEE; }}
  .strip {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:14px; }}
  .strip figure {{ margin:0; }}
  .strip img {{ width:100%; border-radius:6px; border:1px solid #DDD; display:block; }}
  .strip figcaption {{ font-size:12px; margin-top:4px; color:var(--text); }}
  .strip figcaption small {{ color:var(--muted); }}
  .note {{ font-size:12.5px; color:var(--muted); }}
  footer {{ text-align:center; color:var(--muted); font-size:12px; margin-top:40px; }}
</style></head>
<body><div class="wrap">
<header><h1>PhaseForge — Iron–Carbon (Fe–Fe₃C) Report</h1>
<div class="sub">Alloy: <b>{c0:.2f} wt% C</b> &nbsp;|&nbsp; Generated: {timestamp} &nbsp;|&nbsp; Developed by <b>Mohib Mehdi(MME 24-28)</b></div></header>

<div class="card"><h2>Alloy summary</h2><table class="kv">{summary_rows}</table></div>

<div class="card"><h2>Fe–Fe₃C phase diagram</h2>
<img class="wide" src="data:image/png;base64,{diagram_b64}">
<p class="note">Numbered markers on the red line match the cooling sequence below; the star marks the analysis point.</p></div>

<div class="card"><h2>Equilibrium analysis at {T:.0f} °C</h2><table class="kv">{analysis_rows}</table></div>

<div class="card"><h2>Cooling sequence (equilibrium)</h2>
<table><tr><th style="width:4%">#</th><th style="width:11%">T (°C)</th><th style="width:27%">Event / reaction</th>
<th style="width:18%">Phases after</th><th>What forms</th></tr>{sequence_rows}</table></div>

<div class="card"><h2>Microstructure evolution</h2><div class="strip">{strip}</div>
<p class="note">Schematic micrographs — not to scale. Area fractions equal the lever-rule weight fractions.</p></div>

<div class="card"><h2>Room-temperature constituents</h2><table class="kv">{constituent_rows}</table></div>

<div class="card"><h2>Modelling notes</h2>
<p class="note">Metastable Fe–Fe₃C diagram (steels and white cast irons) with straight-line, textbook-level
boundaries; slow (equilibrium) cooling only — non-equilibrium products such as bainite and martensite, and the
stable Fe–graphite system of grey/ductile irons, are not modelled. Values are illustrative teaching
approximations, not research-grade thermodynamic data.</p></div>

<footer>Generated automatically by <b>PhaseForge</b> — Computational Materials Science Toolkit.<br>
Developed by Mohib Mehdi(MME 24-28)</footer>
</div></body></html>
"""


def build_fec_html_report(d: IronCarbonDiagram, C0: float, T: float, seed: int = 7) -> str:
    events = d.transformation_path(C0)
    cls = d.alloy_class(C0)

    crit = d.critical_temperatures(C0)
    summary = [
        ("Carbon content (C₀)", f"{C0:.2f} wt% C"),
        ("Classification", cls.value),
        ("Common name", d.common_name(C0)),
    ] + [(k, f"{v:.0f} °C") for k, v in crit.items()]

    # --- analysis at T
    ms = d.micro_state(C0, T)
    region = d.phase_at(C0, T)
    analysis = [("Temperature", f"{T:.0f} °C"), ("Phase region", region.value)]
    try:
        tl = d.tie_line(C0, T + (1e-3 if abs(T - 727.0) < 1e-6 or abs(T - 1147.0) < 1e-6 or abs(T - 1495.0) < 1e-6 else 0))
        for lab in tl.labels:
            analysis.append(("Tie-line end", lab))
    except FeCError:
        pass
    for ph, fr in ms.fractions.items():
        analysis.append((f"Fraction — {ph}", f"{fr*100:.1f} %"))
    for name, fr in ms.constituents.items():
        analysis.append((f"Constituent — {name}", f"{fr*100:.1f} %"))

    # --- sequence table
    seq = []
    for i, e in enumerate(events, 1):
        seq.append(
            f"<tr><td>{i}</td><td>{html.escape(e.temp_text)}</td>"
            f"<td><b>{html.escape(e.title)}</b><br>{html.escape(e.reaction)}</td>"
            f"<td>{html.escape(e.phases_after)}</td><td>{html.escape(e.microstructure)}</td></tr>"
        )

    # --- filmstrip
    figs = []
    for k, (title, t_s, phases) in enumerate(d.evolution_stages(C0)):
        st = d.micro_state(C0, t_s)
        img = render_micrograph(st, size=300, seed=seed)
        figs.append(
            f'<figure><img src="data:image/jpeg;base64,{_img_b64(img)}">'
            f"<figcaption><b>{html.escape(title)}</b><br><small>{t_s:.0f} °C · {html.escape(phases)}</small></figcaption></figure>"
        )

    # --- room temperature
    rt = d.micro_state(C0, 25.0)
    cons = [(n, f"{v*100:.1f} %") for n, v in rt.constituents.items()]
    cons += [(f"Phase — {p}", f"{v*100:.1f} %") for p, v in rt.fractions.items()]

    diagram_b64 = _fig_b64(plot_fec_diagram(d, C0=C0, T=T, events=events))
    return _TEMPLATE.format(
        c0=C0,
        T=T,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        summary_rows=_rows(summary),
        diagram_b64=diagram_b64,
        analysis_rows=_rows(analysis),
        sequence_rows="".join(seq),
        strip="".join(figs),
        constituent_rows=_rows(cons),
    )
