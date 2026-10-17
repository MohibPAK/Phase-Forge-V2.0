"""
cuni_report.py — static Matplotlib figure + self-contained HTML report for Cu–Ni.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from PIL import Image

from .cuni import FIELD_COLORS, CuNiDiagram, T_CU, T_NI
from .microstructure import render_micrograph


def plot_cuni_diagram(d: CuNiDiagram, C0=None, T=None, events=None, figsize=(9.2, 6.2), dpi=130):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for region, verts in d.field_polygons().items():
        ax.add_patch(Polygon(verts, closed=True, facecolor=FIELD_COLORS[region], edgecolor="none", zorder=0))
    for name, pts in d.boundaries().items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="#1B1B1E", lw=2, zorder=3, label=name)

    for c, t, lbl in ((0, T_CU, "Cu\n1085 °C"), (100, T_NI, "Ni\n1453 °C")):
        ax.plot(c, t, "o", ms=4, color="#1B1B1E", zorder=5)
        ax.annotate(lbl, (c, t), xytext=(8 if c == 0 else -8, 6), textcoords="offset points",
                    ha="left" if c == 0 else "right", fontsize=9, fontweight="bold")

    for x, txt in ((70, "Liquid (L)"), (55, "L + α"), (25, "α (FCC solid solution)")):
        ax.text(x, {"Liquid (L)": 1500, "L + α": 1200, "α (FCC solid solution)": 1030}[txt],
                txt, ha="center", fontsize=11)

    if C0 is not None:
        ax.axvline(C0, color="#C0392B", lw=1.4, ls="--", zorder=6)
        if events:
            for i, e in enumerate(events, 1):
                if not (950.0 <= e.temperature <= 1600.0):
                    continue  # off-chart (e.g. the room-temperature event) — would blow up the tight bbox
                ax.plot(C0, e.temperature, "o", ms=11, color="#C0392B", zorder=7)
                ax.text(C0, e.temperature, str(i), ha="center", va="center", fontsize=6.6,
                        color="white", fontweight="bold", zorder=8, clip_on=True)
        if T is not None:
            ax.plot(C0, T, "*", ms=15, color="black", markeredgecolor="white", markeredgewidth=0.8, zorder=9)

    ax.set_xlim(0, 100)
    ax.set_ylim(950, 1600)
    ax.set_xlabel("Composition (wt% Ni)", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_title("Copper–Nickel (Cu–Ni) Isomorphous Phase Diagram", fontsize=13, fontweight="bold", pad=12)
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
<title>PhaseForge Report — Cu–Ni, {c0:.1f} wt% Ni</title>
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
<header><h1>PhaseForge — Copper–Nickel (Cu–Ni) Report</h1>
<div class="sub">Alloy: <b>{c0:.1f} wt% Ni</b> &nbsp;|&nbsp; Generated: {timestamp} &nbsp;|&nbsp; Developed by <b>Mohib Mehdi(MME 24-28)</b></div></header>

<div class="card"><h2>Alloy summary</h2><table class="kv">{summary_rows}</table></div>

<div class="card"><h2>Cu–Ni phase diagram</h2>
<img class="wide" src="data:image/png;base64,{diagram_b64}">
<p class="note">Numbered markers on the red line match the cooling sequence below; the star marks the analysis point.</p></div>

<div class="card"><h2>Equilibrium analysis at {T:.0f} °C</h2><table class="kv">{analysis_rows}</table></div>

<div class="card"><h2>Cooling sequence (equilibrium)</h2>
<table><tr><th style="width:4%">#</th><th style="width:11%">T (°C)</th><th style="width:27%">Event / reaction</th>
<th style="width:18%">Phases after</th><th>What forms</th></tr>{sequence_rows}</table></div>

<div class="card"><h2>Microstructure evolution</h2><div class="strip">{strip}</div>
<p class="note">Schematic micrographs — not to scale. Area fractions equal the lever-rule weight fractions.</p></div>

<div class="card"><h2>Room-temperature result</h2><table class="kv">{constituent_rows}</table></div>

<div class="card"><h2>Modelling notes</h2>
<p class="note">Cu–Ni is a genuine isomorphous system: complete liquid and solid solubility, no invariant
reactions. Liquidus/solidus are fit to standard classroom data (Callister &amp; Rethwisch, adapted from
ASM's <i>Phase Diagrams of Binary Nickel Alloys</i>, P. Nash, ed., 1991). Equilibrium (infinitely slow)
cooling is assumed, so the model shows uniform, homogeneous α grains at room temperature — real castings
cool too fast for full solid-state diffusion and instead show cored (Ni-rich core / Cu-rich rim) dendritic
grains, which this teaching model does not simulate.</p></div>

<footer>Generated automatically by <b>PhaseForge</b> — Computational Materials Science Toolkit.<br>
Developed by Mohib Mehdi(MME 24-28)</footer>
</div></body></html>
"""


def build_cuni_html_report(d: CuNiDiagram, C0: float, T: float, seed: int = 7) -> str:
    events = d.transformation_path(C0)
    crit = d.critical_temperatures(C0)
    summary = [
        ("Composition (C₀)", f"{C0:.1f} wt% Ni"),
        ("Common name", d.common_name(C0)),
        ("Liquidus", f"{crit['Liquidus']:.0f} °C"),
        ("Solidus", f"{crit['Solidus']:.0f} °C"),
        ("Freezing range", f"{crit['Liquidus'] - crit['Solidus']:.0f} °C"),
    ]

    ms = d.micro_state(C0, T)
    region = d.phase_at(C0, T)
    analysis = [("Temperature", f"{T:.0f} °C"), ("Phase region", region.value)]
    try:
        tl = d.tie_line(C0, T)
        for lab in tl.labels:
            analysis.append(("Tie-line end", lab))
    except Exception:
        pass
    for ph, fr in ms.fractions.items():
        analysis.append((f"Fraction — {ph}", f"{fr*100:.1f} %"))

    seq = []
    for i, e in enumerate(events, 1):
        seq.append(
            f"<tr><td>{i}</td><td>{html.escape(e.temp_text)}</td>"
            f"<td><b>{html.escape(e.title)}</b><br>{html.escape(e.reaction)}</td>"
            f"<td>{html.escape(e.phases_after)}</td><td>{html.escape(e.microstructure)}</td></tr>"
        )

    figs = []
    for title, t_s, phases in d.evolution_stages(C0):
        st = d.micro_state(C0, t_s)
        img = render_micrograph(st, size=300, seed=seed)
        figs.append(
            f'<figure><img src="data:image/jpeg;base64,{_img_b64(img)}">'
            f"<figcaption><b>{html.escape(title)}</b><br><small>{t_s:.0f} °C · {html.escape(phases)}</small></figcaption></figure>"
        )

    rt = d.micro_state(C0, 25.0)
    cons = [(n, f"{v*100:.1f} %") for n, v in rt.constituents.items()]

    diagram_b64 = _fig_b64(plot_cuni_diagram(d, C0=C0, T=T, events=events))
    return _TEMPLATE.format(
        c0=C0, T=T, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        summary_rows=_rows(summary), diagram_b64=diagram_b64, analysis_rows=_rows(analysis),
        sequence_rows="".join(seq), strip="".join(figs), constituent_rows=_rows(cons),
    )
