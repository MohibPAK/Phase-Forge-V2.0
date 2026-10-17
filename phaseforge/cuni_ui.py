"""
cuni_ui.py — Streamlit interface for the Cu–Ni isomorphous system.

Same four-tab pattern as `fec_ui.py`: Diagram / Tie-Line & Lever Rule /
Microstructure Evolution / Report — scaled down because Cu–Ni has no
invariant reactions at all.
"""

from __future__ import annotations

import math
from typing import List

import plotly.graph_objects as go
import streamlit as st

from .cuni import CuNiDiagram, CuNiError, FIELD_COLORS, T_CU, T_NI
from .cuni_report import build_cuni_html_report
from .microstructure import legend_html, render_micrograph

CUNI_LABEL = "Cu–Ni  (Isomorphous: complete solid solubility)"
ACCENT = "#2E5EAA"
TIE_RED = "#C0392B"

ALLOY_PRESETS = {
    "— jump to a typical alloy —": None,
    "Cupronickel 90/10 (10 wt% Ni)": 10.0,
    "Cupronickel 70/30 (30 wt% Ni)": 30.0,
    "Equal mix (50 wt% Ni)": 50.0,
    "Monel-type range (65 wt% Ni)": 65.0,
    "Nickel-rich (90 wt% Ni)": 90.0,
}


@st.cache_resource(show_spinner=False)
def _diagram() -> CuNiDiagram:
    return CuNiDiagram()


@st.cache_data(show_spinner=False)
def _micro_image(C0: float, T: float, size: int, seed: int):
    d = _diagram()
    return render_micrograph(d.micro_state(C0, T), size=size, seed=seed)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def _floor_half(t: float) -> float:
    return math.floor(t * 2) / 2


def build_figure(d: CuNiDiagram, C0=None, T=None, events=None, tie=None, height=620) -> go.Figure:
    fig = go.Figure()
    for region, verts in d.field_polygons().items():
        xs = [v[0] for v in verts] + [verts[0][0]]
        ys = [v[1] for v in verts] + [verts[0][1]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=FIELD_COLORS[region],
                                 line=dict(width=0), hoveron="fills", text=region.value, hoverinfo="text",
                                 showlegend=False))
    for name, pts in d.boundaries().items():
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines", name=name,
                                 line=dict(color="#1B1B1E", width=2.2), hovertemplate=f"{name}<extra></extra>",
                                 showlegend=False))
    for c, t, lbl in ((0, T_CU, "Cu — 1085 °C"), (100, T_NI, "Ni — 1453 °C")):
        fig.add_trace(go.Scatter(x=[c], y=[t], mode="markers", marker=dict(size=7, color="#1B1B1E"),
                                 hovertemplate=f"{lbl}<extra></extra>", showlegend=False))

    annotations = [
        dict(x=72, y=1520, text="Liquid (L)", showarrow=False, font=dict(size=13, color="#1B1B1E")),
        dict(x=52, y=1220, text="L + α", showarrow=False, font=dict(size=12, color="#1B1B1E")),
        dict(x=25, y=1020, text="α (FCC solid solution)", showarrow=False, font=dict(size=12, color="#1B1B1E")),
    ]

    if C0 is not None:
        fig.add_trace(go.Scatter(x=[C0, C0], y=[950, 1600], mode="lines", name=f"Alloy {C0:.1f} wt% Ni",
                                 line=dict(color=TIE_RED, width=2, dash="dash"), hoverinfo="skip"))
        if events:
            fig.add_trace(go.Scatter(
                x=[C0] * len(events), y=[e.temperature for e in events], mode="markers+text",
                text=[str(i) for i in range(1, len(events) + 1)], textfont=dict(color="white", size=11),
                marker=dict(size=19, color=TIE_RED, line=dict(color="white", width=1)),
                customdata=[[e.title, e.reaction] for e in events],
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>", showlegend=False,
            ))
        if T is not None:
            fig.add_trace(go.Scatter(x=[C0], y=[T], mode="markers", name=f"Now: {T:.0f} °C",
                                     marker=dict(size=17, symbol="star", color="black",
                                                 line=dict(color="white", width=1.5)),
                                     hovertemplate=f"{C0:.1f} wt% Ni, {T:.0f} °C<extra></extra>"))
    if tie is not None:
        xs = [c for _, c in tie.points]
        fig.add_trace(go.Scatter(x=[xs[0], xs[-1]], y=[tie.temperature] * 2, mode="lines",
                                 line=dict(color=TIE_RED, width=3), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=[tie.temperature] * 2, mode="markers+text",
                                 text=[p for p, _ in tie.points], textposition="top center",
                                 marker=dict(size=9, color=TIE_RED, line=dict(color="white", width=1)),
                                 hovertemplate="%{text}: %{x:.2f} wt% Ni<extra></extra>", showlegend=False))

    fig.update_layout(
        template="plotly_white", height=height, annotations=annotations,
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis=dict(title="Composition (wt% Ni)", range=[0, 100], showgrid=True, gridcolor="#E5E7EB"),
        yaxis=dict(title="Temperature (°C)", range=[950, 1600], showgrid=True, gridcolor="#E5E7EB"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="closest", plot_bgcolor="white",
    )
    return fig


def _lever_latex(tl, C0: float) -> str:
    cl, cs = tl.c_liquid, tl.c_solid
    f_l = (cs - C0) / (cs - cl)
    f_a = (C0 - cl) / (cs - cl)
    return (
        rf"W_L = \frac{{C_\alpha - C_0}}{{C_\alpha - C_L}} = "
        rf"\frac{{{cs:.2f} - {C0:.2f}}}{{{cs:.2f} - {cl:.2f}}} = {f_l:.3f}"
        "\n\n"
        rf"W_\alpha = \frac{{C_0 - C_L}}{{C_\alpha - C_L}} = "
        rf"\frac{{{C0:.2f} - {cl:.2f}}}{{{cs:.2f} - {cl:.2f}}} = {f_a:.3f}"
    )


def _tab_diagram(d: CuNiDiagram):
    st.markdown("#### Copper–Nickel (Cu–Ni) isomorphous phase diagram")
    st.caption("The classic isomorphous binary: complete liquid *and* solid solubility, so there is only "
               "ever one solid phase (α, FCC) and no eutectic, peritectic or eutectoid reaction.")
    fig = build_figure(d)
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Melting points**")
    st.markdown(_md_table(["Component", "Melting point"], [["Cu", "1085 °C"], ["Ni", "1453 °C"]]))
    with st.expander("Why Cu–Ni has no invariant reactions"):
        st.markdown(
            "Cu and Ni are both FCC, similar atomic size, and adjacent in electronegativity — the Hume-Rothery "
            "rules for complete solid solubility are satisfied, so a single α solid solution field spans the "
            "whole composition range instead of splitting into separate α/β solubility limits with a eutectic, "
            "peritectic or eutectoid reaction between them (contrast with Fe–Fe₃C)."
        )
    with st.expander("Data source"):
        st.caption("Liquidus/solidus fit to standard classroom values (Callister & Rethwisch, adapted from "
                   "*Phase Diagrams of Binary Nickel Alloys*, P. Nash, ed., ASM International, 1991).")


def _tab_tie(d: CuNiDiagram):
    st.markdown("#### Tie-line & lever rule")
    c1, c2 = st.columns([1.35, 1])
    with c1:
        C0 = st.slider("Composition C₀ (wt% Ni)", 0.5, 99.5, 35.0, 0.5, key="cuni2_c0")
        T = st.slider("Temperature (°C)", 1000.0, 1500.0, 1250.0, 1.0, key="cuni2_T")
        region = d.phase_at(C0, T)
        tl = None
        if region.value == "L + α":
            tl = d.tie_line(C0, T)
        fig = build_figure(d, C0=C0, T=T, tie=tl, height=560)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.metric("Phase region", region.value)
        st.caption(d.common_name(C0))
        if tl is None:
            st.success(f"Single phase: 100 % {region.value}.")
        else:
            st.markdown(_md_table(["Tie-line end", "wt% Ni"],
                                  [["L", f"{tl.c_liquid:.2f}"], ["α", f"{tl.c_solid:.2f}"]]))
            fr = d.phase_fractions(C0, T)
            cols = st.columns(2)
            cols[0].metric("W(L)", f"{fr['L']*100:.1f} %")
            cols[1].metric("W(α)", f"{fr['α']*100:.1f} %")
            with st.expander("Lever rule — worked out", expanded=True):
                st.latex(_lever_latex(tl, C0))
        st.image(_micro_image(round(C0, 2), round(T, 1), 300, 7), caption=f"Schematic micrograph at {T:.0f} °C")


def _tab_micro(d: CuNiDiagram):
    st.session_state.setdefault("cuni_c0", 35.0)
    st.session_state.setdefault("cuni_T", 25.0)
    st.session_state.setdefault("cuni_preset", next(iter(ALLOY_PRESETS)))

    def _preset_cb():
        v = ALLOY_PRESETS.get(st.session_state.cuni_preset)
        if v is not None:
            st.session_state.cuni_c0 = v
            st.session_state.cuni_T = 25.0

    def _step(direction: int):
        temps = [s[1] for s in d.evolution_stages(st.session_state.cuni_c0)]
        T = st.session_state.cuni_T
        if direction > 0:
            nxt = [t for t in temps if t < T - 0.6]
            st.session_state.cuni_T = _floor_half(nxt[0]) if nxt else 25.0
        else:
            prv = [t for t in temps if t > T + 0.6]
            st.session_state.cuni_T = _floor_half(prv[-1]) if prv else _floor_half(temps[0])

    st.markdown("#### Microstructure evolution on cooling")
    st.caption("Slow (equilibrium) cooling from the melt. Since Cu–Ni has no solid-state reaction, everything "
               "interesting happens between the liquidus and the solidus — below that, the grains just sit there.")

    top1, top2 = st.columns([1.2, 1])
    with top1:
        st.selectbox("Typical alloys", list(ALLOY_PRESETS), key="cuni_preset", on_change=_preset_cb,
                     label_visibility="collapsed")
        C0 = st.slider("Composition C₀ (wt% Ni)", 0.5, 99.5, step=0.5, key="cuni_c0")
    with top2:
        b1, b2 = st.columns(2)
        b1.button("⬆ Previous stage", key="cuni_prev", on_click=_step, args=(-1,), use_container_width=True)
        b2.button("Next stage ⬇", key="cuni_next", on_click=_step, args=(1,), use_container_width=True)
        T = st.slider("Temperature (°C)", 25.0, 1600.0, step=0.5, key="cuni_T", format="%.1f")

    try:
        events = d.transformation_path(C0)
    except CuNiError as exc:
        st.error(str(exc))
        return
    ms = d.micro_state(C0, T)
    crit = d.critical_temperatures(C0)

    m1, m2, m3 = st.columns([1.2, 1, 1])
    m1.metric("Alloy", d.common_name(C0))
    m2.metric("Phase region at T", ms.region)
    m3.metric("Liquidus / solidus", f"{crit['Liquidus']:.0f} / {crit['Solidus']:.0f} °C")

    left, right = st.columns([1.15, 1])
    with left:
        fig = build_figure(d, C0=C0, T=T, events=events, height=520)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.image(_micro_image(round(C0, 2), round(T, 1), 440, 7),
                 caption=f"{C0:.1f} wt% Ni at {T:.0f} °C — {ms.region}  (schematic, not to scale)")
        st.markdown(legend_html(ms), unsafe_allow_html=True)

    st.markdown("##### Cooling sequence")
    rows = [[str(i), e.temp_text, f"**{e.title}**<br>{e.reaction}", e.phases_after, e.microstructure]
            for i, e in enumerate(events, 1)]
    st.markdown(_md_table(["#", "T (°C)", "Event / reaction", "Phases after", "What forms"], rows),
                unsafe_allow_html=True)

    st.markdown("##### Evolution strip")
    stages = d.evolution_stages(C0)
    cols = st.columns(len(stages))
    for col, (title, t_s, phases) in zip(cols, stages):
        col.image(_micro_image(round(C0, 2), round(_floor_half(t_s), 1), 260, 7))
        col.caption(f"**{title}**  \n{t_s:.0f} °C · {phases}")

    st.markdown("##### Room-temperature result")
    rt = d.micro_state(C0, 25.0)
    r1, r2 = st.columns([1, 1])
    with r1:
        st.markdown(_md_table(["Constituent", "wt %"], [[n, f"{v*100:.1f}"] for n, v in rt.constituents.items()]))
    with r2:
        st.image(_micro_image(round(C0, 2), 25.0, 360, 7), caption="Room-temperature microstructure")
        st.markdown(legend_html(rt), unsafe_allow_html=True)
    with st.expander("Note: coring in real castings", expanded=False):
        st.markdown(
            "This model assumes equilibrium cooling, so it predicts uniform grains of composition C₀. "
            "Real Cu–Ni castings cool faster than carbon can diffuse through the solid, so the first-formed "
            "(Ni-rich) dendrite cores never fully equalise with the later, more Cu-rich shell — a banded, "
            "**cored** structure that a homogenising anneal is needed to remove."
        )


def _tab_report(d: CuNiDiagram):
    st.markdown("#### Automated report")
    C0 = st.session_state.get("cuni_c0", 35.0)
    T = st.session_state.get("cuni_T", 25.0)
    st.write(f"Report for **{C0:.1f} wt% Ni** analysed at **{T:.0f} °C** — change these in the "
             "*Microstructure Evolution* tab.")
    if st.button("Generate report", key="cuni_gen"):
        with st.spinner("Rendering diagram and micrographs…"):
            st.session_state["cuni_report_html"] = build_cuni_html_report(d, C0, T)
            st.session_state["cuni_report_name"] = f"PhaseForge_CuNi_{C0:.1f}wtNi.html"
    html_doc = st.session_state.get("cuni_report_html")
    if html_doc:
        st.download_button("⬇ Download HTML report", html_doc, file_name=st.session_state["cuni_report_name"],
                           mime="text/html")
        st.caption("Self-contained file: diagram, cooling sequence, micrographs and results.")


def render_cu_ni():
    d = _diagram()
    t1, t2, t3, t4 = st.tabs(["📊 Cu–Ni Diagram", "📏 Tie-Line & Lever Rule",
                              "🔬 Microstructure Evolution", "📄 Report"])
    with t1:
        _tab_diagram(d)
    with t2:
        _tab_tie(d)
    with t3:
        _tab_micro(d)
    with t4:
        _tab_report(d)
