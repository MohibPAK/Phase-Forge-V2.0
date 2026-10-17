"""
monotectic_ui.py — Streamlit interface for monotectic systems (Cu-Pb, Al-Pb).

One generic module drives both: `render_monotectic(preset_key)` builds the
same four-tab layout as `fec_ui.py` / `cuni_ui.py`, parametrized by which
`monotectic.PRESETS` entry is selected.
"""

from __future__ import annotations

import math
from typing import List

import plotly.graph_objects as go
import streamlit as st

from .microstructure import legend_html, render_micrograph
from .monotectic import FIELD_COLORS, MonotecticDiagram, MonotecticError, PRESETS
from .monotectic_report import build_monotectic_html_report

TIE_RED = "#C0392B"

LABELS = {"Cu-Pb": "Cu–Pb  (Monotectic: liquid immiscibility)",
          "Al-Pb": "Al–Pb  (Monotectic: liquid immiscibility)"}

DATA_NOTES = {
    "Cu-Pb": ("Monotectic/eutectic invariants and three points on the L₂ liquidus are from Carpenter & "
              "Kullerud, *Am. Mineralogist* 53 (1968), and F.C. Campbell (ed.), *Phase Diagrams — "
              "Understanding the Basics*, ASM International (2012). The critical composition (61.5 wt% Pb) "
              "is not given in either source and is an engineering estimate."),
    "Al-Pb": ("Monotectic and critical points are from Ratke et al., *IOP Conf. Series: Mater. Sci. Eng.* "
              "27 (2011) 012005. The L₂ composition at the monotectic and the eutectic point are not given "
              "there and are engineering estimates consistent with the reported gap width."),
}

ALLOY_PRESETS = {
    "Cu-Pb": {
        "— jump to a typical alloy —": None,
        "Leaded bronze (10 wt% Pb)": 10.0,
        "High-leaded bronze (24 wt% Pb)": 24.0,
        "Monotectic liquid (36 wt% Pb)": 36.0,
        "Mid-gap (62 wt% Pb)": 62.0,
        "Bearing alloy (90 wt% Pb)": 90.0,
    },
    "Al-Pb": {
        "— jump to a typical alloy —": None,
        "Dilute Al-Pb bearing alloy (5 wt% Pb)": 5.0,
        "Mid-gap (40 wt% Pb)": 40.0,
        "Critical composition (79.3 wt% Pb)": 79.3,
        "Pb-rich (95 wt% Pb)": 95.0,
    },
}


@st.cache_resource(show_spinner=False)
def _diagram(key: str) -> MonotecticDiagram:
    return MonotecticDiagram(**PRESETS[key])


@st.cache_data(show_spinner=False)
def _micro_image(key: str, C0: float, T: float, size: int, seed: int):
    d = _diagram(key)
    return render_micrograph(d.micro_state(C0, T), size=size, seed=seed)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def _floor_half(t: float) -> float:
    return math.floor(t * 2) / 2


def build_figure(d: MonotecticDiagram, C0=None, T=None, events=None, tie=None, height=640) -> go.Figure:
    top = max(d.T_A, d.T_c) + 40.0
    fig = go.Figure()
    for region, verts in d.field_polygons().items():
        xs = [v[0] for v in verts] + [verts[0][0]]
        ys = [v[1] for v in verts] + [verts[0][1]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=FIELD_COLORS[region],
                                 line=dict(width=0), hoveron="fills", text=region.value, hoverinfo="text",
                                 showlegend=False))
    for name, pts in d.boundaries().items():
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines", name=name,
                                 line=dict(color="#1B1B1E", width=2), hovertemplate=f"{name}<extra></extra>",
                                 showlegend=False))
    fig.add_trace(go.Scatter(x=[d.C_L1, d.C_L2], y=[d.T_mon, d.T_mon], mode="lines", name="Monotectic",
                             line=dict(color="#8E44AD", width=3), hovertemplate="Monotectic isotherm<extra></extra>",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[d.C_alpha_max, d.C_beta_min], y=[d.T_eu, d.T_eu], mode="lines", name="Eutectic",
                             line=dict(color="#1F7A5A", width=3), hovertemplate="Eutectic isotherm<extra></extra>",
                             showlegend=False))
    for c, t, lbl in ((0, d.T_A, f"{d.component_A} — {d.T_A:.0f} °C"),
                      (100, d.T_B, f"{d.component_B} — {d.T_B:.0f} °C")):
        fig.add_trace(go.Scatter(x=[c], y=[t], mode="markers", marker=dict(size=7, color="#1B1B1E"),
                                 hovertemplate=f"{lbl}<extra></extra>", showlegend=False))

    annotations = [dict(x=25, y=top - 60, text="Liquid (L)", showarrow=False, font=dict(size=13)),
                  dict(x=(d.C_L1 + d.C_crit) / 2, y=(d.T_mon + d.T_c) / 2, text="L₁+L₂", showarrow=False,
                       font=dict(size=11)),
                  dict(x=(d.C_L1 + d.C_beta_min) / 2, y=(d.T_mon + d.T_eu) / 2 - 30, text="α + L₂",
                       showarrow=False, font=dict(size=12)),
                  dict(x=(d.C_alpha_max + d.C_beta_min) / 2, y=(d.T_eu + 25) * 0.4, text="α + β",
                       showarrow=False, font=dict(size=12))]

    if C0 is not None:
        fig.add_trace(go.Scatter(x=[C0, C0], y=[25, top], mode="lines", name=f"Alloy {C0:.2f} wt%",
                                 line=dict(color=TIE_RED, width=2, dash="dash"), hoverinfo="skip"))
        if events:
            ev_plot = [e for e in events if 25.0 <= e.temperature <= top]
            fig.add_trace(go.Scatter(
                x=[C0] * len(ev_plot), y=[e.temperature for e in ev_plot], mode="markers+text",
                text=[str(i) for i in range(1, len(ev_plot) + 1)], textfont=dict(color="white", size=11),
                marker=dict(size=19, color=TIE_RED, line=dict(color="white", width=1)),
                customdata=[[e.title, e.reaction] for e in ev_plot],
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>", showlegend=False,
            ))
        if T is not None:
            fig.add_trace(go.Scatter(x=[C0], y=[T], mode="markers", name=f"Now: {T:.0f} °C",
                                     marker=dict(size=17, symbol="star", color="black",
                                                 line=dict(color="white", width=1.5)),
                                     hovertemplate=f"{C0:.2f} wt%, {T:.0f} °C<extra></extra>"))
    if tie is not None:
        xs = [c for _, c in tie.points]
        fig.add_trace(go.Scatter(x=[min(xs), max(xs)], y=[tie.temperature] * 2, mode="lines",
                                 line=dict(color=TIE_RED, width=3), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=[tie.temperature] * len(xs), mode="markers+text",
                                 text=[p for p, _ in tie.points], textposition="top center",
                                 marker=dict(size=9, color=TIE_RED, line=dict(color="white", width=1)),
                                 hovertemplate="%{text}: %{x:.2f}%<extra></extra>", showlegend=False))

    fig.update_layout(
        template="plotly_white", height=height, annotations=annotations,
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis=dict(title=f"Composition (wt% {d.component_B})", range=[0, 100], showgrid=True, gridcolor="#E5E7EB"),
        yaxis=dict(title="Temperature (°C)", range=[25, top], showgrid=True, gridcolor="#E5E7EB"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="closest", plot_bgcolor="white",
    )
    return fig


def _tab_diagram(d: MonotecticDiagram, key: str):
    st.markdown(f"#### {d.component_A}–{d.component_B} monotectic phase diagram")
    st.caption(f"Above {d.T_mon:.0f} °C, {d.component_A}-{d.component_B} alloys between {d.C_L1:.2f} and "
               f"{d.C_L2:.2f} wt% {d.component_B} split into two immiscible liquids — a genuine liquid "
               f"miscibility gap, closing at {d.T_c:.0f} °C.")
    fig = build_figure(d)
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Invariant reactions**")
    st.markdown(_md_table(
        ["Reaction", "T (°C)", "Equation (wt%)", "Product"],
        [["Monotectic", f"{d.T_mon:.0f}",
          f"L₁ ({d.C_L1:.2f}) → α ({d.C_alpha_max:.2f}) + L₂ ({d.C_L2:.2f})", "α + a second liquid"],
         ["Eutectic (divorced)", f"{d.T_eu:.0f}",
          f"L₂ ({d.C_eu:.2f}) → α ({d.C_alpha_max:.2f}) + β ({d.C_beta_min:.2f})",
          "fine α particles in continuous β"]]))
    with st.expander("Why the eutectic is 'divorced'"):
        st.markdown(
            f"The eutectic composition ({d.C_eu:.2f} wt% {d.component_B}) sits extremely close to pure "
            f"{d.component_B} — both terminal solids dissolve almost none of the other element. So instead "
            "of the usual alternating lamellae, the eutectic liquid deposits an overwhelming amount of β "
            "with only scattered, fine α particles: a 'divorced' eutectic."
        )
    with st.expander("Data source"):
        st.caption(DATA_NOTES[key])


def _tab_tie(d: MonotecticDiagram, key: str):
    st.markdown("#### Tie-line & lever rule")
    top = max(d.T_A, d.T_c) + 40.0
    c1, c2 = st.columns([1.35, 1])
    with c1:
        C0 = st.slider(f"Composition C₀ (wt% {d.component_B})", 0.2, 99.8, 20.0, 0.2, key=f"{key}2_c0")
        T = st.slider("Temperature (°C)", 25.0, top, min(950.0, top - 10), 1.0, key=f"{key}2_T")
        region = d.phase_at(C0, T)
        tie = None
        try:
            tie = d.tie_line(C0, T)
        except MonotecticError:
            pass
        fig = build_figure(d, C0=C0, T=T, tie=tie, height=560)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.metric("Phase region", region.value)
        st.caption(d.common_name(C0))
        if tie is None:
            st.success(f"Single phase: 100 % {region.value}.")
        else:
            st.markdown(_md_table(["Tie-line end", "wt%"], [[p, f"{c:.2f}"] for p, c in tie.points]))
            fr = d.phase_fractions(C0, T)
            cols = st.columns(len(fr))
            for col, (p, v) in zip(cols, fr.items()):
                col.metric(f"W({p})", f"{v*100:.1f} %")
        st.image(_micro_image(key, round(C0, 2), round(T, 1), 300, 7), caption=f"Schematic micrograph at {T:.0f} °C")


def _tab_micro(d: MonotecticDiagram, key: str):
    top = max(d.T_A, d.T_c) + 40.0
    c0_key, t_key, preset_key = f"{key}_c0", f"{key}_T", f"{key}_preset"
    st.session_state.setdefault(c0_key, 20.0)
    st.session_state.setdefault(t_key, 25.0)
    st.session_state.setdefault(preset_key, next(iter(ALLOY_PRESETS[key])))

    def _preset_cb():
        v = ALLOY_PRESETS[key].get(st.session_state[preset_key])
        if v is not None:
            st.session_state[c0_key] = v
            st.session_state[t_key] = 25.0

    def _step(direction: int):
        temps = [s[1] for s in d.evolution_stages(st.session_state[c0_key])]
        T = st.session_state[t_key]
        if direction > 0:
            nxt = [t for t in temps if t < T - 0.6]
            st.session_state[t_key] = _floor_half(nxt[0]) if nxt else 25.0
        else:
            prv = [t for t in temps if t > T + 0.6]
            st.session_state[t_key] = _floor_half(prv[-1]) if prv else _floor_half(temps[0])

    st.markdown("#### Microstructure evolution on cooling")
    st.caption("Slow (equilibrium) cooling from the melt. Watch for the liquid miscibility gap (droplets of "
               "one liquid dispersed in the other), the monotectic reaction, and the divorced eutectic.")

    top1, top2 = st.columns([1.2, 1])
    with top1:
        st.selectbox("Typical alloys", list(ALLOY_PRESETS[key]), key=preset_key, on_change=_preset_cb,
                     label_visibility="collapsed")
        C0 = st.slider(f"Composition C₀ (wt% {d.component_B})", 0.2, 99.8, step=0.2, key=c0_key)
    with top2:
        b1, b2 = st.columns(2)
        b1.button("⬆ Previous stage", key=f"{key}_prev", on_click=_step, args=(-1,), use_container_width=True)
        b2.button("Next stage ⬇", key=f"{key}_next", on_click=_step, args=(1,), use_container_width=True)
        T = st.slider("Temperature (°C)", 25.0, top, step=0.5, key=t_key, format="%.1f")

    try:
        events = d.transformation_path(C0)
    except MonotecticError as exc:
        st.error(str(exc))
        return
    ms = d.micro_state(C0, T)

    m1, m2 = st.columns([1.3, 1])
    m1.metric("Alloy", d.common_name(C0))
    m2.metric("Phase region at T", ms.region)

    left, right = st.columns([1.15, 1])
    with left:
        fig = build_figure(d, C0=C0, T=T, events=events, height=520)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.image(_micro_image(key, round(C0, 2), round(T, 1), 440, 7),
                 caption=f"{C0:.2f} wt% {d.component_B} at {T:.0f} °C — {ms.region}  (schematic, not to scale)")
        st.markdown(legend_html(ms), unsafe_allow_html=True)

    st.markdown("##### Cooling sequence")
    rows = [[str(i), e.temp_text, f"**{e.title}**<br>{e.reaction}", e.phases_after, e.microstructure]
            for i, e in enumerate(events, 1)]
    st.markdown(_md_table(["#", "T (°C)", "Event / reaction", "Phases after", "What forms"], rows),
                unsafe_allow_html=True)

    st.markdown("##### Evolution strip")
    stages = d.evolution_stages(C0)
    for start in range(0, len(stages), 4):
        cols = st.columns(4)
        for col, (title, t_s, phases) in zip(cols, stages[start:start + 4]):
            col.image(_micro_image(key, round(C0, 2), round(_floor_half(t_s), 1), 260, 7))
            col.caption(f"**{title}**  \n{t_s:.0f} °C · {phases}")

    st.markdown("##### Room-temperature result")
    rt = d.micro_state(C0, 25.0)
    r1, r2 = st.columns([1, 1])
    with r1:
        st.markdown(_md_table(["Constituent", "wt %"], [[n, f"{v*100:.1f}"] for n, v in rt.constituents.items()]))
    with r2:
        st.image(_micro_image(key, round(C0, 2), 25.0, 360, 7), caption="Room-temperature microstructure")
        st.markdown(legend_html(rt), unsafe_allow_html=True)


def _tab_report(d: MonotecticDiagram, key: str):
    st.markdown("#### Automated report")
    C0 = st.session_state.get(f"{key}_c0", 20.0)
    T = st.session_state.get(f"{key}_T", 25.0)
    st.write(f"Report for **{C0:.2f} wt% {d.component_B}** analysed at **{T:.0f} °C** — change these in the "
             "*Microstructure Evolution* tab.")
    if st.button("Generate report", key=f"{key}_gen"):
        with st.spinner("Rendering diagram and micrographs…"):
            st.session_state[f"{key}_report_html"] = build_monotectic_html_report(d, C0, T, data_note=DATA_NOTES[key])
            st.session_state[f"{key}_report_name"] = f"PhaseForge_{key}_{C0:.1f}wt.html"
    html_doc = st.session_state.get(f"{key}_report_html")
    if html_doc:
        st.download_button("⬇ Download HTML report", html_doc, file_name=st.session_state[f"{key}_report_name"],
                           mime="text/html")
        st.caption("Self-contained file: diagram, cooling sequence, micrographs and results.")


def render_monotectic(key: str):
    d = _diagram(key)
    t1, t2, t3, t4 = st.tabs([f"📊 {key} Diagram", "📏 Tie-Line & Lever Rule",
                              "🔬 Microstructure Evolution", "📄 Report"])
    with t1:
        _tab_diagram(d, key)
    with t2:
        _tab_tie(d, key)
    with t3:
        _tab_micro(d, key)
    with t4:
        _tab_report(d, key)
