"""
ordering_ui.py — Streamlit tabs for the isomorphous-with-ordering systems
(Cu-Au, Cu-Pd). Same four-tab layout as the other special systems.
"""

from __future__ import annotations

import math
from typing import List

import plotly.graph_objects as go
import streamlit as st

from .microstructure import legend_html, render_micrograph
from .ordering import OrderingDiagram, OrderingError, PRESETS
from .ordering_report import build_ordering_html_report, region_color

TIE_RED = "#C0392B"
LABELS = {"Cu-Au": "Cu–Au  (Isomorphous + ordering)", "Cu-Pd": "Cu–Pd  (Isomorphous + ordering)"}

ALLOY_PRESETS = {
    "Cu-Au": {"— jump to a typical alloy —": None,
              "Cu₃Au stoichiometric (50.8 wt% Au)": 50.83,
              "CuAu stoichiometric (75.6 wt% Au)": 75.62,
              "CuAu₃ stoichiometric (90.3 wt% Au)": 90.30,
              "Congruent-minimum alloy (80 wt% Au)": 80.0,
              "18 carat gold (75 wt% Au)": 75.0,
              "Never orders (30 wt% Au)": 30.0},
    "Cu-Pd": {"— jump to a typical alloy —": None,
              "Cu₃Pd stoichiometric (35.8 wt% Pd)": 35.8,
              "CuPd (B2) stoichiometric (62.6 wt% Pd)": 62.6,
              "H₂-membrane alloy (~60 wt% Pd)": 60.0,
              "Never orders (15 wt% Pd)": 15.0},
}


@st.cache_resource(show_spinner=False)
def _diagram(key: str) -> OrderingDiagram:
    return OrderingDiagram(**PRESETS[key])


@st.cache_data(show_spinner=False)
def _micro_image(key: str, C0: float, T: float, size: int, seed: int):
    return render_micrograph(_diagram(key).micro_state(C0, T), size=size, seed=seed)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def _floor_half(t: float) -> float:
    return math.floor(t * 2) / 2


def build_figure(d: OrderingDiagram, C0=None, T=None, events=None, tie=None, height=640) -> go.Figure:
    top = max(d.T_A, d.T_B) + 60.0
    fig = go.Figure()
    for name, verts in d.field_polygons().items():
        xs = [v[0] for v in verts] + [verts[0][0]]
        ys = [v[1] for v in verts] + [verts[0][1]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=region_color(d, name),
                                 line=dict(width=0), hoveron="fills", text=name, hoverinfo="text", showlegend=False))
    for name, pts in d.boundaries().items():
        dashed = "order" in name
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines", name=name,
                                 line=dict(color="#1B1B1E", width=1.6 if dashed else 2.2, dash="dash" if dashed else None),
                                 hovertemplate=f"{name}<extra></extra>", showlegend=False))
    fig.add_trace(go.Scatter(x=[0, 100], y=[d.T_A, d.T_B], mode="markers", marker=dict(size=7, color="#1B1B1E"),
                             hovertemplate=[f"{d.component_A} — {d.T_A:.0f} °C<extra></extra>",
                                            f"{d.component_B} — {d.T_B:.0f} °C<extra></extra>"], showlegend=False))
    ann = [dict(x=50, y=top - 30, text="Liquid (L)", showarrow=False, font=dict(size=13))]
    for dome in d.domes:
        ann.append(dict(x=dome.center, y=dome.Tc + 45, text=f"{dome.label}<br>{dome.Tc:.0f} °C",
                        showarrow=False, font=dict(size=10)))
    if C0 is not None:
        fig.add_trace(go.Scatter(x=[C0, C0], y=[25, top], mode="lines", name=f"Alloy {C0:.2f} wt%",
                                 line=dict(color=TIE_RED, width=2, dash="dash"), hoverinfo="skip"))
        ev_plot = [e for e in (events or []) if 25.0 <= e.temperature <= top]
        if ev_plot:
            fig.add_trace(go.Scatter(
                x=[C0] * len(ev_plot), y=[e.temperature for e in ev_plot], mode="markers+text",
                text=[str(i) for i in range(1, len(ev_plot) + 1)], textfont=dict(color="white", size=11),
                marker=dict(size=19, color=TIE_RED, line=dict(color="white", width=1)),
                customdata=[[e.title, e.reaction] for e in ev_plot],
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>", showlegend=False))
        if T is not None:
            fig.add_trace(go.Scatter(x=[C0], y=[T], mode="markers", name=f"Now: {T:.0f} °C",
                                     marker=dict(size=17, symbol="star", color="black", line=dict(color="white", width=1.5)),
                                     hovertemplate=f"{C0:.2f} wt%, {T:.0f} °C<extra></extra>"))
    if tie is not None:
        xs = [c for _, c in tie.points]
        fig.add_trace(go.Scatter(x=xs, y=[tie.temperature] * 2, mode="lines+markers+text",
                                 text=[p for p, _ in tie.points], textposition="top center",
                                 line=dict(color=TIE_RED, width=3), marker=dict(size=9, color=TIE_RED),
                                 hovertemplate="%{text}: %{x:.2f}%<extra></extra>", showlegend=False))
    fig.update_layout(
        template="plotly_white", height=height, annotations=ann, margin=dict(l=60, r=20, t=30, b=50),
        xaxis=dict(title=f"Composition (wt% {d.component_B})", range=[0, 100], showgrid=True, gridcolor="#E5E7EB"),
        yaxis=dict(title="Temperature (°C)", range=[25, top], showgrid=True, gridcolor="#E5E7EB"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="closest", plot_bgcolor="white")
    return fig


def _tab_diagram(d: OrderingDiagram, key: str):
    st.markdown(f"#### {d.component_A}–{d.component_B} phase diagram (isomorphous with ordering)")
    st.caption("Above the ordering temperatures this is an ordinary isomorphous system — one disordered FCC solid "
               "solution. Below them, the atoms sort themselves into regular superlattices at specific compositions.")
    st.plotly_chart(build_figure(d), width="stretch")
    rows = [[f"{x.label}", x.structure, f"{x.center:.1f}", f"{x.Tc:.0f}"] for x in d.domes]
    st.markdown("**Ordered phases**")
    st.markdown(_md_table(["Phase", "Structure", f"wt% {d.component_B}", "Order→disorder (°C)"], rows))
    if d.c_min is not None:
        st.info(f"The liquidus and solidus touch at a congruent minimum near {d.c_min:.0f} wt% {d.component_B} "
                f"({float(d.liquidus_T(d.c_min)):.0f} °C): the melting range splits into two lenses, and tie lines "
                "on each side use that side's curves.")
    with st.expander("Data source and simplifications"):
        st.caption(d.data_note)


def _tab_tie(d: OrderingDiagram, key: str):
    st.markdown("#### Tie-line & lever rule")
    top = max(d.T_A, d.T_B) + 60.0
    c1, c2 = st.columns([1.35, 1])
    with c1:
        C0 = st.slider(f"Composition C₀ (wt% {d.component_B})", 0.5, 99.5, 40.0, 0.5, key=f"{key}2_c0")
        T = st.slider("Temperature (°C)", 25.0, top, float(min(max(d.solidus_T(40.0), 25.0) + 10, top)), 1.0,
                      key=f"{key}2_T")
        tie = None
        try:
            tie = d.tie_line(C0, T)
        except OrderingError:
            pass
        st.plotly_chart(build_figure(d, C0=C0, T=T, tie=tie, height=560), width="stretch")
    with c2:
        region = d.phase_at(C0, T)
        st.metric("Phase region", region)
        st.caption(d.common_name(C0))
        if tie is None:
            st.success(f"Single phase: 100 % {region}.")
        else:
            st.markdown(_md_table(["Tie-line end", "wt%"], [[p, f"{c:.2f}"] for p, c in tie.points]))
            fr = d.phase_fractions(C0, T)
            cols = st.columns(len(fr))
            for col, (p, v) in zip(cols, fr.items()):
                col.metric(f"W({p})", f"{v*100:.1f} %")
        st.image(_micro_image(key, round(C0, 2), round(T, 1), 300, 7), caption=f"Schematic micrograph at {T:.0f} °C")


def _tab_micro(d: OrderingDiagram, key: str):
    top = max(d.T_A, d.T_B) + 60.0
    c0_key, t_key, p_key = f"{key}_c0", f"{key}_T", f"{key}_preset"
    st.session_state.setdefault(c0_key, d.domes[0].center)
    st.session_state.setdefault(t_key, 25.0)
    st.session_state.setdefault(p_key, next(iter(ALLOY_PRESETS[key])))

    def _preset_cb():
        v = ALLOY_PRESETS[key].get(st.session_state[p_key])
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
    st.caption("Freezing is the same as Cu-Ni: dendrites, then disordered grains. What is new is the ordering "
               "transition — same grains, but the atoms inside them rearrange into a superlattice (drawn as a "
               "fine domain pattern).")
    t1, t2 = st.columns([1.2, 1])
    with t1:
        st.selectbox("Typical alloys", list(ALLOY_PRESETS[key]), key=p_key, on_change=_preset_cb,
                     label_visibility="collapsed")
        C0 = st.slider(f"Composition C₀ (wt% {d.component_B})", 0.5, 99.5, step=0.01, key=c0_key)
    with t2:
        b1, b2 = st.columns(2)
        b1.button("⬆ Previous stage", key=f"{key}_prev", on_click=_step, args=(-1,), use_container_width=True)
        b2.button("Next stage ⬇", key=f"{key}_next", on_click=_step, args=(1,), use_container_width=True)
        T = st.slider("Temperature (°C)", 25.0, top, step=0.5, key=t_key, format="%.1f")

    try:
        events = d.transformation_path(C0)
    except OrderingError as exc:
        st.error(str(exc))
        return
    ms = d.micro_state(C0, T)
    m1, m2 = st.columns([1.3, 1])
    m1.metric("Alloy", d.common_name(C0))
    m2.metric("Phase region at T", ms.region)

    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(build_figure(d, C0=C0, T=T, events=events, height=520), width="stretch")
        st.caption("Numbered red markers match the cooling sequence below; the star is the current temperature.")
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


def _tab_report(d: OrderingDiagram, key: str):
    st.markdown("#### Automated report")
    C0 = st.session_state.get(f"{key}_c0", d.domes[0].center)
    T = st.session_state.get(f"{key}_T", 25.0)
    st.write(f"Report for **{C0:.2f} wt% {d.component_B}** analysed at **{T:.0f} °C** — change these in the "
             "*Microstructure Evolution* tab.")
    if st.button("Generate report", key=f"{key}_gen"):
        with st.spinner("Rendering diagram and micrographs…"):
            st.session_state[f"{key}_report_html"] = build_ordering_html_report(d, C0, T)
            st.session_state[f"{key}_report_name"] = f"PhaseForge_{key}_{C0:.1f}wt.html"
    doc = st.session_state.get(f"{key}_report_html")
    if doc:
        st.download_button("⬇ Download HTML report", doc, file_name=st.session_state[f"{key}_report_name"],
                           mime="text/html")


def render_ordering(key: str):
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
