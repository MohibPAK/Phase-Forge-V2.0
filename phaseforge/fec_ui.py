"""
fec_ui.py — Streamlit interface for the Iron–Carbon (Fe–Fe₃C) system.

`render_iron_carbon()` draws four tabs inside the main PhaseForge app:

    1. Fe–Fe₃C Diagram          interactive diagram, critical points, invariant reactions
    2. Tie-Line & Lever Rule    phase fractions at any (C₀, T) with the working shown
    3. Microstructure Evolution cool an alloy and watch the micrograph change stage by stage
    4. Report                   downloadable HTML report
"""

from __future__ import annotations

import math
from typing import List, Optional

import plotly.graph_objects as go
import streamlit as st

from .fec_layout import CLASS_BANDS, INVARIANT_COLORS, LETTER_POSITION, VIEWS
from .fec_report import build_fec_html_report
from .ironcarbon import (
    C_C, C_CEM, C_E, C_P, C_S, FIELD_COLORS, POINT_NOTES, POINTS, T_EUD, T_EUT, T_PER,
    FeCError, FeCRegion, IronCarbonDiagram, fmt_c,
)
from .microstructure import legend_html, render_micrograph

ACCENT = "#2E5EAA"
TIE_RED = "#C0392B"
FEC_LABEL = "Fe–C  (Iron–Carbon: steels & cast irons)"

VIEW_KEYS = {v["title"]: k for k, v in VIEWS.items()}

ALLOY_PRESETS = {
    "— jump to a typical alloy —": None,
    "Ferritic iron  (0.01 wt% C)": 0.01,
    "Peritectic alloy  (0.17 wt% C)": 0.17,
    "Low-carbon steel  (0.20 wt% C)": 0.20,
    "Medium-carbon steel  (0.40 wt% C)": 0.40,
    "Eutectoid steel  (0.76 wt% C)": 0.76,
    "Hypereutectoid steel  (1.20 wt% C)": 1.20,
    "Hypoeutectic cast iron  (3.00 wt% C)": 3.00,
    "Eutectic cast iron  (4.30 wt% C)": 4.30,
    "Hypereutectic cast iron  (5.50 wt% C)": 5.50,
}

_LATEX = {"L": "L", "δ": r"\delta", "γ": r"\gamma", "α": r"\alpha", "Fe₃C": r"\mathrm{Fe_3C}"}


# ---------------------------------------------------------------------------
# cached helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _diagram() -> IronCarbonDiagram:
    return IronCarbonDiagram()


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
    """Round down to the nearest 0.5 °C so a stage never lands exactly on an invariant isotherm."""
    return math.floor(t * 2) / 2


def _marker_T(e) -> float:
    return e.t_sample if e.kind == "Acm_ci" else e.temperature


# ---------------------------------------------------------------------------
# Plotly diagram
# ---------------------------------------------------------------------------
def build_figure(
    d: IronCarbonDiagram,
    view_key: str = "full",
    show_letters: bool = True,
    show_bands: bool = True,
    show_magnetic: bool = False,
    C0: Optional[float] = None,
    T: Optional[float] = None,
    events=None,
    tie=None,
    height: int = 640,
) -> go.Figure:
    view = VIEWS[view_key]
    x0, x1 = view["x"]
    fig = go.Figure()

    for region, verts in d.field_polygons().items():
        xs = [v[0] for v in verts] + [verts[0][0]]
        ys = [v[1] for v in verts] + [verts[0][1]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself", fillcolor=FIELD_COLORS[region],
            line=dict(width=0), hoveron="fills", text=region.value, hoverinfo="text", showlegend=False,
        ))

    for name, pts in d.boundaries().items():
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines", name=name,
            line=dict(color="#1B1B1E", width=2), hovertemplate=f"{name}<extra></extra>", showlegend=False,
        ))
    for name, (c0_, c1_, t_) in d.invariant_lines().items():
        fig.add_trace(go.Scatter(
            x=[c0_, c1_], y=[t_, t_], mode="lines", name=name,
            line=dict(color=INVARIANT_COLORS[name], width=3),
            hovertemplate=f"{name}<extra></extra>", showlegend=False,
        ))
    if show_magnetic:
        for name, (c0_, c1_, t_) in d.magnetic_lines().items():
            fig.add_trace(go.Scatter(
                x=[c0_, c1_], y=[t_, t_], mode="lines", name=name,
                line=dict(color="#6B7280", width=1.2, dash="dash"),
                hovertemplate=f"{name}<extra></extra>", showlegend=False,
            ))

    annotations = []
    for x, y, txt, size in view["labels"]:
        annotations.append(dict(x=x, y=y, text=txt, showarrow=False, font=dict(size=size, color="#1B1B1E")))
    for tx, ty, txt, dx, dy in view["tags"]:
        annotations.append(dict(
            x=tx, y=ty, text=txt, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="#555555",
            ax=dx, ay=dy, font=dict(size=11, color="#1B1B1E"),
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#999999", borderwidth=1, borderpad=2,
        ))

    if show_letters:
        letters = view["letters"]
        fig.add_trace(go.Scatter(
            x=[POINTS[k][0] for k in letters], y=[POINTS[k][1] for k in letters],
            mode="markers+text", text=letters, textposition=[LETTER_POSITION[k] for k in letters],
            textfont=dict(size=11, color="#333333"), marker=dict(size=6, color="#1B1B1E"),
            customdata=[[k, POINT_NOTES[k]] for k in letters],
            hovertemplate="<b>%{customdata[0]}</b> — %{customdata[1]}<br>%{x} wt% C, %{y} °C<extra></extra>",
            showlegend=False,
        ))

    shapes = []
    if show_bands:
        for c_lo, c_hi, label in CLASS_BANDS:
            if c_lo < x1:
                shapes.append(dict(type="line", xref="x", yref="paper", x0=c_lo, x1=c_lo, y0=0, y1=1,
                                   line=dict(color="#777777", width=1, dash="dot")))
            mid = (max(c_lo, x0) + min(c_hi, x1)) / 2
            if c_lo < x1 and c_hi > x0 and (min(c_hi, x1) - max(c_lo, x0)) > 0.12 * (x1 - x0):
                annotations.append(dict(x=mid, y=1.0, xref="x", yref="paper", yanchor="bottom",
                                        text=label, showarrow=False, font=dict(size=10, color="#555555")))

    if C0 is not None:
        fig.add_trace(go.Scatter(
            x=[C0, C0], y=[25, 1600], mode="lines", name=f"Alloy {C0:.2f} wt% C",
            line=dict(color=TIE_RED, width=2, dash="dash"), hoverinfo="skip",
        ))
        if events:
            fig.add_trace(go.Scatter(
                x=[C0] * len(events), y=[_marker_T(e) for e in events], mode="markers+text",
                text=[str(i) for i in range(1, len(events) + 1)], textfont=dict(color="white", size=11),
                marker=dict(size=19, color=TIE_RED, line=dict(color="white", width=1)),
                customdata=[[e.title, e.reaction] for e in events],
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
                name="Transformation events", showlegend=False,
            ))
        if T is not None:
            fig.add_trace(go.Scatter(
                x=[C0], y=[T], mode="markers", name=f"Now: {T:.0f} °C",
                marker=dict(size=17, symbol="star", color="black", line=dict(color="white", width=1.5)),
                hovertemplate=f"{C0:.2f} wt% C, {T:.0f} °C<extra></extra>",
            ))

    if tie is not None:
        xs = [c for _, c in tie.points]
        fig.add_trace(go.Scatter(
            x=[xs[0], xs[-1]], y=[tie.temperature] * 2, mode="lines", line=dict(color=TIE_RED, width=3),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=[tie.temperature] * len(xs), mode="markers+text",
            text=[p for p, _ in tie.points], textposition="top center",
            marker=dict(size=9, color=TIE_RED, line=dict(color="white", width=1)),
            hovertemplate="%{text}: %{x:.3f} wt% C<extra></extra>", showlegend=False,
        ))

    fig.update_layout(
        template="plotly_white", height=height, annotations=annotations, shapes=shapes,
        margin=dict(l=60, r=20, t=80 if show_bands else 30, b=50),
        xaxis=dict(title="Carbon content (wt% C)", range=list(view["x"]), showgrid=True, gridcolor="#E5E7EB", zeroline=False),
        yaxis=dict(title="Temperature (°C)", range=list(view["y"]), showgrid=True, gridcolor="#E5E7EB", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02 if not show_bands else -0.2, x=0),
        hovermode="closest", plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# worked examples
# ---------------------------------------------------------------------------
def _lever_latex(tl, C0: float) -> str:
    (pl, cl), (pr, cr) = tl.points[0], tl.points[-1]
    L, R = _LATEX[pl], _LATEX[pr]
    f_l = (cr - C0) / (cr - cl)
    f_r = (C0 - cl) / (cr - cl)
    return (
        rf"W_{{{L}}} = \frac{{C_{{{R}}} - C_0}}{{C_{{{R}}} - C_{{{L}}}}} = "
        rf"\frac{{{cr:.3g} - {C0:.3g}}}{{{cr:.3g} - {cl:.3g}}} = {f_l:.3f}"
        "\n\n"
        rf"W_{{{R}}} = \frac{{C_0 - C_{{{L}}}}}{{C_{{{R}}} - C_{{{L}}}}} = "
        rf"\frac{{{C0:.3g} - {cl:.3g}}}{{{cr:.3g} - {cl:.3g}}} = {f_r:.3f}"
    )


def _constituent_working(d: IronCarbonDiagram, C0: float) -> str:
    cls = d.alloy_class(C0).value
    if C0 < C_P:
        return ("Below 0.022 wt% C the alloy is essentially all ferrite; only a trace of tertiary Fe₃C "
                "appears once the solvus is crossed.")
    if abs(C0 - C_S) <= 0.005:
        return "Eutectoid composition: 100 % of the austenite becomes pearlite — no proeutectoid phase."
    if C0 < C_S:
        f = (C_S - C0) / (C_S - C_P)
        return (f"**{cls}** — just above 727 °C the tie line is α (0.022) – γ (0.76):\n\n"
                f"- Proeutectoid α = (0.76 − {C0:.2f}) / (0.76 − 0.022) = **{f*100:.1f} %**\n"
                f"- Pearlite (the γ that is left) = 1 − {f:.3f} = **{(1-f)*100:.1f} %**")
    if C0 <= C_E:
        f = (C0 - C_S) / (C_CEM - C_S)
        return (f"**{cls}** — just above 727 °C the tie line is γ (0.76) – Fe₃C (6.67):\n\n"
                f"- Proeutectoid Fe₃C = ({C0:.2f} − 0.76) / (6.67 − 0.76) = **{f*100:.1f} %**\n"
                f"- Pearlite = 1 − {f:.3f} = **{(1-f)*100:.1f} %**")
    if abs(C0 - C_C) <= 0.005:
        return "Eutectic composition: the whole melt freezes as ledeburite (γ + Fe₃C) at 1147 °C."
    if C0 < C_C:
        f_p = (C_C - C0) / (C_C - C_E)
        f_s = (C_E - C_S) / (C_CEM - C_S)
        return (f"**{cls}** — just below 1147 °C the tie line is γ (2.14) – L (4.30):\n\n"
                f"- Primary γ = (4.30 − {C0:.2f}) / (4.30 − 2.14) = **{f_p*100:.1f} %**\n"
                f"- Ledeburite = 1 − {f_p:.3f} = **{(1-f_p)*100:.1f} %**\n"
                f"- By 727 °C the primary γ has shed Fe₃C: (2.14 − 0.76) / (6.67 − 0.76) = {f_s*100:.1f} % of it "
                f"is secondary cementite, the rest becomes pearlite ({f_p*(1-f_s)*100:.1f} % of the alloy).")
    f_pl = (C0 - C_C) / (C_CEM - C_C)
    return (f"**{cls}** — just above 1147 °C the tie line is L (4.30) – Fe₃C (6.67):\n\n"
            f"- Primary Fe₃C = ({C0:.2f} − 4.30) / (6.67 − 4.30) = **{f_pl*100:.1f} %**\n"
            f"- Ledeburite = 1 − {f_pl:.3f} = **{(1-f_pl)*100:.1f} %**")


# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------
def _tab_diagram(d: IronCarbonDiagram):
    st.markdown("#### Iron–Iron Carbide (Fe–Fe₃C) phase diagram")
    st.caption("Metastable equilibrium diagram of steels and white cast irons. Hover a field for its name "
               "and a critical point for its meaning.")
    c1, c2, c3, c4 = st.columns([2.4, 1, 1, 1])
    with c1:
        view_title = st.radio("View", list(VIEW_KEYS), horizontal=True, key="fec_view1")
    with c2:
        letters = st.checkbox("Point letters", True, key="fec_letters")
    with c3:
        bands = st.checkbox("Alloy classes", True, key="fec_bands")
    with c4:
        mag = st.checkbox("A₀ / A₂ lines", False, key="fec_mag")
    fig = build_figure(d, VIEW_KEYS[view_title], letters, bands, mag)
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("**Invariant reactions**")
        st.markdown(_md_table(
            ["Reaction", "T (°C)", "Equation (wt% C)", "Product"],
            [
                ["Peritectic", "1495", "L (0.53) + δ (0.09) → γ (0.17)", "Austenite"],
                ["Eutectic", "1147", "L (4.30) → γ (2.14) + Fe₃C (6.67)", "Ledeburite"],
                ["Eutectoid", "727", "γ (0.76) → α (0.022) + Fe₃C (6.67)", "Pearlite"],
            ]))
    with b:
        st.markdown("**Alloy classes**")
        st.markdown(_md_table(
            ["Class", "wt% C"],
            [["Ferritic iron", "< 0.022"], ["Hypoeutectoid steel", "0.022 – 0.76"],
             ["Eutectoid steel", "0.76"], ["Hypereutectoid steel", "0.76 – 2.14"],
             ["Hypoeutectic white cast iron", "2.14 – 4.30"], ["Eutectic white cast iron", "4.30"],
             ["Hypereutectic white cast iron", "4.30 – 6.67"]]))

    with st.expander("Critical points (A … Q) and critical temperatures"):
        rows = [[k, fmt_c(POINTS[k][0]), f"{POINTS[k][1]:.0f}", POINT_NOTES[k]] for k in POINTS]
        st.markdown(_md_table(["Point", "wt% C", "T (°C)", "Meaning"], rows))
        st.markdown("**Critical lines:** A₁ = 727 °C (eutectoid line PSK) · A₃ = line GS · A_cm = line ES · "
                    "A₂ = 770 °C (Curie point of ferrite) · A₀ = 210 °C (Curie point of cementite).")

    with st.expander("Phases and microconstituents"):
        st.markdown(_md_table(
            ["Name", "What it is", "Carbon"],
            [
                ["δ-ferrite", "BCC iron stable at high temperature", "≤ 0.09 wt% at 1495 °C"],
                ["γ-austenite", "FCC iron, non-magnetic, dissolves a lot of carbon", "≤ 2.14 wt% at 1147 °C"],
                ["α-ferrite", "BCC iron, soft and ductile", "≤ 0.022 wt% at 727 °C"],
                ["Cementite (Fe₃C)", "Hard, brittle iron carbide compound", "6.67 wt%"],
                ["Pearlite", "Lamellae of α + Fe₃C from the eutectoid reaction", "0.76 wt% overall"],
                ["Ledeburite", "Eutectic mixture of γ + Fe₃C (turns into pearlite + Fe₃C below 727 °C)", "4.30 wt% overall"],
            ]))
        st.caption("Model limits: straight-line, textbook-level boundaries; equilibrium (slow) cooling only. "
                   "Bainite, martensite and the stable Fe–graphite system of grey/ductile irons are not modelled.")


def _tab_tie(d: IronCarbonDiagram):
    st.markdown("#### Tie-line & lever rule")
    c1, c2 = st.columns([1.35, 1])
    with c1:
        C0 = st.slider("Carbon content C₀ (wt% C)", 0.01, 6.60, 0.40, 0.01, key="fec2_c0")
        T = st.slider("Temperature (°C)", 25.0, 1600.0, 780.0, 1.0, key="fec2_T")
        region = d.phase_at(C0, T)
        try:
            tl = d.tie_line(C0, T)
        except FeCError:
            tl = None
        fig = build_figure(d, "full", True, False, False, C0=C0, T=T, tie=tl, height=560)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.metric("Phase region", region.value)
        st.caption(f"{d.alloy_class(C0).value} · {d.common_name(C0)}")
        if tl is None:
            st.success(f"Single phase: 100 % {region.value}.")
        elif tl.is_invariant:
            st.warning("Exactly on an invariant isotherm: three phases coexist while the reaction runs.")
            st.markdown(_md_table(["Phase", "wt% C"], [[p, fmt_c(c)] for p, c in tl.points]))
            above = d.phase_fractions(C0, T, "above")
            below = d.phase_fractions(C0, T, "below")
            st.markdown(_md_table(
                ["", "Just above", "Just below"],
                [[p, f"{above.get(p, 0)*100:.1f} %", f"{below.get(p, 0)*100:.1f} %"]
                 for p in dict.fromkeys(list(above) + list(below))]))
        else:
            st.markdown(_md_table(["Tie-line end", "wt% C"], [[p, fmt_c(c)] for p, c in tl.points]))
            fr = d.phase_fractions(C0, T)
            cols = st.columns(len(fr))
            for col, (p, v) in zip(cols, fr.items()):
                col.metric(f"W({p})", f"{v*100:.1f} %")
            with st.expander("Lever rule — worked out", expanded=True):
                st.latex(_lever_latex(tl, C0))
        st.image(_micro_image(round(C0, 3), round(T, 1), 300, 7), caption=f"Schematic micrograph at {T:.0f} °C")


def _tab_micro(d: IronCarbonDiagram):
    st.session_state.setdefault("fec_c0", 0.40)
    st.session_state.setdefault("fec_T", 25.0)
    st.session_state.setdefault("fec_preset", next(iter(ALLOY_PRESETS)))

    def _preset_cb():
        v = ALLOY_PRESETS.get(st.session_state.fec_preset)
        if v is not None:
            st.session_state.fec_c0 = v
            st.session_state.fec_T = 25.0

    def _step(direction: int):
        temps = [s[1] for s in d.evolution_stages(st.session_state.fec_c0)]     # descending
        T = st.session_state.fec_T
        if direction > 0:
            nxt = [t for t in temps if t < T - 0.6]
            st.session_state.fec_T = _floor_half(nxt[0]) if nxt else 25.0
        else:
            prv = [t for t in temps if t > T + 0.6]
            st.session_state.fec_T = _floor_half(prv[-1]) if prv else _floor_half(temps[0])

    st.markdown("#### Microstructure evolution on cooling")
    st.caption("Slow (equilibrium) cooling from the melt. Drag the temperature slider, or step through the "
               "transformation events. The same piece of metal is drawn at every temperature.")

    top1, top2 = st.columns([1.2, 1])
    with top1:
        st.selectbox("Typical alloys", list(ALLOY_PRESETS), key="fec_preset", on_change=_preset_cb,
                     label_visibility="collapsed")
        C0 = st.slider("Carbon content C₀ (wt% C)", 0.01, 6.60, step=0.01, key="fec_c0")
    with top2:
        b1, b2 = st.columns(2)
        b1.button("⬆ Previous stage", on_click=_step, args=(-1,), use_container_width=True)
        b2.button("Next stage ⬇", on_click=_step, args=(1,), use_container_width=True)
        T = st.slider("Temperature (°C)", 25.0, 1600.0, step=0.5, key="fec_T", format="%.1f")

    try:
        events = d.transformation_path(C0)
    except FeCError as exc:
        st.error(str(exc))
        return
    ms = d.micro_state(C0, T)
    crit = d.critical_temperatures(C0)

    m1, m2, m3 = st.columns([1.2, 1, 1])
    m1.metric("Alloy", d.alloy_class(C0).value)
    m1.caption(d.common_name(C0))
    m2.metric("Phase region at T", ms.region)
    m3.metric("Liquidus / solidus", f"{crit['Liquidus']:.0f} / {crit['Solidus']:.0f} °C")
    if any(abs(T - t) < 1e-6 for t in (T_PER, T_EUT, T_EUD)):
        st.info("Exactly on an invariant isotherm — showing the state just before the reaction starts.")

    left, right = st.columns([1.15, 1])
    with left:
        view_title = st.radio("Diagram view", list(VIEW_KEYS), horizontal=True, key="fec_view3")
        fig = build_figure(d, VIEW_KEYS[view_title], False, False, False, C0=C0, T=T, events=events, height=520)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.image(_micro_image(round(C0, 3), round(T, 1), 440, 7),
                 caption=f"{C0:.2f} wt% C at {T:.0f} °C — {ms.region}  (schematic, not to scale)")
        st.markdown(legend_html(ms), unsafe_allow_html=True)

    st.markdown("##### Cooling sequence")
    rows = [[str(i), e.temp_text, f"**{e.title}**<br>{e.reaction}", e.phases_after, e.microstructure]
            for i, e in enumerate(events, 1)]
    st.markdown(_md_table(["#", "T (°C)", "Event / reaction", "Phases after", "What forms"], rows),
                unsafe_allow_html=True)
    st.caption("Numbers match the red markers on the diagram.")

    st.markdown("##### Evolution strip")
    stages = d.evolution_stages(C0)
    for start in range(0, len(stages), 4):
        cols = st.columns(4)
        for col, (title, t_s, phases) in zip(cols, stages[start:start + 4]):
            col.image(_micro_image(round(C0, 3), round(_floor_half(t_s), 1), 260, 7))
            col.caption(f"**{title}**  \n{t_s:.0f} °C · {phases}")

    st.markdown("##### Room-temperature result")
    rt = d.micro_state(C0, 25.0)
    r1, r2 = st.columns([1, 1])
    with r1:
        st.markdown("**Microconstituents** (weight fraction)")
        st.markdown(_md_table(["Constituent", "wt %"], [[n, f"{v*100:.1f}"] for n, v in rt.constituents.items()]))
        st.markdown("**Phases** (lever rule at 25 °C)")
        st.markdown(_md_table(["Phase", "wt %"], [[p, f"{v*100:.2f}"] for p, v in rt.fractions.items()]))
    with r2:
        st.image(_micro_image(round(C0, 3), 25.0, 360, 7), caption="Room-temperature microstructure")
        st.markdown(legend_html(rt), unsafe_allow_html=True)
    with st.expander("How these fractions are calculated", expanded=True):
        st.markdown(_constituent_working(d, C0))


def _tab_report(d: IronCarbonDiagram):
    st.markdown("#### Automated report")
    C0 = st.session_state.get("fec_c0", 0.40)
    T = st.session_state.get("fec_T", 25.0)
    st.write(f"Report for **{C0:.2f} wt% C** analysed at **{T:.0f} °C** — change these in the "
             "*Microstructure Evolution* tab.")
    if st.button("Generate report", key="fec_gen"):
        with st.spinner("Rendering diagram and micrographs…"):
            st.session_state["fec_report_html"] = build_fec_html_report(d, C0, T)
            st.session_state["fec_report_name"] = f"PhaseForge_FeC_{C0:.2f}wtC.html"
    html_doc = st.session_state.get("fec_report_html")
    if html_doc:
        st.download_button("⬇ Download HTML report", html_doc, file_name=st.session_state["fec_report_name"],
                           mime="text/html")
        st.caption("Self-contained file: diagram, cooling sequence, micrographs and constituents.")


def render_iron_carbon():
    """Draw the four Fe–C tabs in the main area."""
    d = _diagram()
    t1, t2, t3, t4 = st.tabs(["📊 Fe–Fe₃C Diagram", "📏 Tie-Line & Lever Rule",
                              "🔬 Microstructure Evolution", "📄 Report"])
    with t1:
        _tab_diagram(d)
    with t2:
        _tab_tie(d)
    with t3:
        _tab_micro(d)
    with t4:
        _tab_report(d)
