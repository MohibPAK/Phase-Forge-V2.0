"""
eutectic_micro_ui.py — "Microstructure Evolution" tab content for the standard
binary eutectic mode (Pb-Sn, Cu-Ag, Al-Si, Custom). Called from app.py with
the currently-selected `EutecticPhaseDiagram` instance.
"""

from __future__ import annotations

import math
from typing import List

import streamlit as st

from . import eutectic_micro as em
from .microstructure import legend_html, render_micrograph


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def _floor_half(t: float) -> float:
    return math.floor(t * 2) / 2


def render_microstructure_tab(pd_system, diagram_fn=None):
    d = pd_system
    key = "eut"

    c0_key, t_key, preset_key = f"{key}_c0", f"{key}_T", f"{key}_preset"
    default_c0 = min(max(d.C_E, 0.2), 99.8)
    st.session_state.setdefault(c0_key, default_c0)
    st.session_state.setdefault(t_key, d.T_room)

    presets = {
        "— jump to a composition —": None,
        f"{d.component_A}-rich (single-phase α)": d.C_alpha_E * 0.5,
        "Hypoeutectic": (d.C_alpha_E + d.C_E) / 2,
        "Eutectic": d.C_E,
        "Hypereutectic": (d.C_E + d.C_beta_E) / 2,
        f"{d.component_B}-rich (single-phase β)": min(d.C_beta_E + (100 - d.C_beta_E) * 0.5, 99.7),
    }
    st.session_state.setdefault(preset_key, next(iter(presets)))

    def _preset_cb():
        v = presets.get(st.session_state[preset_key])
        if v is not None:
            st.session_state[c0_key] = round(min(max(v, 0.2), 99.8), 2)
            st.session_state[t_key] = d.T_room

    def _step(direction: int):
        temps = [s[1] for s in em.evolution_stages(d, st.session_state[c0_key])]
        T = st.session_state[t_key]
        if direction > 0:
            nxt = [t for t in temps if t < T - 0.6]
            st.session_state[t_key] = _floor_half(nxt[0]) if nxt else d.T_room
        else:
            prv = [t for t in temps if t > T + 0.6]
            st.session_state[t_key] = _floor_half(prv[-1]) if prv else _floor_half(temps[0])

    st.subheader("Microstructure Evolution")
    st.caption("Slow (equilibrium) cooling from the melt. Watch for primary dendrites forming first "
               "(off-eutectic alloys), then the eutectic reaction filling in the rest as fine α/β lamellae.")

    top1, top2 = st.columns([1.2, 1])
    with top1:
        st.selectbox("Typical compositions", list(presets), key=preset_key, on_change=_preset_cb,
                     label_visibility="collapsed")
        C0 = st.slider(f"Composition C₀ (wt% {d.component_B})", 0.2, 99.8, step=0.2, key=c0_key)
    with top2:
        b1, b2 = st.columns(2)
        b1.button("⬆ Previous stage", key=f"{key}_prev", on_click=_step, args=(-1,), use_container_width=True)
        b2.button("Next stage ⬇", key=f"{key}_next", on_click=_step, args=(1,), use_container_width=True)
        T_top = min(max(d.T_A, d.T_B) + 40.0, 2000.0)
        T = st.slider("Temperature (°C)", float(d.T_room), T_top, step=0.5, key=t_key, format="%.1f")

    try:
        events = em.transformation_path(d, C0)
    except Exception as exc:
        st.error(str(exc))
        return
    ms = em.micro_state(d, C0, T)

    m1, m2 = st.columns([1.3, 1])
    m1.metric("Alloy", em.common_name(d, C0))
    m2.metric("Phase region at T", ms.region)

    left, right = st.columns([1.15, 1])
    with left:
        if diagram_fn is not None:
            import plotly.graph_objects as go
            fig = diagram_fn(d, marker=(C0, T))
            fig.update_layout(height=520)
            T_top = min(max(d.T_A, d.T_B) + 40.0, 2000.0)
            fig.add_trace(go.Scatter(x=[C0, C0], y=[d.T_room, T_top], mode="lines", name=f"Alloy {C0:.2f} wt%",
                                     line=dict(color="#C0392B", width=2, dash="dash"), hoverinfo="skip"))
            ev_plot = [e for e in events if d.T_room <= e.temperature <= T_top]
            fig.add_trace(go.Scatter(
                x=[C0] * len(ev_plot), y=[e.temperature for e in ev_plot], mode="markers+text",
                text=[str(i) for i in range(1, len(ev_plot) + 1)], textfont=dict(color="white", size=11),
                marker=dict(size=19, color="#C0392B", line=dict(color="white", width=1)),
                customdata=[[e.title, e.reaction] for e in ev_plot],
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
                showlegend=False))
            st.plotly_chart(fig, width="stretch")
            st.caption("Numbered red markers match the cooling sequence below; the star is the current temperature.")
    with right:
        st.image(render_micrograph(ms, size=440, seed=7),
                 caption=f"{C0:.2f} wt% {d.component_B} at {T:.0f} °C — {ms.region}  (schematic, not to scale)")
        st.markdown(legend_html(ms), unsafe_allow_html=True)

    st.markdown("##### Cooling sequence")
    rows = [[str(i), e.temp_text, f"**{e.title}**<br>{e.reaction}", e.phases_after, e.microstructure]
            for i, e in enumerate(events, 1)]
    st.markdown(_md_table(["#", "T (°C)", "Event / reaction", "Phases after", "What forms"], rows),
                unsafe_allow_html=True)

    st.markdown("##### Evolution strip")
    stages = em.evolution_stages(d, C0)
    for start in range(0, len(stages), 4):
        cols = st.columns(4)
        for col, (title, t_s, phases) in zip(cols, stages[start:start + 4]):
            st_ms = em.micro_state(d, C0, _floor_half(t_s))
            col.image(render_micrograph(st_ms, size=260, seed=7))
            col.caption(f"**{title}**  \n{t_s:.0f} °C · {phases}")

    st.markdown("##### Room-temperature result")
    rt = em.micro_state(d, C0, d.T_room)
    r1, r2 = st.columns([1, 1])
    with r1:
        st.markdown(_md_table(["Constituent", "wt %"], [[n, f"{v*100:.1f}"] for n, v in rt.constituents.items()]))
    with r2:
        st.image(render_micrograph(rt, size=360, seed=7), caption="Room-temperature microstructure", width=360)
        st.markdown(legend_html(rt), unsafe_allow_html=True)
