"""
app.py — PhaseForge Streamlit application.

Run with:
    streamlit run app.py

A guided, tab-based interface over the phaseforge computational engine:
  1. Phase Diagram      — build/preview the binary eutectic system
  2. Tie-Line & Lever Rule — click-free point analysis at (C0, T)
  3. Solidification Path — equilibrium cooling composition trace
  4. Cooling Curve       — simulated T(t) thermal-arrest curve
  5. Report              — one-click HTML report + PNG downloads
"""

from __future__ import annotations

import io
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from phaseforge.core import EutecticPhaseDiagram, PhaseDiagramError
from phaseforge.simulation import trace_solidification_path, simulate_cooling_curve
from phaseforge.presets import PRESETS, build_preset
from phaseforge.report import build_html_report, plot_phase_diagram, fig_to_base64
from phaseforge.eutectic_micro_ui import render_microstructure_tab
from phaseforge.fec_ui import FEC_LABEL, render_iron_carbon
from phaseforge.cuni_ui import CUNI_LABEL, render_cu_ni
from phaseforge.monotectic_ui import LABELS as MONO_LABELS, render_monotectic
from phaseforge.ordering_ui import LABELS as ORD_LABELS, render_ordering

# Extra (non-eutectic) systems: label -> render function. Each one owns its
# own tabs and is fully responsible for its part of the sidebar + main area.
SPECIAL_SYSTEMS = {
    FEC_LABEL: dict(
        render=render_iron_carbon,
        caption="Iron–carbon system: steels and white cast irons on the metastable Fe–Fe₃C diagram, "
                "with peritectic, eutectic and eutectoid reactions and microstructure evolution.",
        note="Boundaries are textbook-level straight lines; equilibrium (slow) cooling only. "
             "Micrographs are schematic.",
    ),
    CUNI_LABEL: dict(
        render=render_cu_ni,
        caption="Copper-nickel: the classic isomorphous binary — complete liquid and solid solubility, "
                "a single α solid solution, and no invariant reactions at all.",
        note="Liquidus/solidus fit to standard classroom data (Callister & Rethwisch). Equilibrium "
             "cooling only; real castings show cored grains that this model does not simulate.",
    ),
    MONO_LABELS["Cu-Pb"]: dict(
        render=lambda: render_monotectic("Cu-Pb"),
        caption="Copper-lead: a monotectic system — above 954 degC, Cu-Pb alloys between 36 and 87 wt% Pb "
                "split into two immiscible liquids, then a monotectic reaction and a divorced eutectic.",
        note="Monotectic/eutectic data from Carpenter & Kullerud (1968) and ASM's Phase Diagrams reference "
             "(Campbell, ed., 2012); the critical composition is an engineering estimate.",
    ),
    ORD_LABELS["Cu-Au"]: dict(
        render=lambda: render_ordering("Cu-Au"),
        caption="Copper-gold: isomorphous with a congruent melting minimum, plus three solid-state ordered "
                "phases (Cu3Au, CuAu, CuAu3) that form below about 200-410 degC.",
        note="Liquidus/solidus from the Callister problem table; ordering temperatures from the Cu-Au literature. "
             "Dome widths are schematic; equilibrium cooling only.",
    ),
    ORD_LABELS["Cu-Pd"]: dict(
        render=lambda: render_ordering("Cu-Pd"),
        caption="Copper-palladium: isomorphous with ordered Cu3Pd (L12) and CuPd (B2) phases below ~500-600 degC.",
        note="Liquidus/solidus between the melting points is an engineering estimate (no tabulated data found); "
             "ordering data from Subramanian & Laughlin and Popov et al.",
    ),
    MONO_LABELS["Al-Pb"]: dict(
        render=lambda: render_monotectic("Al-Pb"),
        caption="Aluminum-lead: a monotectic system with an especially wide liquid miscibility gap, "
                "closing only at 1422 degC.",
        note="Monotectic/critical-point data from Ratke et al. (2011); the L2 and eutectic compositions "
             "are engineering estimates consistent with the reported gap width.",
    ),
}


# keep related systems together in the dropdown
_ORDER = [FEC_LABEL, CUNI_LABEL, MONO_LABELS["Cu-Pb"], MONO_LABELS["Al-Pb"], ORD_LABELS["Cu-Au"], ORD_LABELS["Cu-Pd"]]
SPECIAL_SYSTEMS = {k: SPECIAL_SYSTEMS[k] for k in _ORDER}

st.set_page_config(
    page_title="PhaseForge — Computational Materials Science",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
ACCENT = "#2E5EAA"
REGION_COLORS = {
    0: "#EAF4FB",  # Liquid
    1: "#E7F5EA",  # alpha
    2: "#FBEDE8",  # beta  (reuse similar palette to matplotlib version)
    3: "#CFE9F7",  # alpha+L
    4: "#F6D9CE",  # beta+L
    5: "#EFE3F5",  # alpha+beta
}

st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.6rem; max-width: 1250px; }}
    h1, h2, h3 {{ color: #1B2A4A; }}
    .pf-header {{
        display:flex; align-items:center; justify-content:space-between;
        border-bottom: 3px solid {ACCENT}; padding-bottom: 10px; margin-bottom: 6px;
    }}
    .pf-title {{ font-size: 30px; font-weight: 800; color:{ACCENT}; margin:0;}}
    .pf-subtitle {{ color:#6B7280; font-size:14px; margin-top:2px;}}
    .pf-badge {{
        background: {ACCENT}; color:white; padding:4px 12px; border-radius:999px;
        font-size:12px; font-weight:600;
    }}
    div[data-testid="stMetricValue"] {{ font-size: 22px; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #F2F5FA; border-radius: 8px 8px 0 0; padding: 10px 18px; font-weight:600;
    }}
    .stTabs [aria-selected="true"] {{ background-color: {ACCENT}; color: white; }}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="pf-header">
  <div>
    <p class="pf-title">🧪 PhaseForge</p>
    <p class="pf-subtitle">Computational Materials Science Toolkit — Binary Alloy Phase Diagrams &amp; Phase Transformations</p>
  </div>
  <div><span class="pf-badge">Developed by Mohib Mehdi(MME 24-28)</span></div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — system definition
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Alloy System")
    _systems = [k for k in PRESETS if k != "Custom"] + list(SPECIAL_SYSTEMS) + ["Custom"]
    preset_name = st.selectbox("Preset system", _systems, index=0)
    special = SPECIAL_SYSTEMS.get(preset_name)

    if special is not None:
        st.caption(special["caption"])
        st.divider()
        st.caption(special["note"])
        st.caption("Technologies: Python · NumPy · SciPy · Matplotlib · Streamlit · Plotly")
    else:
        if preset_name == "Custom":
            st.caption("Define your own binary eutectic system.")
            colA, colB = st.columns(2)
            component_A = colA.text_input("Component A", "A")
            component_B = colB.text_input("Component B", "B")
            T_A = st.number_input("Melting point of A, T_A (°C)", value=327.5, step=1.0)
            T_B = st.number_input("Melting point of B, T_B (°C)", value=231.9, step=1.0)
            T_E = st.number_input("Eutectic temperature, T_E (°C)", value=183.0, step=1.0)
            C_E = st.slider(f"Eutectic composition, C_E (wt% {component_B})", 0.1, 99.9, 61.9)
            C_alpha_E = st.slider("Max solubility of B in α at T_E (wt%)", 0.0, float(C_E) - 0.1, min(19.2, C_E - 0.1))
            C_beta_E = st.slider("Composition of β at T_E (wt%)", float(C_E) + 0.1, 100.0, max(97.5, C_E + 0.1))
            C_alpha_RT = st.slider("Room-temp solubility of B in α (wt%)", 0.0, max(float(C_alpha_E), 0.01), min(2.0, C_alpha_E))
            C_beta_RT = st.slider("Room-temp composition of β (wt%)", float(C_beta_E), 100.0, max(99.0, C_beta_E))
            params = dict(
                T_A=T_A, T_B=T_B, T_E=T_E, C_E=C_E, C_alpha_E=C_alpha_E, C_beta_E=C_beta_E,
                C_alpha_RT=C_alpha_RT, C_beta_RT=C_beta_RT,
                component_A=component_A, component_B=component_B, name=f"{component_A}-{component_B}",
            )
        else:
            params = dict(PRESETS[preset_name])
            with st.expander("View / edit parameters"):
                colA, colB = st.columns(2)
                params["T_A"] = colA.number_input("T_A (°C)", value=float(params["T_A"]))
                params["T_B"] = colB.number_input("T_B (°C)", value=float(params["T_B"]))
                params["T_E"] = st.number_input("T_E (°C)", value=float(params["T_E"]))
                params["C_E"] = st.slider("C_E (wt%)", 0.1, 99.9, float(params["C_E"]))
                params["C_alpha_E"] = st.slider("C_alpha_E (wt%)", 0.0, float(params["C_E"]) - 0.1, float(min(params["C_alpha_E"], params["C_E"] - 0.1)))
                params["C_beta_E"] = st.slider("C_beta_E (wt%)", float(params["C_E"]) + 0.1, 100.0, float(max(params["C_beta_E"], params["C_E"] + 0.1)))

        try:
            pd_system = EutecticPhaseDiagram(**params)
            st.success(f"System ready: {pd_system.name}", icon="✅")
        except PhaseDiagramError as e:
            st.error(f"Invalid parameters: {e}")
            st.stop()

        st.divider()
        st.caption(
            "PhaseForge models a standard binary eutectic topology (terminal solid "
            "solutions α, β + eutectic invariant). Preset values are illustrative "
            "textbook approximations, not measured research data."
        )
        st.caption("Technologies: Python · NumPy · SciPy · Matplotlib · Streamlit · Plotly")


if special is not None:
    special["render"]()
    st.stop()


# ---------------------------------------------------------------------------
# Shared interactive Plotly diagram builder
# ---------------------------------------------------------------------------
def make_plotly_diagram(pd: EutecticPhaseDiagram, tie_line=None, marker=None, solidification=None):
    curves = pd.sample_curves(250)
    T_top = curves["T_top"]

    Cg = np.linspace(0, 100, 140)
    Tg = np.linspace(pd.T_room, T_top, 140)
    CC, TT = np.meshgrid(Cg, Tg)
    Z = pd.region_grid(CC, TT)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=Cg, y=Tg, z=Z,
        colorscale=[[i / 5, REGION_COLORS[i]] for i in range(6)],
        zmin=0, zmax=5, showscale=False, hoverinfo="skip",
    ))

    def add_curve(key, name, dash=None, color="#1B1B1E", width=2.4):
        C, T = curves[key]
        fig.add_trace(go.Scatter(
            x=C, y=T, mode="lines", name=name,
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"%{{x:.2f}} wt%%<br>%{{y:.1f}} °C<extra>{name}</extra>",
        ))

    add_curve("liquidus_left", "Liquidus", color="#1B1B1E", width=2.6)
    add_curve("liquidus_right", "Liquidus (B-rich)", color="#1B1B1E", width=2.6)
    add_curve("solidus_alpha", "Solidus", dash="dash", color="#333333")
    add_curve("solidus_beta", "Solidus (β)", dash="dash", color="#333333")
    add_curve("solvus_alpha", "Solvus", dash="dot", color="#555555")
    add_curve("solvus_beta", "Solvus (β)", dash="dot", color="#555555")

    eC, eT = curves["eutectic_line"]
    fig.add_trace(go.Scatter(x=eC, y=eT, mode="lines", name="Eutectic isotherm",
                              line=dict(color="#8E44AD", width=2.6)))
    fig.add_trace(go.Scatter(x=[pd.C_E], y=[pd.T_E], mode="markers+text", name="Eutectic point",
                              text=[f"  Eutectic {pd.C_E:.1f}%, {pd.T_E:.0f}°C"], textposition="bottom right",
                              marker=dict(color="#8E44AD", size=9), showlegend=False))

    if solidification is not None:
        Ts = [s.temperature for s in solidification.steps if s.liquid_comp is not None]
        Cs_liq = [s.liquid_comp for s in solidification.steps if s.liquid_comp is not None]
        Cs_sol = [s.solid_comp for s in solidification.steps if s.solid_comp is not None]
        Ts_sol = [s.temperature for s in solidification.steps if s.solid_comp is not None]
        fig.add_trace(go.Scatter(x=Cs_liq, y=Ts, mode="lines", name="Liquid composition path",
                                  line=dict(color="#2E86AB", width=3)))
        fig.add_trace(go.Scatter(x=Cs_sol, y=Ts_sol, mode="lines", name="Solid composition path",
                                  line=dict(color="#B8860B", width=3)))

    if tie_line is not None:
        fig.add_trace(go.Scatter(
            x=[tie_line.c_left, tie_line.c_right], y=[tie_line.temperature] * 2,
            mode="lines+markers+text", name="Tie line",
            line=dict(color="#C0392B", width=3),
            marker=dict(size=8, color="#C0392B"),
            text=[tie_line.label_left, tie_line.label_right],
            textposition=["middle left", "middle right"],
        ))

    if marker is not None:
        fig.add_trace(go.Scatter(
            x=[marker[0]], y=[marker[1]], mode="markers", name="Selected point",
            marker=dict(symbol="star", size=16, color="black", line=dict(color="white", width=1)),
        ))

    fig.update_layout(
        xaxis_title=f"Composition (wt% {pd.component_B})",
        yaxis_title="Temperature (°C)",
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[pd.T_room, T_top]),
        template="plotly_white",
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_diagram, tab_tieline, tab_solid, tab_cooling, tab_micro, tab_report = st.tabs(
    ["📊 Phase Diagram", "📏 Tie-Line & Lever Rule", "🌊 Solidification Path", "🌡️ Cooling Curve",
     "🔬 Microstructure Evolution", "📄 Report"]
)

# -- Tab 1: Phase Diagram ----------------------------------------------------
with tab_diagram:
    st.subheader(f"{pd_system.name} Binary Phase Diagram")
    left, right = st.columns([3, 1])
    with left:
        fig = make_plotly_diagram(pd_system)
        st.plotly_chart(fig, width='stretch')
    with right:
        st.markdown("**System summary**")
        st.metric(f"Melting point of {pd_system.component_A}", f"{pd_system.T_A:.0f} °C")
        st.metric(f"Melting point of {pd_system.component_B}", f"{pd_system.T_B:.0f} °C")
        st.metric("Eutectic temperature", f"{pd_system.T_E:.0f} °C")
        st.metric("Eutectic composition", f"{pd_system.C_E:.1f} wt% {pd_system.component_B}")
        st.caption(
            "Shaded fields (left→right): Liquid, α, β, α+Liquid, β+Liquid, α+β. "
            "Hover any curve for exact values."
        )

# -- Tab 2: Tie-Line & Lever Rule -------------------------------------------
with tab_tieline:
    st.subheader("Tie-Line Construction & Lever-Rule Phase Fractions")
    st.caption("Pick an overall composition and temperature — PhaseForge identifies the phase region, "
               "constructs the tie line, and computes phase fractions automatically.")

    c1, c2 = st.columns(2)
    C0 = c1.slider("Overall composition, C₀ (wt% " + pd_system.component_B + ")", 0.1, 99.9, 30.0, 0.1)
    T0 = c2.slider("Temperature (°C)", float(pd_system.T_room), float(max(pd_system.T_A, pd_system.T_B) * 1.05), 220.0, 1.0)

    region = pd_system.phase_at(C0, T0)
    st.info(f"**Phase region:** {region.value}", icon="🔎")

    tie_line = None
    fractions_display = None
    try:
        tie_line = pd_system.tie_line(C0, T0)
        f_left, f_right = pd_system.lever_rule(C0, tie_line.c_left, tie_line.c_right)
        fractions_display = {tie_line.label_left: f_left, tie_line.label_right: f_right}
    except PhaseDiagramError as e:
        st.warning(str(e))

    left, right = st.columns([3, 1])
    with left:
        fig = make_plotly_diagram(pd_system, tie_line=tie_line, marker=(C0, T0))
        st.plotly_chart(fig, width='stretch')
    with right:
        st.markdown("**Lever-rule results**")
        if fractions_display:
            for label, frac in fractions_display.items():
                st.metric(label, f"{frac*100:.1f} %")
            st.progress(list(fractions_display.values())[0])
        else:
            st.write("Single-phase point — 100% of one phase, no tie line to construct.")

# -- Tab 3: Solidification Path ---------------------------------------------
with tab_solid:
    st.subheader("Equilibrium Solidification Path")
    st.caption("Traces the evolving liquid/solid compositions and phase fractions as an alloy of "
               "composition C₀ cools from the liquidus to full solidification (lever-rule / equilibrium cooling).")

    C0_sol = st.slider("Overall composition, C₀ (wt% " + pd_system.component_B + ")", 0.1, 99.9, 30.0, 0.1, key="c0_sol")

    path = trace_solidification_path(pd_system, C0_sol)

    left, right = st.columns([3, 2])
    with left:
        fig = make_plotly_diagram(pd_system, solidification=path, marker=(C0_sol, path.T_liquidus))
        st.plotly_chart(fig, width='stretch')

    with right:
        m1, m2 = st.columns(2)
        m1.metric("Liquidus temperature", f"{path.T_liquidus:.1f} °C")
        m2.metric("Solidification ends at", f"{path.T_solidus_end:.1f} °C")
        st.metric("Passes through eutectic invariant?", "Yes" if path.reaches_eutectic else "No")

        st.markdown("**Final phase fractions**")
        for label, frac in path.final_phase_fractions.items():
            st.metric(label, f"{frac*100:.1f} %")

        T_arr, fL_arr, fS_arr = path.as_arrays()
        frac_fig = go.Figure()
        frac_fig.add_trace(go.Scatter(x=T_arr, y=fL_arr, name="Liquid fraction", line=dict(color="#2E86AB", width=3)))
        frac_fig.add_trace(go.Scatter(x=T_arr, y=fS_arr, name="Solid fraction", line=dict(color="#B8860B", width=3)))
        frac_fig.update_layout(
            xaxis=dict(title="Temperature (°C)", autorange="reversed"),
            yaxis=dict(title="Phase fraction", range=[-0.02, 1.02]),
            template="plotly_white", height=320, margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", y=1.15, font=dict(size=10)),
        )
        st.plotly_chart(frac_fig, width='stretch')

# -- Tab 4: Cooling Curve -----------------------------------------------------
with tab_cooling:
    st.subheader("Cooling-Curve Simulation")
    st.caption("A lumped-parameter thermal model (SciPy ODE integration) showing the characteristic "
               "slope change at the liquidus and — for off-eutectic alloys — a thermal arrest (plateau) "
               "at the eutectic temperature as latent heat is released.")

    c1, c2, c3 = st.columns(3)
    C0_cool = c1.slider("Composition, C₀ (wt% " + pd_system.component_B + ")", 0.1, 99.9, 30.0, 0.1, key="c0_cool")
    k_cool = c2.slider("Cooling rate constant, k", 0.002, 0.03, 0.008, 0.001, help="Higher = faster cooling to ambient.")
    latent = c3.slider("Latent heat ratio (L/Cp)", 1.0, 20.0, 9.0, 0.5, help="Higher = more pronounced arrest/slope change.")
    T_ambient = st.slider("Ambient / mold temperature (°C)", 0.0, 100.0, 25.0, 5.0)

    cc = simulate_cooling_curve(pd_system, C0_cool, T_ambient=T_ambient, k=k_cool, latent_heat_ratio=latent)

    m1, m2, m3 = st.columns(3)
    m1.metric("Liquidus arrest", f"{cc.T_liquidus:.1f} °C")
    m2.metric("Eutectic / solidus", f"{cc.T_arrest_end:.1f} °C")
    if cc.has_eutectic_plateau and cc.t_eutectic_plateau_end:
        m3.metric("Eutectic arrest duration", f"{cc.t_eutectic_plateau_end - cc.t_solidus_reached:.1f}")
    else:
        m3.metric("Eutectic arrest", "none (terminal solid solution)")

    cool_fig = go.Figure()
    cool_fig.add_trace(go.Scatter(x=cc.t, y=cc.T, mode="lines", line=dict(color="#1B1B1E", width=3), name="T(t)"))
    cool_fig.add_hline(y=cc.T_liquidus, line=dict(color="#2E86AB", dash="dash"),
                        annotation_text=f"Liquidus {cc.T_liquidus:.1f}°C", annotation_position="top left")
    cool_fig.add_hline(y=cc.T_arrest_end, line=dict(color="#8E44AD", dash="dash"),
                        annotation_text=f"{'Eutectic' if cc.has_eutectic_plateau else 'Solidus'} {cc.T_arrest_end:.1f}°C",
                        annotation_position="bottom left")
    if cc.has_eutectic_plateau and cc.t_eutectic_plateau_end:
        cool_fig.add_vrect(x0=cc.t_solidus_reached, x1=cc.t_eutectic_plateau_end,
                            fillcolor="#8E44AD", opacity=0.12, line_width=0,
                            annotation_text="Eutectic thermal arrest", annotation_position="top left")
    cool_fig.update_layout(
        xaxis_title="Time", yaxis_title="Temperature (°C)",
        template="plotly_white", height=480, margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(cool_fig, width='stretch')
    for note in cc.notes:
        st.warning(note)

# -- Tab 5: Microstructure Evolution ------------------------------------------
with tab_micro:
    render_microstructure_tab(pd_system, diagram_fn=make_plotly_diagram)

# -- Tab 6: Automated Report ---------------------------------------------------
with tab_report:
    st.subheader("Automated Computational Report")
    st.caption("Generates a single self-contained HTML report bundling the phase diagram, the current "
               "tie-line/lever-rule analysis, the solidification path, and the cooling-curve simulation — "
               "ready to download and share.")

    st.markdown("**Report will include the analysis point below:**")
    c1, c2 = st.columns(2)
    C0_rep = c1.slider("Overall composition, C₀ (wt% " + pd_system.component_B + ")", 0.1, 99.9, 30.0, 0.1, key="c0_rep")
    T_rep = c2.slider("Temperature for tie-line analysis (°C)", float(pd_system.T_room),
                       float(max(pd_system.T_A, pd_system.T_B) * 1.05), 220.0, 1.0, key="t_rep")

    if st.button("🚀 Generate Report", type="primary"):
        with st.spinner("Running phase-diagram, tie-line, solidification, and cooling-curve calculations..."):
            analysis = None
            try:
                tl = pd_system.tie_line(C0_rep, T_rep)
                fr = pd_system.phase_fractions(C0_rep, T_rep)
                analysis = {
                    "C0": C0_rep, "T": T_rep, "tie_line": tl, "region": fr["region"],
                    "fractions": {tl.label_left: fr[tl.label_left], tl.label_right: fr[tl.label_right]},
                }
            except PhaseDiagramError:
                pass

            path = trace_solidification_path(pd_system, C0_rep)
            cc = simulate_cooling_curve(pd_system, C0_rep)
            html = build_html_report(pd_system, analysis=analysis, solidification=path, cooling=cc,
                                      microstructure_C0=C0_rep)

        st.success("Report generated.", icon="✅")
        st.download_button(
            "⬇️ Download HTML Report",
            data=html.encode("utf-8"),
            file_name=f"phaseforge_report_{pd_system.name.replace(' ', '_')}.html",
            mime="text/html",
        )

        diagram_fig = plot_phase_diagram(pd_system, tie_line=analysis["tie_line"] if analysis else None,
                                          marker=(C0_rep, T_rep), solidification_path=path)
        buf = io.BytesIO()
        diagram_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            "⬇️ Download Phase Diagram (PNG)",
            data=buf.getvalue(),
            file_name=f"phaseforge_diagram_{pd_system.name.replace(' ', '_')}.png",
            mime="image/png",
        )

        with st.expander("Preview report"):
            st.components.v1.html(html, height=800, scrolling=True)

st.divider()
st.caption("PhaseForge · Python, NumPy, SciPy, Matplotlib, Streamlit, Plotly · Developed by **Mohib Mehdi(MME 24-28)** · "
           "Future development: thermodynamic database integration, multicomponent alloy systems, ML-based property prediction.")
