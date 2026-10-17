# 🧪 PhaseForge — Computational Materials Science Toolkit

A Python toolkit for analyzing **binary alloy phase diagrams and phase
transformations** — with an interactive, browser-based UI so it's easy to
use without touching code.

**Developed by Mohib Mehdi(MME 24-28)**

---

## Features

| Feature | Where |
|---|---|
| Automated binary phase-diagram visualization | *Phase Diagram* tab |
| Tie-line construction and phase-region identification | *Tie-Line & Lever Rule* tab |
| Lever-rule phase fraction calculations | *Tie-Line & Lever Rule* tab |
| Alloy composition and temperature analysis | *Tie-Line & Lever Rule* tab |
| Solidification-path visualization | *Solidification Path* tab |
| Cooling-curve simulation | *Cooling Curve* tab |
| Automated computational reports and plots | *Report* tab (one-click HTML report + PNG) |
| **Microstructure evolution** for Pb-Sn, Cu-Ag, Al-Si (and Custom) | *Microstructure Evolution* tab |
| **Iron–carbon (Fe–Fe₃C) diagram** with peritectic, eutectic and eutectoid reactions | Sidebar → *Fe–C* preset |
| **Microstructure evolution** of any steel / cast iron, melt → room temperature | *Microstructure Evolution* tab (Fe–C mode) |
| **Cu–Ni isomorphous system** — complete solid solubility, no invariant reactions | Sidebar → *Cu–Ni* preset |
| **Cu–Au and Cu–Pd** — isomorphous with solid-state ordering (superlattices) | Sidebar → *Cu–Au* / *Cu–Pd* presets |
| **Cu–Pb and Al–Pb monotectic systems** — liquid immiscibility, monotectic + divorced-eutectic reactions | Sidebar → *Cu–Pb* / *Al–Pb* presets |

**Technologies:** Python, NumPy, SciPy, Matplotlib, Streamlit, Plotly

**Built-in alloy presets:** Pb-Sn (solder), Cu-Ag, Al-Si (casting alloy), Fe–C (steels and cast irons), Cu–Ni (isomorphous), Cu–Pb and Al–Pb (monotectic) —
plus a fully custom system builder (set your own melting points, eutectic
point, and solubility limits).

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser automatically at `http://localhost:8501`.
No further setup needed — pick a preset alloy from the sidebar and start
exploring.

### Requirements
- Python 3.9+
- The packages in `requirements.txt` (installed above)

### Getting a standalone .exe (no Python needed on the instructor's machine)

This repo is already set up to build a one-click `PhaseForge.exe` with
PyInstaller (`launcher.py` + `PhaseForge.spec`). Building a Windows `.exe`
requires a Windows machine (or CI) — PyInstaller cannot cross-compile. Two ways
to get one:

**Automatic (no Windows machine needed):** push this repo to GitHub. The
included workflow (`.github/workflows/build.yml`) builds Windows, macOS and
Linux versions automatically on every push to `main`, or on demand from the
Actions tab ("Run workflow"). Download the finished `PhaseForge-windows` zip
from the workflow run's Artifacts section — unzip it and double-click
`PhaseForge.exe`.

**Manual (on an actual Windows machine):**
```powershell
pip install -r requirements.txt
pip install pyinstaller
python launcher.py --selftest      # should print SELFTEST OK
pyinstaller --noconfirm PhaseForge.spec
```
The finished app is `dist\PhaseForge\PhaseForge.exe` — copy the whole
`dist\PhaseForge` folder (not just the .exe) to wherever it'll run from.
Double-clicking the exe opens a console window and your browser; closing the
console quits the app.

I've built and smoke-tested the Linux version of this exact bundle here
(imports, launches, serves the app, passes `--selftest`) — the Windows build
goes through the identical PyInstaller pipeline, just on a Windows runner.

---

## How to use it

1. **Pick a system** in the left sidebar — choose a preset (Pb-Sn, Cu-Ag,
   Al-Si) or select **Custom** and enter your own melting points, eutectic
   temperature/composition, and solid-solubility limits.
2. **Phase Diagram tab** — see the full binary eutectic diagram with shaded
   phase fields (Liquid, α, β, α+L, β+L, α+β). Hover any curve for exact
   values.
3. **Tie-Line & Lever Rule tab** — drag the composition/temperature sliders;
   the phase region, tie line, and phase fractions update live.
4. **Solidification Path tab** — pick a composition and watch the
   liquid/solid composition trajectory and phase-fraction evolution as the
   alloy solidifies under equilibrium (lever-rule) cooling.
5. **Cooling Curve tab** — simulate a T(t) cooling curve; adjust the cooling
   rate and latent-heat ratio to see the liquidus slope-change and (for
   off-eutectic alloys) the eutectic thermal-arrest plateau.
6. **Report tab** — click **Generate Report** to produce a single
   self-contained HTML file bundling the phase diagram, tie-line/lever-rule
   analysis, solidification path, and cooling curve — ready to download and
   share. A standalone PNG of the phase diagram is also available.

### Standard eutectic mode: Pb-Sn, Cu-Ag, Al-Si (and Custom)

Every eutectic-family system — the three built-in presets plus anything you
build in **Custom** — now has a **Microstructure Evolution** tab alongside
the existing Phase Diagram / Tie-Line & Lever Rule / Solidification Path /
Cooling Curve / Report tabs. Pick a composition and cool it from the melt:

- **Off-eutectic alloys** (hypo- or hypereutectic): primary α or β dendrites
  form first, then the eutectic reaction fills the remaining liquid with a
  fine, lamellar α+β mixture around them.
- **Eutectic composition**: the whole melt freezes at one temperature into
  100% lamellar eutectic — no primary phase at all.
- **Near-pure compositions**: freezes to a single terminal solid solution,
  with a note on secondary-phase precipitation if the alloy is close enough
  to its solvus limit.

This reuses the same lever-rule/tie-line engine as the other four tabs (`core.py`),
so it works automatically for whatever `EutecticPhaseDiagram` you build in Custom.

### Iron–carbon (Fe–Fe₃C) mode

Choose **Fe–C (Iron–Carbon…)** in the sidebar. Four tabs appear:

1. **Fe–Fe₃C Diagram** — interactive diagram with three zoom views (full, steel
   region, peritectic corner), critical points A…Q, alloy-class bands, and the
   A₀/A₂ magnetic lines.
2. **Tie-Line & Lever Rule** — phase region, tie line and phase fractions at any
   (C₀, T), with the lever-rule working shown, including exactly on the three
   invariant isotherms (1495 / 1147 / 727 °C).
3. **Microstructure Evolution** — pick an alloy, cool it stage by stage, and see a
   schematic micrograph at every temperature: δ/γ dendrites, austenite grains,
   proeutectoid ferrite or cementite, lamellar pearlite, ledeburite, primary
   cementite plates. Includes the full cooling sequence table, an evolution
   strip, and room-temperature constituent fractions.
4. **Report** — self-contained HTML report for the chosen alloy.

```python
from phaseforge import IronCarbonDiagram
from phaseforge.microstructure import render_micrograph

d = IronCarbonDiagram()
d.phase_fractions(0.40, 780)                 # {'α': 0.27, 'γ': 0.73}
for e in d.transformation_path(0.40):        # every event from the melt to 25 °C
    print(e.temp_text, e.reaction)
img = render_micrograph(d.micro_state(0.40, 25))   # RGB array
```

### Cu–Ni (isomorphous) mode

Choose **Cu–Ni (Isomorphous…)** in the sidebar. Same four-tab pattern as Fe–C,
scaled down because Cu–Ni has no invariant reactions — just a liquidus/solidus
"lens" between the two melting points (Cu 1085 °C, Ni 1453 °C):

1. **Cu–Ni Diagram** — the diagram plus why the system has complete solubility
   (Hume-Rothery rules) and no eutectic/peritectic/eutectoid reaction.
2. **Tie-Line & Lever Rule** — phase region, tie line and lever-rule fractions
   at any (C₀, T), worked out step by step.
3. **Microstructure Evolution** — pick an alloy, cool it from the melt: α
   dendrites grow in the liquid, then solidification finishes into uniform
   equiaxed grains that are stable all the way to room temperature (with a note
   on coring — the non-equilibrium segregation this equilibrium model omits).
4. **Report** — self-contained HTML report.

More systems are planned: Cu-Pb and Al-Pb (monotectic), Fe-Ni / Cu-Au / Pd-Cu
(isomorphous with ordering), Ag-Al (eutectic + intermediate compound) and
Co-W (multiple intermetallics) — each needs its own engine, since none of
those diagram shapes are interchangeable with the eutectic, Fe–C or Cu–Ni ones
already here.

### Cu–Pb and Al–Pb (monotectic) mode

Choose **Cu–Pb** or **Al–Pb** in the sidebar. These are genuine monotectic systems:
above the monotectic temperature, alloys in a composition band split into two
immiscible liquids (a real liquid miscibility gap), which then undergo

    Monotectic:  L1 -> alpha + L2
    Eutectic:    L2 -> alpha + beta   (strongly "divorced" — fine alpha
                                        particles in a continuous beta matrix,
                                        since the eutectic composition sits
                                        right next to pure Pb)

Same four-tab pattern as the other systems, plus a new microstructure recipe
(round droplets of the minority liquid dispersed in the majority one) for the
miscibility-gap stage. See each preset's "Data source" expander in the
Diagram tab for exactly which numbers are literature values versus
engineering estimates (mainly the critical/consolute composition, which
neither of the two sources used gives directly).

More systems are still planned: Fe-Ni, Ag-Al (eutectic + peritectics + intermediate
phases) and Co-W (multiple intermetallics) — each needs its own engine, since none of those diagram
shapes are interchangeable with the ones already here.

### Cu–Au and Cu–Pd (isomorphous with ordering)

Choose **Cu–Au** or **Cu–Pd** in the sidebar. Above the ordering temperatures these
behave like Cu–Ni (one disordered FCC solid solution); below them, atoms sort into
superlattices at stoichiometric compositions — Cu₃Au (L1₂, 390 °C), CuAu (L1₀, ~410 °C)
and CuAu₃ (L1₂, ~197 °C) in Cu–Au; Cu₃Pd (L1₂) and CuPd (B2) in Cu–Pd. Cu–Au's liquidus
has a congruent minimum at 80 wt% Au (911 °C), which splits the melting range into two
lenses; tie lines handle each side separately. Ordered phases are drawn as a fine
antiphase-domain pattern. Data sources and simplifications (dome widths are schematic;
the Cu–Pd liquidus/solidus between the melting points is an engineering estimate) are in
each system's Diagram tab.

---

## Project structure

```
phaseforge/
  __init__.py     # package exports
  core.py         # EutecticPhaseDiagram: liquidus/solidus/solvus curves,
                  #   phase-region classification, tie lines, lever rule
  simulation.py   # solidification-path tracing + SciPy ODE cooling-curve model
  presets.py      # built-in alloy systems (Pb-Sn, Cu-Ag, Al-Si)
  report.py       # Matplotlib plotting + self-contained HTML report builder
  ironcarbon.py   # Fe–Fe₃C engine: phase fields, tie lines, lever rule,
                  #   transformation path, constituents, micro_state()
  microstructure.py  # procedural schematic micrograph renderer + legends
  fec_layout.py   # shared labels / zoom views for the Fe–C diagram
  fec_report.py   # Fe–C static figure + HTML report
  fec_ui.py       # Streamlit tabs for the Fe–C mode
  cuni.py         # Cu–Ni isomorphous engine: liquidus/solidus, tie lines,
                  #   lever rule, transformation path, micro_state()
  cuni_report.py  # Cu–Ni static figure + HTML report
  cuni_ui.py      # Streamlit tabs for the Cu–Ni mode
  monotectic.py   # Generic monotectic engine (Cu-Pb, Al-Pb): miscibility
                  #   gap, monotectic + divorced-eutectic reactions
  monotectic_report.py  # Monotectic static figure + HTML report
  monotectic_ui.py      # Streamlit tabs for the Cu-Pb / Al-Pb modes
  ordering.py     # Isomorphous + order-disorder engine (Cu-Au, Cu-Pd)
  ordering_report.py / ordering_ui.py   # its report and Streamlit tabs
app.py            # Streamlit application (the UI)
requirements.txt
README.md
```

You can also use the computational engine directly, without the UI:

```python
from phaseforge.core import EutecticPhaseDiagram
from phaseforge.simulation import trace_solidification_path, simulate_cooling_curve

pd = EutecticPhaseDiagram(
    T_A=327.5, T_B=231.9, T_E=183.0, C_E=61.9,
    C_alpha_E=19.2, C_beta_E=97.5,
    component_A="Pb", component_B="Sn", name="Pb-Sn",
)

region = pd.phase_at(C=30, T=220)          # -> PhaseRegion.ALPHA_LIQUID
tie = pd.tie_line(30, 220)                  # tie-line endpoints
f_alpha, f_liquid = pd.lever_rule(30, tie.c_left, tie.c_right)

path = trace_solidification_path(pd, C0=30)     # equilibrium solidification path
cooling = simulate_cooling_curve(pd, C0=30)      # T(t) cooling curve
```

---

## Modeling notes

- **Fe–C mode** uses the *metastable* Fe–Fe₃C diagram (steels and white cast
  irons) with straight-line, textbook-level boundaries and equilibrium (slow)
  cooling. Bainite, martensite (TTT/CCT behaviour) and the stable Fe–graphite
  system of grey/ductile irons are **not** modelled. Micrographs are schematic
  pictures whose area fractions follow the lever rule — not real image data.
- PhaseForge models the standard **binary eutectic** topology: two terminal
  solid-solution phases (α, β), a eutectic invariant reaction, and the six
  classic phase fields (Liquid, α, β, α+Liquid, β+Liquid, α+β).
- Liquidus/solidus curves use closed-form power-law expressions (forward
  *and* inverse), so tie-line construction at any temperature is exact and
  fast — no numerical root-finding needed for the diagram geometry itself.
- The cooling-curve simulation integrates a lumped-parameter thermal model
  with SciPy's `solve_ivp`; latent heat released during solidification is
  modeled as an effective slowdown in cooling rate, and the eutectic
  reaction produces an explicit isothermal arrest — the classic shape seen
  in real cooling-curve analysis.
- Preset alloy parameters (Pb-Sn, Cu-Ag, Al-Si) are **illustrative,
  textbook-level approximations** for demonstration — not measured
  research-grade thermodynamic data.

---

## Future development

- Integration of real thermodynamic databases (CALPHAD-style) for
  research-grade accuracy
- Multicomponent (ternary+) alloy systems
- Machine-learning-based materials property prediction

---

*PhaseForge — Developed by Mohib Mehdi(MME 24-28)*
