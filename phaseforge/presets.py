"""
presets.py — Built-in binary eutectic alloy presets.

Illustrative textbook-level parameter sets for well-known binary eutectic
systems, so the app is useful immediately without requiring the user to
look up phase-diagram parameters first. Values are approximate/simplified
for educational use, not measured research-grade data.
"""

from .core import EutecticPhaseDiagram

PRESETS = {
    "Pb-Sn (Lead-Tin, solder)": dict(
        T_A=327.5, T_B=231.9, T_E=183.0, C_E=61.9,
        C_alpha_E=19.2, C_beta_E=97.5, C_alpha_RT=2.0, C_beta_RT=99.0,
        component_A="Pb", component_B="Sn", name="Pb-Sn",
    ),
    "Cu-Ag (Copper-Silver)": dict(
        T_A=1085.0, T_B=961.8, T_E=779.1, C_E=71.9,
        C_alpha_E=8.0, C_beta_E=91.2, C_alpha_RT=0.1, C_beta_RT=99.9,
        component_A="Cu", component_B="Ag", name="Cu-Ag",
    ),
    "Al-Si (Aluminum-Silicon, casting alloy)": dict(
        T_A=660.3, T_B=1414.0, T_E=577.0, C_E=12.6,
        C_alpha_E=1.65, C_beta_E=99.8, C_alpha_RT=0.05, C_beta_RT=99.9,
        component_A="Al", component_B="Si", name="Al-Si",
    ),
    "Custom": None,
}


def build_preset(name: str) -> EutecticPhaseDiagram:
    params = PRESETS.get(name)
    if params is None:
        raise KeyError(f"No such preset: {name}")
    return EutecticPhaseDiagram(**params)
