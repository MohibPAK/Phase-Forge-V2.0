"""
launcher.py — desktop entry point for PhaseForge (used by the PyInstaller .exe build).

Starts the Streamlit server on a free local port, opens the default browser once the
server answers, and keeps running until the console window is closed.

    PhaseForge.exe             -> normal launch
    PhaseForge.exe --selftest  -> import everything, run the engine + renderer, exit 0/1
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser


def _base_dir() -> str:
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def _free_port(preferred: int = 8501) -> int:
    for port in [preferred] + list(range(8502, 8530)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _import_everything() -> None:
    """Explicit imports so PyInstaller bundles everything app.py needs
    (app.py itself is shipped as a data file, so it is not analysed)."""
    import numpy  # noqa: F401
    import scipy.integrate  # noqa: F401
    import scipy.ndimage  # noqa: F401
    import scipy.spatial  # noqa: F401
    import matplotlib  # noqa: F401
    matplotlib.use("Agg")
    import matplotlib.pyplot  # noqa: F401
    import plotly.graph_objects  # noqa: F401
    import PIL.Image  # noqa: F401
    import streamlit  # noqa: F401
    import phaseforge  # noqa: F401
    import phaseforge.core  # noqa: F401
    import phaseforge.simulation  # noqa: F401
    import phaseforge.presets  # noqa: F401
    import phaseforge.report  # noqa: F401
    import phaseforge.ironcarbon  # noqa: F401
    import phaseforge.microstructure  # noqa: F401
    import phaseforge.fec_layout  # noqa: F401
    import phaseforge.fec_report  # noqa: F401
    import phaseforge.fec_ui  # noqa: F401
    import phaseforge.cuni  # noqa: F401
    import phaseforge.cuni_report  # noqa: F401
    import phaseforge.cuni_ui  # noqa: F401
    import phaseforge.monotectic  # noqa: F401
    import phaseforge.monotectic_report  # noqa: F401
    import phaseforge.monotectic_ui  # noqa: F401
    import phaseforge.ordering  # noqa: F401
    import phaseforge.ordering_report  # noqa: F401
    import phaseforge.ordering_ui  # noqa: F401
    import phaseforge.eutectic_micro  # noqa: F401
    import phaseforge.eutectic_micro_ui  # noqa: F401


def _selftest() -> int:
    try:
        _import_everything()
        from phaseforge.ironcarbon import IronCarbonDiagram
        from phaseforge.microstructure import render_micrograph
        from phaseforge.fec_report import build_fec_html_report
        from phaseforge.presets import build_preset
        from phaseforge.simulation import trace_solidification_path
        from phaseforge.cuni import CuNiDiagram
        from phaseforge.monotectic import MonotecticDiagram, PRESETS as MONO_PRESETS
        from phaseforge.ordering import OrderingDiagram, PRESETS as ORD_PRESETS
        from phaseforge import eutectic_micro as em

        d = IronCarbonDiagram()
        fr = d.phase_fractions(0.40, 780)
        assert abs(fr["α"] - 0.27) < 0.01, fr
        assert render_micrograph(d.micro_state(3.0, 25.0), size=120).shape == (120, 120, 3)
        assert len(build_fec_html_report(d, 0.76, 25.0)) > 10_000
        pb = build_preset("Pb-Sn (Lead-Tin, solder)")
        trace_solidification_path(pb, C0=30)
        em.micro_state(pb, 30.0, 25.0)
        CuNiDiagram().phase_fractions(35.0, 1250.0)
        MonotecticDiagram(**MONO_PRESETS["Cu-Pb"]).phase_at(50.0, 970.0)
        OrderingDiagram(**ORD_PRESETS["Cu-Au"]).phase_at(50.83, 300.0)
        assert os.path.exists(os.path.join(_base_dir(), "app.py")), "app.py missing from bundle"
        print("SELFTEST OK")
        return 0
    except Exception as exc:  # pragma: no cover
        import traceback
        traceback.print_exc()
        print(f"SELFTEST FAILED: {exc!r}")
        return 1


def _open_browser_when_ready(url: str, timeout: float = 90.0) -> None:
    health = url + "/_stcore/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(health, timeout=1.5) as r:
                if r.status == 200:
                    webbrowser.open(url)
                    return
        except Exception:
            time.sleep(0.5)


def main() -> int:
    if "--selftest" in sys.argv:
        return _selftest()

    _import_everything()
    import streamlit.web.cli as stcli

    port = _free_port()
    url = f"http://localhost:{port}"
    app_path = os.path.join(_base_dir(), "app.py")

    print("=" * 62)
    print("  PhaseForge — Computational Materials Science Toolkit")
    print(f"  Running at {url}")
    print("  Your browser will open automatically.")
    print("  Close this window to quit PhaseForge.")
    print("=" * 62, flush=True)

    threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    sys.argv = [
        "streamlit", "run", app_path,
        "--global.developmentMode=false",
        "--server.headless=true",
        "--server.address=localhost",
        f"--server.port={port}",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
