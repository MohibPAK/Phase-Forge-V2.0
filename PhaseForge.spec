# PyInstaller spec — builds a one-folder app:  dist/PhaseForge/PhaseForge(.exe)
# Build with:  pyinstaller --noconfirm PhaseForge.spec
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = [("app.py", ".")]
binaries = []
hiddenimports = []

# Streamlit ships a compiled web front-end and reads its own package metadata
for pkg in ("streamlit",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
datas += copy_metadata("streamlit", recursive=True)

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "pytest", "IPython", "notebook"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="PhaseForge",
    console=True,          # keep the console: closing it quits the app
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PhaseForge", upx=False)
