"""
microstructure.py — schematic micrographs for the Fe–Fe₃C diagram.

`render_micrograph(state)` turns a `MicroState` (produced by
`IronCarbonDiagram.micro_state(C0, T)`) into an RGB image that looks like an
etched section: polygonal grains, dendrites in liquid, proeutectoid ferrite or
cementite along prior-austenite boundaries, lamellar pearlite colonies, and the
"leopard-skin" texture of ledeburite.

How the picture is tied to the physics
--------------------------------------
* Area fractions of each drawn constituent equal the equilibrium weight fractions
  from the lever rule (density differences between the phases are ignored), and
  are hit *exactly* by thresholding a distance field at the right quantile.
* The random layout (grain seeds, dendrite skeleton, pearlite colony orientations)
  depends only on the alloy, not the temperature — so sweeping T for one alloy
  shows the *same* piece of metal evolving.

The images are schematic teaching pictures (not to scale, no real image data).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

from .ironcarbon import CEM, MicroState

# ---------------------------------------------------------------------------
# Palette (RGB 0-1)
# ---------------------------------------------------------------------------
COLOR = {
    "L": (0.93, 0.61, 0.27),
    "L₁": (0.93, 0.61, 0.27),
    "L₂": (0.40, 0.62, 0.88),
    "δ": (0.79, 0.73, 0.91),
    "γ": (0.86, 0.80, 0.60),
    "α": (0.90, 0.91, 0.93),
    "β": (0.70, 0.82, 0.62),
    "Cu3Au": (0.93, 0.80, 0.45),
    "CuAu": (0.85, 0.70, 0.38),
    "CuAu3": (0.95, 0.87, 0.58),
    "Cu3Pd": (0.80, 0.78, 0.85),
    "CuPd": (0.70, 0.68, 0.80),
    CEM: (1.00, 1.00, 1.00),
    "pearlite_dark": (0.33, 0.33, 0.38),
    "pearlite_light": (0.87, 0.87, 0.90),
}


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(255 * c)) for c in rgb)


# ---------------------------------------------------------------------------
# A "sample": deterministic random fields shared by all temperatures
# ---------------------------------------------------------------------------
class _Sample:
    def __init__(self, size: int, seed: int):
        self.N = size
        self.seed = seed
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        self.X, self.Y = xx, yy
        self._cache: Dict[str, object] = {}

    def _rng(self, salt: int) -> np.random.Generator:
        return np.random.default_rng(self.seed * 101 + salt)

    # -- generic helpers -----------------------------------------------------
    def noise(self, sigma: float, salt: int = 1) -> np.ndarray:
        key = f"noise{sigma}_{salt}"
        if key not in self._cache:
            n = gaussian_filter(self._rng(salt).standard_normal((self.N, self.N)), sigma)
            self._cache[key] = (n / (n.std() + 1e-9)).astype(np.float32)
        return self._cache[key]

    def _voronoi(self, g: int, salt: int, jitter: float = 0.38):
        N = self.N
        rng = self._rng(salt)
        cell = N / g
        ij = np.stack(np.meshgrid(np.arange(g), np.arange(g)), -1).reshape(-1, 2)
        seeds = (ij + 0.5 + rng.uniform(-jitter, jitter, ij.shape)) * cell
        tree = cKDTree(seeds)
        pts = np.stack([self.X.ravel(), self.Y.ravel()], 1)
        d, i = tree.query(pts, k=2)
        labels = i[:, 0].reshape(N, N)
        bd = ((d[:, 1] - d[:, 0]) / 2).reshape(N, N).astype(np.float32)   # ≈ distance to boundary
        return labels, bd, len(seeds)

    # -- grains (prior-austenite / δ / α grains) --------------------------------
    def grains(self):
        if "grains" not in self._cache:
            g = max(4, int(round(self.N / 80)))
            labels, bd, n = self._voronoi(g, salt=11)
            tone = 1.0 + 0.030 * self._rng(12).standard_normal(n)
            wob = 0.55 + 0.9 * self._rng(13).random(n)               # per-grain rim thickness multiplier
            self._cache["grains"] = (labels, bd, tone.astype(np.float32), wob.astype(np.float32))
        return self._cache["grains"]

    # -- pearlite colonies -----------------------------------------------------------
    def colonies(self):
        if "col" not in self._cache:
            g = max(8, int(round(self.N / 32)))
            labels, bd, n = self._voronoi(g, salt=21, jitter=0.5)
            rng = self._rng(22)
            theta = rng.uniform(0, np.pi, n)
            spacing = rng.uniform(5.0, 8.0, n) * self.N / 440.0
            offset = rng.uniform(0, 1, n)
            self._cache["col"] = (labels, bd, theta, spacing, offset)
        return self._cache["col"]

    def pearlite_texture(self) -> np.ndarray:
        """Lamellar pearlite RGB image (dark ferrite with light cementite lamellae)."""
        if "pearl" not in self._cache:
            labels, bd, theta, spacing, offset = self.colonies()
            th, sp, off = theta[labels], spacing[labels], offset[labels]
            phase = (self.X * np.cos(th) + self.Y * np.sin(th)) / sp + off
            frac = phase - np.floor(phase)
            band = np.exp(-0.5 * ((frac - 0.5) / 0.115) ** 2)
            dark = np.array(COLOR["pearlite_dark"])
            light = np.array(COLOR["pearlite_light"])
            tex = dark[None, None, :] * (1 - band[..., None]) + light[None, None, :] * band[..., None]
            tex *= (1.0 + 0.02 * self.noise(6, 31))[..., None]
            tex *= (1.0 - 0.45 * np.clip(1 - bd / 1.3, 0, 1))[..., None]      # colony boundaries
            self._cache["pearl"] = tex.astype(np.float32)
        return self._cache["pearl"]

    def domain_texture(self, base_color) -> np.ndarray:
        """Antiphase-domain texture: a checkerboard of two near-identical tints of
        `base_color`, one per domain variant, with domain grains from colonies()."""
        key = f"dom_{base_color}"
        if key not in self._cache:
            labels, bd, theta, spacing, offset = self.colonies()
            th, sp, off = theta[labels], spacing[labels], offset[labels]
            u = (self.X * np.cos(th) + self.Y * np.sin(th)) / sp + off
            v = (-self.X * np.sin(th) + self.Y * np.cos(th)) / sp
            checker = (np.floor(u) + np.floor(v)) % 2
            base = np.array(base_color, dtype=np.float32)
            light = np.clip(base * 1.10, 0, 1)
            dark = np.clip(base * 0.90, 0, 1)
            tex = dark[None, None, :] * (1 - checker[..., None]) + light[None, None, :] * checker[..., None]
            tex *= (1.0 + 0.015 * self.noise(6, 33))[..., None]
            tex *= (1.0 - 0.35 * np.clip(1 - bd / 1.3, 0, 1))[..., None]     # domain (antiphase) boundaries
            self._cache[key] = tex.astype(np.float32)
        return self._cache[key]

    # -- dendrite skeleton distance field ----------------------------------------------
    def dendrite_u(self) -> np.ndarray:
        if "dend" not in self._cache:
            N = self.N
            rng = self._rng(41)
            main = Image.new("L", (N, N), 0)
            sec = Image.new("L", (N, N), 0)
            dm, ds = ImageDraw.Draw(main), ImageDraw.Draw(sec)
            g = 3
            cell = N / g
            for i in range(g):
                for j in range(g):
                    cx = (i + 0.5 + rng.uniform(-0.25, 0.25)) * cell
                    cy = (j + 0.5 + rng.uniform(-0.25, 0.25)) * cell
                    th = rng.uniform(0, np.pi / 2)
                    L = cell * rng.uniform(0.80, 1.05)
                    for k in range(4):
                        ang = th + k * np.pi / 2 + rng.normal(0, 0.05)
                        dx, dy = np.cos(ang), np.sin(ang)
                        dm.line([(cx, cy), (cx + dx * L, cy + dy * L)], fill=255, width=1)
                        sp0 = cell * 0.17
                        s = sp0 * rng.uniform(0.6, 1.0)
                        while s < L * 0.92:
                            Ls = ((L - s) * 0.62 + cell * 0.04) * rng.uniform(0.75, 1.1)
                            px, py = cx + dx * s, cy + dy * s
                            for sgn in (-1, 1):
                                a2 = ang + sgn * np.pi / 2 + rng.normal(0, 0.10)
                                ds.line([(px, py), (px + np.cos(a2) * Ls, py + np.sin(a2) * Ls)],
                                        fill=255, width=1)
                            s += sp0 * rng.uniform(0.85, 1.15)
            dmain = distance_transform_edt(np.asarray(main) == 0)
            dsec = distance_transform_edt(np.asarray(sec) == 0)
            u = np.minimum(dmain / 1.9, dsec / 1.0)
            self._cache["dend"] = (u + 0.35 * self.noise(3, 42)).astype(np.float32)
        return self._cache["dend"]

    # -- cementite plate skeleton distance field ----------------------------------------
    def plates_u(self) -> np.ndarray:
        if "plates" not in self._cache:
            N = self.N
            rng = self._rng(51)
            img = Image.new("L", (N, N), 0)
            dr = ImageDraw.Draw(img)
            families = rng.uniform(0, np.pi, 3)
            for _ in range(38):
                cx, cy = rng.uniform(0, N, 2)
                ang = rng.choice(families) + rng.normal(0, 0.12)
                L = rng.uniform(0.16, 0.42) * N
                dx, dy = np.cos(ang) * L / 2, np.sin(ang) * L / 2
                dr.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=255, width=1)
            u = distance_transform_edt(np.asarray(img) == 0)
            self._cache["plates"] = (u + 0.2 * self.noise(3, 52)).astype(np.float32)
        return self._cache["plates"]

    # -- ledeburite particles -----------------------------------------------------------------
    def led_particles(self, fraction: float, region: np.ndarray) -> np.ndarray:
        N = self.N
        rng = self._rng(61)
        step = max(5.0, 6.5 * N / 440.0)
        g = int(N / step) + 1
        ij = np.stack(np.meshgrid(np.arange(g), np.arange(g)), -1).reshape(-1, 2)
        centers = (ij + 0.5 + rng.uniform(-0.5, 0.5, ij.shape)) * step
        radius = rng.uniform(0.70, 1.30, len(centers))
        tree = cKDTree(centers)
        pts = np.stack([self.X.ravel(), self.Y.ravel()], 1)
        d, i = tree.query(pts, k=1)
        dn = (d / radius[i]).reshape(N, N) + 0.30 * self.noise(1.6, 62)
        if not region.any():
            return np.zeros_like(region)
        thr = np.quantile(dn[region], min(max(fraction, 0.0), 1.0))
        return (dn <= thr) & region


# ---------------------------------------------------------------------------
# compositing helpers
# ---------------------------------------------------------------------------
def _fill(color, mottle: np.ndarray, amp: float = 0.03) -> np.ndarray:
    base = np.array(color, dtype=np.float32)[None, None, :]
    return base * (1.0 + amp * mottle[..., None])


def _outline(mat: np.ndarray, strength: float = 0.5) -> np.ndarray:
    e = np.zeros(mat.shape, dtype=np.float32)
    e[1:, :] = np.maximum(e[1:, :], (mat[1:, :] != mat[:-1, :]).astype(np.float32))
    e[:, 1:] = np.maximum(e[:, 1:], (mat[:, 1:] != mat[:, :-1]).astype(np.float32))
    e = gaussian_filter(e, 0.65)
    return 1.0 - strength * np.clip(e * 2.2, 0, 1)


def _aa(mask: np.ndarray, s: float = 0.7) -> np.ndarray:
    return np.clip(gaussian_filter(mask.astype(np.float32), s), 0, 1)


def _rim_mask(S: _Sample, f: float) -> np.ndarray:
    labels, bd, tone, wob = S.grains()
    if f <= 0:
        return np.zeros((S.N, S.N), dtype=bool)
    u = bd / wob[labels]
    return u <= np.quantile(u, min(f, 1.0))


def _grain_base(S: _Sample, color) -> np.ndarray:
    labels, bd, tone, wob = S.grains()
    base = np.array(color, dtype=np.float32)[None, None, :] * tone[labels][..., None]
    return base * (1.0 + 0.012 * S.noise(8, 71))[..., None]


def _gb_lines(S: _Sample, strength=0.55) -> np.ndarray:
    labels, bd, tone, wob = S.grains()
    return 1.0 - strength * np.clip(1 - bd / 1.15, 0, 1)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def render_micrograph(state: MicroState, size: int = 440, seed: int = 7) -> np.ndarray:
    """Return an (size, size, 3) uint8 schematic micrograph for `state`."""
    N = size
    S = _Sample(N, seed + int(round(state.C0 * 100)))
    rec = state.recipe
    mot = S.noise(10, 5)

    if rec == "liquid":
        img = _fill(COLOR["L"], mot, 0.05) * (1.0 + 0.03 * S.noise(3, 6))[..., None]
        mat = np.zeros((N, N), int)

    elif rec in ("dendrites", "plates_in_liquid"):
        u = S.dendrite_u() if rec == "dendrites" else S.plates_u()
        mask = u <= np.quantile(u, min(max(state.f, 0.0), 1.0))
        a = _aa(mask)[..., None]
        liquid_col = COLOR.get(state.phase, COLOR["L"])
        solid = COLOR[state.rim_phase]
        img = _fill(liquid_col, mot, 0.05) * (1 - a) + _fill(solid, S.noise(8, 7), 0.02) * a
        mat = mask.astype(int)

    elif rec == "eutectic_dendrites":
        ex = state.extras
        f_prim = float(ex.get("f_primary", 0.0))
        prim_color = COLOR[ex.get("primary_phase", "α")]
        u = S.dendrite_u()
        prim = u <= np.quantile(u, min(max(f_prim, 0.0), 1.0))
        apr = _aa(prim)[..., None]
        img = S.pearlite_texture() * (1 - apr) + _fill(prim_color, S.noise(8, 9), 0.02) * apr
        mat = prim.astype(int)

    elif rec == "ordered_domains":
        img = S.domain_texture(COLOR[state.phase])
        mat = np.zeros((N, N), int)

    elif rec == "grains":
        labels, bd, tone, wob = S.grains()
        img = _grain_base(S, COLOR[state.phase])
        mat = np.zeros((N, N), int)
        if state.rim_phase and state.f > 0:
            rim = _rim_mask(S, state.f)
            a = _aa(rim)[..., None]
            img = img * (1 - a) + _fill(COLOR[state.rim_phase], mot, 0.0) * a
            mat = rim.astype(int)
            img = img * _gb_lines(S)[..., None] ** (1 - a)
        else:
            img = img * _gb_lines(S)[..., None]

    elif rec == "rim":
        labels, bd, tone, wob = S.grains()
        rim = _rim_mask(S, state.f)
        a = _aa(rim)[..., None]
        img = _grain_base(S, COLOR[state.phase]) * (1 - a) + _fill(COLOR[state.rim_phase], mot, 0.015) * a
        img = img * (_gb_lines(S)[..., None] ** (1 - a))
        mat = rim.astype(int)

    elif rec == "pearlite_rim":
        rim = _rim_mask(S, state.f)
        a = _aa(rim)[..., None]
        img = S.pearlite_texture() * (1 - a) + _fill(COLOR[state.rim_phase], mot, 0.015) * a
        mat = rim.astype(int)

    elif rec == "pearlite":
        img = S.pearlite_texture()
        mat = np.zeros((N, N), int)

    elif rec == "droplets":
        matrix_col, drop_col = COLOR[state.phase], COLOR[state.rim_phase]
        full_mask = np.ones((N, N), dtype=bool)
        part = S.led_particles(float(state.f), full_mask)
        ap = _aa(part, 0.6)[..., None]
        img = _fill(matrix_col, mot, 0.05) * (1 - ap) + _fill(drop_col, S.noise(6, 10), 0.04) * ap
        mat = part.astype(int)

    elif rec == "ledeburite":
        ex = state.extras
        kind = ex.get("primary", "none")
        f_prim = float(ex.get("f_primary", 0.0))
        f_sec = float(ex.get("f_sec", 0.0))
        transformed = ex.get("particle") == "pearlite"
        matrix_key = ex.get("matrix_color", CEM)
        primary_key = ex.get("primary_color", "γ")
        particle_key = ex.get("particle_color")  # None -> Fe-C's transformed-dependent default

        prim = np.zeros((N, N), bool)
        if kind == "dendrites":
            u = S.dendrite_u()
            prim = u <= np.quantile(u, min(max(f_prim, 0.0), 1.0))
        elif kind == "plates":
            u = S.plates_u()
            prim = u <= np.quantile(u, min(max(f_prim, 0.0), 1.0))
        led = ~prim
        part = S.led_particles(float(ex.get("f_particle", 0.52)), led)

        white = _fill(COLOR[matrix_key], mot, 0.01)
        if particle_key is not None:
            part_col = np.array(COLOR[particle_key], dtype=np.float32)
        else:
            part_col = np.array((0.50, 0.50, 0.55) if transformed else COLOR["γ"], dtype=np.float32)
        part_rgb = part_col[None, None, :] * (1.0 + 0.05 * S.noise(2, 8))[..., None]
        ap = _aa(part, 0.6)[..., None]
        img = white * (1 - ap) + part_rgb * ap
        mat = part.astype(int)

        if kind == "dendrites":
            if transformed and particle_key is None:
                interior = S.pearlite_texture() * 0.70
            else:
                interior = _fill(COLOR[primary_key], S.noise(8, 9), 0.02)
            apr = _aa(prim)[..., None]
            prim_img = interior
            if f_sec > 0:
                din = distance_transform_edt(prim)
                rim = prim & (din <= np.quantile(din[prim], min(f_sec, 1.0)))
                ar = _aa(rim)[..., None]
                prim_img = interior * (1 - ar) + white * ar
                mat = np.where(rim, 4, mat)
            img = img * (1 - apr) + prim_img * apr
            mat = np.where(prim & (mat != 4), 2, mat)
        elif kind == "plates":
            apr = _aa(prim)[..., None]
            img = img * (1 - apr) + white * apr
            mat = np.where(prim, 3, mat)

    else:                                                     # safety net
        img = _fill(COLOR["α"], mot, 0.02)
        mat = np.zeros((N, N), int)

    img = img * _outline(mat, 0.42)[..., None]
    grain = np.random.default_rng(seed).normal(0, 0.010, img.shape).astype(np.float32)
    img = np.clip(img + grain, 0, 1)
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# legend
# ---------------------------------------------------------------------------
def _pearlite_css() -> str:
    d, l = _hex(COLOR["pearlite_dark"]), _hex(COLOR["pearlite_light"])
    return f"repeating-linear-gradient(45deg,{d} 0 2px,{l} 2px 4px)"


def _led_css(transformed: bool) -> str:
    dot = _hex((0.50, 0.50, 0.55) if transformed else COLOR["γ"])
    return f"radial-gradient(circle,{dot} 38%,#ffffff 42%) 0 0/7px 7px"


def legend_items(state: MicroState) -> List[Tuple[str, str, float]]:
    """[(name, css-background, fraction)] for every constituent drawn in `state`."""
    items = []
    for name, frac in state.constituents.items():
        low = name.lower()
        if frac <= 1e-9:
            continue
        if "ledeburite" in low:
            css = _led_css("transformed" in low)
        elif "divorced eutectic" in low:
            css = _hex(COLOR["β"])
        elif "eutectic" in low:
            css = _pearlite_css()
        elif "pearlite" in low:
            css = _pearlite_css()
        elif "fe₃c" in low or "cementite" in low:
            css = "#ffffff"
        elif "cu3au" in low:
            css = _hex(COLOR["Cu3Au"])
        elif "cuau3" in low:
            css = _hex(COLOR["CuAu3"])
        elif "cuau" in low:
            css = _hex(COLOR["CuAu"])
        elif "cu3pd" in low:
            css = _hex(COLOR["Cu3Pd"])
        elif "cupd" in low:
            css = _hex(COLOR["CuPd"])
        elif "l₁" in low:
            css = _hex(COLOR["L₁"])
        elif "l₂" in low:
            css = _hex(COLOR["L₂"])
        elif "liquid" in low:
            css = _hex(COLOR["L"])
        elif "δ" in low:
            css = _hex(COLOR["δ"])
        elif "austenite" in low:
            css = _hex(COLOR["γ"])
        elif "β" in low:
            css = _hex(COLOR["β"])
        else:
            css = _hex(COLOR["α"])
        items.append((name, css, frac))
    return items


def legend_html(state: MicroState) -> str:
    rows = []
    for name, css, frac in legend_items(state):
        rows.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:13px;">'
            f'<span style="width:22px;height:14px;border-radius:2px;border:1px solid #9aa3b2;'
            f'background:{css};flex:none;"></span>'
            f'<span>{name} <span style="color:#6B7280;">— {frac*100:.1f} %</span></span></div>'
        )
    return "".join(rows)
