"""
Interactive 3D explosion diagram for thick-panel origami trimming.

Shows design-plane panel outlines and dual-curve shaded regions stacked at
each thickness ``layer_h``, with an explosion slider that pulls layers apart
along z. Drag to orbit / zoom with the mouse. Right sidebar: show/hide
individual thickness layers (checkboxes + All / None).

Each shaded (collided) region is tied to a **specific panel** and **layer**:
  dual_curve_v1 fields: panel, unit, layer_h, layer_idx, side, unit_a/b
  shell_kind: "physical" (real thick shell), "ghost" (collision-only intermediate),
              or "side" (collision-only vertical band, unique panel index)
  If panel/layer are missing (legacy JSON), they are inferred from geometry.

Usage:
  python panel-trimming/visualize/visualize_3d.py
  python panel-trimming/visualize/visualize_3d.py --name mountain-thick-trimmed
  python panel-trimming/visualize/visualize_3d.py --name mountain-thick-trimmed --factor 2.5
  python panel-trimming/visualize/visualize_3d.py --name mountain-thick-trimmed --no-show -o out.png
  python panel-trimming/visualize/visualize_3d.py --name mountain-thick-trimmed --report-only

Config (optional, same dir as this file):
  config_3d.yml  or fall back to config.yml simulations[].name
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DESCRIPTION_DIR = os.path.join(_PROJECT_ROOT, "descriptionData")
_PANEL_TRIM_DIR = os.path.dirname(_THIS_DIR)
TRIMMED_DIR = os.path.join(_PANEL_TRIM_DIR, "trimmedData")
DEFAULT_CONFIG = os.path.join(_THIS_DIR, "config.yml")
DEFAULT_CONFIG_3D = os.path.join(_THIS_DIR, "config_3d.yml")
DEFAULT_OUTPUT_DIR = os.path.join(_THIS_DIR, "output")

TYPE_MOUNTAIN = 0
TYPE_VALLEY = 1

# Colors
PANEL_FACE = "#e8e8ee"
PANEL_EDGE = "#6a6a78"
ACTIVE_PANEL_FACE = "#f5c26b"
ACTIVE_PANEL_EDGE = "#c47d1a"
TRIM_FACE = "#e45756"
TRIM_EDGE = "#a32020"
# Ghost intermediate shells (collision-only stack samples)
GHOST_FACE = "#4c78a8"
GHOST_EDGE = "#1f4e79"
# Vertical side panels (collision-only, unique indices)
SIDE_FACE = "#59a14f"
SIDE_EDGE = "#2d6a2d"
CREASE_MV = "#444444"
CREASE_BORDER = "#bbbbbb"
GUIDE_COLOR = "#9aa0a8"
TEXT_COLOR = "#222222"
LAYER_CMAP = "coolwarm"


def _shell_kind_of(entry: Optional[dict]) -> str:
    """Normalize shell_kind: 'ghost' | 'side' | 'physical' (legacy → physical)."""
    if not entry:
        return "physical"
    raw = entry.get("shell_kind")
    if raw is None:
        return "physical"
    s = str(raw).strip().lower()
    if s in ("ghost", "g", "intermediate", "collision_only"):
        return "ghost"
    if s in ("side", "s", "vertical", "wall"):
        return "side"
    return "physical"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_in_dirs(name: str, dirs: Sequence[str]) -> Optional[str]:
    if os.path.isfile(name):
        return name
    basenames = [name]
    if not name.endswith(".json"):
        basenames.append(name + ".json")
    for d in dirs:
        for bn in basenames:
            for candidate in (
                os.path.join(d, bn),
                os.path.join(d, os.path.basename(bn)),
            ):
                if os.path.isfile(candidate):
                    return candidate
    return None


def _resolve_json_path(name: str, *, prefer_trimmed: bool = False) -> str:
    legacy_trimmed = os.path.join(_PROJECT_ROOT, "trimmedData")
    legacy_cleaned = os.path.join(_PANEL_TRIM_DIR, "cleanedData")
    legacy_root_cleaned = os.path.join(_PROJECT_ROOT, "cleanedData")
    dirs = (
        [TRIMMED_DIR, legacy_cleaned, legacy_root_cleaned, legacy_trimmed, DESCRIPTION_DIR]
        if prefer_trimmed
        else [DESCRIPTION_DIR, TRIMMED_DIR, legacy_cleaned, legacy_root_cleaned, legacy_trimmed]
    )
    path = _resolve_in_dirs(name, dirs)
    if path is not None:
        return path
    raise FileNotFoundError(
        f"JSON not found for name/path: {name} (searched {', '.join(dirs)})"
    )


def _guess_original_name(trimmed_name: str) -> Optional[str]:
    base = os.path.basename(trimmed_name)
    if base.endswith(".json"):
        base = base[:-5]
    if base.endswith("-trimmed"):
        return base[: -len("-trimmed")]
    if base.endswith("_trimmed"):
        return base[: -len("_trimmed")]
    return None


def _layer_key(h: float) -> str:
    return str(round(float(h), 6))


def _poly_xy(unit: Sequence) -> np.ndarray:
    return np.asarray([[float(p[0]), float(p[1])] for p in unit], dtype=float)


def _ribbon_from_dual_curves(c0: Sequence, c1: Sequence) -> Optional[np.ndarray]:
    """Same dual-curve ribbon as 2D visualize.py (c0 then reverse(c1))."""
    if not c0 or not c1:
        return None
    n = min(len(c0), len(c1))
    if n < 2:
        return None
    ring = [[float(c0[i][0]), float(c0[i][1])] for i in range(n)]
    ring += [[float(c1[i][0]), float(c1[i][1])] for i in range(n - 1, -1, -1)]
    if len(ring) < 3:
        return None
    return np.asarray(ring, dtype=float)


def _poly_area_xy(poly: Optional[np.ndarray]) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _simplify_closed_poly(poly: np.ndarray, max_verts: int = 48) -> np.ndarray:
    """
    Downsample a closed dual-curve ribbon for fast 3D drawing.

    Keeps first/last of each half so the ribbon ends stay accurate.
    """
    if poly is None or len(poly) <= max_verts:
        return poly
    n = len(poly)
    # Even stride subsample; ensure last vertex of open chain is kept before close
    idx = np.linspace(0, n - 1, num=max_verts, dtype=int)
    idx = np.unique(idx)
    return poly[idx]


# ---------------------------------------------------------------------------
# Geometry / layers
# ---------------------------------------------------------------------------

def _xy_key(pt, nd: int = 3) -> Tuple[float, float]:
    return (round(float(pt[0]), nd), round(float(pt[1]), nd))


def _edge_key(a, b, nd: int = 3) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ka, kb = _xy_key(a, nd), _xy_key(b, nd)
    return (ka, kb) if ka <= kb else (kb, ka)


def _line_edge_table(data: dict) -> Dict[Tuple, List[Dict[str, Any]]]:
    """Undirected design-xy edge → list of {type, height} from lines/features."""
    lines = data.get("lines") or []
    feats = data.get("line_features") or []
    table: Dict[Tuple, List[Dict[str, Any]]] = {}
    for i, ln in enumerate(lines):
        if not ln or len(ln) < 2:
            continue
        a, b = ln[0], ln[1]
        t = 2
        th = None
        if i < len(feats):
            t = int(feats[i].get("type", 2))
            if feats[i].get("thick_panel_height") is not None:
                th = float(feats[i]["thick_panel_height"])
        if th is None and len(a) > 2:
            th = float(a[2])
        if th is None:
            th = 0.0
        ek = _edge_key(a, b)
        table.setdefault(ek, []).append({"type": t, "height": float(th), "index": i})
    return table


def panel_thickness_offsets(
    units: Sequence,
    data: dict,
    shaded_regions: Optional[Sequence[dict]] = None,
) -> List[List[float]]:
    """
    Per design-panel thickness offsets (mm), matching phys_sim thick-mode build.

    For each panel, sim collects unique ``thick_panel_height`` of **non-border**
    creases on that panel's outline, then instantiates one sim-unit layer at
    each height (planar copy at z = height).

    Also merges ``layer_h`` from shaded_regions for that panel so collision
    layers never disappear if edge matching is slightly off.
    """
    table = _line_edge_table(data)
    n = len(units)
    per_panel: List[List[float]] = [[] for _ in range(n)]

    for pi, unit in enumerate(units):
        if not unit or len(unit) < 2:
            continue
        heights: List[float] = []
        nu = len(unit)
        for j in range(nu):
            a, b = unit[j], unit[(j + 1) % nu]
            for ent in table.get(_edge_key(a, b), []):
                # Same rule as phys_sim_pd14.start thick branch:
                # only mountain/valley (non-border) crease heights define layers.
                if ent["type"] in (TYPE_MOUNTAIN, TYPE_VALLEY):
                    h = float(ent["height"])
                    if not any(abs(h - x) < 1e-9 for x in heights):
                        heights.append(h)
        heights.sort()
        per_panel[pi] = heights

    # Ensure shade layer_h is present for its panel (source of truth for paint)
    for sh in shaded_regions or []:
        p = sh.get("panel")
        lh = sh.get("layer_h")
        if p is None or lh is None:
            continue
        try:
            pi = int(p)
            h = float(lh)
        except (TypeError, ValueError):
            continue
        if pi < 0 or pi >= n:
            continue
        if not any(abs(h - x) < 1e-9 for x in per_panel[pi]):
            per_panel[pi].append(h)
            per_panel[pi].sort()

    # Panels with no crease heights: fall back to z=0 (flat design)
    for pi in range(n):
        if not per_panel[pi]:
            per_panel[pi] = [0.0]

    return per_panel


def _collect_layer_heights(
    trimmed: dict,
    base: dict,
    layers: Optional[Sequence[float]] = None,
    panel_offsets: Optional[Sequence[Sequence[float]]] = None,
) -> List[float]:
    """
    Unique **panel thickness offsets** used in the stack (not border-line z).

    Prefer reconstructed per-panel crease heights (+ shade layer_h). Avoids
    spurious planes like border z=-10 / 0 that are not thick-panel layers.
    """
    heights: List[float] = []

    if panel_offsets is not None:
        for offs in panel_offsets:
            heights.extend(float(h) for h in offs)
    else:
        units = list(trimmed.get("units") or base.get("units") or [])
        shades = list(trimmed.get("shaded_regions") or [])
        for offs in panel_thickness_offsets(units, base or trimmed, shades):
            heights.extend(offs)
    # Always include every shade layer_h (parity with 2D subplots)
    for sh in trimmed.get("shaded_regions") or []:
        if sh.get("layer_h") is not None:
            heights.append(float(sh["layer_h"]))
    stats = trimmed.get("collision_stats") or {}
    for bucket_name in ("shaded_areas", "closed_polygons", "segments"):
        for ent in stats.get(bucket_name) or []:
            if ent.get("layer_h") is not None:
                heights.append(float(ent["layer_h"]))

    uniq: List[float] = []
    seen = set()
    for h in sorted(heights):
        key = round(h, 6)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(h))

    if layers is not None:
        want = [float(h) for h in layers]
        uniq = [h for h in uniq if any(abs(h - w) < 1e-6 for w in want)]

    if not uniq:
        uniq = [0.0]
    return uniq


def _panel_has_offset(panel_offsets: Sequence[Sequence[float]], panel: int, h: float, eps: float = 1e-6) -> bool:
    if panel < 0 or panel >= len(panel_offsets):
        return False
    return any(abs(float(h) - float(x)) <= eps for x in panel_offsets[panel])


def _point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-cast point-in-polygon (xy)."""
    x, y = float(pt[0]), float(pt[1])
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
        ):
            inside = not inside
        j = i
    return inside


def _infer_panel_from_poly(
    poly: np.ndarray,
    panel_xy: Sequence[np.ndarray],
) -> Optional[int]:
    """Pick design panel whose outline best contains the shade centroid."""
    if poly is None or len(poly) < 1 or not panel_xy:
        return None
    c = poly.mean(axis=0)
    # Prefer containment; else nearest centroid
    best_i, best_d = None, float("inf")
    for i, uxy in enumerate(panel_xy):
        if uxy is None or len(uxy) < 3:
            continue
        if _point_in_poly(c, uxy):
            return i
        d = float(np.linalg.norm(c - uxy.mean(axis=0)))
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def _raw_shaded_regions(trimmed: dict) -> List[dict]:
    regions = list(trimmed.get("shaded_regions") or [])
    if regions:
        return regions
    stats = trimmed.get("collision_stats") or {}
    return list(stats.get("shaded_regions") or stats.get("shaded_areas") or [])


def _shade_poly_from_region(sh: dict) -> Optional[np.ndarray]:
    """Polygon geometry — same sources as 2D visualize._entries_from_shaded_regions."""
    c0, c1 = sh.get("c0") or [], sh.get("c1") or []
    if c0 and c1:
        poly = _ribbon_from_dual_curves(c0, c1)
        if poly is not None:
            return poly
    for key in ("coordinates", "shaded_polygon", "paint_polygon", "polygon"):
        raw = sh.get(key)
        if raw and len(raw) >= 3:
            return np.asarray(raw, dtype=float)[:, :2]
    return None


def resolve_shaded_associations(
    trimmed: dict,
    units: Optional[Sequence] = None,
) -> List[Dict[str, Any]]:
    """
    Resolve each shaded/collided region to a concrete (panel, layer_h).

    Geometry matches 2D visualize.py:
      dual_curve ribbon from c0/c1, plus sweep_samples for stroke drawing.
    Missing panel/layer fields are inferred when possible.
    """
    base_units = list(units if units is not None else (trimmed.get("units") or []))
    panel_xy = [_poly_xy(u) for u in base_units] if base_units else []
    n_panels = len(base_units)

    records: List[Dict[str, Any]] = []
    for idx, sh in enumerate(_raw_shaded_regions(trimmed)):
        c0 = sh.get("c0") or []
        c1 = sh.get("c1") or []
        poly = _shade_poly_from_region(sh)
        # Closed ribbon only (no per-sample strokes — those make 3D laggy)
        if poly is not None and len(poly) >= 3:
            poly = _simplify_closed_poly(poly, max_verts=48)
        area = sh.get("area")
        if area is None:
            area = _poly_area_xy(poly)
        kind = sh.get("kind") or sh.get("paint_kind") or (
            "sweep" if poly is not None and float(area or 0) > 1e-3 else "line"
        )

        # --- layer ---
        layer_h = sh.get("layer_h")
        layer_idx = sh.get("layer_idx")
        layer_source = "export"
        if layer_h is None and layer_idx is not None:
            layer_source = "layer_idx_only"
        elif layer_h is None:
            layer_source = "missing"

        # --- panel ---
        panel = sh.get("panel")
        unit = sh.get("unit")
        panel_source = "export"
        if panel is not None:
            try:
                panel = int(panel)
                if panel < 0:
                    panel = None
            except (TypeError, ValueError):
                panel = None
                panel_source = "invalid"
        if panel is None and unit is not None:
            try:
                ui = int(unit)
                if 0 <= ui < n_panels:
                    panel = ui
                    panel_source = "unit_as_panel"
            except (TypeError, ValueError):
                pass
        if panel is None and poly is not None and panel_xy:
            panel = _infer_panel_from_poly(poly, panel_xy)
            panel_source = "inferred_geometry" if panel is not None else "missing"
        if panel is None:
            panel_source = "missing"

        # Closed polygon only (same fill as 2D; skip zero-area line contacts)
        drawable = (
            poly is not None
            and len(poly) >= 3
            and float(area or 0.0) >= 1e-3
        )

        shell_kind = _shell_kind_of(sh)
        parent_panel = sh.get("parent_panel")
        if parent_panel is not None:
            try:
                parent_panel = int(parent_panel)
            except (TypeError, ValueError):
                parent_panel = None
        rec = {
            "index": idx,
            "panel": panel,
            "parent_panel": parent_panel,
            "unit": unit if unit is not None else sh.get("unit"),
            "layer_h": float(layer_h) if layer_h is not None else None,
            "layer_idx": int(layer_idx) if layer_idx is not None else None,
            "shell_kind": shell_kind,
            "h_lo": sh.get("h_lo"),
            "h_hi": sh.get("h_hi"),
            "depth_from_top_mm": sh.get("depth_from_top_mm"),
            "depth_from_bottom_mm": sh.get("depth_from_bottom_mm"),
            "stock_span_mm": sh.get("stock_span_mm"),
            "side": sh.get("side"),
            "unit_a": sh.get("unit_a"),
            "unit_b": sh.get("unit_b"),
            "tri_a": sh.get("tri_a"),
            "tri_b": sh.get("tri_b"),
            "kind": kind,
            "area": float(area) if area is not None else 0.0,
            "poly": poly if drawable else None,
            "panel_source": panel_source,
            "layer_source": layer_source,
            "associated": panel is not None and layer_h is not None,
            "drawable": drawable,
        }
        records.append(rec)
    return records


def association_report(
    records: Sequence[Dict[str, Any]],
    panel_offsets: Optional[Sequence[Sequence[float]]] = None,
) -> str:
    """Human-readable panel × layer association summary (incl. ghost/side)."""
    n = len(records)
    n_ok = sum(1 for r in records if r.get("associated"))
    n_panel_miss = sum(1 for r in records if r.get("panel") is None)
    n_layer_miss = sum(1 for r in records if r.get("layer_h") is None)
    n_mismatch = sum(1 for r in records if r.get("offset_mismatch"))
    n_ghost = sum(1 for r in records if _shell_kind_of(r) == "ghost")
    n_side = sum(1 for r in records if _shell_kind_of(r) == "side")
    n_phys = sum(1 for r in records if _shell_kind_of(r) == "physical")
    sources = defaultdict(int)
    for r in records:
        sources[r.get("panel_source", "?")] += 1

    by_pl: Dict[Tuple[Any, Any], int] = defaultdict(int)
    by_pl_kind: Dict[Tuple[Any, Any, str], int] = defaultdict(int)
    by_layer: Dict[Any, set] = defaultdict(set)
    by_panel: Dict[Any, set] = defaultdict(set)
    ghost_panels: set = set()
    side_panels: set = set()
    phys_panels: set = set()
    for r in records:
        p, h = r.get("panel"), r.get("layer_h")
        sk = _shell_kind_of(r)
        by_pl[(p, h)] += 1
        by_pl_kind[(p, h, sk)] += 1
        if h is not None:
            by_layer[h].add(p)
        if p is not None:
            by_panel[p].add(h)
            if sk == "ghost":
                ghost_panels.add(int(p))
            elif sk == "side":
                side_panels.add(int(p))
            else:
                phys_panels.add(int(p))

    lines = [
        f"shaded/collided regions: {n}",
        f"  shell_kind: physical={n_phys}  ghost={n_ghost}  side={n_side}",
        f"  panels with physical shades: {sorted(phys_panels)}",
        f"  panels with ghost shades:    {sorted(ghost_panels)}",
        f"  side panel unique indices:   {sorted(side_panels)}",
        f"  fully associated (panel + layer_h): {n_ok}/{n}",
        f"  missing panel: {n_panel_miss}   missing layer_h: {n_layer_miss}",
        f"  panel source counts: {dict(sources)}",
        f"  distinct panels with shades: {sorted(p for p in by_panel if p is not None)}",
        f"  shade layer_h values: {sorted(h for h in by_layer if h is not None)}",
    ]
    if panel_offsets is not None:
        lines.append("  per-panel thickness offsets (crease thick_panel_height):")
        for pi, offs in enumerate(panel_offsets):
            shade_hs = sorted(by_panel.get(pi, set()) - {None})
            mark = ""
            for h in shade_hs:
                if not any(abs(float(h) - float(x)) < 1e-6 for x in offs):
                    mark = "  ⚠ shade layer_h not in offsets"
                    break
            lines.append(
                f"    P{pi}: offsets={list(offs)}  "
                f"shades@={shade_hs}{mark}"
            )
        if n_mismatch:
            lines.append(
                f"  ⚠ {n_mismatch} shade(s) have layer_h outside reconstructed offsets"
            )
    lines.append("  panel × layer_h × shell_kind  (count):")
    def _kind_rank(sk: str) -> int:
        if sk == "physical":
            return 0
        if sk == "ghost":
            return 1
        if sk == "side":
            return 2
        return 3

    for (p, h, sk), c in sorted(
        by_pl_kind.items(),
        key=lambda kv: (
            float(kv[0][1]) if kv[0][1] is not None else 1e99,
            kv[0][0] if kv[0][0] is not None else -1,
            _kind_rank(kv[0][2]),
        ),
    ):
        p_s = f"P{p}" if p is not None else "P?"
        h_s = f"h={h:g}" if h is not None else "h=?"
        if sk == "ghost":
            sk_s = "ghost"
        elif sk == "side":
            sk_s = "side"
        else:
            sk_s = "phys"
        ok = ""
        if sk == "side":
            ok = "  (collision-only vertical; unique idx)"
        elif sk == "ghost":
            ok = "  (collision-only intermediate)"
        elif (
            panel_offsets is not None
            and p is not None
            and h is not None
            and 0 <= int(p) < len(panel_offsets)
        ):
            if not _panel_has_offset(panel_offsets, int(p), float(h)):
                ok = "  ⚠ not in panel offsets"
        lines.append(f"    {p_s:>4} @ {h_s:<12} {sk_s:<5} ×{c}{ok}")
    return "\n".join(lines)


def _shade_polys_by_layer(
    trimmed: dict,
    units: Optional[Sequence] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Map layer_key → list of shade items (same dual-curve set as 2D).

    Includes filled ribbons **and** line/stroke-only contacts so every
    dual_curve export entry that 2D can show is present in 3D.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for rec in resolve_shaded_associations(trimmed, units=units):
        lh = rec.get("layer_h")
        if lh is None:
            continue
        if not rec.get("drawable"):
            continue
        k = _layer_key(lh)
        out.setdefault(k, []).append(rec)
    return out


def _active_panels_by_layer(shades: Dict[str, List[Dict[str, Any]]]) -> Dict[str, set]:
    active: Dict[str, set] = {}
    for k, items in shades.items():
        s = set()
        for it in items:
            if it.get("panel") is not None:
                try:
                    s.add(int(it["panel"]))
                except (TypeError, ValueError):
                    pass
        active[k] = s
    return active


def _lines_xyz(
    data: dict,
    layer_heights: Sequence[float],
    eps: float = 1e-3,
) -> List[Dict[str, Any]]:
    """Crease/border segments with native z (or thick_panel_height)."""
    lines = data.get("lines") or []
    feats = data.get("line_features") or []
    height_set = {round(float(h), 6) for h in layer_heights}
    out: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines):
        if not ln or len(ln) < 2:
            continue
        p0, p1 = ln[0], ln[1]
        z0 = float(p0[2]) if len(p0) > 2 else None
        z1 = float(p1[2]) if len(p1) > 2 else None
        th = None
        t = 2
        if i < len(feats):
            t = int(feats[i].get("type", 2))
            if feats[i].get("thick_panel_height") is not None:
                th = float(feats[i]["thick_panel_height"])
        if z0 is None:
            z0 = th if th is not None else 0.0
        if z1 is None:
            z1 = th if th is not None else 0.0
        # Keep segments whose z is near a known layer (or both ends equal)
        z_mid = 0.5 * (z0 + z1)
        if height_set and not any(abs(z_mid - h) <= eps for h in layer_heights):
            # still keep if both ends share a height near any layer endpoint
            if not any(abs(z0 - h) <= eps or abs(z1 - h) <= eps for h in layer_heights):
                continue
        out.append({
            "a": np.asarray([float(p0[0]), float(p0[1]), z0], dtype=float),
            "b": np.asarray([float(p1[0]), float(p1[1]), z1], dtype=float),
            "type": t,
            "z_ref": z_mid if abs(z0 - z1) < eps else z_mid,
        })
    return out


def _explode_z(h: float, heights: Sequence[float], factor: float) -> float:
    """
    Map true thickness height → display z.

    factor=0 → true stack (z = h)
    factor>0 → layers pull apart around the mid-height
    """
    if not heights:
        return float(h)
    hs = np.asarray(heights, dtype=float)
    mid = float(hs.mean())
    # Extra gap proportional to rank spacing so sparse layers still separate
    span = float(hs.max() - hs.min()) if len(hs) > 1 else 1.0
    span = max(span, 1.0)
    # Rank-based offset: evenly space layers when factor is large
    order = {round(float(x), 6): i for i, x in enumerate(sorted(set(float(x) for x in hs)))}
    n = max(len(order) - 1, 1)
    rank = order.get(round(float(h), 6), 0)
    rank_z = (rank / n - 0.5) * span  # centered rank coordinate
    # Blend true geometry with rank spacing
    true = float(h) - mid
    blended = true * (1.0 + 0.35 * factor) + rank_z * factor
    return mid + blended


def _is_reddish(rgb: Tuple[float, float, float]) -> bool:
    """True if color is red-ish (reserved for collided / shaded regions)."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    # Strong red channel relative to green/blue (covers tab10 red, dark red, etc.)
    return r > 0.55 and r > g + 0.15 and r > b + 0.15


def _layer_solid_colors(
    heights: Sequence[float],
) -> Dict[float, Tuple[float, float, float]]:
    """
    One solid RGB color per thickness offset for **panels only**.

    Skips red hues — red is reserved for collided / shaded regions.
    """
    import matplotlib.pyplot as plt

    n = len(heights)
    if n <= 0:
        return {}
    # tab10/tab20; filter out red so collision paint stays unique
    cmap = plt.get_cmap("tab20")
    palette: List[Tuple[float, float, float]] = []
    for i in range(cmap.N):
        rgba = cmap(i)
        rgb = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
        if not _is_reddish(rgb):
            palette.append(rgb)
    # Fallbacks if filter is too aggressive
    if not palette:
        palette = [
            (0.12, 0.47, 0.71),  # blue
            (0.20, 0.63, 0.17),  # green
            (0.89, 0.47, 0.76),  # pink
            (0.50, 0.50, 0.50),  # grey
            (0.74, 0.74, 0.13),  # olive
            (0.09, 0.75, 0.81),  # cyan
        ]
    out: Dict[float, Tuple[float, float, float]] = {}
    for i, h in enumerate(heights):
        out[float(h)] = palette[i % len(palette)]
    return out


def _color_for_height(
    layer_colors: Dict[float, Tuple[float, float, float]],
    h: float,
    eps: float = 1e-6,
) -> Tuple[float, float, float]:
    hf = float(h)
    if hf in layer_colors:
        return layer_colors[hf]
    for kh, rgb in layer_colors.items():
        if abs(float(kh) - hf) <= eps:
            return rgb
    return (0.75, 0.75, 0.78)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _xy_to_verts3d(xy: np.ndarray, z: float) -> np.ndarray:
    zcol = np.full((len(xy), 1), float(z), dtype=float)
    return np.hstack([xy.astype(float), zcol])


def _set_equal_aspect_3d(ax, xs, ys, zs):
    xs, ys, zs = np.asarray(xs), np.asarray(ys), np.asarray(zs)
    if xs.size == 0:
        return
    xmid, ymid, zmid = xs.mean(), ys.mean(), zs.mean()
    r = 0.5 * max(
        xs.max() - xs.min(),
        ys.max() - ys.min(),
        zs.max() - zs.min(),
        1.0,
    )
    ax.set_xlim(xmid - r, xmid + r)
    ax.set_ylim(ymid - r, ymid + r)
    ax.set_zlim(zmid - r, zmid + r)


def build_explosion_scene(
    trimmed: dict,
    original: Optional[dict] = None,
    layers: Optional[Sequence[float]] = None,
    explosion_factor: float = 1.5,
    show_panels: bool = True,
    show_shades: bool = True,
    show_lines: bool = True,
    show_guides: bool = True,
    show_panel_ids: bool = False,
    show_layer_labels: bool = True,
    show_shade_labels: bool = False,
    color_shades_by_panel: bool = False,  # ignored: color is by layer only
    panel_alpha: float = 0.42,
    shade_alpha: float = 1.0,
    title: Optional[str] = None,
):
    """
    Build an interactive (or static) 3D explosion figure.

    Solid color per thickness layer for **panels** (each layer a different
    color). **Collided** regions = simplified closed dual-curve polygons
    (batched), solid red on every layer — no per-sample stroke swarm.

    Returns (fig, ax, state) where state holds redraw helpers for the slider.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.widgets import Button, CheckButtons, Slider
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    base = original or trimmed
    units = list(trimmed.get("units") or base.get("units") or [])
    if not units:
        raise ValueError("No units in JSON — nothing to draw.")

    shade_raw = list(trimmed.get("shaded_regions") or [])
    # Per-panel thickness offsets (true z of each thick layer for that panel)
    panel_offsets = panel_thickness_offsets(units, base, shade_raw)
    heights = _collect_layer_heights(
        trimmed, base, layers=layers, panel_offsets=panel_offsets
    )
    shade_records = resolve_shaded_associations(trimmed, units=units)
    # Validate shade layer_h against that panel's offsets; snap if needed
    for rec in shade_records:
        p, lh = rec.get("panel"), rec.get("layer_h")
        if p is None or lh is None:
            continue
        # Side panels use unique indices outside design panels; skip offset check.
        if _shell_kind_of(rec) == "side":
            rec["offset_mismatch"] = False
            continue
        if not _panel_has_offset(panel_offsets, int(p), float(lh)):
            # Keep export layer_h (authoritative for paint) but flag mismatch
            rec["offset_mismatch"] = True
        else:
            rec["offset_mismatch"] = False
    shades = _shade_polys_by_layer(trimmed, units=units)
    lines3d = _lines_xyz(trimmed if trimmed.get("lines") else base, heights)
    if not lines3d and base is not trimmed:
        lines3d = _lines_xyz(base, heights)

    n_panels = len(units)
    panel_xy = [_poly_xy(u) for u in units]
    layer_colors = _layer_solid_colors(heights)
    n_mismatch = sum(1 for r in shade_records if r.get("offset_mismatch"))
    state_assoc = {
        "records": shade_records,
        "n_associated": sum(1 for r in shade_records if r.get("associated")),
        "n_total": len(shade_records),
        "panel_offsets": panel_offsets,
        "n_offset_mismatch": n_mismatch,
    }

    # Bounds in xy
    all_xy = np.vstack(panel_xy) if panel_xy else np.zeros((1, 2))
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    pad = 0.06 * max(xmax - xmin, ymax - ymin, 1.0)
    xmin, xmax = xmin - pad, xmax + pad
    ymin, ymax = ymin - pad, ymax + pad

    # Shade count per height (for sidebar labels)
    shade_count_at: Dict[float, int] = {float(h): 0 for h in heights}
    for h in heights:
        shade_count_at[float(h)] = len(shades.get(_layer_key(h), []))

    fig = plt.figure(figsize=(12.5, 8.8))
    # 3D view | right sidebar for layer visibility
    ax = fig.add_axes([0.03, 0.14, 0.68, 0.80], projection="3d")
    slider_ax = fig.add_axes([0.12, 0.04, 0.50, 0.035])

    # Sidebar layout (right column)
    help_ax = fig.add_axes([0.74, 0.78, 0.24, 0.16])
    help_ax.axis("off")
    btn_all_ax = fig.add_axes([0.74, 0.725, 0.11, 0.035])
    btn_none_ax = fig.add_axes([0.87, 0.725, 0.11, 0.035])
    # Check list height scales with layer count (capped)
    n_h = max(len(heights), 1)
    check_h = min(0.48, 0.055 * n_h + 0.08)
    check_bottom = 0.72 - check_h - 0.01
    check_ax = fig.add_axes([0.74, check_bottom, 0.24, check_h])

    state: Dict[str, Any] = {
        "factor": float(explosion_factor),
        "heights": heights,
        # layer_h → visible (explosion spacing still uses full height list)
        "layer_visible": {float(h): True for h in heights},
        "collections": [],
        "line_artists": [],
        "text_artists": [],
        "guide_artists": [],
        "associations": state_assoc,
    }

    def _layer_visible(h: float, eps: float = 1e-6) -> bool:
        vis = state["layer_visible"]
        hf = float(h)
        if hf in vis:
            return bool(vis[hf])
        for kh, on in vis.items():
            if abs(float(kh) - hf) <= eps:
                return bool(on)
        return True

    def _visible_heights() -> List[float]:
        return [float(h) for h in heights if _layer_visible(h)]

    def _clear_artists():
        for coll in state["collections"]:
            try:
                coll.remove()
            except Exception:
                pass
        for art in state["line_artists"] + state["guide_artists"]:
            try:
                art.remove()
            except Exception:
                pass
        for t in state["text_artists"]:
            try:
                t.remove()
            except Exception:
                pass
        state["collections"] = []
        state["line_artists"] = []
        state["guide_artists"] = []
        state["text_artists"] = []

    def _draw(factor: float):
        _clear_artists()
        factor = float(factor)
        state["factor"] = factor

        vis_heights = _visible_heights()
        zs_all: List[float] = []
        xs_all: List[float] = list(all_xy[:, 0])
        ys_all: List[float] = list(all_xy[:, 1])

        # Explosion mapping keeps full stack so toggles don't jump positions
        z_of = {h: _explode_z(h, heights, factor) for h in heights}

        # --- panels: batch one Poly3DCollection per layer color (fast) ---
        if show_panels:
            panels_by_h: Dict[float, List[np.ndarray]] = {}
            for i, xy in enumerate(panel_xy):
                if len(xy) < 3:
                    continue
                offs = panel_offsets[i] if i < len(panel_offsets) else [0.0]
                for h in offs:
                    if not _layer_visible(h):
                        continue
                    if layers is not None and not any(
                        abs(float(h) - float(w)) < 1e-6 for w in layers
                    ):
                        continue
                    if not any(abs(float(h) - float(g)) < 1e-6 for g in heights):
                        continue
                    hf = float(h)
                    panels_by_h.setdefault(hf, []).append(xy)
            for h, xylist in panels_by_h.items():
                z = _explode_z(h, heights, factor)
                zs_all.append(z)
                rgb = _color_for_height(layer_colors, h)
                edge = (rgb[0] * 0.55, rgb[1] * 0.55, rgb[2] * 0.55)
                verts = [_xy_to_verts3d(xy, z) for xy in xylist]
                coll = Poly3DCollection(
                    verts,
                    facecolors=[(*rgb, panel_alpha)] * len(verts),
                    edgecolors=[edge] * len(verts),
                    linewidths=0.5,
                    zsort="average",
                )
                ax.add_collection3d(coll)
                state["collections"].append(coll)

        # --- shaded: closed dual-curve polygons (phys=red, ghost=blue, side=green) ---
        n_shade_vis = 0
        n_shade_ghost_vis = 0
        n_shade_side_vis = 0
        n_shade_phys_vis = 0
        if show_shades:
            phys_rgb = to_rgb(TRIM_FACE)
            phys_edge = to_rgb(TRIM_EDGE)
            ghost_rgb = to_rgb(GHOST_FACE)
            ghost_edge = to_rgb(GHOST_EDGE)
            side_rgb = to_rgb(SIDE_FACE)
            side_edge = to_rgb(SIDE_EDGE)
            # Sit almost on the panel (tiny lift only to avoid z-fighting)
            if len(heights) >= 2:
                z_span = abs(
                    _explode_z(heights[-1], heights, factor)
                    - _explode_z(heights[0], heights, factor)
                )
            else:
                z_span = max(abs(float(heights[0])), 1.0) if heights else 1.0
            z_lift = max(0.02, 0.0015 * max(z_span, 1.0))

            phys_verts: List[np.ndarray] = []
            ghost_verts: List[np.ndarray] = []
            side_verts: List[np.ndarray] = []
            for key, items in shades.items():
                try:
                    h = float(key)
                except (TypeError, ValueError):
                    continue
                if not _layer_visible(h):
                    continue
                z_panel = _explode_z(h, heights, factor)
                z_bias = z_panel + z_lift
                for item in items:
                    poly = item.get("poly")
                    if poly is None or len(poly) < 3:
                        continue
                    if float(item.get("area") or 0.0) < 1e-3:
                        continue
                    sk = _shell_kind_of(item)
                    verts3 = _xy_to_verts3d(poly, z_bias)
                    if sk == "ghost":
                        ghost_verts.append(verts3)
                    elif sk == "side":
                        side_verts.append(verts3)
                    else:
                        phys_verts.append(verts3)
                    if show_shade_labels and item.get("panel") is not None:
                        c = poly.mean(axis=0)
                        if sk == "ghost":
                            tag = f"P{int(item['panel'])}·G"
                            tcolor = "#0a2a5a"
                        elif sk == "side":
                            tag = f"P{int(item['panel'])}·S"
                            tcolor = "#1a4a1a"
                        else:
                            tag = f"P{int(item['panel'])}"
                            tcolor = "#5a0a0a"
                        t = ax.text(
                            float(c[0]), float(c[1]), z_bias,
                            tag,
                            color=tcolor,
                            fontsize=5.5, ha="center", va="center",
                        )
                        state["text_artists"].append(t)

            n_shade_phys_vis = len(phys_verts)
            n_shade_ghost_vis = len(ghost_verts)
            n_shade_side_vis = len(side_verts)
            n_shade_vis = n_shade_phys_vis + n_shade_ghost_vis + n_shade_side_vis
            if phys_verts:
                coll = Poly3DCollection(
                    phys_verts,
                    facecolors=[(*phys_rgb, shade_alpha)] * len(phys_verts),
                    edgecolors=[phys_edge] * len(phys_verts),
                    linewidths=0.7,
                    zsort="average",
                )
                ax.add_collection3d(coll)
                state["collections"].append(coll)
            if ghost_verts:
                coll = Poly3DCollection(
                    ghost_verts,
                    facecolors=[(*ghost_rgb, shade_alpha)] * len(ghost_verts),
                    edgecolors=[ghost_edge] * len(ghost_verts),
                    linewidths=0.7,
                    zsort="average",
                )
                ax.add_collection3d(coll)
                state["collections"].append(coll)
            if side_verts:
                coll = Poly3DCollection(
                    side_verts,
                    facecolors=[(*side_rgb, shade_alpha)] * len(side_verts),
                    edgecolors=[side_edge] * len(side_verts),
                    linewidths=0.7,
                    zsort="average",
                )
                ax.add_collection3d(coll)
                state["collections"].append(coll)

        # --- collision-only vertical side panel walls (fill layer-height gaps) ---
        side_panels_reg = list(trimmed.get("side_panels") or [])
        if show_panels and side_panels_reg:
            side_wall_verts: List[np.ndarray] = []
            side_rgb_w = to_rgb(SIDE_FACE)
            side_edge_w = to_rgb(SIDE_EDGE)
            for sp in side_panels_reg:
                try:
                    h_lo = float(sp.get("h_lo"))
                    h_hi = float(sp.get("h_hi"))
                except (TypeError, ValueError):
                    continue
                # Draw if either bounding layer (or mid) is visible
                h_mid = float(sp.get("layer_h", 0.5 * (h_lo + h_hi)))
                if not (
                    _layer_visible(h_lo)
                    or _layer_visible(h_hi)
                    or _layer_visible(h_mid)
                ):
                    continue
                outline = sp.get("outline_xy") or []
                if len(outline) < 2:
                    continue
                z0 = _explode_z(h_lo, heights, factor)
                z1 = _explode_z(h_hi, heights, factor)
                zs_all.extend([z0, z1])
                n_o = len(outline)
                for j in range(n_o):
                    a = outline[j]
                    b = outline[(j + 1) % n_o]
                    # Vertical quad between consecutive outline edges
                    side_wall_verts.append(np.asarray([
                        [float(a[0]), float(a[1]), z0],
                        [float(b[0]), float(b[1]), z0],
                        [float(b[0]), float(b[1]), z1],
                        [float(a[0]), float(a[1]), z1],
                    ], dtype=float))
            if side_wall_verts:
                coll = Poly3DCollection(
                    side_wall_verts,
                    facecolors=[(*side_rgb_w, 0.22)] * len(side_wall_verts),
                    edgecolors=[side_edge_w] * len(side_wall_verts),
                    linewidths=0.35,
                    zsort="average",
                )
                ax.add_collection3d(coll)
                state["collections"].append(coll)

        # --- creases / borders (only if either end's native z layer is visible) ---
        if show_lines and lines3d:
            segs = []
            colors = []
            widths = []
            for ln in lines3d:
                a, b = ln["a"], ln["b"]
                ha, hb = float(a[2]), float(b[2])
                if not (_layer_visible(ha) or _layer_visible(hb)):
                    continue
                # If only one end is on a visible layer, still draw (partial edge)
                za = _explode_z(ha, heights, factor)
                zb = _explode_z(hb, heights, factor)
                segs.append([[a[0], a[1], za], [b[0], b[1], zb]])
                zs_all.extend([za, zb])
                if ln["type"] in (TYPE_MOUNTAIN, TYPE_VALLEY):
                    colors.append(CREASE_MV)
                    widths.append(1.1)
                else:
                    colors.append(CREASE_BORDER)
                    widths.append(0.45)
            if segs:
                lc = Line3DCollection(segs, colors=colors, linewidths=widths, alpha=0.85)
                ax.add_collection3d(lc)
                state["line_artists"].append(lc)

        # --- vertical guide rails across visible offsets only ---
        if show_guides:
            guide_segs = []
            for i, xy in enumerate(panel_xy):
                offs = panel_offsets[i] if i < len(panel_offsets) else []
                offs = [
                    float(h) for h in offs
                    if any(abs(float(h) - float(g)) < 1e-6 for g in heights)
                    and _layer_visible(h)
                ]
                if len(offs) < 2:
                    continue
                c = xy.mean(axis=0)
                z_lo = _explode_z(min(offs), heights, factor)
                z_hi = _explode_z(max(offs), heights, factor)
                guide_segs.append([[c[0], c[1], z_lo], [c[0], c[1], z_hi]])
            if guide_segs:
                lc = Line3DCollection(
                    guide_segs, colors=GUIDE_COLOR, linewidths=0.35,
                    linestyles="dashed", alpha=0.45,
                )
                ax.add_collection3d(lc)
                state["guide_artists"].append(lc)

        # --- labels ---
        if show_layer_labels:
            x_lab = xmax + 0.02 * (xmax - xmin)
            for h in heights:
                if not _layer_visible(h):
                    continue
                z = z_of[h]
                t = ax.text(
                    x_lab, ymax, z,
                    f"h={h:g}",
                    color=TEXT_COLOR, fontsize=8, ha="left", va="center",
                )
                state["text_artists"].append(t)
                zs_all.append(z)

        if show_panel_ids:
            for i, xy in enumerate(panel_xy):
                offs = [
                    float(h)
                    for h in (panel_offsets[i] if i < len(panel_offsets) else heights)
                    if _layer_visible(h)
                ]
                if not offs:
                    continue
                mid_h = offs[len(offs) // 2]
                z = _explode_z(float(mid_h), heights, factor)
                c = xy.mean(axis=0)
                t = ax.text(
                    c[0], c[1], z,
                    str(i), color=TEXT_COLOR, fontsize=7,
                    ha="center", va="center",
                )
                state["text_artists"].append(t)

        if not zs_all:
            # keep a stable frame when everything is hidden
            zs_all = [z_of[h] for h in heights] if heights else [0.0]
        _set_equal_aspect_3d(ax, xs_all, ys_all, zs_all)

        n_shade = sum(len(v) for v in shades.values())
        n_ghost_all = sum(
            1 for items in shades.values()
            for it in items if _shell_kind_of(it) == "ghost"
        )
        n_side_all = sum(
            1 for items in shades.values()
            for it in items if _shell_kind_of(it) == "side"
        )
        n_phys_all = sum(
            1 for items in shades.values()
            for it in items if _shell_kind_of(it) == "physical"
        )
        n_assoc = state_assoc["n_associated"]
        n_tot = state_assoc["n_total"]
        n_vis = len(vis_heights)
        n_side_reg = len(trimmed.get("side_panels") or [])
        ttl = title or "Thick-panel explosion"
        ax.set_title(
            f"{ttl}\n"
            f"layers {n_vis}/{len(heights)} visible  panels={n_panels}  "
            f"side_walls={n_side_reg}  "
            f"shades={n_shade_vis}/{n_shade} "
            f"(phys={n_shade_phys_vis}/{n_phys_all} "
            f"ghost={n_shade_ghost_vis}/{n_ghost_all} "
            f"side={n_shade_side_vis}/{n_side_all})  "
            f"assoc {n_assoc}/{n_tot}  explosion={factor:.2f}",
            fontsize=11, pad=10,
        )
        fig.canvas.draw_idle()

    # Axes cosmetics
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z (thickness, exploded)")
    ax.view_init(elev=22, azim=-58)
    try:
        ax.set_box_aspect((1, 1, 0.85))
    except Exception:
        pass

    # Slider
    slider = Slider(
        slider_ax,
        "Explosion",
        valmin=0.0,
        valmax=6.0,
        valinit=float(explosion_factor),
        valstep=0.05,
        color="#6c8ebf",
    )

    def _on_slider(val):
        _draw(val)

    slider.on_changed(_on_slider)
    state["slider"] = slider
    state["draw"] = _draw

    # ---- Sidebar: help + All/None + layer checkboxes ----
    # UI order: tallest (highest h) → lowest (smallest h). Internal heights stay ascending.
    heights_ui = sorted((float(h) for h in heights), reverse=True)

    n_assoc = state_assoc["n_associated"]
    n_tot = state_assoc["n_total"]
    n_ghost_help = sum(
        1 for r in state_assoc["records"] if _shell_kind_of(r) == "ghost"
    )
    n_side_help = sum(
        1 for r in state_assoc["records"] if _shell_kind_of(r) == "side"
    )
    n_phys_help = sum(
        1 for r in state_assoc["records"] if _shell_kind_of(r) == "physical"
    )
    n_side_walls = len(trimmed.get("side_panels") or [])
    help_ax.text(
        0.0, 1.0,
        "Layers\n"
        "────────\n"
        "Top = tallest h\n"
        "Bottom = lowest h\n"
        "Panels: color / layer\n"
        "Phys shade: red  P#\n"
        "Ghost shade: blue P#·G\n"
        "Side walls: green P#·S\n"
        f"phys={n_phys_help} ghost={n_ghost_help}\n"
        f"side={n_side_help} walls={n_side_walls}\n"
        f"assoc {n_assoc}/{n_tot}",
        transform=help_ax.transAxes,
        va="top", ha="left", fontsize=7.5, color="#333333",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="#f7f7f9",
            edgecolor="#cccccc", alpha=0.95,
        ),
    )

    check_labels = [
        f"h={h:g}  ({shade_count_at.get(float(h), 0)} sh)"
        for h in heights_ui
    ]
    # All checked initially
    check = CheckButtons(
        check_ax,
        check_labels,
        actives=[True] * len(check_labels) if check_labels else [],
    )
    # Label color matches the solid layer color
    try:
        for i, label in enumerate(check.labels):
            label.set_fontsize(8.5)
            if i < len(heights_ui):
                rgb = _color_for_height(layer_colors, heights_ui[i])
                label.set_color(rgb)
                label.set_fontweight("bold")
    except Exception:
        pass

    def _sync_visibility_from_checks():
        status = list(check.get_status()) if check_labels else []
        for i, h in enumerate(heights_ui):
            on = bool(status[i]) if i < len(status) else True
            state["layer_visible"][float(h)] = on

    def _on_check(label):
        _sync_visibility_from_checks()
        _draw(state["factor"])

    check.on_clicked(_on_check)

    def _set_all_checks(value: bool):
        """Force every checkbox to on/off and redraw."""
        if not check_labels:
            return
        status = list(check.get_status())
        for i, on in enumerate(status):
            if bool(on) != bool(value):
                check.set_active(i)
        _sync_visibility_from_checks()
        _draw(state["factor"])

    btn_all = Button(btn_all_ax, "All", color="#dce8f5", hovercolor="#b8d0ec")
    btn_none = Button(btn_none_ax, "None", color="#f0e0e0", hovercolor="#e0c0c0")
    btn_all.on_clicked(lambda _evt: _set_all_checks(True))
    btn_none.on_clicked(lambda _evt: _set_all_checks(False))

    state["layer_check"] = check
    state["heights_ui"] = heights_ui  # tallest → lowest (sidebar order)
    state["btn_all"] = btn_all
    state["btn_none"] = btn_none
    state["set_layer_visible"] = lambda h, on: (
        state["layer_visible"].__setitem__(float(h), bool(on)),
        _draw(state["factor"]),
    )

    _draw(explosion_factor)
    return fig, ax, state


def visualize_explosion_3d(
    trimmed: dict,
    original: Optional[dict] = None,
    layers: Optional[Sequence[float]] = None,
    explosion_factor: float = 1.5,
    show_panels: bool = True,
    show_shades: bool = True,
    show_lines: bool = True,
    show_guides: bool = True,
    show_panel_ids: bool = False,
    show_layer_labels: bool = True,
    show_shade_labels: bool = False,
    color_shades_by_panel: bool = False,
    title: Optional[str] = None,
    interactive: bool = True,
    save_path: Optional[str] = None,
    print_report: bool = True,
) -> Any:
    """
    Create the explosion figure; optionally save a PNG and/or show interactive UI.

    Returns the matplotlib Figure.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    units = list(
        (trimmed.get("units") or (original or {}).get("units") or [])
    )
    base = original or trimmed
    panel_offsets = panel_thickness_offsets(
        units, base, trimmed.get("shaded_regions") or []
    )
    records = resolve_shaded_associations(trimmed, units=units)
    for rec in records:
        p, lh = rec.get("panel"), rec.get("layer_h")
        if p is not None and lh is not None:
            rec["offset_mismatch"] = not _panel_has_offset(
                panel_offsets, int(p), float(lh)
            )
        else:
            rec["offset_mismatch"] = False
    if print_report:
        print(association_report(records, panel_offsets=panel_offsets))

    if interactive:
        backend = matplotlib.get_backend().lower()
        if backend == "agg":
            for candidate in ("TkAgg", "QtAgg", "Qt5Agg"):
                try:
                    plt.switch_backend(candidate)
                    break
                except Exception:
                    continue

    fig, ax, state = build_explosion_scene(
        trimmed=trimmed,
        original=original,
        layers=layers,
        explosion_factor=explosion_factor,
        show_panels=show_panels,
        show_shades=show_shades,
        show_lines=show_lines,
        show_guides=show_guides,
        show_panel_ids=show_panel_ids,
        show_layer_labels=show_layer_labels,
        show_shade_labels=show_shade_labels,
        color_shades_by_panel=color_shades_by_panel,
        title=title,
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
        print(f"[explosion-3d] wrote {save_path}")

    if interactive:
        print(
            "[explosion-3d] interactive window open — "
            "drag to orbit, use the Explosion slider, close window to exit."
        )
        plt.show()
    else:
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_one(
    sim: dict,
    output_dir: str,
    *,
    show: bool,
    save: bool,
    report_only: bool = False,
) -> Optional[str]:
    name = sim["name"]
    trimmed_path = _resolve_json_path(name, prefer_trimmed=True)
    trimmed = _load_json(trimmed_path)

    original = None
    orig_name = sim.get("original") or _guess_original_name(name)
    if orig_name:
        try:
            original = _load_json(_resolve_json_path(orig_name, prefer_trimmed=False))
        except FileNotFoundError:
            print(f"[warn] original JSON '{orig_name}' not found; using trimmed only")

    units = list(trimmed.get("units") or (original or {}).get("units") or [])
    base = original or trimmed
    panel_offsets = panel_thickness_offsets(
        units, base, trimmed.get("shaded_regions") or []
    )
    heights = _collect_layer_heights(
        trimmed, base, layers=sim.get("layers"), panel_offsets=panel_offsets
    )
    records = resolve_shaded_associations(trimmed, units=units)
    for rec in records:
        p, lh = rec.get("panel"), rec.get("layer_h")
        if _shell_kind_of(rec) == "side":
            rec["offset_mismatch"] = False
        elif p is not None and lh is not None:
            rec["offset_mismatch"] = not _panel_has_offset(
                panel_offsets, int(p), float(lh)
            )
        else:
            rec["offset_mismatch"] = False
    n_shade = len(records)
    n_ok = sum(1 for r in records if r.get("associated"))
    n_ghost = sum(1 for r in records if _shell_kind_of(r) == "ghost")
    n_side = sum(1 for r in records if _shell_kind_of(r) == "side")
    n_phys = sum(1 for r in records if _shell_kind_of(r) == "physical")
    n_side_panels = len(trimmed.get("side_panels") or [])
    print(
        f"[explosion-3d] {os.path.basename(trimmed_path)}  "
        f"thickness_offsets={heights}  units={len(units)}  "
        f"shaded_regions={n_shade}  phys={n_phys} ghost={n_ghost} "
        f"side={n_side}  side_panels={n_side_panels}  "
        f"associated={n_ok}/{n_shade}"
    )
    print(association_report(records, panel_offsets=panel_offsets))

    if report_only:
        return None

    out_path = None
    if save:
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(trimmed_path))[0]
        out_path = os.path.join(output_dir, f"{stem}_explosion_3d.png")

    visualize_explosion_3d(
        trimmed=trimmed,
        original=original,
        layers=sim.get("layers"),
        explosion_factor=float(sim.get("explosion_factor", sim.get("factor", 1.5))),
        show_panels=bool(sim.get("show_panels", True)),
        show_shades=bool(sim.get("show_shades", True)),
        show_lines=bool(sim.get("show_lines", True)),
        show_guides=bool(sim.get("show_guides", True)),
        show_panel_ids=bool(sim.get("show_panel_ids", False)),
        show_layer_labels=bool(sim.get("show_layer_labels", True)),
        show_shade_labels=bool(sim.get("show_shade_labels", False)),
        color_shades_by_panel=bool(sim.get("color_shades_by_panel", False)),
        title=os.path.basename(trimmed_path),
        interactive=show,
        save_path=out_path,
        print_report=False,  # already printed above
    )
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive 3D explosion diagram of thick-panel layers "
            "and dual-curve trim shades."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config (default: config_3d.yml or config.yml)",
    )
    parser.add_argument("--name", default=None, help="Design / trimmed JSON name")
    parser.add_argument(
        "--factor", type=float, default=None,
        help="Initial explosion factor (0 = true thickness stack)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not open interactive window (still can save PNG)",
    )
    parser.add_argument(
        "--save", action="store_true", default=None,
        help="Save a PNG snapshot (default: on when --no-show, else off)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write PNG",
    )
    parser.add_argument("-o", "--output", default=None, help="Output dir or .png path")
    parser.add_argument(
        "--no-guides", action="store_true", help="Hide vertical centroid guides"
    )
    parser.add_argument(
        "--panel-ids", action="store_true", help="Label panel indices"
    )
    parser.add_argument(
        "--no-shade-labels",
        action="store_true",
        help="Hide P# labels on shaded regions",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only print panel×layer association table (no figure)",
    )
    args = parser.parse_args(argv)

    if args.name:
        simulations = [{"name": args.name}]
    else:
        cfg_path = args.config
        if cfg_path is None:
            if os.path.isfile(DEFAULT_CONFIG_3D):
                cfg_path = DEFAULT_CONFIG_3D
            elif os.path.isfile(DEFAULT_CONFIG):
                cfg_path = DEFAULT_CONFIG
            else:
                cfg_path = os.path.join(_THIS_DIR, "config.example.yml")
        if not os.path.isfile(cfg_path):
            print(f"Config not found: {cfg_path}")
            return 1
        cfg = _load_yaml(cfg_path)
        simulations = list(cfg.get("simulations") or [])
        if not simulations:
            print("No simulations in config.")
            return 1

    # CLI overrides applied to every sim
    for sim in simulations:
        if args.factor is not None:
            sim["explosion_factor"] = args.factor
        if args.no_guides:
            sim["show_guides"] = False
        if args.panel_ids:
            sim["show_panel_ids"] = True
        if args.no_shade_labels:
            sim["show_shade_labels"] = False

    show = not args.no_show and not args.report_only
    if args.report_only:
        save = False
    elif args.no_save:
        save = False
    elif args.save:
        save = True
    else:
        # Default: save when headless, skip PNG when interactive-only
        save = (not show) or (args.output is not None)

    output_dir = DEFAULT_OUTPUT_DIR
    single_out = None
    if args.output:
        if args.output.lower().endswith(".png") and len(simulations) == 1:
            single_out = args.output
            output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
            save = True
        else:
            output_dir = args.output
            save = True

    paths = []
    for sim in simulations:
        p = _run_one(
            sim, output_dir, show=show, save=save, report_only=args.report_only
        )
        if p:
            paths.append(p)

    if single_out and paths:
        src = paths[0]
        if os.path.abspath(src) != os.path.abspath(single_out):
            os.makedirs(os.path.dirname(os.path.abspath(single_out)) or ".", exist_ok=True)
            if os.path.isfile(single_out):
                os.remove(single_out)
            os.replace(src, single_out)
            print(f"[explosion-3d] moved → {single_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
