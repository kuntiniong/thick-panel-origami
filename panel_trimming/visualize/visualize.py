"""
Panel-trimming visualizer — 2D contact-line sweep paint + first/last fallback.

Primary shade (design plane only, no 3D) = **dual-curve export** (preferred):
  shaded_regions[] with c0 / c1 / theta  (schema dual_curve_v1)
  ribbon ring = c0 + reverse(c1); stroke i = segment(c0[i], c1[i])

Legacy shade keys still accepted:
  1) coordinates / shaded_polygon / paint_polygon
  2) closed_triangle — first crease edge + max-area last apex

  first      = first dual-curve sample (green)
  last       = last dual-curve sample (red)
  sweep strokes drawn as thin segments when present

Data sources (in order):
  1) shaded_regions[]  (dual_curve_v1, top-level on design JSON)
  2) collision_stats.closed_polygons[] / shaded_areas[]  (legacy)
  3) collision_stats.segments  (recomputed max-area triangle)
  4) Reconstructed: shared crease (first) + matched cut (last)

Layer mapping:
  Each subplot is a thickness offset (layer_h). Only panels that actually
  have that offset (crease thick_panel_height, same rule as phys_sim thick
  mode) are drawn. Shades are filtered by export panel + layer_h.

Usage:
  python panel-trimming/visualize/visualize.py
  python panel-trimming/visualize/visualize.py --name mountain-thick-trimmed

Trimmed JSON is read from project trimmedData/ (sim export folder);
source designs still resolve from descriptionData/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIS_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_VIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 9

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
PANEL_FACE = "#f4f4f6"
PANEL_EDGE = "#b0b0b8"
TRIM_POLY_FACE = "#e45756"
TRIM_POLY_ALPHA = 0.50
TRIM_POLY_EDGE = "#a32020"
FIRST_LINE_COLOR = "#2ca02c"
LAST_LINE_COLOR = "#d62728"
CUT_LINE_COLOR = "#c41e3a"
CREASE_COLOR = "#888888"
BORDER_LINE_COLOR = "#bbbbbb"
TEXT_COLOR = "#222222"

TYPE_MOUNTAIN = 0
TYPE_VALLEY = 1

DESCRIPTION_DIR = os.path.join(_PROJECT_ROOT, "descriptionData")
# Sim export target (phys_sim_pd14.export_trimmed_json)
TRIMMED_DIR = os.path.join(_PROJECT_ROOT, "trimmedData")
DEFAULT_CONFIG = os.path.join(_THIS_DIR, "config.yml")
DEFAULT_OUTPUT_DIR = os.path.join(_THIS_DIR, "output")


# ---------------------------------------------------------------------------
# IO / geometry
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_in_dirs(name: str, dirs: Sequence[str]) -> Optional[str]:
    """Resolve name or name.json under the given directories (first hit wins)."""
    if os.path.isfile(name):
        return name
    basenames = [name]
    if not name.endswith(".json"):
        basenames.append(name + ".json")
    for d in dirs:
        for bn in basenames:
            candidate = os.path.join(d, bn)
            if os.path.isfile(candidate):
                return candidate
            # bare stem under dir
            candidate = os.path.join(d, os.path.basename(bn))
            if os.path.isfile(candidate):
                return candidate
    return None


def _resolve_json_path(name: str, *, prefer_trimmed: bool = False) -> str:
    """
    Find a design JSON.

    prefer_trimmed=True  → trimmedData first, then descriptionData (exports)
    prefer_trimmed=False → descriptionData first, then trimmedData (source / original)
    """
    if prefer_trimmed:
        dirs = [TRIMMED_DIR, DESCRIPTION_DIR]
    else:
        dirs = [DESCRIPTION_DIR, TRIMMED_DIR]
    path = _resolve_in_dirs(name, dirs)
    if path is not None:
        return path
    raise FileNotFoundError(
        f"JSON not found for name/path: {name} "
        f"(searched {', '.join(dirs)})"
    )


def _guess_original_name(trimmed_name: str) -> Optional[str]:
    base = trimmed_name
    if base.endswith(".json"):
        base = base[:-5]
    base = os.path.basename(base)
    if base.endswith("-trimmed"):
        return base[: -len("-trimmed")]
    if base.endswith("_trimmed"):
        return base[: -len("_trimmed")]
    return None


def _poly_xy(unit: Sequence) -> np.ndarray:
    return np.asarray([[float(p[0]), float(p[1])] for p in unit], dtype=float)


def _xy_key(pt, nd: int = 3) -> Tuple[float, float]:
    return (round(float(pt[0]), nd), round(float(pt[1]), nd))


def _edge_key(a, b, nd: int = 3) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ka, kb = _xy_key(a, nd), _xy_key(b, nd)
    return (ka, kb) if ka <= kb else (kb, ka)


def _layer_key(h: float) -> str:
    return str(round(float(h), 6))


def _line_edge_table(data: dict) -> Dict[Tuple, List[Dict[str, Any]]]:
    """Undirected design-xy edge → {type, height} from lines / line_features."""
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
    Per design-panel thickness offsets (mm), matching phys_sim thick-mode.

    Unique mountain/valley crease ``thick_panel_height`` values on each panel
    outline → one layer unit at each height. Shade ``layer_h`` values are
    merged so paint never references a missing offset.
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
                if ent["type"] in (TYPE_MOUNTAIN, TYPE_VALLEY):
                    h = float(ent["height"])
                    if not any(abs(h - x) < 1e-9 for x in heights):
                        heights.append(h)
        heights.sort()
        per_panel[pi] = heights

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

    for pi in range(n):
        if not per_panel[pi]:
            per_panel[pi] = [0.0]
    return per_panel


def _panels_at_height(
    panel_offsets: Sequence[Sequence[float]],
    h: float,
    eps: float = 1e-6,
) -> set:
    """Design panel indices that have a thick-panel layer at offset h."""
    out = set()
    for i, offs in enumerate(panel_offsets):
        if any(abs(float(h) - float(x)) <= eps for x in offs):
            out.add(i)
    return out


def _layer_keys(units_by_layer: dict) -> List[str]:
    keys = list(units_by_layer.keys())
    try:
        keys.sort(key=lambda k: float(k))
    except ValueError:
        keys.sort()
    return keys


def _layer_match(a, b, eps: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= eps


def _bounds_from_units(unit_lists: Sequence[Sequence], margin: float = 8.0):
    xs, ys = [], []
    for units in unit_lists:
        for u in units:
            for p in u:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
    if not xs:
        return 0.0, 1.0, 0.0, 1.0
    return min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin


def _poly_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _polygons_differ(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return True
    return sorted(_xy_key(p, 2) for p in a) != sorted(_xy_key(p, 2) for p in b)


def _as_xy2(p) -> Optional[np.ndarray]:
    if p is None:
        return None
    try:
        a = np.asarray(p, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if a.size < 2:
        return None
    return a[:2].copy()


# ---------------------------------------------------------------------------
# Closed first + last triangle (max area over the two last endpoints)
# ---------------------------------------------------------------------------

def _order_last_to_first(
    f0: np.ndarray, f1: np.ndarray, c0: np.ndarray, c1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    cost_a = float(np.linalg.norm(c0 - f0) + np.linalg.norm(c1 - f1))
    cost_b = float(np.linalg.norm(c0 - f1) + np.linalg.norm(c1 - f0))
    if cost_b < cost_a:
        return c1.copy(), c0.copy()
    return c0.copy(), c1.copy()


def closed_poly_first_last(f0, f1, last0, last1) -> np.ndarray:
    """Legacy 4-gon: first_p0 → first_p1 → last_p1 → last_p0."""
    f0 = np.asarray(f0, dtype=float)[:2]
    f1 = np.asarray(f1, dtype=float)[:2]
    l0, l1 = _order_last_to_first(
        f0, f1,
        np.asarray(last0, dtype=float)[:2],
        np.asarray(last1, dtype=float)[:2],
    )
    return np.asarray([f0, f1, l1, l0], dtype=float)


def closed_triangle_max_area(f0, f1, last0, last1) -> Optional[Dict[str, Any]]:
    """
    Triangle [first_p0, first_p1, apex] picking apex ∈ {last0, last1}
    with the larger area. Returns dict or None.
    """
    f0 = np.asarray(f0, dtype=float)[:2].copy()
    f1 = np.asarray(f1, dtype=float)[:2].copy()
    lasts = [
        np.asarray(last0, dtype=float)[:2].copy(),
        np.asarray(last1, dtype=float)[:2].copy(),
    ]
    best = None
    best_area = -1.0
    for idx, apex in enumerate(lasts):
        tri = np.asarray([f0, f1, apex], dtype=float)
        area = _poly_area(tri)
        if area > best_area:
            best_area = area
            best = {
                "poly": tri,
                "area": float(area),
                "last_index": int(idx),
                "last_point": apex,
                "first": (f0, f1),
                "last_both": (lasts[0], lasts[1]),
            }
    if best is None or best_area < 1e-18:
        return None
    return best


def _side_first_last(side: dict):
    if not side:
        return None
    f0 = _as_xy2(side.get("first_p0"))
    f1 = _as_xy2(side.get("first_p1"))
    l0 = _as_xy2(side.get("p0"))
    l1 = _as_xy2(side.get("p1"))
    if f0 is None or f1 is None:
        fr = side.get("first") or {}
        f0 = f0 or _as_xy2(fr.get("p0"))
        f1 = f1 or _as_xy2(fr.get("p1"))
    if l0 is None or l1 is None:
        la = side.get("last") or {}
        l0 = l0 or _as_xy2(la.get("p0"))
        l1 = l1 or _as_xy2(la.get("p1"))
    if f0 is None or f1 is None or l0 is None or l1 is None:
        return None
    return f0, f1, l0, l1


# ---------------------------------------------------------------------------
# Creases / cuts
# ---------------------------------------------------------------------------

def _mountain_valley_creases(data: dict) -> List[Dict[str, Any]]:
    lines = data.get("lines") or []
    feats = data.get("line_features") or []
    out = []
    for i, ln in enumerate(lines):
        if not ln or len(ln) < 2:
            continue
        t = 2
        if i < len(feats):
            t = int(feats[i].get("type", 2))
        if t not in (TYPE_MOUNTAIN, TYPE_VALLEY):
            continue
        a = np.asarray(ln[0][:2], dtype=float)
        b = np.asarray(ln[1][:2], dtype=float)
        out.append({"a": a, "b": b, "index": i, "type": t})
    return out


def _unit_vert_keys(unit: Sequence, nd: int = 2) -> set:
    return {_xy_key(p, nd) for p in unit}


def _shared_crease_json(
    unit_a: Sequence,
    unit_b: Sequence,
    creases: List[Dict[str, Any]],
    nd: int = 2,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    ka = _unit_vert_keys(unit_a, nd)
    kb = _unit_vert_keys(unit_b, nd)
    shared = ka & kb
    if len(shared) < 2:
        return None
    best = None
    best_len = -1.0
    for c in creases:
        a_k, b_k = _xy_key(c["a"], nd), _xy_key(c["b"], nd)
        if a_k in shared and b_k in shared:
            L = float(np.linalg.norm(c["b"] - c["a"]))
            if L > best_len:
                best_len = L
                best = (c["a"].copy(), c["b"].copy())
    return best


def _panel_own_creases(
    unit: Sequence,
    creases: List[Dict[str, Any]],
    nd: int = 2,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    keys = _unit_vert_keys(unit, nd)
    out = []
    for c in creases:
        if _xy_key(c["a"], nd) in keys and _xy_key(c["b"], nd) in keys:
            out.append((c["a"].copy(), c["b"].copy()))
    return out


def _cut_segments_for_layer(
    data: dict, layer_h: float, eps: float = 1e-3
) -> List[Tuple[np.ndarray, np.ndarray]]:
    lines = data.get("lines") or []
    feats = data.get("line_features") or []
    meta = data.get("trim_3d_metadata") or {}
    n_cuts = int(meta.get("n_cuts") or 0)

    def _seg_if_layer(ln):
        if not ln or len(ln) < 2:
            return None
        p0, p1 = ln[0], ln[1]
        if len(p0) <= 2 or len(p1) <= 2:
            return None
        if abs(float(p0[2]) - float(layer_h)) > eps or abs(float(p1[2]) - float(layer_h)) > eps:
            return None
        return (np.asarray(p0[:2], dtype=float), np.asarray(p1[:2], dtype=float))

    segs: List[Tuple[np.ndarray, np.ndarray]] = []
    if n_cuts > 0 and n_cuts <= len(lines):
        for ln in lines[-n_cuts:]:
            seg = _seg_if_layer(ln)
            if seg is not None:
                segs.append(seg)
        if segs:
            return segs
    for i, ln in enumerate(lines):
        seg = _seg_if_layer(ln)
        if seg is not None:
            segs.append(seg)
        elif i < len(feats):
            th = feats[i].get("thick_panel_height")
            if th is not None and abs(float(th) - float(layer_h)) <= eps and ln and len(ln) >= 2:
                segs.append((
                    np.asarray(ln[0][:2], dtype=float),
                    np.asarray(ln[1][:2], dtype=float),
                ))
    return segs


def _classifications_for_layer(
    classifications: List[dict], layer_h: float, eps: float = 1e-6
) -> List[dict]:
    out = []
    for c in classifications:
        th = c.get("trimmed_layer_h")
        if th is None:
            continue
        if abs(float(th) - float(layer_h)) <= eps and c.get("trimmed_unit") is not None:
            out.append(c)
    return out


def _point_on_segment(p, a, b, tol: float = 1.0) -> bool:
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    lab2 = float(np.dot(ab, ab))
    if lab2 < 1e-18:
        return float(np.linalg.norm(p - a)) <= tol
    t = float(np.dot(p - a, ab) / lab2)
    if t < -1e-6 or t > 1.0 + 1e-6:
        return False
    return float(np.linalg.norm(p - (a + t * ab))) <= tol


def _point_on_poly_boundary(p, poly: np.ndarray, tol: float = 1.0) -> bool:
    n = len(poly)
    for i in range(n):
        if _point_on_segment(p, poly[i], poly[(i + 1) % n], tol=tol):
            return True
    return False


def _match_cut_to_unit(
    orig: np.ndarray,
    kept: np.ndarray,
    cuts: List[Tuple[np.ndarray, np.ndarray]],
    tol: float = 1.2,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not cuts:
        return None
    new_pts = []
    for p in kept:
        if all(float(np.linalg.norm(p - q)) > 0.6 for q in orig):
            new_pts.append(p)

    best, best_score = None, -1.0
    for a, b in cuts:
        score = 0.0
        if _point_on_poly_boundary(a, orig, tol=tol):
            score += 2.0
        if _point_on_poly_boundary(b, orig, tol=tol):
            score += 2.0
        if _point_on_poly_boundary(a, kept, tol=tol):
            score += 1.5
        if _point_on_poly_boundary(b, kept, tol=tol):
            score += 1.5
        for p in new_pts:
            score += max(0.0, 2.5 - float(np.linalg.norm(p - a)))
            score += max(0.0, 2.5 - float(np.linalg.norm(p - b)))
        if score > best_score:
            best_score = score
            best = (a, b)
    return best if best_score >= 2.0 else None


def _pick_first_crease(
    orig: np.ndarray,
    unit_list: Sequence,
    unit_idx: int,
    creases: List[Dict[str, Any]],
    cut: Tuple[np.ndarray, np.ndarray],
    preferred_host: Optional[int] = None,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[int]]:
    cut_mid = 0.5 * (np.asarray(cut[0], float) + np.asarray(cut[1], float))
    candidates: List[Tuple[float, Tuple[np.ndarray, np.ndarray], int]] = []

    neighbors = []
    if preferred_host is not None:
        neighbors.append(int(preferred_host))
    for i in range(len(unit_list)):
        if i != unit_idx and i not in neighbors:
            neighbors.append(i)

    for hi in neighbors:
        if hi < 0 or hi >= len(unit_list) or hi == unit_idx:
            continue
        shared = _shared_crease_json(unit_list[unit_idx], unit_list[hi], creases)
        if shared is None:
            continue
        f0, f1 = shared
        fmid = 0.5 * (f0 + f1)
        sep = float(np.linalg.norm(fmid - cut_mid))
        flen = float(np.linalg.norm(f1 - f0))
        tri = closed_triangle_max_area(f0, f1, cut[0], cut[1])
        if tri is None or tri["area"] < 1e-2:
            continue
        area = tri["area"]
        score = area + 0.5 * flen + 0.3 * sep
        if preferred_host is not None and hi == int(preferred_host):
            score += 1e4
        candidates.append((score, (f0, f1), hi))

    if not candidates:
        for f0, f1 in _panel_own_creases(unit_list[unit_idx], creases):
            tri = closed_triangle_max_area(f0, f1, cut[0], cut[1])
            if tri is None or tri["area"] < 1e-2:
                continue
            fmid = 0.5 * (f0 + f1)
            sep = float(np.linalg.norm(fmid - cut_mid))
            score = tri["area"] + 0.3 * sep
            candidates.append((score, (f0, f1), -1))

    if not candidates:
        return None, None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, first, host = candidates[0]
    return first, (host if host >= 0 else None)


# ---------------------------------------------------------------------------
# Build closed-poly items
# ---------------------------------------------------------------------------

def _ribbon_from_dual_curves(
    c0: Sequence, c1: Sequence
) -> Optional[np.ndarray]:
    """Rebuild shaded ring: walk c0 then reverse(c1)."""
    if not c0 or not c1 or len(c0) != len(c1) or len(c0) < 2:
        return None
    ring = [[float(p[0]), float(p[1])] for p in c0]
    ring += [[float(p[0]), float(p[1])] for p in reversed(c1)]
    if len(ring) < 3:
        return None
    return np.asarray(ring, dtype=float)


def _samples_from_dual_curves(
    theta: Sequence, c0: Sequence, c1: Sequence
) -> List[Dict[str, Any]]:
    """Rebuild stroke samples from parallel dual-curve arrays."""
    n = min(len(c0), len(c1))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        ang = float(theta[i]) if theta is not None and i < len(theta) else 0.0
        out.append({
            "angle": ang,
            "p0": [float(c0[i][0]), float(c0[i][1])],
            "p1": [float(c1[i][0]), float(c1[i][1])],
        })
    return out


def _entries_from_shaded_regions(
    regions: Sequence[dict],
) -> List[Dict[str, Any]]:
    """
    dual_curve_v1 → visualizer entry shape.

    Geometry source of truth is (c0, c1, theta); coordinates / sweep_samples
    are rebuilt here (no duplicated polygon keys required in JSON).
    """
    entries: List[Dict[str, Any]] = []
    for sh in regions or []:
        c0 = sh.get("c0") or []
        c1 = sh.get("c1") or []
        theta = sh.get("theta") or []
        ring = _ribbon_from_dual_curves(c0, c1)
        samples = _samples_from_dual_curves(theta, c0, c1)
        coords = ring.tolist() if ring is not None else sh.get("coordinates")
        kind = sh.get("kind") or ("sweep" if ring is not None else "line")
        entries.append({
            "layer_h": sh.get("layer_h"),
            "coordinates": coords,
            "shaded_polygon": coords,
            "paint_polygon": coords,
            "paint_kind": kind,
            "shaded_kind": kind,
            "paint_area": sh.get("area"),
            "shaded_area": sh.get("area"),
            "panel": sh.get("panel"),
            "unit": sh.get("unit"),
            "side": sh.get("side"),
            "intruder_panel": sh.get("intruder_panel"),
            "other_panel": sh.get("other_panel"),
            "tri_a": sh.get("tri_a"),
            "tri_b": sh.get("tri_b"),
            "sweep_n_samples": sh.get("n") or len(c0),
            "sweep_samples": samples,
            "first_p0": c0[0] if c0 else None,
            "first_p1": c1[0] if c1 else None,
            "last_p0": c0[-1] if c0 else None,
            "last_p1": c1[-1] if c1 else None,
            "c0": c0,
            "c1": c1,
            "theta": theta,
        })
    return entries


def _items_from_collision_stats(
    stats: dict, layer_h: float, eps: float = 1e-6,
    shaded_regions: Optional[Sequence[dict]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # dual_curve_v1: top-level shaded_regions (preferred) or stats pointer
    dual = list(shaded_regions or [])
    if not dual:
        dual = list(stats.get("shaded_regions") or [])
    if dual:
        source_entries = _entries_from_shaded_regions(dual)
    else:
        # Legacy: closed_polygons / shaded_areas with full polygon copies
        shaded_list = stats.get("shaded_areas") or []
        poly_list = stats.get("closed_polygons") or []
        source_entries = poly_list if poly_list else [
            {
                "layer_h": sh.get("layer_h"),
                "coordinates": sh.get("coordinates"),
                "shaded_polygon": sh.get("coordinates"),
                "paint_polygon": sh.get("coordinates"),
                "paint_kind": sh.get("kind"),
                "shaded_kind": sh.get("kind"),
                "paint_area": sh.get("area"),
                "shaded_area": sh.get("area"),
                "panel": sh.get("panel"),
                "unit": sh.get("unit"),
                "side": sh.get("side"),
                "intruder_panel": sh.get("intruder_panel"),
                "other_panel": sh.get("other_panel"),
                "tri_a": sh.get("tri_a"),
                "tri_b": sh.get("tri_b"),
                "sweep_n_samples": sh.get("sweep_n_samples"),
                "sweep_samples": sh.get("sweep_samples") or [],
            }
            for sh in shaded_list
        ]

    for cp in source_entries:
        if not _layer_match(cp.get("layer_h"), layer_h, eps):
            continue

        # Prefer exported shaded-area coordinates (sweep ribbon on design plane)
        paint_kind = cp.get("shaded_kind") or cp.get("paint_kind") or ""
        poly = None
        source = "collision_stats.closed_triangle"
        for key in (
            "coordinates",
            "shaded_polygon",
            "paint_polygon",
            "sweep_polygon",
            "closed_triangle",
            "polygon",
        ):
            raw = cp.get(key)
            if raw and len(raw) >= 3:
                poly = np.asarray(raw, dtype=float)[:, :2]
                if key in (
                    "coordinates",
                    "shaded_polygon",
                    "paint_polygon",
                    "sweep_polygon",
                ) or paint_kind == "sweep":
                    source = "collision_stats.sweep"
                elif key == "polygon":
                    source = "collision_stats.polygon"
                else:
                    source = "collision_stats.closed_triangle"
                break

        if poly is None:
            # Backward compat: recompute from first + both last if present
            f0 = _as_xy2(cp.get("first_p0"))
            f1 = _as_xy2(cp.get("first_p1"))
            poly4 = cp.get("polygon")
            if f0 is not None and f1 is not None and poly4 and len(poly4) >= 4:
                l0 = np.asarray(poly4[3], dtype=float)[:2]
                l1 = np.asarray(poly4[2], dtype=float)[:2]
                tri = closed_triangle_max_area(f0, f1, l0, l1)
                if tri is None:
                    continue
                poly = tri["poly"]
                source = "collision_stats.closed_triangle"
            else:
                continue
        if _poly_area(poly) < 1e-3:
            continue
        panel = cp.get("panel")
        role = "intruder"
        if panel is not None and cp.get("intruder_panel") is not None:
            role = "intruder" if int(panel) == int(cp["intruder_panel"]) else "host"
        f0 = _as_xy2(cp.get("first_p0"))
        f1 = _as_xy2(cp.get("first_p1"))
        if f0 is None or f1 is None:
            if len(poly) >= 2:
                f0, f1 = poly[0].copy(), poly[1].copy()
        apex = _as_xy2(cp.get("triangle_last_point"))
        if apex is None and len(poly) >= 3 and source != "collision_stats.sweep":
            apex = poly[2].copy()
        # Show both last endpoints when available (legacy 4-gon / last sample)
        last_both = None
        poly4 = cp.get("polygon")
        if poly4 and len(poly4) >= 4:
            last_both = (
                np.asarray(poly4[3], dtype=float)[:2],
                np.asarray(poly4[2], dtype=float)[:2],
            )
        sweep_samples = cp.get("sweep_samples") or []
        if last_both is None and len(sweep_samples) >= 1:
            last_s = sweep_samples[-1]
            if "p0" in last_s and "p1" in last_s:
                last_both = (
                    np.asarray(last_s["p0"], dtype=float)[:2],
                    np.asarray(last_s["p1"], dtype=float)[:2],
                )
        if f0 is None or f1 is None:
            if len(sweep_samples) >= 1 and "p0" in sweep_samples[0]:
                f0 = np.asarray(sweep_samples[0]["p0"], dtype=float)[:2]
                f1 = np.asarray(sweep_samples[0]["p1"], dtype=float)[:2]
        out.append({
            "poly": poly,
            "first": (f0, f1) if f0 is not None and f1 is not None else None,
            "last": last_both if last_both is not None else (
                (apex, apex) if apex is not None else None
            ),
            "apex": apex,
            "triangle_area": cp.get("triangle_area") or cp.get("paint_area"),
            "triangle_last_index": cp.get("triangle_last_index"),
            "role": role,
            "panel": panel,
            "host_panel": cp.get("other_panel"),
            "source": source,
            "paint_kind": paint_kind or (
                "sweep" if source == "collision_stats.sweep" else "triangle"
            ),
            "sweep_samples": sweep_samples,
            "tri_a": cp.get("tri_a"),
            "tri_b": cp.get("tri_b"),
        })

    if out:
        return out

    segs = list(stats.get("segments") or [])
    if not segs:
        for g in stats.get("groups") or []:
            segs.extend(g.get("segments") or [])
            segs.extend(g.get("lines") or [])

    for seg in segs:
        lh = seg.get("layer_h")
        if lh is None:
            lh = (seg.get("flat") or {}).get("layer_h")
        if lh is not None and not _layer_match(lh, layer_h, eps):
            continue

        sides = []
        if seg.get("side_a"):
            sides.append(("side_a", seg["side_a"]))
        if seg.get("side_b"):
            sides.append(("side_b", seg["side_b"]))
        flat = seg.get("flat") or {}
        for sk in ("side_a", "side_b"):
            if flat.get(sk):
                sides.append((sk, flat[sk]))

        for side_key, side in sides:
            fl = _side_first_last(side)
            if fl is None:
                continue
            f0, f1, l0, l1 = fl
            tri = closed_triangle_max_area(f0, f1, l0, l1)
            if tri is None or tri["area"] < 1e-3:
                continue
            panel = side.get("panel")
            role = "unknown"
            if panel is not None and seg.get("intruder_panel") is not None:
                role = (
                    "intruder"
                    if int(panel) == int(seg["intruder_panel"])
                    else "host"
                )
            out.append({
                "poly": tri["poly"],
                "first": (f0, f1),
                "last": (l0, l1),
                "apex": tri["last_point"],
                "triangle_area": tri["area"],
                "triangle_last_index": tri["last_index"],
                "role": role,
                "panel": panel,
                "host_panel": seg.get("other_panel"),
                "source": "collision_stats.segments",
                "tri_a": seg.get("tri_a"),
                "tri_b": seg.get("tri_b"),
                "side": side_key,
            })
    return out


def _items_reconstructed(
    orig_units: List,
    layer_units: List,
    layer_h: float,
    classifications: List[dict],
    cut_segs: List[Tuple[np.ndarray, np.ndarray]],
    creases: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = min(len(orig_units), len(layer_units))
    used_cuts = set()

    changed = []
    for i in range(n):
        if _polygons_differ(_poly_xy(orig_units[i]), _poly_xy(layer_units[i])):
            changed.append(i)

    trim_ids = []
    for c in classifications:
        tu = c.get("trimmed_unit")
        if tu is None:
            continue
        tu = int(tu)
        if 0 <= tu < n:
            trim_ids.append(tu)
    for i in changed:
        if i not in trim_ids:
            trim_ids.append(i)

    for tu in trim_ids:
        orig = _poly_xy(orig_units[tu])
        kept = _poly_xy(layer_units[tu])
        cut = _match_cut_to_unit(orig, kept, cut_segs)
        if cut is None:
            c = orig.mean(axis=0)
            best, best_d = None, float("inf")
            for a, b in cut_segs:
                key = (_xy_key(a, 3), _xy_key(b, 3))
                if key in used_cuts:
                    continue
                d = float(np.linalg.norm(0.5 * (a + b) - c))
                if d < best_d:
                    best_d, best = d, (a, b)
            cut = best
        if cut is None:
            continue

        cut_key = (_xy_key(cut[0], 3), _xy_key(cut[1], 3))
        used_cuts.add(cut_key)
        used_cuts.add((cut_key[1], cut_key[0]))

        first, host = _pick_first_crease(
            orig, orig_units, tu, creases, cut, preferred_host=None
        )
        if first is None:
            continue
        f0, f1 = first
        tri = closed_triangle_max_area(f0, f1, cut[0], cut[1])
        if tri is None or tri["area"] < 1e-2:
            continue

        out.append({
            "poly": tri["poly"],
            "first": (f0, f1),
            "last": (np.asarray(cut[0], float), np.asarray(cut[1], float)),
            "apex": tri["last_point"],
            "triangle_area": tri["area"],
            "triangle_last_index": tri["last_index"],
            "role": "intruder",
            "panel": tu,
            "host_panel": host,
            "source": "reconstructed",
        })
    return out


def _active_layers_from_stats(
    stats: dict, shaded_regions: Optional[Sequence[dict]] = None
) -> set:
    active = set()
    for sh in shaded_regions or stats.get("shaded_regions") or []:
        if sh.get("layer_h") is not None:
            active.add(_layer_key(sh["layer_h"]))
    for cp in stats.get("closed_polygons") or []:
        if cp.get("layer_h") is not None:
            active.add(_layer_key(cp["layer_h"]))
    for sh in stats.get("shaded_areas") or []:
        if sh.get("layer_h") is not None:
            active.add(_layer_key(sh["layer_h"]))
    for seg in stats.get("segments") or []:
        if seg.get("layer_h") is not None:
            active.add(_layer_key(seg["layer_h"]))
    return active


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_poly(ax, xy: np.ndarray, **kwargs):
    if xy is None or len(xy) < 3:
        return
    ax.add_patch(Polygon(xy, closed=True, **kwargs))


def _draw_seg(ax, a, b, **kwargs):
    if a is None or b is None:
        return
    ax.plot(
        [float(a[0]), float(b[0])],
        [float(a[1]), float(b[1])],
        **kwargs,
    )


def _draw_layer(
    ax,
    orig_units: List,
    layer_units: List,
    layer_h: float,
    closed_polys: List[Dict[str, Any]],
    cut_segs: List[Tuple[np.ndarray, np.ndarray]],
    all_lines: Optional[List] = None,
    line_features: Optional[List] = None,
    show_lines: bool = True,
    show_cut_lines: bool = False,
    show_panel_ids: bool = True,
    show_closed_polys: bool = True,
    show_first_last_edges: bool = True,
    panels_on_layer: Optional[set] = None,
):
    """
    Draw one thickness-offset subplot.

    ``panels_on_layer``: design panel indices that exist at this layer_h
    (thick-panel offset). If None, all panels in layer_units are drawn
    (legacy behaviour).
    """
    # Prefer full design unit count so panel index == design panel index
    n = max(len(orig_units), len(layer_units))
    if orig_units and layer_units:
        n = max(len(orig_units), len(layer_units))
    elif orig_units:
        n = len(orig_units)
    else:
        n = len(layer_units)

    # Panels that own a shade here (for ids / draw set only — no special fill color)
    shade_panel_ids = set()
    for p in closed_polys:
        if p.get("panel") is not None:
            try:
                shade_panel_ids.add(int(p["panel"]))
            except (TypeError, ValueError):
                pass

    # Only draw panels that have this thickness offset
    draw_ids = set(range(n)) if panels_on_layer is None else set(panels_on_layer)
    # Always include panels that own a shade here (export is source of truth)
    draw_ids |= shade_panel_ids

    n_drawn = 0
    for i in sorted(draw_ids):
        if i < 0:
            continue
        if i < len(layer_units) and layer_units[i]:
            xy = _poly_xy(layer_units[i])
        elif i < len(orig_units) and orig_units[i]:
            xy = _poly_xy(orig_units[i])
        else:
            continue
        n_drawn += 1
        _draw_poly(
            ax, xy, facecolor=PANEL_FACE, edgecolor=PANEL_EDGE,
            linewidth=0.7, zorder=1,
        )

    if show_lines and all_lines:
        for idx, ln in enumerate(all_lines):
            if not ln or len(ln) < 2:
                continue
            p0, p1 = ln[0], ln[1]
            # Prefer endpoint z; fall back to thick_panel_height on the feature
            z0 = float(p0[2]) if len(p0) > 2 else None
            z1 = float(p1[2]) if len(p1) > 2 else None
            th = None
            t = 2
            if line_features and idx < len(line_features):
                t = int(line_features[idx].get("type", 2))
                if line_features[idx].get("thick_panel_height") is not None:
                    th = float(line_features[idx]["thick_panel_height"])
            if z0 is None:
                z0 = th if th is not None else 0.0
            if z1 is None:
                z1 = th if th is not None else 0.0
            # Keep creases/borders whose geometry lives at this offset
            if abs(z0 - float(layer_h)) > 1e-3 or abs(z1 - float(layer_h)) > 1e-3:
                # Allow border (type 2) with flat z if it frames panels on this layer
                if t in (TYPE_MOUNTAIN, TYPE_VALLEY):
                    continue
                if abs(z0 - float(layer_h)) > 1e-3 and abs(z1 - float(layer_h)) > 1e-3:
                    continue
            color, lw = CREASE_COLOR, 0.55
            if t in (TYPE_MOUNTAIN, TYPE_VALLEY):
                color, lw = "#555555", 0.95
            elif t == 2:
                color, lw = BORDER_LINE_COLOR, 0.4
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]],
                color=color, linewidth=lw, zorder=4, solid_capstyle="round",
            )

    if show_cut_lines:
        for a, b in cut_segs:
            _draw_seg(
                ax, a, b, color=CUT_LINE_COLOR, linewidth=1.2,
                linestyle="--", zorder=5, alpha=0.7, solid_capstyle="round",
            )

    n_shaded = 0
    if show_closed_polys:
        for item in closed_polys:
            poly = item["poly"]
            if poly is None or len(poly) < 3:
                continue
            n_shaded += 1
            is_sweep = (
                item.get("paint_kind") == "sweep"
                or str(item.get("source", "")).endswith("sweep")
            )
            _draw_poly(
                ax, poly,
                facecolor=TRIM_POLY_FACE, edgecolor=TRIM_POLY_EDGE,
                linewidth=1.2, alpha=TRIM_POLY_ALPHA,
                hatch=None if is_sweep else "///",
                zorder=6,
            )
            # Panel tag on shade (export association)
            pid = item.get("panel")
            if pid is not None and len(poly) >= 1:
                c = poly.mean(axis=0)
                ax.text(
                    float(c[0]), float(c[1]), f"P{int(pid)}",
                    ha="center", va="center", fontsize=6.5,
                    color="#5a0a0a", fontweight="bold", zorder=11,
                )
            # Draw intermediate two-node line samples (the actual sweep stroke)
            samples = item.get("sweep_samples") or []
            if is_sweep and len(samples) >= 2:
                for s in samples:
                    if "p0" not in s or "p1" not in s:
                        continue
                    _draw_seg(
                        ax, s["p0"], s["p1"],
                        color="#e45756", linewidth=0.55, alpha=0.35,
                        zorder=7, solid_capstyle="round",
                    )

    if show_first_last_edges:
        for item in closed_polys:
            fr = item.get("first")
            la = item.get("last")
            apex = item.get("apex")
            if fr is not None:
                _draw_seg(
                    ax, fr[0], fr[1], color=FIRST_LINE_COLOR,
                    linewidth=2.4, zorder=8, solid_capstyle="round",
                )
            if la is not None:
                # Full last (cut) segment when both ends differ
                try:
                    same = (
                        abs(float(la[0][0]) - float(la[1][0])) < 1e-12
                        and abs(float(la[0][1]) - float(la[1][1])) < 1e-12
                    )
                except (TypeError, IndexError):
                    same = True
                if not same:
                    _draw_seg(
                        ax, la[0], la[1], color=LAST_LINE_COLOR,
                        linewidth=2.4, zorder=8, solid_capstyle="round",
                    )
                for pt in la:
                    ax.plot(
                        float(pt[0]), float(pt[1]), "o",
                        color=LAST_LINE_COLOR, markersize=4.5, zorder=9,
                    )
            # Highlight the last apex chosen for the max-area triangle
            if apex is not None:
                ax.plot(
                    float(apex[0]), float(apex[1]), "o",
                    color=LAST_LINE_COLOR, markersize=7.0,
                    markeredgecolor="#5a0a0a", markeredgewidth=0.8, zorder=10,
                )

    if show_panel_ids:
        for i in sorted(draw_ids):
            if i < len(layer_units) and layer_units[i]:
                xy = _poly_xy(layer_units[i])
            elif i < len(orig_units) and orig_units[i]:
                xy = _poly_xy(orig_units[i])
            else:
                continue
            c = xy.mean(axis=0)
            ax.text(
                c[0], c[1], str(i), ha="center", va="center",
                fontsize=7.5, color=TEXT_COLOR, zorder=12,
                fontweight="bold" if i in shade_panel_ids else "normal",
            )

    ax.set_aspect("equal", adjustable="box")
    n_panels = len(draw_ids)
    if closed_polys:
        src = closed_polys[0]["source"].split(".")[0]
        n_sw = sum(
            1 for it in closed_polys
            if it.get("paint_kind") == "sweep"
            or str(it.get("source", "")).endswith("sweep")
        )
        kind = f"{n_sw} sweeps" if n_sw else "tris/quads"
        ax.set_title(
            f"offset h = {layer_h:g}   "
            f"({n_panels} panels, {n_shaded} paints, {kind}, src={src})",
            fontsize=11,
        )
    else:
        ax.set_title(
            f"offset h = {layer_h:g}   "
            f"({n_panels} panels, no trims on this offset)",
            fontsize=11, color="#666666",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return n_shaded


def _synthesize_units_by_layer(
    trimmed: dict,
    base: dict,
    shaded_regions: Optional[Sequence[dict]] = None,
    panel_offsets: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, List]:
    """
    Build a units_by_layer map for drawing.

    dual_curve_v1 export keeps design ``units`` only (no per-height copies).
    Subplots are one per distinct thickness offset (from shades + panel
    crease heights). Each layer still stores the full design unit list so
    panel index stays stable; drawing filters via ``panels_on_layer``.
    """
    units = list(trimmed.get("units") or base.get("units") or [])
    heights: List[float] = []

    for sh in shaded_regions or []:
        if sh.get("layer_h") is not None:
            heights.append(float(sh["layer_h"]))
    stats = trimmed.get("collision_stats") or {}
    for bucket in (
        stats.get("shaded_areas") or [],
        stats.get("closed_polygons") or [],
        stats.get("segments") or [],
    ):
        for ent in bucket:
            if ent.get("layer_h") is not None:
                heights.append(float(ent["layer_h"]))

    # Prefer real panel thickness offsets (not border z) when available
    if panel_offsets is not None:
        for offs in panel_offsets:
            heights.extend(float(h) for h in offs)

    # Unique heights (rounded) sorted
    uniq: List[float] = []
    seen = set()
    for h in sorted(heights):
        key = round(h, 6)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(h))

    if not uniq:
        # No layer meta → single design-plane view
        uniq = [0.0]

    if not units:
        # Still allow shade-only plots (empty panel outlines)
        return {_layer_key(h): [] for h in uniq}

    # Same design polygons for every offset; filter at draw time
    return {_layer_key(h): units for h in uniq}


def visualize_trimmed(
    trimmed: dict,
    original: Optional[dict] = None,
    layers: Optional[Sequence[float]] = None,
    show_lines: bool = True,
    show_cut_lines: bool = False,
    show_panel_ids: bool = True,
    show_closed_polys: bool = True,
    show_first_last_edges: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    base = original or trimmed
    stats = trimmed.get("collision_stats")
    # dual_curve_v1 lives at top-level; fall back to stats-embedded copy
    shaded_regions = trimmed.get("shaded_regions") or (
        (stats or {}).get("shaded_regions") if stats else None
    )

    orig_units = base.get("units") or trimmed.get("units") or []
    # Per-panel thickness offsets (crease thick_panel_height), same as sim
    panel_offsets = panel_thickness_offsets(
        orig_units or list(trimmed.get("units") or []),
        base,
        shaded_regions,
    )

    ubl = trimmed.get("units_by_layer")
    if not ubl:
        # dual_curve / crease-only export: synthesize from units + offsets
        ubl = _synthesize_units_by_layer(
            trimmed, base, shaded_regions, panel_offsets=panel_offsets
        )
        if not any(ubl.values()) and not (trimmed.get("units") or base.get("units")):
            raise ValueError(
                "JSON has neither units_by_layer nor units — nothing to draw."
            )

    if not orig_units:
        first_key = _layer_keys(ubl)[0]
        orig_units = ubl[first_key]
        panel_offsets = panel_thickness_offsets(
            orig_units, base, shaded_regions
        )

    classifications = trimmed.get("intruder_classifications") or []
    all_lines = trimmed.get("lines") or base.get("lines") or []
    line_features = trimmed.get("line_features") or base.get("line_features") or []
    creases = _mountain_valley_creases(base)

    keys = _layer_keys(ubl)
    if layers is not None:
        keys = [k for k in keys if any(abs(float(k) - float(h)) < 1e-6 for h in layers)]
        if not keys:
            raise ValueError(f"No matching layers for {layers}; available {list(ubl.keys())}")
    else:
        # Prefer subplots only for offsets that have shades (trim view).
        # Fall back to all reconstructed panel offsets if no shades.
        shade_hs = set()
        for sh in shaded_regions or []:
            if sh.get("layer_h") is not None:
                shade_hs.add(round(float(sh["layer_h"]), 6))
        if shade_hs:
            keys = [k for k in keys if round(float(k), 6) in shade_hs]
            if not keys:
                keys = _layer_keys(ubl)

    n = len(keys)
    ncols = min(3, max(n, 1))
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.2 * ncols, 4.9 * nrows),
        squeeze=False,
    )

    xmin, xmax, ymin, ymax = _bounds_from_units([orig_units] + [ubl[k] for k in keys])

    total_polys = 0
    sources = set()

    for idx, key in enumerate(keys):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        layer_h = float(key)
        layer_units = ubl.get(key) or orig_units
        layer_cls = _classifications_for_layer(classifications, layer_h)
        cut_segs = _cut_segments_for_layer(trimmed, layer_h)

        closed: List[Dict[str, Any]] = []
        if stats or shaded_regions:
            closed = _items_from_collision_stats(
                stats or {}, layer_h, shaded_regions=shaded_regions
            )
        if not closed:
            closed = _items_reconstructed(
                orig_units, layer_units, layer_h,
                layer_cls, cut_segs, creases,
            )

        total_polys += len(closed)
        for p in closed:
            sources.add(p.get("source", "?"))

        # Panels that physically exist at this thickness offset
        panels_on_layer = _panels_at_height(panel_offsets, layer_h)
        for p in closed:
            if p.get("panel") is not None:
                try:
                    panels_on_layer.add(int(p["panel"]))
                except (TypeError, ValueError):
                    pass

        n_sh = _draw_layer(
            ax,
            orig_units=orig_units,
            layer_units=layer_units,
            layer_h=layer_h,
            closed_polys=closed,
            cut_segs=cut_segs,
            all_lines=all_lines,
            line_features=line_features,
            show_lines=show_lines,
            show_cut_lines=show_cut_lines,
            show_panel_ids=show_panel_ids,
            show_closed_polys=show_closed_polys,
            show_first_last_edges=show_first_last_edges,
            panels_on_layer=panels_on_layer,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        panel_list = ",".join(str(i) for i in sorted(panels_on_layer))
        ax.text(
            0.02, 0.98,
            f"panels @ h: [{panel_list}]\n"
            f"shades: {n_sh}  (P# on ribbon)\n"
            f"first=green · last=red",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    legend_handles = [
        Patch(facecolor=PANEL_FACE, edgecolor=PANEL_EDGE, label="Panel at this offset"),
        Patch(
            facecolor=TRIM_POLY_FACE, edgecolor=TRIM_POLY_EDGE, alpha=TRIM_POLY_ALPHA,
            label="Shaded area (panel P# @ layer_h)",
        ),
        Line2D([0], [0], color=FIRST_LINE_COLOR, lw=2.4, label="first"),
        Line2D([0], [0], color=LAST_LINE_COLOR, lw=2.4, label="last (cut / contact)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        frameon=True, fontsize=8.5, bbox_to_anchor=(0.5, 0.0),
    )

    fig_title = title or "Panel trimming"
    fig_title += (
        "\nshaded = dual-curve ribbon · each subplot = thickness offset · "
        "only panels with that offset"
    )
    meta = trimmed.get("trim_3d_metadata") or {}
    bits = [f"shaded={total_polys}"]
    if meta:
        bits.append(f"trimmer_cuts={meta.get('n_cuts', '?')}")
    n_dual = len(shaded_regions or [])
    if n_dual:
        bits.append(f"dual_curve={n_dual}")
    if stats:
        bits.append(f"stats_segs={stats.get('n_segments', '?')}")
        bits.append(
            f"stats_shaded={stats.get('n_shaded_regions', stats.get('n_shaded_areas', stats.get('n_closed_polygons', '?')))}"
        )
    if sources:
        bits.append("src=" + ",".join(sorted(s.split(".")[0] for s in sources)))
    fig_title += "  (" + ", ".join(bits) + ")"
    fig.suptitle(fig_title, fontsize=12, y=1.03)
    fig.tight_layout(rect=[0, 0.08, 1, 1.0])
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_one(sim: dict, output_dir: str) -> str:
    name = sim["name"]
    # Prefer trimmedData/ for exported *-trimmed.json
    trimmed_path = _resolve_json_path(name, prefer_trimmed=True)
    trimmed = _load_json(trimmed_path)

    original = None
    orig_name = sim.get("original") or _guess_original_name(name)
    if orig_name:
        try:
            # Source design lives in descriptionData/
            original = _load_json(_resolve_json_path(orig_name, prefer_trimmed=False))
        except FileNotFoundError:
            print(f"[warn] original JSON '{orig_name}' not found; using trimmed.units")

    has_stats = bool(trimmed.get("collision_stats"))
    stats = trimmed.get("collision_stats") or {}
    n_segs = stats.get("n_segments", 0)
    n_poly = stats.get(
        "n_shaded_regions",
        stats.get(
            "n_shaded_areas",
            stats.get("n_closed_triangles", stats.get("n_closed_polygons", 0)),
        ),
    )
    n_dual = len(trimmed.get("shaded_regions") or [])
    has_ubl = bool(trimmed.get("units_by_layer"))
    print(
        f"[panel-trimming] {os.path.basename(trimmed_path)}  "
        f"collision_stats={'yes' if has_stats else 'no'}  "
        f"segs={n_segs} shaded={n_poly} dual_curve={n_dual}  "
        f"units_by_layer={'yes' if has_ubl else 'synth-from-units'}"
    )

    fig = visualize_trimmed(
        trimmed=trimmed,
        original=original,
        layers=sim.get("layers"),
        show_lines=bool(sim.get("show_lines", True)),
        show_cut_lines=bool(sim.get("show_cut_lines", False)),
        show_panel_ids=bool(sim.get("show_panel_ids", True)),
        show_closed_polys=bool(sim.get("show_closed_polys", True)),
        show_first_last_edges=bool(sim.get("show_first_last_edges", True)),
        title=os.path.basename(trimmed_path),
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(trimmed_path))[0]
    out_path = os.path.join(output_dir, f"{stem}_panel_trimming.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[panel-trimming] wrote {out_path}")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize trimmed JSON; shaded regions use exported coordinates "
            "(contact-line sweep ribbon). Falls back to first/last triangle."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG if os.path.isfile(DEFAULT_CONFIG)
        else os.path.join(_THIS_DIR, "config.example.yml"),
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args(argv)

    if args.name:
        simulations = [{"name": args.name}]
    else:
        if not os.path.isfile(args.config):
            print(f"Config not found: {args.config}")
            return 1
        cfg = _load_yaml(args.config)
        simulations = list(cfg.get("simulations") or [])
        if not simulations:
            print("No simulations in config.")
            return 1

    output_dir = DEFAULT_OUTPUT_DIR
    single_out = None
    if args.output:
        if args.output.lower().endswith(".png") and len(simulations) == 1:
            single_out = args.output
            output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        else:
            output_dir = args.output

    paths = []
    for sim in simulations:
        paths.append(_run_one(sim, output_dir))

    if single_out and paths:
        src = paths[0]
        if os.path.abspath(src) != os.path.abspath(single_out):
            os.makedirs(os.path.dirname(os.path.abspath(single_out)) or ".", exist_ok=True)
            if os.path.isfile(single_out):
                os.remove(single_out)
            os.replace(src, single_out)
            print(f"[panel-trimming] moved → {single_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
