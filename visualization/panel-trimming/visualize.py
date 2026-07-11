"""
Panel-trimming visualizer — 2D contact-line sweep paint + first/last fallback.

Primary shade (design plane only, no 3D) = **exported coordinates**:
  1) coordinates / shaded_polygon / paint_polygon — sweep ribbon of the
     two-node contact line (preferred; matches sim GUI locus)
  2) closed_triangle — first crease edge + max-area last apex (fallback)

  first      = JSON mountain/valley crease endpoints (green)
  last       = locked cut endpoints (red)
  sweep line samples drawn as thin strokes when present

Data sources (in order):
  1) collision_stats.closed_polygons[] / shaded_areas[]  (sim export)
  2) collision_stats.segments  (recomputed max-area triangle)
  3) Reconstructed: shared crease (first) + matched cut (last)

Usage:
  python visualization/panel-trimming/visualize.py
  python visualization/panel-trimming/visualize.py --name mountain-thick-trimmed
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
HOST_FACE = "#4c78a8"
HOST_ALPHA = 0.22
INTRUDER_FACE = "#f58518"
INTRUDER_ALPHA = 0.28
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


def _resolve_json_path(name: str) -> str:
    if os.path.isfile(name):
        return name
    candidate = os.path.join(DESCRIPTION_DIR, name)
    if os.path.isfile(candidate):
        return candidate
    if not name.endswith(".json"):
        candidate = os.path.join(DESCRIPTION_DIR, name + ".json")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"JSON not found for name/path: {name}")


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


def _layer_key(h: float) -> str:
    return str(round(float(h), 6))


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

def _items_from_collision_stats(
    stats: dict, layer_h: float, eps: float = 1e-6
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # Prefer dedicated shaded_areas list when present (primary export payload)
    shaded_list = stats.get("shaded_areas") or []
    poly_list = stats.get("closed_polygons") or []
    # If shaded_areas is non-empty, still walk closed_polygons (has more meta);
    # fall back to shaded_areas alone when closed_polygons empty.
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


def _active_layers_from_stats(stats: dict) -> set:
    active = set()
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
    show_host_polys: bool = False,
):
    n = min(len(orig_units), len(layer_units))

    intruder_ids = set()
    host_ids = set()
    for p in closed_polys:
        if p.get("panel") is not None and p.get("role") != "host":
            intruder_ids.add(int(p["panel"]))
        if p.get("panel") is not None and p.get("role") == "host":
            host_ids.add(int(p["panel"]))
        if p.get("host_panel") is not None:
            try:
                host_ids.add(int(p["host_panel"]))
            except (TypeError, ValueError):
                pass

    for i in range(n):
        xy = _poly_xy(layer_units[i] if i < len(layer_units) else orig_units[i])
        _draw_poly(
            ax, xy, facecolor=PANEL_FACE, edgecolor=PANEL_EDGE,
            linewidth=0.7, zorder=1,
        )

    for i in sorted(host_ids):
        if 0 <= i < n:
            xy = _poly_xy(layer_units[i])
            _draw_poly(
                ax, xy, facecolor=HOST_FACE, edgecolor="#2f4b7c",
                linewidth=0.8, alpha=HOST_ALPHA, zorder=2,
            )

    for i in sorted(intruder_ids):
        if 0 <= i < n:
            xy = _poly_xy(layer_units[i])
            _draw_poly(
                ax, xy, facecolor=INTRUDER_FACE, edgecolor="#b35c00",
                linewidth=0.9, alpha=INTRUDER_ALPHA, zorder=3,
            )

    if show_lines and all_lines:
        for idx, ln in enumerate(all_lines):
            if not ln or len(ln) < 2:
                continue
            p0, p1 = ln[0], ln[1]
            if len(p0) > 2 and len(p1) > 2:
                z0, z1 = float(p0[2]), float(p1[2])
                if abs(z0) > 1e-6 or abs(z1) > 1e-6:
                    if abs(z0 - float(layer_h)) > 1e-3 or abs(z1 - float(layer_h)) > 1e-3:
                        continue
            color, lw = CREASE_COLOR, 0.55
            if line_features and idx < len(line_features):
                t = line_features[idx].get("type", 2)
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
            role = item.get("role", "intruder")
            if role == "host" and not show_host_polys:
                continue
            poly = item["poly"]
            if poly is None or len(poly) < 3:
                continue
            n_shaded += 1
            is_sweep = (
                item.get("paint_kind") == "sweep"
                or str(item.get("source", "")).endswith("sweep")
            )
            if role == "host":
                _draw_poly(
                    ax, poly,
                    facecolor="#72b7b2", edgecolor="#3d7a76",
                    linewidth=0.9, alpha=0.25, zorder=5,
                )
            else:
                _draw_poly(
                    ax, poly,
                    facecolor=TRIM_POLY_FACE, edgecolor=TRIM_POLY_EDGE,
                    linewidth=1.2, alpha=TRIM_POLY_ALPHA,
                    hatch=None if is_sweep else "///",
                    zorder=6,
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
            if item.get("role") == "host" and not show_host_polys:
                continue
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
        for i in range(n):
            xy = _poly_xy(layer_units[i])
            c = xy.mean(axis=0)
            tag = str(i)
            if i in intruder_ids:
                tag += " I"
            if i in host_ids:
                tag += " H"
            ax.text(
                c[0], c[1], tag, ha="center", va="center",
                fontsize=7.5, color=TEXT_COLOR, zorder=12,
                fontweight="bold" if i in intruder_ids else "normal",
            )

    ax.set_aspect("equal", adjustable="box")
    if closed_polys:
        src = closed_polys[0]["source"].split(".")[0]
        n_sw = sum(
            1 for it in closed_polys
            if it.get("paint_kind") == "sweep"
            or str(it.get("source", "")).endswith("sweep")
        )
        kind = f"{n_sw} sweeps" if n_sw else "tris/quads"
        ax.set_title(
            f"layer h = {layer_h:g}   ({n_shaded} 2D paints, {kind}, src={src})",
            fontsize=11,
        )
    else:
        ax.set_title(
            f"layer h = {layer_h:g}   (no trims on this layer)",
            fontsize=11, color="#666666",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return n_shaded


def visualize_trimmed(
    trimmed: dict,
    original: Optional[dict] = None,
    layers: Optional[Sequence[float]] = None,
    show_lines: bool = True,
    show_cut_lines: bool = False,
    show_panel_ids: bool = True,
    show_closed_polys: bool = True,
    show_first_last_edges: bool = True,
    show_host_polys: bool = False,
    title: Optional[str] = None,
) -> plt.Figure:
    ubl = trimmed.get("units_by_layer")
    if not ubl:
        raise ValueError("JSON has no units_by_layer — expected a trimmed design.")

    base = original or trimmed
    orig_units = base.get("units") or []
    if not orig_units:
        first_key = _layer_keys(ubl)[0]
        orig_units = ubl[first_key]

    classifications = trimmed.get("intruder_classifications") or []
    all_lines = trimmed.get("lines") or base.get("lines") or []
    line_features = trimmed.get("line_features") or base.get("line_features") or []
    creases = _mountain_valley_creases(base)
    stats = trimmed.get("collision_stats")

    keys = _layer_keys(ubl)
    if layers is not None:
        keys = [k for k in keys if any(abs(float(k) - float(h)) < 1e-6 for h in layers)]
        if not keys:
            raise ValueError(f"No matching layers for {layers}; available {list(ubl.keys())}")
    # else: show ALL thickness layers

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
        layer_units = ubl[key]
        layer_cls = _classifications_for_layer(classifications, layer_h)
        cut_segs = _cut_segments_for_layer(trimmed, layer_h)

        closed: List[Dict[str, Any]] = []
        if stats:
            closed = _items_from_collision_stats(stats, layer_h)
        if not closed:
            closed = _items_reconstructed(
                orig_units, layer_units, layer_h,
                layer_cls, cut_segs, creases,
            )

        total_polys += sum(1 for p in closed if p.get("role") != "host")
        for p in closed:
            sources.add(p.get("source", "?"))

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
            show_host_polys=show_host_polys,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.text(
            0.02, 0.98,
            f"closed polys: {n_sh}\n"
            f"first=green crease · last=red cut",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    legend_handles = [
        Patch(facecolor=PANEL_FACE, edgecolor=PANEL_EDGE, label="Panel"),
        Patch(
            facecolor=INTRUDER_FACE, edgecolor="#b35c00", alpha=INTRUDER_ALPHA,
            label="Intruder panel",
        ),
        Patch(
            facecolor=HOST_FACE, edgecolor="#2f4b7c", alpha=HOST_ALPHA,
            label="Host panel",
        ),
        Patch(
            facecolor=TRIM_POLY_FACE, edgecolor=TRIM_POLY_EDGE, alpha=TRIM_POLY_ALPHA,
            label="Shaded area (exported sweep coordinates)",
        ),
        Line2D([0], [0], color=FIRST_LINE_COLOR, lw=2.4, label="first (JSON crease)"),
        Line2D([0], [0], color=LAST_LINE_COLOR, lw=2.4, label="last (cut / contact)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        frameon=True, fontsize=8.5, bbox_to_anchor=(0.5, 0.0),
    )

    fig_title = title or "Panel trimming"
    fig_title += (
        "\nshaded = exported coordinates (contact-line sweep ribbon on design xy)"
    )
    meta = trimmed.get("trim_3d_metadata") or {}
    bits = [f"shaded={total_polys}"]
    if meta:
        bits.append(f"trimmer_cuts={meta.get('n_cuts', '?')}")
    if stats:
        bits.append(f"stats_segs={stats.get('n_segments', '?')}")
        bits.append(
            f"stats_shaded={stats.get('n_shaded_areas', stats.get('n_closed_polygons', '?'))}"
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
    trimmed_path = _resolve_json_path(name)
    trimmed = _load_json(trimmed_path)

    original = None
    orig_name = sim.get("original") or _guess_original_name(name)
    if orig_name:
        try:
            original = _load_json(_resolve_json_path(orig_name))
        except FileNotFoundError:
            print(f"[warn] original JSON '{orig_name}' not found; using trimmed.units")

    has_stats = bool(trimmed.get("collision_stats"))
    n_segs = (trimmed.get("collision_stats") or {}).get("n_segments", 0)
    n_poly = (trimmed.get("collision_stats") or {}).get(
        "n_closed_triangles",
        (trimmed.get("collision_stats") or {}).get("n_closed_polygons", 0),
    )
    print(
        f"[panel-trimming] {os.path.basename(trimmed_path)}  "
        f"collision_stats={'yes' if has_stats else 'no'}  "
        f"segs={n_segs} closed_tris={n_poly}"
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
        show_host_polys=bool(sim.get("show_host_polys", False)),
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
