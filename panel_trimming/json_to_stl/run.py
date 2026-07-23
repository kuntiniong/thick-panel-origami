"""
JSON → three STLs (panel stock, collision trim, final).

Reuses panel_trimming/visualize/visualize_3d associations:
  - panel outline from units[]
  - thickness offsets from non-border crease heights (+ shade layer_h)
  - collision polygons from resolve_shaded_associations
    (cleaned → fabric/cut; trimmed → dual-curve ribbon)
  - physical + ghost shades included; side ribbons skipped

Geometry:
  1) original   — full panel stock (unit outline extruded over panel Z span)
  2) collision  — clean.py merge (h_lo→h_hi continuous stack) when present;
                  else fallback min→max of sample layer_h (no panel clip)
  3) final      — stock − that collision band

Writes under panel_trimming/trimmedData/:
  <stem>-original.stl
  <stem>-collision.stl
  <stem>-final.stl

Usage:
  python panel_trimming/json_to_stl/run.py
  python panel_trimming/json_to_stl/run.py --name miura-thick-trimmed
  python panel_trimming/json_to_stl/run.py --config panel_trimming/json_to_stl/config.yml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PANEL_TRIM_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_PANEL_TRIM_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# Same association / offset logic as the 3D visualizer
from panel_trimming.visualize.visualize_3d import (  # noqa: E402
    _poly_xy,
    _shell_kind_of,
    panel_thickness_offsets,
    resolve_shaded_associations,
)

try:
    from scipy.spatial import Delaunay as _Delaunay
except Exception:  # pragma: no cover
    _Delaunay = None

TRIMMED_DIR = os.path.join(_PANEL_TRIM_DIR, "trimmedData")
DESCRIPTION_DIR = os.path.join(_PROJECT_ROOT, "descriptionData")
DEFAULT_OUTPUT_DIR = TRIMMED_DIR
DEFAULT_CONFIG = os.path.join(_THIS_DIR, "config.yml")
DEFAULT_CONFIG_EXAMPLE = os.path.join(_THIS_DIR, "config.example.yml")

DEFAULT_THICKNESS = 6.0
VERTEX_EPS = 1e-9


# ---------------------------------------------------------------------------
# IO / path resolve
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(path: Optional[str] = None) -> Optional[str]:
    if path and os.path.isfile(path):
        return os.path.abspath(path)
    if path:
        cand = os.path.join(_THIS_DIR, path)
        if os.path.isfile(cand):
            return os.path.abspath(cand)
        if not path.endswith((".yml", ".yaml")):
            for ext in (".yml", ".yaml"):
                cand = os.path.join(_THIS_DIR, path + ext)
                if os.path.isfile(cand):
                    return os.path.abspath(cand)
        return None
    if os.path.isfile(DEFAULT_CONFIG):
        return DEFAULT_CONFIG
    if os.path.isfile(DEFAULT_CONFIG_EXAMPLE):
        return DEFAULT_CONFIG_EXAMPLE
    return None


def _resolve_input(name_or_path: str) -> str:
    """
    Explicit *-trimmed / *-cleaned used as given.
    Bare stem tries cleaned, then trimmed.
    """
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)

    stems = [name_or_path]
    if name_or_path.endswith(".json"):
        stems = [os.path.splitext(name_or_path)[0]]

    candidates: List[str] = []
    for stem in stems:
        base = os.path.basename(stem)
        lower = base.lower()
        if lower.endswith("-cleaned") or lower.endswith("_cleaned"):
            candidates.append(base + ".json")
        elif lower.endswith("-trimmed") or lower.endswith("_trimmed"):
            candidates.append(base + ".json")
        else:
            candidates.append(base + "-cleaned.json")
            candidates.append(base + "-trimmed.json")
            candidates.append(base + ".json")

    ordered: List[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    for d in (TRIMMED_DIR, DESCRIPTION_DIR, _PROJECT_ROOT):
        for bn in ordered:
            cand = os.path.join(d, bn)
            if os.path.isfile(cand):
                return os.path.abspath(cand)
    raise FileNotFoundError(
        f"JSON not found for: {name_or_path} (searched {TRIMMED_DIR}; {ordered})"
    )


def _stl_basename(input_path: str) -> str:
    stem, _ = os.path.splitext(os.path.basename(input_path))
    return stem + ".stl"


# ---------------------------------------------------------------------------
# 2D geometry helpers
# ---------------------------------------------------------------------------

def _dedupe_closed(poly: Sequence, *, eps: float = VERTEX_EPS) -> List[List[float]]:
    if poly is None:
        return []
    try:
        n = len(poly)
    except TypeError:
        return []
    if n == 0:
        return []
    raw = [[float(p[0]), float(p[1])] for p in poly]
    if not raw:
        return []
    out: List[List[float]] = []
    eps2 = float(eps) * float(eps)
    for p in raw:
        if not out:
            out.append(p)
            continue
        dx = p[0] - out[-1][0]
        dy = p[1] - out[-1][1]
        if dx * dx + dy * dy > eps2:
            out.append(p)
    if len(out) >= 2:
        dx = out[0][0] - out[-1][0]
        dy = out[0][1] - out[-1][1]
        if dx * dx + dy * dy <= eps2:
            out = out[:-1]
    return out if len(out) >= 3 else []


def _poly_signed_area(poly: Sequence) -> float:
    acc = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = float(poly[i][0]), float(poly[i][1])
        x1, y1 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        acc += x0 * y1 - x1 * y0
    return 0.5 * acc


def _geom_to_polygon_list(geom) -> List[Polygon]:
    """Flatten a shapely geometry into simple Polygons with positive area."""
    if geom is None:
        return []
    try:
        if geom.is_empty:
            return []
    except Exception:
        return []
    try:
        if not geom.is_valid:
            geom = make_valid(geom)
    except Exception:
        try:
            geom = geom.buffer(0)
        except Exception:
            return []
    pieces: List[Polygon] = []
    try:
        gt = geom.geom_type
    except Exception:
        return []
    if gt == "Polygon":
        if geom.area > 1e-12:
            pieces.append(geom)
    elif gt == "MultiPolygon":
        pieces.extend(
            p for p in geom.geoms
            if p.geom_type == "Polygon" and p.area > 1e-12
        )
    else:
        # GeometryCollection / others
        try:
            for g in getattr(geom, "geoms", []):
                pieces.extend(_geom_to_polygon_list(g))
        except Exception:
            return []
    return pieces


def _repair_ring_polygons(poly: Sequence) -> List[Polygon]:
    """
    Turn a (possibly self-intersecting dual-curve) ring into valid Polygons.

    Dual-curve ribbons are often invalid; buffer(0) / make_valid recovers them.
    Returns all positive-area polygon parts (not just the largest).
    """
    pts = _dedupe_closed(poly)
    if len(pts) < 3:
        return []
    if abs(_poly_signed_area(pts)) < 1e-18:
        return []
    # Prefer CCW exterior
    if _poly_signed_area(pts) < 0:
        pts = list(reversed(pts))
    try:
        g = Polygon(pts)
    except Exception:
        return []
    if g.is_empty:
        return []

    if g.is_valid and g.area > 1e-12:
        return [g]

    # Invalid: try buffer(0) first (handles many bow-ties better than make_valid)
    for fixer in (
        lambda x: x.buffer(0),
        lambda x: make_valid(x),
        lambda x: make_valid(x.buffer(0)),
    ):
        try:
            fixed = fixer(g)
        except Exception:
            continue
        parts = _geom_to_polygon_list(fixed)
        if parts:
            return parts
    return []


def _ring_to_shapely(poly: Sequence) -> Optional[Polygon]:
    """Largest repaired polygon from a ring (legacy single-poly helper)."""
    parts = _repair_ring_polygons(poly)
    if not parts:
        return None
    return max(parts, key=lambda p: p.area)


def _panel_z_span(
    offsets: Sequence[float],
    *,
    default_thickness: float = DEFAULT_THICKNESS,
) -> Tuple[float, float]:
    """Stock Z from panel thickness offsets (same idea as visualize_3d stack)."""
    offs = [float(h) for h in (offsets or [])]
    if len(offs) >= 2:
        return float(min(offs)), float(max(offs))
    if len(offs) == 1:
        mid = float(offs[0])
        half = 0.5 * float(default_thickness)
        return mid - half, mid + half
    half = 0.5 * float(default_thickness)
    return -half, half


def _polygon_to_rings(
    p: Polygon,
) -> Optional[Tuple[List[List[float]], List[List[List[float]]]]]:
    if p is None or p.is_empty or p.area < 1e-12:
        return None
    outer = [[float(x), float(y)] for x, y in p.exterior.coords[:-1]]
    outer = _dedupe_closed(outer)
    if len(outer) < 3:
        return None
    hole_rings: List[List[List[float]]] = []
    for interior in p.interiors:
        hr = [[float(x), float(y)] for x, y in interior.coords[:-1]]
        hr = _dedupe_closed(hr)
        if len(hr) >= 3:
            hole_rings.append(hr)
    return outer, hole_rings


def _collision_cut_geom(
    outer_xy: Sequence,
    collision_polys: Sequence[Sequence],
) -> Optional[Any]:
    """
    Build a shapely cut geometry from collision rings.

    Repairs invalid dual-curve polys, then clips to panel when possible.
    If clip empties a piece, keep the unclipped repaired poly so the boolean
    still removes volume (those rings are still real collision footprints).
    """
    panel = _ring_to_shapely(outer_xy)
    holes_g: List[Polygon] = []
    for c in collision_polys or []:
        parts = _repair_ring_polygons(c)
        if not parts:
            continue
        for g in parts:
            if panel is None:
                holes_g.append(g)
                continue
            try:
                inter = g.intersection(panel)
            except Exception:
                holes_g.append(g)
                continue
            inter_parts = _geom_to_polygon_list(inter)
            if inter_parts:
                holes_g.extend(inter_parts)
            else:
                # Intersection empty (topology glitch) but ring is valid — keep it
                # if it roughly overlaps the panel bounds
                try:
                    if g.bounds and panel.bounds:
                        # AABB overlap test
                        minx, miny, maxx, maxy = g.bounds
                        pminx, pminy, pmaxx, pmaxy = panel.bounds
                        if maxx >= pminx and maxy >= pminy and minx <= pmaxx and miny <= pmaxy:
                            holes_g.append(g)
                except Exception:
                    holes_g.append(g)
    if not holes_g:
        return None
    try:
        cut = unary_union(holes_g)
    except Exception:
        # sequential union fallback
        cut = holes_g[0]
        for h in holes_g[1:]:
            try:
                cut = cut.union(h)
            except Exception:
                continue
    if cut is None or cut.is_empty:
        return None
    if not cut.is_valid:
        try:
            cut = make_valid(cut)
        except Exception:
            try:
                cut = cut.buffer(0)
            except Exception:
                return None
    return cut if not cut.is_empty else None


def boolean_panel_minus_collisions(
    outer_xy: Sequence,
    collision_polys: Sequence[Sequence],
) -> List[Tuple[List[List[float]], List[List[List[float]]]]]:
    """
    2D boolean: panel outline − collision footprints.

    Robust to invalid dual-curve rings (buffer(0) / make_valid + multi-part).
    Subtracts sequentially if a single union/difference fails.
    """
    panel_parts = _repair_ring_polygons(outer_xy)
    if not panel_parts:
        return []
    panel = panel_parts[0] if len(panel_parts) == 1 else unary_union(panel_parts)
    try:
        if not panel.is_valid:
            panel = make_valid(panel)
    except Exception:
        pass

    # Collect repaired cut pieces
    cut_parts: List[Polygon] = []
    for c in collision_polys or []:
        cut_parts.extend(_repair_ring_polygons(c))

    if not cut_parts:
        result = panel
    else:
        result = panel
        # Prefer one-shot union difference
        try:
            cut = unary_union(cut_parts)
            if not cut.is_empty:
                if not cut.is_valid:
                    cut = make_valid(cut)
                # Clip cut to panel when possible (cleaner holes)
                try:
                    clipped = cut.intersection(panel)
                    if not clipped.is_empty and getattr(clipped, "area", 0) > 1e-12:
                        cut = clipped
                except Exception:
                    pass
                result = panel.difference(cut)
        except Exception:
            # Sequential subtract each repaired piece
            result = panel
            for part in cut_parts:
                try:
                    p = part
                    if not p.is_valid:
                        p = p.buffer(0)
                    try:
                        inter = p.intersection(result)
                        if not inter.is_empty and getattr(inter, "area", 0) > 1e-12:
                            result = result.difference(inter)
                        else:
                            result = result.difference(p)
                    except Exception:
                        result = result.difference(p)
                except Exception:
                    continue

    try:
        if result is not None and not result.is_valid:
            result = make_valid(result)
    except Exception:
        pass

    out: List[Tuple[List[List[float]], List[List[List[float]]]]] = []
    for p in _geom_to_polygon_list(result):
        rings = _polygon_to_rings(p)
        if rings is not None:
            out.append(rings)
    return out


def collision_pieces_clipped(
    outer_xy: Sequence,
    collision_polys: Sequence[Sequence],
) -> List[Tuple[List[List[float]], List[List[List[float]]]]]:
    """Collision footprints as rings (repaired; clipped to panel when possible)."""
    cut = _collision_cut_geom(outer_xy, collision_polys)
    out: List[Tuple[List[List[float]], List[List[List[float]]]]] = []
    for p in _geom_to_polygon_list(cut):
        rings = _polygon_to_rings(p)
        if rings is not None:
            out.append(rings)
    return out


def _extrude_pieces(
    pieces: Sequence[Tuple[List[List[float]], List[List[List[float]]]]],
    *,
    z_lo: float,
    z_hi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for outer_r, holes_r in pieces:
        V, F, N = extrude_polygon_mesh(
            outer_r, z_lo=z_lo, z_hi=z_hi, holes=holes_r
        )
        if len(V) and len(F):
            meshes.append((V, F, N))
    return merge_meshes(meshes)


# ---------------------------------------------------------------------------
# Triangulation + extrude (vertical prism)
# ---------------------------------------------------------------------------

def _distance(a: Sequence, b: Sequence) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _max_delta_angle(x0, x1, x2) -> float:
    x0x1 = _distance(x0, x1)
    x1x2 = _distance(x1, x2)
    x2x0 = _distance(x2, x0)
    if x0x1 < 1e-15 or x1x2 < 1e-15 or x2x0 < 1e-15:
        return 3.14

    def _ang(a, b, c):
        cosv = (a * a + b * b - c * c) / (2.0 * a * b)
        cosv = max(min(cosv, 1.0), -1.0)
        return math.acos(cosv)

    a0 = _ang(x0x1, x2x0, x1x2)
    a1 = _ang(x1x2, x0x1, x2x0)
    a2 = _ang(x2x0, x1x2, x0x1)
    return max(abs(a0 - a1), abs(a0 - a2), abs(a1 - a2))


def triangulate_like_ori_sim(poly_xy: Sequence) -> List[List[int]]:
    n = len(poly_xy) if poly_xy is not None else 0
    if n < 3:
        return []
    if n == 3:
        return [[0, 1, 2]]
    indices = list(range(n))
    complete_id: List[int] = []
    tri_indices: List[List[int]] = []
    forbidden: List[List[int]] = []
    unit = [[float(p[0]), float(p[1])] for p in poly_xy]
    while n - len(complete_id) >= 3:
        delta_angle_max = 3.14
        temp_tri = None
        pointer = 0
        active = n - len(complete_id)
        while pointer < active:
            while indices[pointer] in complete_id:
                pointer += 1
                if pointer >= n:
                    break
            if pointer >= n:
                break
            next_pointer = (pointer + 1) % n
            while indices[next_pointer] in complete_id:
                next_pointer = (next_pointer + 1) % n
            next_next = (next_pointer + 1) % n
            while indices[next_next] in complete_id:
                next_next = (next_next + 1) % n
            cand = [indices[pointer], indices[next_pointer], indices[next_next]]
            dlt = _max_delta_angle(unit[cand[0]], unit[cand[1]], unit[cand[2]])
            if dlt < delta_angle_max and cand not in forbidden:
                temp_tri = cand
                delta_angle_max = dlt
            pointer += 1
        if temp_tri is None:
            if not tri_indices or not complete_id:
                return [[0, i, i + 1] for i in range(1, n - 1)]
            complete_id.pop()
            forbidden.append(tri_indices[-1])
            tri_indices.pop()
            continue
        tri_indices.append(temp_tri)
        complete_id.append(temp_tri[1])
    return tri_indices


def _point_in_poly(p: Sequence, poly: Sequence, eps: float = 1e-9) -> bool:
    if not poly or len(poly) < 3:
        return False
    x, y = float(p[0]), float(p[1])
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-30) + xi
        ):
            inside = not inside
        j = i
    return inside


def triangulate_polygon_with_holes(
    outer: Sequence,
    holes: Optional[Sequence[Sequence]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    outer_ccw = _dedupe_closed(outer)
    if len(outer_ccw) >= 3 and _poly_signed_area(outer_ccw) < 0:
        outer_ccw = list(reversed(outer_ccw))
    if len(outer_ccw) < 3:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=np.int32)

    hole_list: List[List[List[float]]] = []
    for h in holes or []:
        hc = _dedupe_closed(h)
        if len(hc) < 3:
            continue
        if _poly_signed_area(hc) > 0:
            hc = list(reversed(hc))
        hole_list.append(hc)

    if not hole_list:
        faces = triangulate_like_ori_sim(outer_ccw)
        V = np.asarray(outer_ccw, dtype=float)
        if not faces:
            return V, np.zeros((0, 3), dtype=np.int32)
        keep = []
        for tri in faces:
            a, b, c = V[tri[0]], V[tri[1]], V[tri[2]]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross < 0:
                keep.append([tri[0], tri[2], tri[1]])
            else:
                keep.append(list(tri))
        return V, np.asarray(keep, dtype=np.int32)

    if _Delaunay is None:
        faces = triangulate_like_ori_sim(outer_ccw)
        V = np.asarray(outer_ccw, dtype=float)
        return V, np.asarray(faces, dtype=np.int32)

    pts: List[List[float]] = [list(p) for p in outer_ccw]
    for h in hole_list:
        pts.extend(list(p) for p in h)
    V = np.asarray(pts, dtype=float)
    try:
        tri = _Delaunay(V)
    except Exception:
        faces = triangulate_like_ori_sim(outer_ccw)
        V = np.asarray(outer_ccw, dtype=float)
        return V, np.asarray(faces, dtype=np.int32)

    keep: List[List[int]] = []
    for simplex in tri.simplices:
        a, b, c = V[int(simplex[0])], V[int(simplex[1])], V[int(simplex[2])]
        cen = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0]
        if not _point_in_poly(cen, outer_ccw):
            continue
        if any(_point_in_poly(cen, h) for h in hole_list):
            continue
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if cross < 0:
            keep.append([int(simplex[0]), int(simplex[2]), int(simplex[1])])
        else:
            keep.append([int(simplex[0]), int(simplex[1]), int(simplex[2])])
    if not keep:
        faces = triangulate_like_ori_sim(outer_ccw)
        V = np.asarray(outer_ccw, dtype=float)
        return V, np.asarray(faces, dtype=np.int32)
    return V, np.asarray(keep, dtype=np.int32)


def _triangle_normal(a, b, c) -> Tuple[float, float, float]:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    nlen = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nlen < 1e-18:
        return 0.0, 0.0, 0.0
    return nx / nlen, ny / nlen, nz / nlen


def extrude_polygon_mesh(
    outer: Sequence,
    *,
    z_lo: float,
    z_hi: float,
    holes: Optional[Sequence[Sequence]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z0 = float(min(z_lo, z_hi))
    z1 = float(max(z_lo, z_hi))
    if z1 - z0 < 1e-12:
        z1 = z0 + 1e-3

    V2, faces2d = triangulate_polygon_with_holes(outer, holes)
    if len(V2) == 0 or len(faces2d) == 0:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=float),
        )

    n = len(V2)
    V3 = np.zeros((2 * n, 3), dtype=float)
    V3[:n, 0] = V2[:, 0]
    V3[:n, 1] = V2[:, 1]
    V3[:n, 2] = z0
    V3[n:, 0] = V2[:, 0]
    V3[n:, 1] = V2[:, 1]
    V3[n:, 2] = z1

    faces: List[List[int]] = []
    for tri in faces2d:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        faces.append([a, c, b])
        faces.append([a + n, b + n, c + n])

    rings: List[List[List[float]]] = [_dedupe_closed(outer)]
    for h in holes or []:
        hc = _dedupe_closed(h)
        if len(hc) >= 3:
            rings.append(hc)

    def _find_idx(p: Sequence) -> int:
        px, py = float(p[0]), float(p[1])
        d2 = (V2[:, 0] - px) ** 2 + (V2[:, 1] - py) ** 2
        return int(np.argmin(d2))

    for ring in rings:
        m = len(ring)
        for i in range(m):
            i0 = _find_idx(ring[i])
            i1 = _find_idx(ring[(i + 1) % m])
            faces.append([i0, i1, i1 + n])
            faces.append([i0, i1 + n, i0 + n])

    F = np.asarray(faces, dtype=np.int32)
    N = np.zeros((len(F), 3), dtype=float)
    for i, tri in enumerate(F):
        a, b, c = V3[int(tri[0])], V3[int(tri[1])], V3[int(tri[2])]
        N[i] = _triangle_normal(a, b, c)
    return V3, F, N


def merge_meshes(
    meshes: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not meshes:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=float),
        )
    vs: List[np.ndarray] = []
    fs: List[np.ndarray] = []
    ns: List[np.ndarray] = []
    offset = 0
    for V, F, N in meshes:
        if len(V) == 0 or len(F) == 0:
            continue
        vs.append(V)
        fs.append(F + offset)
        ns.append(N)
        offset += len(V)
    if not vs:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=float),
        )
    return np.vstack(vs), np.vstack(fs).astype(np.int32), np.vstack(ns)


def write_stl_binary(
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    *,
    header: str = "panel solid minus collision",
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n_tri = int(len(faces))
    hdr = header.encode("ascii", errors="replace")[:80]
    hdr = hdr + b"\0" * (80 - len(hdr))
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(struct.pack("<I", n_tri))
        for i in range(n_tri):
            a = vertices[int(faces[i, 0])]
            b = vertices[int(faces[i, 1])]
            c = vertices[int(faces[i, 2])]
            if normals is not None and i < len(normals):
                nx, ny, nz = (
                    float(normals[i, 0]),
                    float(normals[i, 1]),
                    float(normals[i, 2]),
                )
            else:
                nx, ny, nz = _triangle_normal(a, b, c)
            f.write(struct.pack("<3f", nx, ny, nz))
            f.write(struct.pack("<3f", float(a[0]), float(a[1]), float(a[2])))
            f.write(struct.pack("<3f", float(b[0]), float(b[1]), float(b[2])))
            f.write(struct.pack("<3f", float(c[0]), float(c[1]), float(c[2])))
            f.write(struct.pack("<H", 0))


# ---------------------------------------------------------------------------
# Build: panel solid − collision (visualize_3d associations)
# ---------------------------------------------------------------------------

# Default half-thickness (mm) for a lone ghost layer band when neighbors absent.
DEFAULT_LAYER_HALF_THICKNESS = 0.1


def collect_panel_collisions_by_layer(
    data: dict,
    *,
    kinds: Optional[Sequence[str]] = None,
) -> Dict[int, Dict[float, List[List[List[float]]]]]:
    """
    Collision footprints: panel → layer_h → rings.

    kinds: None → physical + ghost; or e.g. ("ghost",) for ghost-only.
    Side ribbons always skipped. Same polys as visualize_3d.

    Note: cleaned data from clean.py already merges layer stacks and stamps
    h_lo/h_hi; prefer :func:`collect_panel_collision_slabs` for extrusion.
    """
    want_kinds = None
    if kinds is not None:
        want_kinds = {str(k).strip().lower() for k in kinds}

    units = list(data.get("units") or [])
    records = resolve_shaded_associations(data, units)
    by_panel: Dict[int, Dict[float, List[List[List[float]]]]] = {}
    for rec in records:
        if not rec.get("drawable"):
            continue
        sk = _shell_kind_of(rec)
        if sk == "side":
            continue
        if want_kinds is not None and sk not in want_kinds:
            continue
        panel = rec.get("panel")
        poly = rec.get("poly")
        layer_h = rec.get("layer_h")
        if panel is None or poly is None:
            continue
        try:
            pid = int(panel)
        except (TypeError, ValueError):
            continue
        if layer_h is None:
            h = 0.0
        else:
            try:
                h = float(layer_h)
            except (TypeError, ValueError):
                h = 0.0
        ring = _dedupe_closed(np.asarray(poly, dtype=float)[:, :2])
        if len(ring) < 3:
            continue
        by_panel.setdefault(pid, {}).setdefault(h, []).append(ring)
    return by_panel


def collect_panel_collision_slabs(
    data: dict,
    *,
    kinds: Optional[Sequence[str]] = None,
    half_thickness: float = DEFAULT_LAYER_HALF_THICKNESS,
) -> Dict[int, List[Tuple[float, float, List[List[List[float]]]]]]:
    """
    panel → list of (z_lo, z_hi, rings) slabs ready to extrude.

    Prefers clean.py merge fields ``h_lo`` / ``h_hi`` (continuous min→max).
    When only sample ``layer_h`` values remain (unmerged / trimmed input),
    falls back to one continuous slab spanning min→max layer_h for that panel.
    """
    want_kinds = None
    if kinds is not None:
        want_kinds = {str(k).strip().lower() for k in kinds}

    units = list(data.get("units") or [])
    records = resolve_shaded_associations(data, units)
    half = max(float(half_thickness), 1e-6)

    # panel → list of (za, zb, ring, has_explicit_span)
    raw: Dict[int, List[Tuple[float, float, List[List[float]], bool]]] = {}
    for rec in records:
        if not rec.get("drawable"):
            continue
        sk = _shell_kind_of(rec)
        if sk == "side":
            continue
        if want_kinds is not None and sk not in want_kinds:
            continue
        panel = rec.get("panel")
        poly = rec.get("poly")
        if panel is None or poly is None:
            continue
        try:
            pid = int(panel)
        except (TypeError, ValueError):
            continue
        ring = _dedupe_closed(np.asarray(poly, dtype=float)[:, :2])
        if len(ring) < 3:
            continue

        h_lo = rec.get("h_lo")
        h_hi = rec.get("h_hi")
        layer_h = rec.get("layer_h")
        explicit = False
        try:
            if h_lo is not None and h_hi is not None:
                za, zb = float(h_lo), float(h_hi)
                if zb < za:
                    za, zb = zb, za
                explicit = True
            elif layer_h is not None:
                h = float(layer_h)
                za = zb = h
            else:
                za = zb = 0.0
        except (TypeError, ValueError):
            za = zb = 0.0

        raw.setdefault(pid, []).append((za, zb, ring, explicit))

    out: Dict[int, List[Tuple[float, float, List[List[List[float]]]]]] = {}
    for pid, entries in raw.items():
        if not entries:
            continue
        any_explicit = any(e[3] for e in entries)
        if any_explicit:
            # Group rings that share the same continuous span (clean merge parts)
            groups: Dict[Tuple[float, float], List[List[List[float]]]] = {}
            for za, zb, ring, explicit in entries:
                if not explicit:
                    # Rare: mixed — treat as zero-thickness at its sample height
                    key = (za, zb) if zb > za + 1e-12 else (za - half, za + half)
                else:
                    if zb - za < 1e-12:
                        key = (za - half, za + half)
                    else:
                        key = (za, zb)
                groups.setdefault(key, []).append(ring)
            slabs = [
                (float(k[0]), float(k[1]), rings)
                for k, rings in sorted(groups.items(), key=lambda kv: kv[0][0])
            ]
        else:
            # Unmerged intermediate layers: one continuous min→max band
            by_h: Dict[float, List[List[List[float]]]] = {}
            for za, zb, ring, _ in entries:
                h = 0.5 * (za + zb)
                by_h.setdefault(h, []).append(ring)
            slabs = ghost_layer_slabs(by_h, half_thickness=half)
        if slabs:
            out[pid] = slabs
    return out


def _union_rings_clipped(
    outer_xy: Sequence,
    rings: Sequence[Sequence],
) -> List[Tuple[List[List[float]], List[List[List[float]]]]]:
    """Clip rings to panel and return polygon pieces (outer, holes)."""
    return collision_pieces_clipped(outer_xy, rings)


def _union_rings_unclipped(
    rings: Sequence[Sequence],
) -> List[Tuple[List[List[float]], List[List[List[float]]]]]:
    """
    Repair + 2D-union collision rings (no panel clip).

    Used so every ghost/main footprint in a panel group becomes one solid
    before the continuous min→max Z extrude.
    """
    parts: List[Polygon] = []
    for ring in rings or []:
        parts.extend(_repair_ring_polygons(ring))
    if not parts:
        return []
    try:
        merged = unary_union(parts)
    except Exception:
        merged = parts[0]
        for p in parts[1:]:
            try:
                merged = merged.union(p)
            except Exception:
                continue
    if merged is None:
        return []
    try:
        if getattr(merged, "is_empty", True):
            return []
    except Exception:
        return []
    if not getattr(merged, "is_valid", True):
        try:
            merged = make_valid(merged)
        except Exception:
            try:
                merged = merged.buffer(0)
            except Exception:
                return []
    out: List[Tuple[List[List[float]], List[List[List[float]]]]] = []
    for p in _geom_to_polygon_list(merged):
        rings_out = _polygon_to_rings(p)
        if rings_out is not None:
            out.append(rings_out)
    return out


def ghost_layer_slabs(
    by_layer: Dict[float, List[List[List[float]]]],
    *,
    half_thickness: float = DEFAULT_LAYER_HALF_THICKNESS,
) -> List[Tuple[float, float, List[List[List[float]]]]]:
    """
    Merge all physical + ghost rings for one panel into a single continuous
    Z slab spanning min(layer_h) → max(layer_h).

    Intermediate ghost / main heights no longer produce separate thin bands
    with empty gaps between them — everything in the group is one solid.
    A lone height uses ±half_thickness about that layer.
    """
    if not by_layer:
        return []
    heights = sorted(float(h) for h in by_layer.keys())
    half = max(float(half_thickness), 1e-6)

    all_rings: List[List[List[float]]] = []
    for h in heights:
        all_rings.extend(list(by_layer[h]))
    if not all_rings:
        return []

    if len(heights) == 1:
        z_a = heights[0] - half
        z_b = heights[0] + half
    else:
        z_a = float(heights[0])
        z_b = float(heights[-1])
    if z_b - z_a < 1e-12:
        mid = 0.5 * (z_a + z_b)
        z_a, z_b = mid - half, mid + half

    return [(float(z_a), float(z_b), all_rings)]


def _extrude_rings_raw(
    rings: Sequence[Sequence],
    *,
    z_lo: float,
    z_hi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Extrude each ring as its own solid (no panel clip, no union).

    Returns mesh + count of rings that produced faces.
    """
    meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    n_ok = 0
    for ring in rings or []:
        r = _dedupe_closed(ring)
        if len(r) < 3:
            continue
        V, F, N = extrude_polygon_mesh(r, z_lo=z_lo, z_hi=z_hi, holes=None)
        if len(V) and len(F):
            meshes.append((V, F, N))
            n_ok += 1
    V, F, N = merge_meshes(meshes)
    return V, F, N, n_ok


def build_panel_meshes(
    data: dict,
    *,
    default_thickness: float = DEFAULT_THICKNESS,
    thickness: Optional[float] = None,
    panels: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Build three mesh groups per design unit:

      original   — full panel stock (no cuts)
      collision  — clean.py layer-stack merge (h_lo→h_hi) when present;
                   else fallback min→max of sample layer_h
      final      — stock − that collision band

    Thickness offsets match visualize_3d.panel_thickness_offsets.
    """
    units = list(data.get("units") or [])
    shades = list(data.get("shaded_regions") or [])
    offsets = panel_thickness_offsets(units, data, shades)
    # Prefer clean.py continuous spans (h_lo/h_hi); fallback min→max merge
    all_slabs = collect_panel_collision_slabs(data, kinds=None)

    want: Optional[set] = set(int(p) for p in panels) if panels is not None else None

    groups: Dict[str, List[Dict[str, Any]]] = {
        "original": [],
        "collision": [],
        "final": [],
    }
    report: Dict[str, Any] = {
        "n_panels": 0,
        "n_collisions_total": 0,
        "n_collision_layers": 0,
        "n_triangles_original": 0,
        "n_triangles_collision": 0,
        "n_triangles_final": 0,
        "panels": [],
        "build": (
            "visualize_3d: original stock | collision from clean "
            "h_lo→h_hi (or min→max fallback) | final = stock − collision"
        ),
    }

    for pi, unit in enumerate(units):
        if want is not None and pi not in want:
            continue
        outer = _dedupe_closed(_poly_xy(unit))
        if len(outer) < 3:
            continue

        offs = list(offsets[pi]) if pi < len(offsets) else [0.0]
        if thickness is not None and float(thickness) > 0:
            z0, z1 = _panel_z_span(offs, default_thickness=float(thickness))
            mid = 0.5 * (z0 + z1)
            half = 0.5 * float(thickness)
            z_lo, z_hi = mid - half, mid + half
            th = float(thickness)
        else:
            z_lo, z_hi = _panel_z_span(offs, default_thickness=default_thickness)
            th = float(z_hi - z_lo)

        slabs = list(all_slabs.get(pi) or [])
        n_coll = sum(len(rings) for _, _, rings in slabs)
        n_layers = len(slabs)

        # 1) original stock
        V_o, F_o, N_o = extrude_polygon_mesh(
            outer, z_lo=z_lo, z_hi=z_hi, holes=None
        )
        if len(V_o) == 0 or len(F_o) == 0:
            continue

        # 2) collision STL — continuous h_lo→h_hi from clean (or fallback)
        coll_meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        n_coll_pieces = 0
        n_coll_failed = 0
        for za, zb, rings in slabs:
            # Prefer 2D union → single extrude; raw fallback
            merged_pieces = _union_rings_unclipped(rings)
            if merged_pieces:
                Vc, Fc, Nc = _extrude_pieces(merged_pieces, z_lo=za, z_hi=zb)
                if len(Vc) and len(Fc):
                    n_coll_pieces += len(merged_pieces)
                    coll_meshes.append((Vc, Fc, Nc))
                else:
                    n_coll_failed += len(rings)
            else:
                Vc, Fc, Nc, n_ok = _extrude_rings_raw(rings, z_lo=za, z_hi=zb)
                n_coll_pieces += n_ok
                n_coll_failed += max(0, len(rings) - n_ok)
                if len(Vc) and len(Fc):
                    coll_meshes.append((Vc, Fc, Nc))
        V_c, F_c, N_c = merge_meshes(coll_meshes)

        # 3) final — stock with one cut through the merged collision band
        if not slabs:
            V_f, F_f, N_f = V_o, F_o, N_o
            n_final_pieces = 1
        else:
            final_meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            n_final_pieces = 0
            cut_slabs_sorted = sorted(slabs, key=lambda s: s[0])
            cursor = z_lo
            for za, zb, rings in cut_slabs_sorted:
                za_c = max(float(za), z_lo)
                zb_c = min(float(zb), z_hi)
                if za_c > cursor + 1e-12:
                    Vgap, Fgap, Ngap = extrude_polygon_mesh(
                        outer, z_lo=cursor, z_hi=za_c, holes=None
                    )
                    if len(Vgap) and len(Fgap):
                        final_meshes.append((Vgap, Fgap, Ngap))
                        n_final_pieces += 1
                if zb_c - za_c > 1e-12 and rings:
                    final_pieces = boolean_panel_minus_collisions(outer, rings)
                    if not final_pieces:
                        # clip wiped everything — keep solid stock for band
                        final_pieces = [(outer, [])]
                    Vf, Ff, Nf = _extrude_pieces(
                        final_pieces, z_lo=za_c, z_hi=zb_c
                    )
                    if len(Vf) and len(Ff):
                        final_meshes.append((Vf, Ff, Nf))
                        n_final_pieces += len(final_pieces)
                cursor = max(cursor, zb_c)
            if z_hi > cursor + 1e-12:
                Vgap, Fgap, Ngap = extrude_polygon_mesh(
                    outer, z_lo=cursor, z_hi=z_hi, holes=None
                )
                if len(Vgap) and len(Fgap):
                    final_meshes.append((Vgap, Fgap, Ngap))
                    n_final_pieces += 1
            V_f, F_f, N_f = merge_meshes(final_meshes)
            if len(V_f) == 0 or len(F_f) == 0:
                V_f, F_f, N_f = V_o, F_o, N_o

        base_meta = {
            "panel": pi,
            "thickness": th,
            "z_lo": z_lo,
            "z_hi": z_hi,
            "offsets": offs,
            "n_collisions": n_coll,
            "n_collision_layers": n_layers,
            "n_z_slabs": len(slabs),
            "n_collision_failed": n_coll_failed,
        }
        groups["original"].append({
            **base_meta,
            "kind": "original",
            "n_pieces": 1,
            "n_verts": int(len(V_o)),
            "n_faces": int(len(F_o)),
            "vertices": V_o,
            "faces": F_o,
            "normals": N_o,
        })
        if len(V_c) and len(F_c):
            groups["collision"].append({
                **base_meta,
                "kind": "collision",
                "n_pieces": n_coll_pieces,
                "n_verts": int(len(V_c)),
                "n_faces": int(len(F_c)),
                "vertices": V_c,
                "faces": F_c,
                "normals": N_c,
            })
        groups["final"].append({
            **base_meta,
            "kind": "final",
            "n_pieces": n_final_pieces,
            "n_verts": int(len(V_f)),
            "n_faces": int(len(F_f)),
            "vertices": V_f,
            "faces": F_f,
            "normals": N_f,
        })

        report["n_panels"] += 1
        report["n_collisions_total"] += n_coll
        report["n_collision_layers"] += n_layers
        report["n_triangles_original"] += int(len(F_o))
        report["n_triangles_collision"] += int(len(F_c))
        report["n_triangles_final"] += int(len(F_f))
        report["panels"].append({
            "panel": pi,
            "thickness": th,
            "z_lo": z_lo,
            "z_hi": z_hi,
            "offsets": offs,
            "n_collisions": n_coll,
            "n_collision_layers": n_layers,
            "n_z_slabs": len(slabs),
            "n_collision_pieces": n_coll_pieces,
            "n_collision_failed": n_coll_failed,
            "n_final_pieces": n_final_pieces,
            "n_faces_original": int(len(F_o)),
            "n_faces_collision": int(len(F_c)),
            "n_faces_final": int(len(F_f)),
        })

    return groups, report


def _output_paths(base_path: str) -> Dict[str, str]:
    """
    Map role → path.

    base_path may be:
      .../stem.stl           → stem-original / -collision / -final
      .../stem-final.stl     → same stem family
      .../stem (no ext)      → treated as stem
    """
    directory = os.path.dirname(os.path.abspath(base_path)) or "."
    name = os.path.basename(base_path)
    stem, ext = os.path.splitext(name)
    if not ext:
        stem = name
    # If user already passed *-final / *-original / *-collision, strip role suffix
    lower = stem.lower()
    for suffix in ("-final", "_final", "-original", "_original",
                   "-collision", "_collision", "-trim", "_trim"):
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return {
        "original": os.path.join(directory, f"{stem}-original.stl"),
        "collision": os.path.join(directory, f"{stem}-collision.stl"),
        "final": os.path.join(directory, f"{stem}-final.stl"),
    }


def export_stl(
    data: dict,
    output_path: str,
    *,
    default_thickness: float = DEFAULT_THICKNESS,
    thickness: Optional[float] = None,
    panels: Optional[Sequence[int]] = None,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    groups, report = build_panel_meshes(
        data,
        default_thickness=default_thickness,
        thickness=thickness,
        panels=panels,
    )
    report["input"] = source_path
    report["outputs"] = {}
    if not groups.get("original"):
        raise RuntimeError("no panel meshes generated")

    paths = _output_paths(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(paths["final"])) or ".", exist_ok=True)

    for role in ("original", "collision", "final"):
        meshes = groups.get(role) or []
        path = paths[role]
        if not meshes:
            # Empty collision set: still write a minimal empty-ish solid? skip file.
            if role == "collision":
                report["outputs"][role] = None
                report[f"n_verts_{role}"] = 0
                report[f"n_faces_{role}"] = 0
                continue
            raise RuntimeError(f"no meshes for role={role}")
        V, F, N = merge_meshes(
            [(m["vertices"], m["faces"], m["normals"]) for m in meshes]
        )
        write_stl_binary(
            path,
            V,
            F,
            N,
            header=f"json_to_stl {role}",
        )
        report["outputs"][role] = path
        report[f"n_verts_{role}"] = int(len(V))
        report[f"n_faces_{role}"] = int(len(F))

    return report


def export_file(input_path: str, output_path: str, **kwargs: Any) -> Dict[str, Any]:
    data = _load_json(input_path)
    return export_stl(data, output_path, source_path=input_path, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write 3 STLs per design: original stock, collision trim-away, "
            "and final (stock − collision). Same associations as visualize_3d. "
            "Outputs: <stem>-original.stl, <stem>-collision.stl, <stem>-final.stl"
        )
    )
    parser.add_argument("path", nargs="?", default=None, help="Input JSON path")
    parser.add_argument("--config", default=None, help="YAML config")
    parser.add_argument(
        "--name",
        default=None,
        help="Stem under panel_trimming/trimmedData/",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Base .stl path (writes <stem>-original / -collision / -final). "
            "Default: trimmedData/<input-stem>-*.stl"
        ),
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=None,
        help=(
            "Force uniform thickness (mm). Default: span of panel thickness "
            f"offsets, or {DEFAULT_THICKNESS:g} if only one offset"
        ),
    )
    parser.add_argument(
        "--default-thickness",
        type=float,
        default=None,
        help=f"Thickness when a panel has a single offset (default {DEFAULT_THICKNESS:g})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert every *-trimmed / *-cleaned JSON in trimmedData/",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg_path = _resolve_config_path(args.config)
    cfg: dict = {}
    if cfg_path:
        try:
            cfg = _load_yaml(cfg_path)
        except Exception as exc:
            print(f"[stl] bad config {cfg_path}: {exc}", file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"[stl] config {cfg_path}")

    stl_cfg = dict(cfg.get("stl") or {})
    defaults = {
        "thickness": stl_cfg.get("thickness", cfg.get("thickness")),
        "default_thickness": float(
            stl_cfg.get(
                "default_thickness",
                cfg.get("default_thickness", DEFAULT_THICKNESS),
            )
        ),
    }
    if args.thickness is not None:
        defaults["thickness"] = float(args.thickness)
    if args.default_thickness is not None:
        defaults["default_thickness"] = float(args.default_thickness)

    exact_output: Optional[str] = None
    if args.output is not None:
        out_arg = str(args.output).strip()
        if not out_arg.lower().endswith(".stl"):
            print(f"[stl] -o must be a .stl path (got {out_arg!r})", file=sys.stderr)
            return 1
        exact_output = os.path.abspath(out_arg)

    work: List[Tuple[str, dict]] = []
    if args.all:
        if not os.path.isdir(TRIMMED_DIR):
            print(f"[stl] trimmedData not found: {TRIMMED_DIR}", file=sys.stderr)
            return 1
        for fn in sorted(os.listdir(TRIMMED_DIR)):
            low = fn.lower()
            if not low.endswith(".json"):
                continue
            if not (
                low.endswith("-cleaned.json")
                or low.endswith("_cleaned.json")
                or low.endswith("-trimmed.json")
                or low.endswith("_trimmed.json")
            ):
                continue
            work.append((os.path.join(TRIMMED_DIR, fn), dict(defaults)))
    elif args.path:
        work.append((_resolve_input(args.path), dict(defaults)))
    elif args.name:
        work.append((_resolve_input(args.name), dict(defaults)))
    else:
        jobs_cfg = list(
            stl_cfg.get("jobs")
            or cfg.get("stl_jobs")
            or cfg.get("jobs")
            or cfg.get("simulations")
            or []
        )
        if jobs_cfg:
            for job in jobs_cfg:
                if not isinstance(job, dict) or not job.get("name"):
                    continue
                settings = dict(defaults)
                for k in ("thickness", "default_thickness"):
                    if k in job and job[k] is not None:
                        settings[k] = job[k]
                try:
                    in_path = _resolve_input(str(job["name"]))
                except FileNotFoundError as exc:
                    print(f"[stl] SKIP {job['name']}: {exc}", file=sys.stderr)
                    continue
                work.append((in_path, settings))
        else:
            if os.path.isdir(TRIMMED_DIR):
                for fn in sorted(os.listdir(TRIMMED_DIR)):
                    low = fn.lower()
                    if low.endswith("-trimmed.json") or low.endswith("_trimmed.json"):
                        work.append((os.path.join(TRIMMED_DIR, fn), dict(defaults)))
                        break
                if not work:
                    for fn in sorted(os.listdir(TRIMMED_DIR)):
                        low = fn.lower()
                        if low.endswith("-cleaned.json") or low.endswith("_cleaned.json"):
                            work.append((os.path.join(TRIMMED_DIR, fn), dict(defaults)))
                            break
            if not work:
                print(
                    "[stl] no input; pass --name, path, --all, or config jobs",
                    file=sys.stderr,
                )
                return 1

    if not work:
        print("[stl] nothing to do", file=sys.stderr)
        return 1

    rc = 0
    for in_path, settings in work:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        if exact_output is not None and len(work) == 1:
            out_path = exact_output
        else:
            out_path = os.path.join(DEFAULT_OUTPUT_DIR, _stl_basename(in_path))
        try:
            report = export_file(
                in_path,
                out_path,
                thickness=settings.get("thickness"),
                default_thickness=float(
                    settings.get("default_thickness", DEFAULT_THICKNESS)
                ),
            )
        except Exception as exc:
            print(f"[stl] FAILED {in_path}: {exc}", file=sys.stderr)
            rc = 1
            continue

        if not args.quiet:
            outs = report.get("outputs") or {}
            print(
                f"[stl] {os.path.basename(in_path)} → "
                f"panels={report.get('n_panels')} "
                f"collisions={report.get('n_collisions_total')} "
                f"tris[orig/coll/final]="
                f"{report.get('n_triangles_original')}/"
                f"{report.get('n_triangles_collision')}/"
                f"{report.get('n_triangles_final')} "
                f"({report.get('build')})"
            )
            for role in ("original", "collision", "final"):
                p = outs.get(role)
                if p:
                    print(f"[stl] wrote {role:10s} {p}")
                else:
                    print(f"[stl] wrote {role:10s} (empty — skipped)")
            for pe in report.get("panels") or []:
                fail = pe.get("n_collision_failed", 0)
                fail_s = f" fail={fail}" if fail else ""
                print(
                    f"      panel {pe['panel']}: "
                    f"z=[{pe['z_lo']:g},{pe['z_hi']:g}] "
                    f"T={pe['thickness']:g} "
                    f"collisions={pe['n_collisions']} "
                    f"layers={pe.get('n_collision_layers', 0)} "
                    f"pieces={pe.get('n_collision_pieces', 0)}{fail_s} "
                    f"faces[o/c/f]="
                    f"{pe['n_faces_original']}/"
                    f"{pe['n_faces_collision']}/"
                    f"{pe['n_faces_final']}"
                )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
