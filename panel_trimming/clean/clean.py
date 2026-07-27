"""
Clean trimmed collision shades: drop collapsed paints, UF-clean nodes, cut, offset.

Reads  panel_trimming/trimmedData/<name>-trimmed.json  (or a path)
Writes panel_trimming/trimmedData/<name>-cleaned.json  (-trimmed ΓåÆ -cleaned suffix)

Pipeline (per kept shaded region):
  1) Snap dual-curve coords onto nearby crease / border segments
     (mountain, valley, border lines + host panel edges) when within snap_tol
  2) Round all dual-curve coords to 6 d.p. (float noise ΓåÆ stable grouping)
  3) Union-Find group messy nodes (esp. vertex clouds) under custom
     constraints; keep the outermost node per cluster (farthest from the
     region sample centroid ΓÇö never the cluster mean)
  4) Straight containing cut = convex hull of UF winners (panel-clipped),
     residual re-hull if any original sample is outside, optional RDP only
     when containment is preserved ΓåÆ never undershoot shaded region
  5) Re-snap cut_ring vertices/edges onto creases (closes thin residual
     walls left by simplify / UF when original ΓêÆ collision is applied)
  6) Manual fabrication offset outward from the cut polygon (barrier /
     panel clamped), then re-snap fabric_ring onto creases
  7) Per panel: merge each stream into one solid prism (largest cleaned layer):
       main    = physical + ghosts inside the physical shell span
       support = support pads + ghosts outside that span (support-collision);
                 Z expanded to abut physical stock
     Same rule both streams: clean every layer, then extrude the single
     largest-area cleaned ring from min→max layer_h (not multi-layer union).
     Side ribbons unchanged. Pre-merge layers kept as shaded_regions_layers.
     Merged rings are re-snapped onto creases after the prism merge.

Also drops:
  - kind == "line"  (exporter collapsed-contact flag)
  - recomputed ribbon area ~ 0  (c0/c1 reverse-trace; shoelace cancels)
  - optional: area < min_area (default 0 = off; keeps small pink fills)

Viz helpers:
  clean_ring   = dual-curve sample nodes (c0 + reverse c1), rounded
  coordinates  = cut_ring, or fabric_ring when offset > 0
  cut_ring     = step-3 straight containing cut
  fabric_ring  = step-4 offset outline
  clean_edges  = straight cut / fabric edges

YAML (panel_trimming/clean/config.yml):
  jobs:
    - name: miura-thick-trimmed
      fabric_offset: 0.5
      group_tol: 3.0

Usage:
  python panel_trimming/clean/clean.py
  python panel_trimming/clean/clean.py --config panel_trimming/clean/config.yml
  python panel_trimming/clean/clean.py --name miura-thick-trimmed --fabric-offset 0.8
  python panel_trimming/clean/clean.py --name mountain-thick-trimmed --min-area 20
  python panel_trimming/clean/clean.py --name miura-thick-trimmed --cut-tol 2.0
  python panel_trimming/clean/clean.py --name miura-thick-trimmed --group-tol 3.0
  python panel_trimming/clean/clean.py --name miura-thick-trimmed --no-straight-cut
  python panel_trimming/clean/clean.py --name miura-thick-trimmed --no-viz
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# panel_trimming/clean/ ΓåÆ panel_trimming ΓåÆ project root
_PANEL_TRIM_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_PANEL_TRIM_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Trimmed + cleaned JSON both under panel_trimming/trimmedData/
TRIMMED_DIR = os.path.join(_PANEL_TRIM_DIR, "trimmedData")
DESCRIPTION_DIR = os.path.join(_PROJECT_ROOT, "descriptionData")
DEFAULT_OUTPUT_DIR = TRIMMED_DIR
DEFAULT_CONFIG = os.path.join(_THIS_DIR, "config.yml")
DEFAULT_CONFIG_EXAMPLE = os.path.join(_THIS_DIR, "config.example.yml")

# Default min filled-ribbon area. 0 = keep all non-collapsed fills (P9/P0 etc.).
DEFAULT_MIN_AREA = 0.0
# Ribbon shoelace below this is treated as a collapsed contact line (crease-like).
COLLAPSED_AREA_EPS = 1e-6
# Near-duplicate vertex eps when building hull / ring
VERTEX_EPS = 1e-9
# Max perpendicular deviation when simplifying dense convex hulls (design units).
# Larger ΓåÆ fewer, longer straight edges (still containment-checked).
DEFAULT_CUT_TOL = 3.5
# Small outward slack before simplify so aggressive straight cuts don't undershoot.
# Clamped by panel / barriers; keeps overshoot modest.
DEFAULT_STRAIGHT_SLACK = 0.6
# Reject simplified cuts that grow area by more than this factor vs hull
DEFAULT_MAX_OVERSHOOT_RATIO = 1.35
# Outward fabrication offset (design units / mm). 0 = off.
DEFAULT_FABRIC_OFFSET = 0.0
# line_features.type: mountain / valley / border ΓÇö expansion may not cross these
TYPE_MOUNTAIN = 0
TYPE_VALLEY = 1
TYPE_BORDER = 2
BARRIER_TYPES = (TYPE_MOUNTAIN, TYPE_VALLEY, TYPE_BORDER)
# Stop a short epsilon before the barrier so we don't sit on top of creases
BARRIER_STOP_EPS = 1e-4
# Union-Find cluster distance for messy nodes (design units)
DEFAULT_GROUP_TOL = 3.0
# Alias kept for older config keys
DEFAULT_CORNER_MERGE_TOL = DEFAULT_GROUP_TOL
# Quantize geometry before UF / hull (decimal places)
DEFAULT_ROUND_DP = 6
# Whether UF refuses to merge pairs whose segment crosses a crease/border
DEFAULT_UF_RESPECT_BARRIERS = False
# Snap dual-curve samples / cut rings onto creases/borders if within this
# distance (mm). 0 = off. Needs to cover the typical dual-curve inset so
# original ΓêÆ collision does not leave a thin wall along the crease.
DEFAULT_SNAP_TOL = 1.5
# Edges nearly parallel to a crease (|cos| >= this) may be fully projected
# onto that crease when their mean distance is within snap_tol.
SNAP_EDGE_PARALLEL_COS = 0.97
# Per-panel: merge ghost + physical (and separately support) layer stacks into
# continuous [min layer_h, max layer_h] solids (no intermediate Z gaps).
DEFAULT_MERGE_LAYER_STACK = True


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(path: Optional[str] = None) -> Optional[str]:
    """Prefer config.yml, else config.example.yml under clean/."""
    if path and os.path.isfile(path):
        return os.path.abspath(path)
    if path:
        # allow stem under clean/
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
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)
    basenames = [name_or_path]
    if not name_or_path.endswith(".json"):
        basenames.append(name_or_path + ".json")
    # Prefer panel_trimming/trimmedData; fall back to legacy project-root dirs
    legacy_trimmed = os.path.join(_PROJECT_ROOT, "trimmedData")
    legacy_cleaned = os.path.join(_PANEL_TRIM_DIR, "cleanedData")
    legacy_root_cleaned = os.path.join(_PROJECT_ROOT, "cleanedData")
    search_dirs = (
        TRIMMED_DIR,
        DESCRIPTION_DIR,
        legacy_trimmed,
        legacy_cleaned,
        legacy_root_cleaned,
        _PROJECT_ROOT,
    )
    for d in search_dirs:
        for bn in basenames:
            cand = os.path.join(d, bn)
            if os.path.isfile(cand):
                return os.path.abspath(cand)
            cand = os.path.join(d, os.path.basename(bn))
            if os.path.isfile(cand):
                return os.path.abspath(cand)
    raise FileNotFoundError(
        f"JSON not found for: {name_or_path} "
        f"(searched {TRIMMED_DIR}, {DESCRIPTION_DIR})"
    )


def _cleaned_basename(input_path: str) -> str:
    """
    mountain-thick-trimmed.json ΓåÆ mountain-thick-cleaned.json
    mountain-thick.json         ΓåÆ mountain-thick-cleaned.json
    mountain-thick-cleaned.json ΓåÆ mountain-thick-cleaned.json  (idempotent)
    """
    base = os.path.basename(input_path)
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".json"
    lower = stem.lower()
    if lower.endswith("-cleaned") or lower.endswith("_cleaned"):
        return stem + ext
    if lower.endswith("-trimmed"):
        stem = stem[: -len("-trimmed")] + "-cleaned"
    elif lower.endswith("_trimmed"):
        stem = stem[: -len("_trimmed")] + "_cleaned"
    else:
        stem = stem + "-cleaned"
    return stem + ext


def _polygon_area_2d(pts: Sequence) -> float:
    n = len(pts) if pts is not None else 0
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x0 = float(pts[i][0])
        y0 = float(pts[i][1])
        x1 = float(pts[(i + 1) % n][0])
        y1 = float(pts[(i + 1) % n][1])
        acc += x0 * y1 - x1 * y0
    return abs(acc) * 0.5


def _ribbon_ring(c0: Sequence, c1: Sequence) -> Optional[List[List[float]]]:
    if not c0 or not c1 or len(c0) != len(c1) or len(c0) < 2:
        return None
    ring: List[List[float]] = [[float(p[0]), float(p[1])] for p in c0]
    ring += [[float(p[0]), float(p[1])] for p in reversed(c1)]
    cleaned = [ring[0]]
    for q in ring[1:]:
        prev = cleaned[-1]
        if abs(q[0] - prev[0]) > 1e-9 or abs(q[1] - prev[1]) > 1e-9:
            cleaned.append(q)
    if len(cleaned) < 3:
        return None
    if _polygon_area_2d(cleaned) < 1e-12:
        return None
    return cleaned


def region_area(region: dict) -> float:
    """Prefer stored area; recompute from dual curves when missing/zero."""
    stored = region.get("area")
    try:
        a = float(stored) if stored is not None else 0.0
    except (TypeError, ValueError):
        a = 0.0
    if a > 1e-12:
        return a
    ring = _ribbon_ring(region.get("c0") or [], region.get("c1") or [])
    if ring is None:
        return 0.0
    return _polygon_area_2d(ring)


def ribbon_area(region: dict) -> float:
    """Shoelace area of dual-curve ribbon only (ignore stored area)."""
    ring = _ribbon_ring(region.get("c0") or [], region.get("c1") or [])
    if ring is None:
        return 0.0
    return _polygon_area_2d(ring)


def is_collapsed_line(
    region: dict,
    *,
    area_eps: float = COLLAPSED_AREA_EPS,
) -> Tuple[bool, str]:
    """
    True for free-standing crease-like paints: dual curves collapse so the
    ribbon has no 2D fill (renders as a single red last segment only).
    """
    kind = str(region.get("kind") or "").strip().lower()
    ra = ribbon_area(region)
    if kind == "line":
        return True, f"kind=line ribbon_area={ra:.6g}"
    if ra < float(area_eps):
        return True, f"collapsed ribbon_area={ra:.6g} < eps={area_eps:g}"
    stored = region.get("area")
    try:
        sa = float(stored) if stored is not None else 0.0
    except (TypeError, ValueError):
        sa = 0.0
    if sa < float(area_eps) and ra < float(area_eps):
        return True, f"zero-area stored={sa:.6g} ribbon={ra:.6g}"
    return False, f"ribbon_area={ra:.6g}"


def is_too_small(
    region: dict,
    *,
    min_area: float,
    drop_line_kind: bool = True,
) -> Tuple[bool, str]:
    """
    Return (drop?, reason).

    Drops:
      - collapsed / kind=line contact (free-standing red crease-like paint)
      - effective area < min_area only when min_area > 0 (optional; keeps
        small pink fills by default)
    """
    if drop_line_kind:
        collapsed, creason = is_collapsed_line(region)
        if collapsed:
            return True, creason
    area = region_area(region)
    if float(min_area) > 0.0 and area < float(min_area):
        return True, f"area={area:.6g} < min_area={min_area:g}"
    return False, f"keep area={area:.6g}"


def filter_shaded_regions(
    regions: Sequence[dict],
    *,
    min_area: float = DEFAULT_MIN_AREA,
    drop_line_kind: bool = True,
) -> Tuple[List[dict], List[Dict[str, Any]]]:
    kept: List[dict] = []
    dropped_log: List[Dict[str, Any]] = []
    for i, r in enumerate(regions or []):
        drop, reason = is_too_small(
            r, min_area=min_area, drop_line_kind=drop_line_kind
        )
        if drop:
            dropped_log.append({
                "index": i,
                "panel": r.get("panel"),
                "layer_h": r.get("layer_h"),
                "kind": r.get("kind"),
                "shell_kind": r.get("shell_kind"),
                "area": region_area(r),
                "reason": reason,
            })
        else:
            # Refresh area if it was zero/stale but dual curves yield a real ribbon
            out = dict(r)
            a = region_area(r)
            if a > 0 and (out.get("area") is None or float(out.get("area") or 0) <= 0):
                out["area"] = float(a)
                if out.get("kind") in (None, "line", "empty") and (
                    min_area <= 0 or a >= min_area
                ):
                    out["kind"] = "sweep"
            # Strip prior simplify fields; dual curves stay source of truth
            for k in (
                "linear_c0",
                "linear_c1",
                "coordinates",
                "cut_ring",
                "fabric_ring",
                "fabric_offset",
                "boundary_order",
                "element_order",
                "curve_order",
                "approx_type",
                "spline",
                "curved_boundary",
                "high_order",
            ):
                out.pop(k, None)
            kept.append(out)
    return kept, dropped_log


# ---------------------------------------------------------------------------
# Straight containing cut = convex hull of all dual-curve samples
# ---------------------------------------------------------------------------

def _xy(p) -> List[float]:
    return [float(p[0]), float(p[1])]


def _round_coord(x: float, *, dp: int = DEFAULT_ROUND_DP) -> float:
    return round(float(x), int(dp))


def _round_pt(p, *, dp: int = DEFAULT_ROUND_DP) -> List[float]:
    return [_round_coord(p[0], dp=dp), _round_coord(p[1], dp=dp)]


def _round_poly(pts: Sequence, *, dp: int = DEFAULT_ROUND_DP) -> List[List[float]]:
    return [_round_pt(p, dp=dp) for p in (pts or [])]


def round_region_geometry(region: dict, *, dp: int = DEFAULT_ROUND_DP) -> None:
    """In-place: quantize dual-curve (and residual poly) coords to ``dp`` d.p."""
    if region.get("c0") is not None:
        region["c0"] = _round_poly(region["c0"], dp=dp)
    if region.get("c1") is not None:
        region["c1"] = _round_poly(region["c1"], dp=dp)
    for key in ("coordinates", "cut_ring", "fabric_ring", "clean_ring"):
        if region.get(key) is not None:
            region[key] = _round_poly(region[key], dp=dp)


def _dist2(a: Sequence, b: Sequence) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


def _cross_o(o: Sequence, a: Sequence, b: Sequence) -> float:
    return (float(a[0]) - float(o[0])) * (float(b[1]) - float(o[1])) - (
        float(a[1]) - float(o[1])
    ) * (float(b[0]) - float(o[0]))


def _dedupe_points(pts: Sequence, *, eps: float = VERTEX_EPS) -> List[List[float]]:
    """Drop near-duplicate points (order-preserving, first kept)."""
    out: List[List[float]] = []
    eps2 = float(eps) * float(eps)
    for p in pts or []:
        q = _xy(p)
        if not any(_dist2(q, r) <= eps2 for r in out):
            out.append(q)
    return out


def convex_hull(pts: Sequence) -> List[List[float]]:
    """
    Convex hull (monotone chain), CCW, straight edges only.
    Smallest convex set containing every input point ΓåÆ full red ribbon inside,
    hull area ΓëÑ ribbon area.
    """
    raw = _dedupe_points(pts)
    n = len(raw)
    if n <= 1:
        return [list(p) for p in raw]
    if n == 2:
        return [list(raw[0]), list(raw[1])]

    order = sorted(range(n), key=lambda i: (raw[i][0], raw[i][1]))
    lower: List[int] = []
    for i in order:
        while len(lower) >= 2 and _cross_o(
            raw[lower[-2]], raw[lower[-1]], raw[i]
        ) <= 0.0:
            lower.pop()
        lower.append(i)
    upper: List[int] = []
    for i in reversed(order):
        while len(upper) >= 2 and _cross_o(
            raw[upper[-2]], raw[upper[-1]], raw[i]
        ) <= 0.0:
            upper.pop()
        upper.append(i)
    hull_idx = lower[:-1] + upper[:-1]
    seen = set()
    hull: List[List[float]] = []
    for i in hull_idx:
        if i in seen:
            continue
        seen.add(i)
        hull.append(list(raw[i]))
    return hull if len(hull) >= 3 else [list(p) for p in raw]


def _point_in_or_on_polygon(p: Sequence, poly: Sequence, eps: float = 1e-7) -> bool:
    """Ray cast + boundary check; used to verify cut contains samples."""
    if not poly or len(poly) < 3:
        return False
    x, y = float(p[0]), float(p[1])
    n = len(poly)
    eps = max(float(eps), 1e-9)
    eps2 = eps * eps
    # Near a vertex? (avoids false outs after 6 d.p. rounding / clamps)
    for i in range(n):
        dx = x - float(poly[i][0])
        dy = y - float(poly[i][1])
        if dx * dx + dy * dy <= eps2:
            return True
    # On boundary edge?
    for i in range(n):
        ax, ay = float(poly[i][0]), float(poly[i][1])
        bx, by = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        abx, aby = bx - ax, by - ay
        apx, apy = x - ax, y - ay
        lab2 = abx * abx + aby * aby
        if lab2 < 1e-24:
            continue
        # Perp distance to infinite line, then clamp to segment
        t = (apx * abx + apy * aby) / lab2
        if t < -1e-9 or t > 1.0 + 1e-9:
            continue
        projx = ax + t * abx
        projy = ay + t * aby
        ddx = x - projx
        ddy = y - projy
        if ddx * ddx + ddy * ddy <= eps2:
            return True
        # Also accept classic cross-product boundary test
        cross = abx * apy - aby * apx
        if abs(cross) <= eps * math.sqrt(lab2):
            return True
    # Interior (even-odd)
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


def _point_seg_dist(p: Sequence, a: Sequence, b: Sequence) -> float:
    """Perpendicular distance from point to the infinite line through AB."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(p[0]), float(p[1])
    dx, dy = bx - ax, by - ay
    lab2 = dx * dx + dy * dy
    if lab2 < 1e-24:
        return math.hypot(px - ax, py - ay)
    return abs(dx * (ay - py) - dy * (ax - px)) / math.sqrt(lab2)


def _project_point_to_segment(
    p: Sequence,
    a: Sequence,
    b: Sequence,
) -> Tuple[List[float], float]:
    """
    Closest point on segment AB to P, and Euclidean distance.

    Projection is clamped to the segment (not the infinite line).
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(p[0]), float(p[1])
    abx, aby = bx - ax, by - ay
    lab2 = abx * abx + aby * aby
    if lab2 < 1e-24:
        return [ax, ay], math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / lab2
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * abx, ay + t * aby
    return [qx, qy], math.hypot(px - qx, py - qy)


def snap_point_to_segments(
    p: Sequence,
    segments: Sequence[Tuple[Sequence, Sequence]],
    *,
    snap_tol: float,
) -> Tuple[List[float], bool, float]:
    """
    If P is within snap_tol of any segment, return the closest projection.

    Returns (point_xy, snapped?, distance_to_segment).
    """
    tol = float(snap_tol)
    if tol <= 0 or not segments:
        q = _xy(p)
        return q, False, 0.0
    best_q = _xy(p)
    best_d = float("inf")
    for a, b in segments:
        q, d = _project_point_to_segment(p, a, b)
        if d < best_d:
            best_d = d
            best_q = q
    if best_d <= tol + 1e-15:
        return best_q, True, float(best_d)
    return _xy(p), False, float(best_d if best_d < float("inf") else 0.0)


def snap_poly_to_segments(
    pts: Sequence,
    segments: Sequence[Tuple[Sequence, Sequence]],
    *,
    snap_tol: float,
) -> Tuple[List[List[float]], int]:
    """Snap each vertex of a polyline/ring. Returns (new_pts, n_snapped)."""
    out: List[List[float]] = []
    n_snap = 0
    for p in pts or []:
        q, did, _ = snap_point_to_segments(p, segments, snap_tol=snap_tol)
        out.append(q)
        if did:
            n_snap += 1
    return out, n_snap


def snap_closed_ring_to_creases(
    ring: Sequence,
    segments: Sequence[Tuple[Sequence, Sequence]],
    *,
    snap_tol: float,
    parallel_cos: float = SNAP_EDGE_PARALLEL_COS,
) -> Tuple[List[List[float]], int]:
    """
    Snap a closed cut/fabric ring tightly onto crease/border segments.

    1) Vertex snap: each vertex within ``snap_tol`` of a segment is projected
       onto that segment (exact on-crease, no barrier stop-eps pullback).
    2) Edge attraction: if a ring edge is nearly parallel to a crease and
       every sample along the edge is within ``snap_tol`` of that crease,
       both endpoints are projected onto it so the whole edge sits on the
       crease (closes thin residual walls after original ΓêÆ collision).

    Returns (new_ring, n_vertices_moved).
    """
    tol = float(snap_tol)
    raw = [_xy(p) for p in ring or []]
    if tol <= 0 or len(raw) < 3 or not segments:
        return raw, 0

    out = [list(p) for p in raw]
    n = len(out)
    n_moved = 0

    # Pass 1: vertex snap onto nearest barrier within tol
    for i in range(n):
        q, did, _ = snap_point_to_segments(out[i], segments, snap_tol=tol)
        if did and _dist2(out[i], q) > 1e-24:
            out[i] = q
            n_moved += 1
        elif did:
            out[i] = q  # exact coincidence / float clean-up

    # Pass 2: project near-parallel edges fully onto matching creases
    cos_min = float(parallel_cos)
    for i in range(n):
        a = out[i]
        b = out[(i + 1) % n]
        ex = float(b[0]) - float(a[0])
        ey = float(b[1]) - float(a[1])
        el = math.hypot(ex, ey)
        if el < 1e-12:
            continue
        ux, uy = ex / el, ey / el
        best: Optional[Tuple[float, Sequence, Sequence]] = None
        for sa, sb in segments:
            sx = float(sb[0]) - float(sa[0])
            sy = float(sb[1]) - float(sa[1])
            sl = math.hypot(sx, sy)
            if sl < 1e-12:
                continue
            sux, suy = sx / sl, sy / sl
            if abs(ux * sux + uy * suy) < cos_min:
                continue
            # Sample edge; require all samples within tol of this segment
            ds: List[float] = []
            ok = True
            for k in range(5):
                t = k / 4.0
                p = [float(a[0]) + t * ex, float(a[1]) + t * ey]
                _, d = _project_point_to_segment(p, sa, sb)
                if d > tol + 1e-12:
                    ok = False
                    break
                ds.append(d)
            if not ok or not ds:
                continue
            mean_d = sum(ds) / len(ds)
            if best is None or mean_d < best[0]:
                best = (mean_d, sa, sb)
        if best is None:
            continue
        _, sa, sb = best
        qa, _ = _project_point_to_segment(a, sa, sb)
        qb, _ = _project_point_to_segment(b, sa, sb)
        if math.hypot(qb[0] - qa[0], qb[1] - qa[1]) < 1e-12:
            continue
        if _dist2(out[i], qa) > 1e-24:
            n_moved += 1
        if _dist2(out[(i + 1) % n], qb) > 1e-24:
            n_moved += 1
        out[i] = qa
        out[(i + 1) % n] = qb

    # Drop near-duplicates introduced by projecting both ends of short edges
    cleaned: List[List[float]] = []
    for p in out:
        if not cleaned or _dist2(cleaned[-1], p) > VERTEX_EPS * VERTEX_EPS:
            cleaned.append(list(p))
    if (
        len(cleaned) >= 2
        and _dist2(cleaned[0], cleaned[-1]) <= VERTEX_EPS * VERTEX_EPS
    ):
        cleaned = cleaned[:-1]
    if len(cleaned) < 3:
        return raw, 0
    if _poly_signed_area(cleaned) < 0:
        cleaned = list(reversed(cleaned))
    return cleaned, int(n_moved)


def snap_region_to_creases(
    region: dict,
    data: Optional[dict],
    *,
    snap_tol: float = DEFAULT_SNAP_TOL,
    rings_only: bool = False,
) -> Dict[str, Any]:
    """
    In-place: snap dual-curve and/or cut/fabric rings onto nearby creases.

    Targets: mountain, valley, and border line segments from the design, plus
    the host panel outline edges (same set as barrier segments).

    When ``rings_only`` is True, only closed export rings (cut/fabric/
    coordinates/clean_ring) are edge-attracted onto creases ΓÇö used after the
    straight cut / fabric offset / merge so simplify cannot leave a thin gap.
    """
    tol = float(snap_tol)
    info: Dict[str, Any] = {
        "snap_tol": tol,
        "n_snapped": 0,
        "n_points": 0,
        "n_ring_moved": 0,
        "skipped": False,
        "rings_only": bool(rings_only),
    }
    if tol <= 0:
        info["skipped"] = True
        info["reason"] = "snap_tol<=0"
        return info

    segs = collect_barrier_segments(data, region=region)
    if not segs:
        info["skipped"] = True
        info["reason"] = "no_crease_segments"
        return info

    n_snap = 0
    n_pts = 0
    n_ring_moved = 0

    if not rings_only:
        for key in ("c0", "c1"):
            raw = region.get(key)
            if not raw:
                continue
            snapped, k = snap_poly_to_segments(raw, segs, snap_tol=tol)
            region[key] = snapped
            n_snap += k
            n_pts += len(snapped)

    # Closed rings: vertex snap + near-parallel edge attraction
    ring_keys = ("cut_ring", "fabric_ring", "coordinates", "clean_ring")
    if not rings_only:
        # Early pipeline may only have residual coordinates
        ring_keys = ("coordinates", "cut_ring", "fabric_ring", "clean_ring")

    for key in ring_keys:
        raw = region.get(key)
        if not raw or len(raw) < 3:
            continue
        snapped, k = snap_closed_ring_to_creases(raw, segs, snap_tol=tol)
        region[key] = snapped
        n_ring_moved += k
        n_pts += len(snapped)
        n_snap += k

    info["n_snapped"] = int(n_snap)
    info["n_points"] = int(n_pts)
    info["n_ring_moved"] = int(n_ring_moved)
    info["n_segments"] = len(segs)
    return info


def _rdp_indices_open(pts: Sequence, eps: float) -> List[int]:
    """DouglasΓÇôPeucker keep indices for an open polyline."""
    n = len(pts) if pts is not None else 0
    if n <= 0:
        return []
    if n <= 2:
        return list(range(n))
    keep = [False] * n
    keep[0] = True
    keep[n - 1] = True
    stack: List[Tuple[int, int]] = [(0, n - 1)]
    tol = float(eps)
    while stack:
        i0, i1 = stack.pop()
        best_d, best_i = -1.0, -1
        a, b = pts[i0], pts[i1]
        for i in range(i0 + 1, i1):
            d = _point_seg_dist(pts[i], a, b)
            if d > best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d > tol:
            keep[best_i] = True
            stack.append((i0, best_i))
            stack.append((best_i, i1))
    return [i for i, k in enumerate(keep) if k]


def simplify_closed_polygon(
    pts: Sequence,
    *,
    tol: float,
) -> List[List[float]]:
    """
    RDP-simplify a closed polygon (order-1 straight edges).

    Used on dense convex hulls so a gently curved outer chain collapses to a
    few long straight cuts instead of dozens of micro-edges.
    """
    raw = [_xy(p) for p in pts or []]
    n = len(raw)
    if n <= 3 or float(tol) <= 0.0:
        return [list(p) for p in raw]

    # Find a stable "start" (leftmost-then-lowest) and rotate so RDP opens there
    start = min(range(n), key=lambda i: (raw[i][0], raw[i][1]))
    rot = raw[start:] + raw[:start]
    # Open polyline = full cycle without duplicating start at end for RDP,
    # but we need to close: RDP the chain rot[0]..rot[-1] then connect lastΓåÆfirst.
    # Use rot + [rot[0]] as open path of length n+1, RDP, drop duplicate end.
    path = rot + [rot[0]]
    idxs = _rdp_indices_open(path, float(tol))
    # Drop the final index if it is the duplicated start
    if len(idxs) >= 2 and idxs[-1] == len(path) - 1:
        idxs = idxs[:-1]
    simplified = [list(path[i]) for i in idxs]
    # Remove consecutive near-duplicates
    out: List[List[float]] = []
    for p in simplified:
        if not out or _dist2(out[-1], p) > VERTEX_EPS * VERTEX_EPS:
            out.append(p)
    if len(out) >= 2 and _dist2(out[0], out[-1]) <= VERTEX_EPS * VERTEX_EPS:
        out = out[:-1]
    return out if len(out) >= 3 else [list(p) for p in raw]


def _polygon_centroid(poly: Sequence) -> Tuple[float, float]:
    # Prefer area-weighted centroid; fall back to vertex mean
    n = len(poly) if poly is not None else 0
    if n == 0:
        return 0.0, 0.0
    if n < 3:
        return (
            sum(float(p[0]) for p in poly) / n,
            sum(float(p[1]) for p in poly) / n,
        )
    acc = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = float(poly[i][0]), float(poly[i][1])
        x1, y1 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        cross = x0 * y1 - x1 * y0
        acc += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(acc) < 1e-18:
        return (
            sum(float(p[0]) for p in poly) / n,
            sum(float(p[1]) for p in poly) / n,
        )
    inv = 1.0 / (3.0 * acc)
    return cx * inv, cy * inv


def collect_barrier_segments(
    data: Optional[dict],
    *,
    region: Optional[dict] = None,
) -> List[Tuple[List[float], List[float]]]:
    """
    Crease + border segments that expansion must not cross.

    Sources:
      - lines / line_features with type in {mountain=0, valley=1, border=2}
      - host panel outline edges (units[panel]) when available
    """
    segs: List[Tuple[List[float], List[float]]] = []
    if not data:
        return segs

    lines = data.get("lines") or []
    feats = data.get("line_features") or []
    for i, ln in enumerate(lines):
        if not ln or len(ln) < 2:
            continue
        t = TYPE_BORDER
        if i < len(feats) and feats[i] is not None:
            try:
                t = int(feats[i].get("type", TYPE_BORDER))
            except (TypeError, ValueError):
                t = TYPE_BORDER
        if t not in BARRIER_TYPES:
            continue
        a, b = _xy(ln[0]), _xy(ln[1])
        if _dist2(a, b) < 1e-18:
            continue
        segs.append((a, b))

    # Host panel outline as hard walls
    units = data.get("units") or []
    panel_idx = None
    if region is not None:
        for key in ("panel", "unit"):
            if region.get(key) is not None:
                try:
                    panel_idx = int(region[key])
                    break
                except (TypeError, ValueError):
                    pass
    if panel_idx is not None and 0 <= panel_idx < len(units):
        unit = units[panel_idx] or []
        nu = len(unit)
        for j in range(nu):
            a, b = _xy(unit[j]), _xy(unit[(j + 1) % nu])
            if _dist2(a, b) < 1e-18:
                continue
            segs.append((a, b))

    return segs


def _ray_hit_segment_scale(
    c: Sequence,
    d: Sequence,
    a: Sequence,
    b: Sequence,
) -> Optional[float]:
    """
    Ray C + s*D (s >= 0) vs segment AB.

    Returns s at intersection, or None if no hit.
    D is the vector from C to the polygon vertex (not necessarily unit).
    """
    # C + s D = A + u (B-A),  0 <= u <= 1, s >= 0
    cx, cy = float(c[0]), float(c[1])
    dx, dy = float(d[0]), float(d[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    ex, ey = bx - ax, by - ay
    # [dx  -ex] [s] = [ax-cx]
    # [dy  -ey] [u]   [ay-cy]
    det = dx * (-ey) - dy * (-ex)
    if abs(det) < 1e-14:
        return None  # parallel
    rx, ry = ax - cx, ay - cy
    s = (rx * (-ey) - ry * (-ex)) / det
    u = (dx * ry - dy * rx) / det
    if s < -1e-12 or u < -1e-9 or u > 1.0 + 1e-9:
        return None
    return float(s)


def max_expand_scale_along_ray(
    c: Sequence,
    v: Sequence,
    barriers: Sequence[Tuple[Sequence, Sequence]],
    *,
    stop_eps: float = BARRIER_STOP_EPS,
) -> float:
    """
    Largest scale s for V' = C + s*(V-C) that does not cross a barrier.

    Original vertex is s=1. Any barrier hit with s >= 1 (including a vertex
    already sitting on a border) caps expansion ΓÇö previously only s>1 was
    capped, so border vertices got s_lim=inf and expanded past the wall.
    """
    cx, cy = float(c[0]), float(c[1])
    vx, vy = float(v[0]), float(v[1])
    dx, dy = vx - cx, vy - cy
    if dx * dx + dy * dy < 1e-24:
        return 1.0

    s_cap = float("inf")
    for a, b in barriers or []:
        s_hit = _ray_hit_segment_scale(c, (dx, dy), a, b)
        if s_hit is None:
            continue
        # Cap at barrier on or beyond the vertex (s >= 1). Also pull back if
        # the barrier lies between C and V (s in (0,1)) ΓÇö vertex already past wall.
        if s_hit > 1e-9:
            s_cap = min(s_cap, s_hit)

    if not math.isfinite(s_cap):
        return float("inf")
    # Stay slightly inside the barrier
    return max(0.0, s_cap * (1.0 - float(stop_eps)) - float(stop_eps) * 0.0)


def _clip_poly_by_edge(
    subject: Sequence,
    a: Sequence,
    b: Sequence,
) -> List[List[float]]:
    """
    SutherlandΓÇôHodgman: clip subject polygon to the half-plane left of directed
    edge AΓåÆB (points with cross(B-A, P-A) >= 0 kept).
    """
    if not subject:
        return []
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    ex, ey = bx - ax, by - ay

    def inside(p: Sequence) -> bool:
        return ex * (float(p[1]) - ay) - ey * (float(p[0]) - ax) >= -1e-9

    def intersect(p: Sequence, q: Sequence) -> List[float]:
        px, py = float(p[0]), float(p[1])
        qx, qy = float(q[0]), float(q[1])
        dx, dy = qx - px, qy - py
        # p + t (q-p) on line a + u (b-a)
        # (p-a) ├ù e + t (d ├ù e) = 0
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-14:
            return _xy(q)
        t = ((ax - px) * ey - (ay - py) * ex) / denom
        return [px + t * dx, py + t * dy]

    out: List[List[float]] = []
    n = len(subject)
    for i in range(n):
        cur = subject[i]
        prev = subject[(i - 1) % n]
        cin, pin = inside(cur), inside(prev)
        if cin:
            if not pin:
                out.append(intersect(prev, cur))
            out.append(_xy(cur))
        elif pin:
            out.append(intersect(prev, cur))
    return out


def clip_polygon_to_convex(
    poly: Sequence,
    clip: Sequence,
) -> List[List[float]]:
    """Clip poly to a convex clip polygon (CCW or CW; half-planes from edges)."""
    if not poly or not clip or len(clip) < 3:
        return [_xy(p) for p in (poly or [])]
    # Ensure clip is CCW for "left of edge" inside test
    clip_pts = [_xy(p) for p in clip]
    area_signed = 0.0
    m = len(clip_pts)
    for i in range(m):
        x0, y0 = clip_pts[i]
        x1, y1 = clip_pts[(i + 1) % m]
        area_signed += x0 * y1 - x1 * y0
    if area_signed < 0:
        clip_pts = list(reversed(clip_pts))

    out = [_xy(p) for p in poly]
    for i in range(len(clip_pts)):
        a = clip_pts[i]
        b = clip_pts[(i + 1) % len(clip_pts)]
        out = _clip_poly_by_edge(out, a, b)
        if len(out) < 3:
            return out
    # Drop near-duplicate verts
    cleaned: List[List[float]] = []
    for p in out:
        if not cleaned or _dist2(cleaned[-1], p) > VERTEX_EPS * VERTEX_EPS:
            cleaned.append(p)
    if len(cleaned) >= 2 and _dist2(cleaned[0], cleaned[-1]) <= VERTEX_EPS * VERTEX_EPS:
        cleaned = cleaned[:-1]
    return cleaned


def _poly_signed_area(poly: Sequence) -> float:
    n = len(poly) if poly is not None else 0
    acc = 0.0
    for i in range(n):
        x0, y0 = float(poly[i][0]), float(poly[i][1])
        x1, y1 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        acc += x0 * y1 - x1 * y0
    return 0.5 * acc


def _ensure_ccw(poly: Sequence) -> List[List[float]]:
    pts = [_xy(p) for p in poly or []]
    if len(pts) >= 3 and _poly_signed_area(pts) < 0:
        pts = list(reversed(pts))
    return pts


def _clamp_point_to_convex(p: Sequence, panel: Sequence) -> List[float]:
    """If p is outside convex panel, project to nearest point on panel boundary."""
    if _point_in_or_on_polygon(p, panel, eps=1e-9):
        return _xy(p)
    px, py = float(p[0]), float(p[1])
    best = _xy(p)
    best_d = float("inf")
    m = len(panel)
    for i in range(m):
        ax, ay = float(panel[i][0]), float(panel[i][1])
        bx, by = float(panel[(i + 1) % m][0]), float(panel[(i + 1) % m][1])
        dx, dy = bx - ax, by - ay
        lab2 = dx * dx + dy * dy
        if lab2 < 1e-24:
            qx, qy = ax, ay
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / lab2))
            qx, qy = ax + t * dx, ay + t * dy
        d = (px - qx) ** 2 + (py - qy) ** 2
        if d < best_d:
            best_d = d
            best = [qx, qy]
    return best


def expand_polygon_to_contain(
    poly: Sequence,
    samples: Sequence,
    *,
    barriers: Optional[Sequence[Tuple[Sequence, Sequence]]] = None,
    panel_poly: Optional[Sequence] = None,
    max_iters: int = 80,
    step: float = 0.35,
) -> Tuple[List[List[float]], int, float, Dict[str, Any]]:
    """
    Push edges **outward** until every sample is inside, without leaving the
    host panel or crossing creases/borders.

    Uses outward edge offsets (not pure radial scale), so a simplified convex
    chain can re-cover interior samples without blasting past borders.
    Vertices are clamped to the panel; rays to barriers further limit travel.
    """
    cut = _ensure_ccw(poly)
    if len(cut) < 3:
        return cut, 0, 1.0, {"barrier_clamped": False, "n_barriers": 0}
    pts = [_xy(p) for p in samples or []]
    if not pts:
        return cut, 0, 1.0, {"barrier_clamped": False, "n_barriers": 0}

    bars = list(barriers or [])
    panel = (
        _ensure_ccw(panel_poly)
        if panel_poly and len(panel_poly) >= 3
        else None
    )

    def all_inside(poly_pts: Sequence) -> bool:
        return all(_point_in_or_on_polygon(p, poly_pts, eps=1e-6) for p in pts)

    def finalize(poly_pts: List[List[float]], it: int, extra: dict):
        out = _ensure_ccw(poly_pts)
        if panel is not None and len(out) >= 3:
            clipped = clip_polygon_to_convex(out, panel)
            if len(clipped) >= 3:
                out = clipped
                extra = dict(extra)
                extra["panel_clipped"] = True
            # clamp any residual outside verts
            out = [_clamp_point_to_convex(p, panel) for p in out]
        return out, it, 1.0, extra

    if all_inside(cut):
        return finalize(cut, 0, {"barrier_clamped": False, "n_barriers": len(bars)})

    # Precompute per-vertex barrier caps from centroid (for ray-limited moves)
    cx, cy = _polygon_centroid(cut)
    c = [cx, cy]

    def barrier_limit_point(origin: Sequence, target: Sequence) -> List[float]:
        """Move from origin toward target, stop before first barrier / panel."""
        ox, oy = float(origin[0]), float(origin[1])
        tx, ty = float(target[0]), float(target[1])
        dx, dy = tx - ox, ty - oy
        lab = math.hypot(dx, dy)
        if lab < 1e-15:
            return _xy(origin)
        # scale s in [0,1] along originΓåÆtarget
        s_cap = 1.0
        for a, b in bars:
            s_hit = _ray_hit_segment_scale(origin, (dx, dy), a, b)
            # hit with s in (0,1] means barrier on the way to target
            if s_hit is not None and 1e-9 < s_hit <= 1.0 + 1e-9:
                s_cap = min(s_cap, s_hit * (1.0 - BARRIER_STOP_EPS))
        s_cap = max(0.0, s_cap)
        q = [ox + dx * s_cap, oy + dy * s_cap]
        if panel is not None:
            q = _clamp_point_to_convex(q, panel)
        return q

    n = len(cut)
    cur = [list(p) for p in cut]
    moved = False
    for it in range(1, max_iters + 1):
        if all_inside(cur):
            return finalize(cur, it - 1, {
                "barrier_clamped": moved,
                "n_barriers": len(bars),
                "edge_push": True,
            })
        outside = [
            p for p in pts
            if not _point_in_or_on_polygon(p, cur, eps=1e-6)
        ]
        # Accumulate outward push per vertex
        push = [[0.0, 0.0] for _ in range(n)]
        for p in outside:
            # nearest edge
            best_i = 0
            best_d = float("inf")
            for i in range(n):
                a, b = cur[i], cur[(i + 1) % n]
                d = _point_seg_dist(p, a, b)
                if d < best_d:
                    best_d = d
                    best_i = i
            a, b = cur[best_i], cur[(best_i + 1) % n]
            ex, ey = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
            el = math.hypot(ex, ey) or 1.0
            # outward normal for CCW poly = right-to-left rotate ΓåÆ (-ey, ex)? 
            # CCW edge AΓåÆB, interior left, outward = (ey, -ex) / el
            nx, ny = ey / el, -ex / el
            # Ensure outward points away from centroid
            mx = 0.5 * (float(a[0]) + float(b[0]))
            my = 0.5 * (float(a[1]) + float(b[1]))
            if (mx - cx) * nx + (my - cy) * ny < 0:
                nx, ny = -nx, -ny
            # Push amount: at least cover this sample's outside distance
            # Signed distance of p from edge line (outward positive if p outside)
            dist = (float(p[0]) - float(a[0])) * nx + (float(p[1]) - float(a[1])) * ny
            # If p is outside polygon, push edge toward p
            amt = max(float(step), abs(dist) + 0.05)
            push[best_i][0] += nx * amt
            push[best_i][1] += ny * amt
            j = (best_i + 1) % n
            push[j][0] += nx * amt
            push[j][1] += ny * amt
            moved = True

        # Average multi-pushes lightly and apply with barrier/panel clamp
        new_cur: List[List[float]] = []
        for i in range(n):
            px, py = push[i]
            pl = math.hypot(px, py)
            if pl < 1e-15:
                new_cur.append(list(cur[i]))
                continue
            # limit step size per iteration
            scale = min(1.0, (float(step) * 3.0) / pl)
            target = [
                float(cur[i][0]) + px * scale,
                float(cur[i][1]) + py * scale,
            ]
            new_cur.append(barrier_limit_point(cur[i], target))
        cur = _ensure_ccw(new_cur)
        if panel is not None:
            cur = [_clamp_point_to_convex(p, panel) for p in cur]

    return finalize(cur, max_iters, {
        "barrier_clamped": True,
        "n_barriers": len(bars),
        "containment_incomplete": not all_inside(cur),
        "edge_push": True,
    })


def drop_near_collinear(
    pts: Sequence,
    *,
    max_dev: float,
) -> List[List[float]]:
    """
    Closed-polygon pass: drop intermediate verts whose deviation from the
    neighbour chord is Γëñ max_dev (handles near-collinear hull runs on convex
    sides that RDP already mostly cleaned).
    """
    raw = [_xy(p) for p in pts or []]
    n = len(raw)
    if n <= 3 or float(max_dev) <= 0.0:
        return raw
    changed = True
    tol = float(max_dev)
    while changed and len(raw) > 3:
        changed = False
        m = len(raw)
        keep = [True] * m
        for i in range(m):
            if sum(keep) <= 3:
                break
            if not keep[i]:
                continue
            # previous / next kept
            j = (i - 1) % m
            while not keep[j]:
                j = (j - 1) % m
            k = (i + 1) % m
            while not keep[k]:
                k = (k + 1) % m
            if j == i or k == i:
                continue
            if _point_seg_dist(raw[i], raw[j], raw[k]) <= tol:
                keep[i] = False
                changed = True
        raw = [raw[i] for i in range(m) if keep[i]]
    return raw


def _all_samples_inside(
    poly: Sequence,
    samples: Sequence,
    *,
    eps: float = 1e-3,
) -> bool:
    if len(poly) < 3:
        return False
    return all(_point_in_or_on_polygon(p, poly, eps=eps) for p in samples or [])


def simplify_containing_polygon(
    pts: Sequence,
    samples: Sequence,
    *,
    tol: float,
    min_verts: int = 3,
) -> List[List[float]]:
    """
    Aggressive straight-edge simplify that **never undershoots** ``samples``.

    1) RDP + collinear drop at ``tol``
    2) Greedy vertex drop (largest chord deviation first among removable)
       while every sample stays inside

    If a candidate would leave any sample outside, that vertex is kept.
    """
    raw = [_xy(p) for p in pts or []]
    if len(raw) <= min_verts or float(tol) <= 0.0:
        return raw
    samp = [_xy(p) for p in samples or []]
    if not samp:
        return drop_near_collinear(
            simplify_closed_polygon(raw, tol=float(tol)),
            max_dev=float(tol),
        )

    # Stage 1: classical simplify if still containing
    cand = simplify_closed_polygon(raw, tol=float(tol))
    cand = drop_near_collinear(cand, max_dev=float(tol))
    if len(cand) >= min_verts and _all_samples_inside(cand, samp):
        raw = cand

    # Stage 2: greedy drop near-collinear / low-deviation verts (forgiving)
    # Prefer dropping the flattest verts first so long straight edges win.
    changed = True
    while changed and len(raw) > min_verts:
        changed = False
        m = len(raw)
        # score each vertex by chord deviation (low = safer to drop)
        scores: List[Tuple[float, int]] = []
        for i in range(m):
            j = (i - 1) % m
            k = (i + 1) % m
            dev = _point_seg_dist(raw[i], raw[j], raw[k])
            scores.append((dev, i))
        scores.sort(key=lambda t: t[0])  # flattest first
        for dev, i in scores:
            if len(raw) <= min_verts:
                break
            # Only drop if within forgiving tol (or very flat)
            if dev > float(tol) * 1.25 and dev > 1e-9:
                continue
            trial = [raw[j] for j in range(len(raw)) if j != i]
            if len(trial) < min_verts:
                continue
            if _poly_signed_area(trial) < 0:
                trial = list(reversed(trial))
            if _all_samples_inside(trial, samp):
                raw = trial
                changed = True
                break
    return raw if len(raw) >= min_verts else [_xy(p) for p in pts or []]


def inflate_polygon_slack(
    poly: Sequence,
    slack: float,
    *,
    barriers: Optional[Sequence[Tuple[Sequence, Sequence]]] = None,
    panel_poly: Optional[Sequence] = None,
    samples: Optional[Sequence] = None,
    max_area_ratio: float = DEFAULT_MAX_OVERSHOOT_RATIO,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Small outward inflate so later straight-edge simplify has room without
    undershooting. Barrier/panel clamped. Rejects if area grows too much.
    """
    raw = _ensure_ccw(poly)
    info: Dict[str, Any] = {
        "slack": float(slack),
        "applied": False,
        "area_in": float(_polygon_area_2d(raw)) if len(raw) >= 3 else 0.0,
    }
    if len(raw) < 3 or float(slack) <= 1e-12:
        return [list(p) for p in raw], info

    expanded, oinfo = offset_polygon_outward(
        raw,
        float(slack),
        barriers=barriers,
        panel_poly=panel_poly,
    )
    if len(expanded) < 3:
        return [list(p) for p in raw], {**info, **oinfo, "reverted": True}

    a0 = info["area_in"]
    a1 = float(_polygon_area_2d(expanded))
    info["area_out"] = a1
    # Containment: inflated poly must still hold samples (should; outward)
    if samples and not _all_samples_inside(expanded, samples, eps=1e-4):
        # Rare numerical case ΓÇö keep original
        return [list(p) for p in raw], {**info, "reverted": True, "reason": "lost_samples"}

    if a0 > 1e-9 and a1 > a0 * float(max_area_ratio):
        # Too much overshoot ΓÇö try half slack once
        half = float(slack) * 0.5
        if half > 1e-12:
            mid, minfo = offset_polygon_outward(
                raw, half, barriers=barriers, panel_poly=panel_poly
            )
            if len(mid) >= 3:
                am = float(_polygon_area_2d(mid))
                if (
                    (samples is None or _all_samples_inside(mid, samples, eps=1e-4))
                    and (a0 <= 1e-9 or am <= a0 * float(max_area_ratio))
                ):
                    info.update({
                        "applied": True,
                        "slack": half,
                        "area_out": am,
                        "halved": True,
                        **{k: minfo.get(k) for k in ("clamped", "panel_clipped")},
                    })
                    return mid, info
        return [list(p) for p in raw], {
            **info,
            "reverted": True,
            "reason": "overshoot_ratio",
            "ratio": a1 / max(a0, 1e-18),
        }

    info["applied"] = True
    info["clamped"] = bool(oinfo.get("clamped"))
    info["panel_clipped"] = bool(oinfo.get("panel_clipped"))
    return expanded, info


def _panel_polygon(data: Optional[dict], region: Optional[dict]) -> Optional[List[List[float]]]:
    if not data or not region:
        return None
    units = data.get("units") or []
    panel_idx = None
    for key in ("panel", "unit"):
        if region.get(key) is not None:
            try:
                panel_idx = int(region[key])
                break
            except (TypeError, ValueError):
                pass
    if panel_idx is None or panel_idx < 0 or panel_idx >= len(units):
        return None
    unit = units[panel_idx] or []
    if len(unit) < 3:
        return None
    return [_xy(p) for p in unit]


def straight_containing_cut(
    c0: Sequence,
    c1: Sequence,
    *,
    tol: float = DEFAULT_CUT_TOL,
    barriers: Optional[Sequence[Tuple[Sequence, Sequence]]] = None,
    panel_poly: Optional[Sequence] = None,
    hull_samples: Optional[Sequence] = None,
    contain_samples: Optional[Sequence] = None,
    round_dp: int = DEFAULT_ROUND_DP,
    straight_slack: float = DEFAULT_STRAIGHT_SLACK,
    max_overshoot_ratio: float = DEFAULT_MAX_OVERSHOOT_RATIO,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Straight-edged cut that contains every dual-curve sample (never undershoot).

    Pipeline:
      1) Convex hull of dual-curve samples (UF winners + all samples)
      2) Small outward ``straight_slack`` (barrier/panel clamped) so simplify
         has room ΓÇö fewer long straight edges without cutting into the shade
      3) Containment-preserving simplify (RDP + greedy flat-vertex drop)
         with forgiving ``tol``; reject any candidate that loses a sample
      4) Cap overshoot via ``max_overshoot_ratio`` vs hull area

    Prefer more straight lines (forgiving tol + slack) while refusing both
    undershoot (sample outside) and large overshoot (area explosion).
    """
    all_samples: List[List[float]] = []
    for p in c0 or []:
        all_samples.append(_round_pt(p, dp=round_dp))
    for p in c1 or []:
        all_samples.append(_round_pt(p, dp=round_dp))

    if hull_samples is not None and len(hull_samples) > 0:
        samples = [_round_pt(p, dp=round_dp) for p in hull_samples]
    else:
        samples = list(all_samples)

    if contain_samples is not None and len(contain_samples) > 0:
        must_contain_all = [_round_pt(p, dp=round_dp) for p in contain_samples]
    else:
        must_contain_all = list(all_samples)

    ribbon = _ribbon_ring(c0, c1)
    ribbon_a = _polygon_area_2d(ribbon) if ribbon else 0.0
    bars = list(barriers or [])
    panel = (
        _round_poly(panel_poly, dp=round_dp)
        if panel_poly and len(panel_poly) >= 3
        else None
    )

    # Containment target: all dual-curve samples. Points outside the panel are
    # projected in so the cut covers the shaded region without needing to leave
    # the panel when the shade only grazes a border.
    if panel is not None:
        must_contain = []
        for p in must_contain_all:
            if _point_in_or_on_polygon(p, panel, eps=1e-3):
                must_contain.append(p)
            else:
                must_contain.append(_clamp_point_to_convex(p, panel))
        # Dedupe projected stack-ups
        must_contain = _dedupe_points(must_contain, eps=10 ** (-int(round_dp)))
    else:
        must_contain = list(must_contain_all)

    def _count_out(poly_pts: Sequence, pts: Sequence, *, eps: float = 1e-3) -> int:
        """Count samples outside poly. Slightly forgiving eps for border grazing."""
        if len(poly_pts) < 3:
            return len(pts)
        return sum(
            1 for p in pts
            if not _point_in_or_on_polygon(p, poly_pts, eps=eps)
        )

    exp_info: Dict[str, Any] = {
        "barrier_clamped": False,
        "n_barriers": len(bars),
        "panel_clipped": False,
    }
    n_expand, scale = 0, 1.0

    # Always contain the (possibly panel-projected) dual-curve samples
    target = list(must_contain) if must_contain else list(must_contain_all)

    # Hull seeds: UF winners + every target sample (never drop a sample)
    hull_pts: List[List[float]] = []
    for p in samples:
        if panel is None or _point_in_or_on_polygon(p, panel, eps=1e-3):
            hull_pts.append(p)
        elif panel is not None:
            hull_pts.append(_clamp_point_to_convex(p, panel))
    for p in target:
        hull_pts.append(p)
    if not hull_pts:
        hull_pts = list(must_contain_all)

    hull = convex_hull(hull_pts)
    n_hull_raw = len(hull)
    hull_a = _polygon_area_2d(hull) if len(hull) >= 3 else 0.0
    cut = [list(p) for p in hull]

    # Degenerate projected hull ΓåÆ full unclipped sample hull
    if hull_a < max(1e-4, 0.05 * max(ribbon_a, 1e-9)) and must_contain_all:
        full_all = convex_hull(must_contain_all)
        fa = _polygon_area_2d(full_all) if len(full_all) >= 3 else 0.0
        if fa > hull_a + 1e-9:
            cut = full_all
            target = list(must_contain_all)
            hull_a = fa
            n_hull_raw = len(cut)
            exp_info["panel_overflow"] = True
            exp_info["fell_back_to_unclipped_hull"] = True

    n_out = _count_out(cut, target)
    if n_out > 0:
        fixed = convex_hull(target)
        if len(fixed) >= 3:
            cut = fixed
            hull_a = _polygon_area_2d(cut)
            n_out = _count_out(cut, target)
            exp_info["fell_back_to_full_hull"] = True

    t = float(tol)
    slack = float(straight_slack)
    # Adaptive slack: small ribbons need less room; dense hulls need more
    if hull_a > 1e-6 and n_hull_raw >= 8:
        slack = max(slack, min(1.25, 0.02 * math.sqrt(hull_a)))
    if hull_a < 5.0:
        slack = min(slack, 0.35)

    # --- Expand-then-simplify: forgiving straight edges, no undershoot ---
    base = [list(p) for p in cut]
    if slack > 1e-12 and len(base) >= 3 and n_out == 0:
        inflated, iinfo = inflate_polygon_slack(
            base,
            slack,
            barriers=bars,
            panel_poly=None if exp_info.get("panel_overflow") else panel,
            samples=target,
            max_area_ratio=float(max_overshoot_ratio),
        )
        exp_info["slack_info"] = iinfo
        if iinfo.get("applied") and _all_samples_inside(inflated, target):
            cut = inflated
            exp_info["straight_slack"] = float(iinfo.get("slack") or slack)
            exp_info["barrier_clamped"] = bool(
                exp_info.get("barrier_clamped") or iinfo.get("clamped")
            )

    n_simplified = len(cut)
    if t > 0.0 and len(cut) > 3 and _all_samples_inside(cut, target):
        # Multi-level tol: try base, then more forgiving, keep fewest verts
        # that still contain samples and respect overshoot cap
        best = [list(p) for p in cut]
        best_n = len(best)
        base_area = max(hull_a, 1e-12)
        for scale_t in (1.0, 1.5, 2.0):
            trial_tol = t * scale_t
            simp = simplify_containing_polygon(cut, target, tol=trial_tol)
            if len(simp) < 3 or not _all_samples_inside(simp, target):
                continue
            sa = _polygon_area_2d(simp)
            if sa > base_area * float(max_overshoot_ratio) * 1.15:
                # Simplified shape grew too much relative to hull ΓÇö skip
                continue
            if len(simp) < best_n:
                best = simp
                best_n = len(simp)
                exp_info["simplified"] = True
                exp_info["simplify_tol_used"] = float(trial_tol)
        cut = best
        n_simplified = len(cut)
        if not exp_info.get("simplified") and len(cut) == len(base):
            exp_info["simplify_rejected_undershoot"] = True

    # Round, then repair containment if needed
    cut = _round_poly(cut, dp=round_dp)
    cut = _dedupe_points(cut, eps=10 ** (-int(round_dp)))
    if len(cut) >= 3 and _poly_signed_area(cut) < 0:
        cut = list(reversed(cut))

    n_out = _count_out(cut, target)
    if n_out > 0 and target:
        repaired = convex_hull(
            _round_poly(list(cut) + list(target), dp=round_dp)
        )
        repaired = _dedupe_points(repaired, eps=10 ** (-int(round_dp)))
        if len(repaired) >= 3:
            # One more forgiving simplify on the repaired hull
            if t > 0.0:
                repaired2 = simplify_containing_polygon(
                    repaired, target, tol=t
                )
                if (
                    len(repaired2) >= 3
                    and _all_samples_inside(repaired2, target)
                ):
                    repaired = repaired2
            cut = repaired
            if _poly_signed_area(cut) < 0:
                cut = list(reversed(cut))
            n_out = _count_out(cut, target)
            exp_info["final_containment_repair"] = True

    exp_info["panel_clipped"] = bool(
        panel is not None and not exp_info.get("panel_overflow")
    )
    cut_a = _polygon_area_2d(cut) if len(cut) >= 3 else 0.0
    n_out = _count_out(cut, target)
    n_out_panel = 0
    if panel is not None:
        n_out_panel = sum(
            1 for p in cut
            if not _point_in_or_on_polygon(p, panel, eps=1e-3)
        )
    n_in_panel = sum(
        1 for p in must_contain_all
        if panel is None or _point_in_or_on_polygon(p, panel, eps=1e-3)
    )
    shape = "convex_dense" if n_hull_raw >= 12 else "concave_or_simple"
    overshoot = (cut_a / hull_a) if hull_a > 1e-12 else 1.0

    info = {
        "n_samples": len(must_contain_all),
        "n_samples_in_panel": int(n_in_panel),
        "n_samples_outside_panel": int(len(must_contain_all) - n_in_panel),
        "n_hull_samples": len(samples),
        "n_hull": int(n_hull_raw),
        "n_cut": len(cut),
        "n_simplified": int(n_simplified),
        "ribbon_area": float(ribbon_a),
        "cut_area": float(cut_a),
        "hull_area": float(hull_a),
        "overshoot_ratio": float(overshoot),
        "n_outside": int(n_out),
        "n_outside_panel": int(n_out_panel),
        "n_expand_iters": int(n_expand),
        "expand_scale": float(scale),
        "cut_tol": float(t),
        "straight_slack": float(exp_info.get("straight_slack") or 0.0),
        "shape": shape,
        "n_barriers": int(exp_info.get("n_barriers") or len(bars)),
        "barrier_clamped": bool(exp_info.get("barrier_clamped")),
        "panel_clipped": bool(exp_info.get("panel_clipped")),
        "containment_incomplete": bool(n_out > 0),
        "merged_outside_samples": bool(exp_info.get("merged_outside_samples")),
        "fell_back_to_full_hull": bool(exp_info.get("fell_back_to_full_hull")),
        "simplified": bool(exp_info.get("simplified")),
        "simplify_tol_used": exp_info.get("simplify_tol_used"),
    }
    return cut, info


# ---------------------------------------------------------------------------
# Union-Find: group messy nodes ΓåÆ outermost representative
# ---------------------------------------------------------------------------

def _segments_properly_intersect(
    a: Sequence, b: Sequence, c: Sequence, d: Sequence,
) -> bool:
    """True if open segment AB properly crosses open segment CD."""
    def orient(p, q, r) -> float:
        return (float(q[0]) - float(p[0])) * (float(r[1]) - float(p[1])) - (
            float(q[1]) - float(p[1])
        ) * (float(r[0]) - float(p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    return False


def can_union(
    i: int,
    j: int,
    pts: Sequence,
    *,
    tol: float,
    barriers: Optional[Sequence[Tuple[Sequence, Sequence]]] = None,
    respect_barriers: bool = False,
    members_i: Optional[Sequence[int]] = None,
    members_j: Optional[Sequence[int]] = None,
) -> bool:
    """
    Custom UF constraints. Union only if **all** pass:

      C1 pairwise |pi - pj| <= tol (seed edge)
      C2 complete-linkage: diameter of merged cluster <= tol
          (prevents single-linkage chains along dense dual curves)
      C3 optional: segment between cluster reps does not cross a barrier
    """
    if i == j:
        return True
    tol2 = float(tol) * float(tol)
    if _dist2(pts[i], pts[j]) > tol2:
        return False

    mi = list(members_i) if members_i is not None else [i]
    mj = list(members_j) if members_j is not None else [j]
    for a in mi:
        for b in mj:
            if _dist2(pts[a], pts[b]) > tol2:
                return False

    if respect_barriers and barriers:
        a, b = pts[i], pts[j]
        for ba, bb in barriers:
            if _segments_properly_intersect(a, b, ba, bb):
                return False
    return True


def group_nodes_outermost(
    pts: Sequence,
    *,
    tol: float = DEFAULT_GROUP_TOL,
    origin: Optional[Sequence] = None,
    barriers: Optional[Sequence[Tuple[Sequence, Sequence]]] = None,
    respect_barriers: bool = False,
    adaptive_tol: bool = True,
    round_dp: int = DEFAULT_ROUND_DP,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Union-Find cluster of near-coincident / messy nodes; one outermost each.

    Clustering is **complete-linkage** (cluster diameter Γëñ tol) so dense dual-
    curve chains do not collapse into one giant component.

    **Outermost** = member maximizing squared distance to the *global* sample
    centroid ``origin`` (default: mean of all input points ΓÇö not the cluster
    mean). Ties: higher x, then higher y.

    Rationale: tip clouds at outer vertices should collapse to the true extreme
    so a later convex hull never pulls inward. Cluster means / first-index
    picks are undershoot-prone.
    """
    raw = [_round_pt(p, dp=round_dp) for p in (pts or [])]
    n = len(raw)
    info: Dict[str, Any] = {
        "n_in": n,
        "n_out": n,
        "n_groups": n,
        "group_tol": float(tol),
        "respect_barriers": bool(respect_barriers),
        "linkage": "complete",
    }
    if n <= 1 or float(tol) <= 0.0:
        return raw, info

    use_tol = float(tol)
    if adaptive_tol and n >= 2:
        xs = [p[0] for p in raw]
        ys = [p[1] for p in raw]
        diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if diag > 1e-9:
            # Cap so tiny ribbons cannot wipe out to one point; never raise tol
            use_tol = min(use_tol, max(0.35, 0.15 * diag))
    info["group_tol_used"] = use_tol

    if origin is None:
        ox = sum(p[0] for p in raw) / n
        oy = sum(p[1] for p in raw) / n
    else:
        ox, oy = float(origin[0]), float(origin[1])
    info["origin"] = [ox, oy]

    parent = list(range(n))
    rank = [0] * n
    members: List[List[int]] = [[i] for i in range(n)]

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union_roots(ri: int, rj: int) -> None:
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            parent[ri] = rj
            members[rj].extend(members[ri])
            members[ri] = []
        elif rank[ri] > rank[rj]:
            parent[rj] = ri
            members[ri].extend(members[rj])
            members[rj] = []
        else:
            parent[rj] = ri
            rank[ri] += 1
            members[ri].extend(members[rj])
            members[rj] = []

    bars = list(barriers or [])
    # Sort candidate pairs by distance (nearest first) for stable complete-linkage
    pairs: List[Tuple[float, int, int]] = []
    tol2 = use_tol * use_tol
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _dist2(raw[i], raw[j])
            if d2 <= tol2:
                pairs.append((d2, i, j))
    pairs.sort(key=lambda t: t[0])

    for _d2, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if can_union(
            i, j, raw,
            tol=use_tol,
            barriers=bars,
            respect_barriers=respect_barriers,
            members_i=members[ri],
            members_j=members[rj],
        ):
            union_roots(ri, rj)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    def outer_key(i: int) -> Tuple[float, float, float]:
        # max distance to global centroid; tie-break higher x, higher y
        dx = raw[i][0] - ox
        dy = raw[i][1] - oy
        return (dx * dx + dy * dy, raw[i][0], raw[i][1])

    winners: List[Tuple[int, List[float]]] = []
    for mems in clusters.values():
        first = min(mems)
        win = max(mems, key=outer_key)
        winners.append((first, list(raw[win])))
    winners.sort(key=lambda t: t[0])
    out = [p for _, p in winners]
    deduped = _dedupe_points(out, eps=10 ** (-int(round_dp)))
    info["n_out"] = len(deduped)
    info["n_groups"] = len(clusters)
    info["n_merged"] = max(0, n - len(deduped))
    return deduped, info


def collapse_nodes_keep_outermost(
    pts: Sequence,
    *,
    tol: float = DEFAULT_GROUP_TOL,
    origin: Optional[Sequence] = None,
    barriers: Optional[Sequence[Tuple[Sequence, Sequence]]] = None,
    respect_barriers: bool = False,
) -> List[List[float]]:
    """Backward-compatible wrapper ΓåÆ :func:`group_nodes_outermost`."""
    out, _ = group_nodes_outermost(
        pts,
        tol=tol,
        origin=origin,
        barriers=barriers,
        respect_barriers=respect_barriers,
    )
    return out


def apply_straight_containing_cut(
    region: dict,
    *,
    tol: float = DEFAULT_CUT_TOL,
    barriers: Optional[Sequence[Tuple[Sequence, Sequence]]] = None,
    data: Optional[dict] = None,
    group_tol: float = DEFAULT_GROUP_TOL,
    corner_merge_tol: Optional[float] = None,
    uf_respect_barriers: bool = DEFAULT_UF_RESPECT_BARRIERS,
    round_dp: int = DEFAULT_ROUND_DP,
    straight_slack: float = DEFAULT_STRAIGHT_SLACK,
    max_overshoot_ratio: float = DEFAULT_MAX_OVERSHOOT_RATIO,
) -> Dict[str, Any]:
    """
    In-place: round ΓåÆ UF outermost samples ΓåÆ straight containing cut.

    Keeps original dual curves (rounded) as c0/c1 for trails / clean_ring.
    Cut is a few long straight edges containing the shade (never undershoot).
    """
    gtol = float(group_tol if corner_merge_tol is None else corner_merge_tol)

    round_region_geometry(region, dp=round_dp)
    c0 = region.get("c0") or []
    c1 = region.get("c1") or []
    if not c0 or not c1 or len(c0) < 2 or len(c1) < 2:
        return {"skipped": True, "reason": "missing dual curves"}

    bars = list(barriers) if barriers is not None else collect_barrier_segments(
        data, region=region
    )
    panel = _panel_polygon(data, region)
    if panel is not None:
        panel = _round_poly(panel, dp=round_dp)

    originals: List[List[float]] = []
    for p in c0:
        originals.append(_round_pt(p, dp=round_dp))
    for p in c1:
        originals.append(_round_pt(p, dp=round_dp))

    # Containment set: project out-of-panel samples onto panel (matches cut)
    if panel is not None:
        contain_check = []
        for p in originals:
            if _point_in_or_on_polygon(p, panel, eps=1e-3):
                contain_check.append(p)
            else:
                contain_check.append(_clamp_point_to_convex(p, panel))
        contain_check = _dedupe_points(contain_check, eps=10 ** (-int(round_dp)))
    else:
        contain_check = list(originals)

    ox = sum(p[0] for p in originals) / len(originals)
    oy = sum(p[1] for p in originals) / len(originals)

    winners, uf_info = group_nodes_outermost(
        originals,
        tol=gtol,
        origin=(ox, oy),
        barriers=bars,
        respect_barriers=bool(uf_respect_barriers),
        round_dp=round_dp,
    )

    cut, info = straight_containing_cut(
        c0,
        c1,
        tol=tol,
        barriers=bars,
        panel_poly=panel,
        hull_samples=winners,
        contain_samples=originals,
        round_dp=round_dp,
        straight_slack=float(straight_slack),
        max_overshoot_ratio=float(max_overshoot_ratio),
    )
    if len(cut) < 3:
        return {
            "skipped": True,
            "reason": "cut < 3 verts",
            **info,
            "uf": uf_info,
        }

    # Corner UF only if containment preserved (prefer fewer nodes)
    n_before = len(cut)
    cut2, cut_uf = group_nodes_outermost(
        cut,
        tol=min(gtol, max(0.5, float(tol))),
        origin=(ox, oy),
        barriers=bars,
        respect_barriers=bool(uf_respect_barriers),
        round_dp=round_dp,
    )
    if len(cut2) >= 3 and (
        not contain_check
        or all(_point_in_or_on_polygon(p, cut2, eps=1e-4) for p in contain_check)
    ):
        # Prefer fewer verts when containment holds
        if len(cut2) <= len(cut):
            cut = cut2
            info = dict(info)
            info["n_cut"] = len(cut)
            info["n_corners_merged"] = max(0, n_before - len(cut))
            info["cut_area"] = float(_polygon_area_2d(cut))
            info["cut_uf"] = cut_uf
        else:
            info = dict(info)
            info["n_corners_merged"] = 0
    else:
        info = dict(info)
        info["n_corners_merged"] = 0

    cut = _round_poly(cut, dp=round_dp)

    def _n_out_check(poly: Sequence, pts: Sequence, *, eps: float = 1e-3) -> int:
        if len(poly) < 3:
            return len(pts)
        return sum(
            1 for p in pts
            if not _point_in_or_on_polygon(p, poly, eps=eps)
        )

    if contain_check and _n_out_check(cut, contain_check) > 0:
        # Force containment: hull of cut Γê¬ samples, then re-simplify only if safe
        fixed = convex_hull(
            _round_poly(list(cut) + list(contain_check), dp=round_dp)
        )
        fixed = _dedupe_points(fixed, eps=10 ** (-int(round_dp)))
        if len(fixed) >= 3 and _poly_signed_area(fixed) < 0:
            fixed = list(reversed(fixed))
        simp = simplify_containing_polygon(fixed, contain_check, tol=float(tol))
        simp = _round_poly(simp, dp=round_dp)
        if len(simp) >= 3 and _n_out_check(simp, contain_check) == 0:
            cut = simp
        else:
            cut = fixed
        info = dict(info)
        info["n_cut"] = len(cut)
        info["cut_area"] = float(_polygon_area_2d(cut))
        info["n_outside"] = _n_out_check(cut, contain_check)
        info["containment_incomplete"] = bool(info["n_outside"] > 0)
        info["post_round_repair"] = True

    # Last-resort: if still short, tiny outward slack then hull with samples
    if contain_check and _n_out_check(cut, contain_check) > 0:
        bumped, _ = inflate_polygon_slack(
            cut,
            max(0.15, float(straight_slack) * 0.5),
            barriers=bars,
            panel_poly=panel,
            samples=None,
            max_area_ratio=float(max_overshoot_ratio) * 1.1,
        )
        fixed = convex_hull(
            _round_poly(list(bumped) + list(contain_check), dp=round_dp)
        )
        if len(fixed) >= 3:
            cut = fixed
            if _poly_signed_area(cut) < 0:
                cut = list(reversed(cut))
            cut = _round_poly(cut, dp=round_dp)
            info = dict(info)
            info["n_cut"] = len(cut)
            info["cut_area"] = float(_polygon_area_2d(cut))
            info["n_outside"] = _n_out_check(cut, contain_check)
            info["containment_incomplete"] = bool(info["n_outside"] > 0)
            info["last_resort_inflate"] = True

    region["coordinates"] = [list(p) for p in cut]
    region["cut_ring"] = [list(p) for p in cut]
    region["area"] = float(info.get("cut_area") or _polygon_area_2d(cut))
    region["ribbon_area"] = float(info.get("ribbon_area") or 0.0)
    region["cut_kind"] = "straight_containing"
    region["cut_tol"] = float(tol)
    region["straight_slack"] = float(info.get("straight_slack") or straight_slack)
    region["group_tol"] = float(gtol)
    region["corner_merge_tol"] = float(gtol)
    region["round_dp"] = int(round_dp)
    region["approx_type"] = "linear"
    region["boundary_order"] = 1
    region["curve_order"] = 1
    region["spline"] = False
    region["curved_boundary"] = False
    if str(region.get("kind") or "").lower() in ("line", "empty", ""):
        if float(region["area"]) > COLLAPSED_AREA_EPS:
            region["kind"] = "sweep"
    return {"skipped": False, **info, "uf": uf_info}


# ---------------------------------------------------------------------------
# Fabrication offset (outward), barrier-clamped
# ---------------------------------------------------------------------------

def _edge_outward_normal(a: Sequence, b: Sequence, *, ccw: bool) -> Tuple[float, float]:
    """Unit outward normal for directed edge AΓåÆB (poly CCW ΓçÆ outward = right)."""
    ex = float(b[0]) - float(a[0])
    ey = float(b[1]) - float(a[1])
    el = math.hypot(ex, ey) or 1.0
    # CCW: left of edge is interior ΓåÆ outward = (ey, -ex) / el
    # CW: flip
    nx, ny = ey / el, -ex / el
    if not ccw:
        nx, ny = -nx, -ny
    return nx, ny


def offset_polygon_outward(
    poly: Sequence,
    distance: float,
    *,
    barriers: Optional[Sequence[Tuple[Sequence, Sequence]]] = None,
    panel_poly: Optional[Sequence] = None,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Constant-ish outward offset of a closed polygon for fabrication allowance.

    Each vertex is moved along the miter of adjacent outward edge normals by
    ``distance``. Movement is then clamped:
      - ray toward the offset target stops before creases / borders
      - result is hard-clipped to the host panel

    Positive distance = grow outward. Zero / negative ΓåÆ no-op copy.
    """
    raw = _ensure_ccw(poly)
    n = len(raw)
    d = float(distance)
    info: Dict[str, Any] = {
        "fabric_offset": d,
        "n_in": n,
        "clamped": False,
        "panel_clipped": False,
    }
    if n < 3 or d <= 1e-12:
        return [list(p) for p in raw], {**info, "skipped": d <= 1e-12, "n_out": n}

    bars = list(barriers or [])
    panel = (
        _ensure_ccw(panel_poly)
        if panel_poly is not None and len(panel_poly) >= 3
        else None
    )
    ccw = _poly_signed_area(raw) >= 0

    # Per-edge outward normals
    edge_n: List[Tuple[float, float]] = []
    for i in range(n):
        a, b = raw[i], raw[(i + 1) % n]
        edge_n.append(_edge_outward_normal(a, b, ccw=ccw))

    def limit_move(origin: Sequence, target: Sequence) -> List[float]:
        ox, oy = float(origin[0]), float(origin[1])
        tx, ty = float(target[0]), float(target[1])
        dx, dy = tx - ox, ty - oy
        if dx * dx + dy * dy < 1e-24:
            return _xy(origin)
        s_cap = 1.0
        for a, b in bars:
            s_hit = _ray_hit_segment_scale(origin, (dx, dy), a, b)
            if s_hit is not None and 1e-9 < s_hit <= 1.0 + 1e-9:
                s_cap = min(s_cap, s_hit * (1.0 - BARRIER_STOP_EPS))
                info["clamped"] = True
        s_cap = max(0.0, s_cap)
        q = [ox + dx * s_cap, oy + dy * s_cap]
        if panel is not None:
            q2 = _clamp_point_to_convex(q, panel)
            if _dist2(q, q2) > 1e-12:
                info["clamped"] = True
            q = q2
        return q

    offset: List[List[float]] = []
    for i in range(n):
        n0 = edge_n[(i - 1) % n]
        n1 = edge_n[i]
        bx, by = n0[0] + n1[0], n0[1] + n1[1]
        bl = math.hypot(bx, by)
        if bl < 1e-12:
            # nearly opposite normals ΓÇö fall back to n1
            bx, by = n1[0], n1[1]
            bl = math.hypot(bx, by) or 1.0
        bx, by = bx / bl, by / bl
        # miter length so offset distance along edge normals Γëê d
        cos_a = max(0.15, min(1.0, n1[0] * bx + n1[1] * by))
        miter = d / cos_a
        target = [raw[i][0] + bx * miter, raw[i][1] + by * miter]
        offset.append(limit_move(raw[i], target))

    if panel is not None and len(offset) >= 3:
        clipped = clip_polygon_to_convex(offset, panel)
        if len(clipped) >= 3:
            offset = clipped
            info["panel_clipped"] = True
        offset = [_clamp_point_to_convex(p, panel) for p in offset]

    # Drop near-duplicates from clamping
    cleaned: List[List[float]] = []
    for p in offset:
        if not cleaned or _dist2(cleaned[-1], p) > VERTEX_EPS * VERTEX_EPS:
            cleaned.append(p)
    if len(cleaned) >= 2 and _dist2(cleaned[0], cleaned[-1]) <= VERTEX_EPS * VERTEX_EPS:
        cleaned = cleaned[:-1]
    if len(cleaned) < 3:
        cleaned = [list(p) for p in raw]
        info["reverted"] = True

    info["n_out"] = len(cleaned)
    info["area_in"] = float(_polygon_area_2d(raw))
    info["area_out"] = float(_polygon_area_2d(cleaned))
    info["skipped"] = False
    return cleaned, info


def apply_fabric_offset(
    region: dict,
    *,
    offset: float,
    barriers: Optional[Sequence[Tuple[Sequence, Sequence]]] = None,
    data: Optional[dict] = None,
    group_tol: float = DEFAULT_GROUP_TOL,
    round_dp: int = DEFAULT_ROUND_DP,
    cut_tol: float = DEFAULT_CUT_TOL,
) -> Dict[str, Any]:
    """
    In-place: offset step-3 cut_ring outward for fabrication (manual offset).

    Guarantees fabric contains the geometric cut and dual-curve samples
    (never undershoot the shade). Keeps long straight edges.

    Stores:
      fabric_ring      ΓÇö offset polygon (straight, containing)
      fabric_offset    ΓÇö requested distance
      cut_ring         ΓÇö geometric cut (unchanged if already set)
      coordinates      ΓÇö set to fabric_ring when offset > 0 (export paint)
    """
    d = float(offset)
    if d <= 1e-12:
        region.pop("fabric_ring", None)
        region.pop("fabric_offset", None)
        return {"skipped": True, "reason": "fabric_offset<=0", "fabric_offset": d}

    # Prefer pure geometric cut from step 3; do not use a prior fabric ring
    poly = region.get("cut_ring")
    if not poly or len(poly) < 3:
        poly = region.get("coordinates")
    if not poly or len(poly) < 3:
        ring = _ribbon_ring(region.get("c0") or [], region.get("c1") or [])
        poly = ring
    if not poly or len(poly) < 3:
        return {"skipped": True, "reason": "no polygon to offset", "fabric_offset": d}

    poly = _round_poly(poly, dp=round_dp)
    bars = list(barriers) if barriers is not None else collect_barrier_segments(
        data, region=region
    )
    panel = _panel_polygon(data, region)
    if panel is not None:
        panel = _round_poly(panel, dp=round_dp)

    # Samples that fabric must cover (cut verts + dual curves, panel-projected)
    must: List[List[float]] = [list(p) for p in poly]
    for p in (region.get("c0") or []) + (region.get("c1") or []):
        q = _round_pt(p, dp=round_dp)
        if panel is not None and not _point_in_or_on_polygon(q, panel, eps=1e-3):
            q = _clamp_point_to_convex(q, panel)
        must.append(q)
    must = _dedupe_points(must, eps=10 ** (-int(round_dp)))

    # Offset without barrier clamp first so we don't pull inward past the shade;
    # then panel-clip only if it still contains must.
    fab, info = offset_polygon_outward(
        poly, d, barriers=None, panel_poly=None
    )
    if len(fab) < 3:
        fab, info = offset_polygon_outward(
            poly, d, barriers=bars, panel_poly=panel
        )
    if len(fab) < 3:
        return {"skipped": True, "reason": "offset collapsed", **info}

    # Keep geometric cut separate from fabrication outline
    if not region.get("cut_ring"):
        region["cut_ring"] = [list(p) for p in poly]

    # Never undershoot: re-hull with anything still outside
    def _n_out(poly_pts: Sequence, pts: Sequence) -> int:
        if len(poly_pts) < 3:
            return len(pts)
        return sum(
            1 for p in pts
            if not _point_in_or_on_polygon(p, poly_pts, eps=1e-3)
        )

    if _n_out(fab, must) > 0:
        fab = convex_hull(list(fab) + list(must))
        info = dict(info)
        info["containment_repair"] = True

    # Optional panel clip only when containment preserved
    if panel is not None and len(fab) >= 3:
        clipped = clip_polygon_to_convex(fab, panel)
        if len(clipped) >= 3 and _n_out(clipped, must) == 0:
            fab = clipped
            info = dict(info)
            info["panel_clipped"] = True
        else:
            # Soft clamp verts only
            clamped = [_clamp_point_to_convex(p, panel) for p in fab]
            if _n_out(clamped, must) == 0 and len(clamped) >= 3:
                fab = clamped
                info = dict(info)
                info["panel_clamped_verts"] = True

    # Straighten: containment-preserving simplify (no heavy UF corner wipe)
    n_before = len(fab)
    fab_s = simplify_containing_polygon(
        fab, must, tol=max(float(cut_tol), float(group_tol) * 0.5)
    )
    if len(fab_s) >= 3 and _n_out(fab_s, must) == 0:
        fab = fab_s
        info = dict(info)
        info["n_corners_merged"] = max(0, n_before - len(fab))
        info["simplified"] = True
    else:
        info = dict(info)
        info["n_corners_merged"] = 0

    # Final containment guarantee
    if _n_out(fab, must) > 0:
        fab = convex_hull(list(fab) + list(must))
        info = dict(info)
        info["final_hull_repair"] = True

    fab = _round_poly(fab, dp=round_dp)
    fab = _dedupe_points(fab, eps=10 ** (-int(round_dp)))
    if len(fab) >= 3 and _poly_signed_area(fab) < 0:
        fab = list(reversed(fab))

    # Drop near-duplicates only (tiny tol) ΓÇö keep real corners
    if len(fab) > 3:
        fab_d = _dedupe_points(fab, eps=max(1e-6, 10 ** (-int(round_dp))))
        if len(fab_d) >= 3 and _n_out(fab_d, must) == 0:
            fab = fab_d

    info["area_out"] = float(_polygon_area_2d(fab))
    info["n_out"] = len(fab)
    info["n_outside_samples"] = _n_out(fab, must)
    info["containment_incomplete"] = bool(info["n_outside_samples"] > 0)

    region["fabric_ring"] = [list(p) for p in fab]
    region["fabric_offset"] = float(d)
    region["coordinates"] = [list(p) for p in fab]
    region["area"] = float(info.get("area_out") or _polygon_area_2d(fab))
    return {"skipped": False, "panel": region.get("panel"), **info}


# ---------------------------------------------------------------------------
# Closed-polygon metadata for viz
# ---------------------------------------------------------------------------

def attach_closed_polygon_meta(region: dict) -> None:
    """
    Attach:
      clean_ring  ΓÇö all dual-curve sample nodes (for blue/yellow node viz)
      clean_edges ΓÇö straight edges of the containing cut (hull) when present,
                    else dual-curve ring edges
    """
    c0 = region.get("c0") or []
    c1 = region.get("c1") or []
    if not c0 or not c1:
        region.pop("clean_ring", None)
        region.pop("clean_edges", None)
        return

    # All sample nodes for visualization
    n = min(len(c0), len(c1))
    ring: List[List[float]] = [_xy(p) for p in c0[:n]]
    ring += [_xy(p) for p in reversed(c1[:n])]
    region["clean_ring"] = ring

    def _len(a: Sequence, b: Sequence) -> float:
        return math.sqrt(_dist2(a, b))

    # Prefer fabrication outline when offset applied, else geometric cut
    cut = (
        region.get("fabric_ring")
        or region.get("cut_ring")
        or region.get("coordinates")
    )
    edges: List[Dict[str, Any]] = []
    if cut and len(cut) >= 3:
        m = len(cut)
        role = "fabric" if region.get("fabric_ring") else "cut"
        for i in range(m):
            p0, p1 = cut[i], cut[(i + 1) % m]
            edges.append({
                "role": role,
                "p0": list(p0),
                "p1": list(p1),
                "length": _len(p0, p1),
                "grouped": False,
                "linear": True,
            })
    elif n >= 2:
        # Fallback: dual-curve ring edges
        full = ring
        rn = len(full)
        for i in range(n - 1):
            p0, p1 = full[i], full[i + 1]
            edges.append({
                "role": "c0",
                "p0": list(p0),
                "p1": list(p1),
                "length": _len(p0, p1),
                "grouped": False,
            })
        if rn >= 2 * n:
            p0, p1 = full[n - 1], full[n]
            edges.append({
                "role": "last",
                "p0": list(p0),
                "p1": list(p1),
                "length": _len(p0, p1),
                "grouped": _len(p0, p1) <= 1e-9,
            })
            for k in range(n - 1):
                p0, p1 = full[n + k], full[n + k + 1]
                edges.append({
                    "role": "c1",
                    "p0": list(p0),
                    "p1": list(p1),
                    "length": _len(p0, p1),
                    "grouped": False,
                })
            p0, p1 = full[2 * n - 1], full[0]
            edges.append({
                "role": "first",
                "p0": list(p0),
                "p1": list(p1),
                "length": _len(p0, p1),
                "grouped": _len(p0, p1) <= 1e-9,
            })

    region["clean_edges"] = edges


def _refresh_collision_stats(data: dict, regions: Sequence[dict]) -> None:
    stats = data.get("collision_stats")
    if not isinstance(stats, dict):
        stats = {}
        data["collision_stats"] = stats
    n_sweep = sum(1 for r in regions if str(r.get("kind") or "").lower() == "sweep")
    n_ghost = sum(
        1 for r in regions if str(r.get("shell_kind") or "").lower() == "ghost"
    )
    n_support = sum(
        1 for r in regions if str(r.get("shell_kind") or "").lower() == "support"
    )
    n_side = sum(
        1 for r in regions if str(r.get("shell_kind") or "").lower() == "side"
    )
    n_phys = sum(
        1 for r in regions
        if str(r.get("shell_kind") or "").lower()
        not in ("ghost", "side", "support")
    )
    n_merged = sum(1 for r in regions if r.get("layer_stack_merged"))
    stats["schema"] = stats.get("schema") or "dual_curve_v1"
    stats["n_shaded_regions"] = len(regions)
    stats["n_shaded_areas"] = len(regions)
    stats["n_sweep_paints"] = n_sweep
    stats["n_closed_polygons"] = n_sweep
    stats["n_physical_regions"] = n_phys
    stats["n_ghost_regions"] = n_ghost
    stats["n_support_regions"] = n_support
    stats["n_side_regions"] = n_side
    stats["n_layer_stack_merged"] = n_merged
    stats["shaded_regions_key"] = "shaded_regions"


# ---------------------------------------------------------------------------
# Per-panel layer-stack merge
#   main    = physical + in-span ghost  → largest-layer solid prism
#   support = support pads + out-of-span ghost → largest-layer solid prism
#             (Z expanded to abut physical stock)
# Both: clean each layer, then ONE prism from the single largest cleaned ring
# extruded min→max layer_h (not a multi-layer 2D union).
# ---------------------------------------------------------------------------

def _shell_kind_of_region(region: Optional[dict]) -> str:
    if not region:
        return "physical"
    raw = region.get("shell_kind")
    if raw is None:
        return "physical"
    s = str(raw).strip().lower()
    if s in ("ghost", "g", "intermediate", "collision_only"):
        return "ghost"
    if s in ("support", "sup", "pad", "filler"):
        return "support"
    if s in ("side", "s", "vertical", "side_ribbon"):
        return "side"
    return "physical"


def _panel_phys_spans_from_data(
    data: Optional[dict],
) -> Dict[int, Tuple[float, float]]:
    """
    Per design-panel physical shell Z span from crease heights only.

    Matches ``collision.py`` / ``collect_support_panel_slabs`` membership.
    Used to expand support-collision prisms so they abut the physical stock.
    """
    if not data:
        return {}
    try:
        from panel_trimming.visualize.visualize_3d import (  # noqa: WPS433
            panel_thickness_offsets,
        )
        offs = panel_thickness_offsets(
            list(data.get("units") or []), data, shaded_regions=None
        )
    except Exception:
        offs = []
    out: Dict[int, Tuple[float, float]] = {}
    for pi, hs in enumerate(offs or []):
        if not hs:
            continue
        try:
            vals = [float(h) for h in hs]
        except (TypeError, ValueError):
            continue
        if not vals:
            continue
        out[int(pi)] = (float(min(vals)), float(max(vals)))
    return out


def _merge_stream_of(
    region: Optional[dict],
    *,
    phys_span: Optional[Tuple[float, float]] = None,
) -> str:
    """
    Which merge bucket a cleaned collision shade belongs to.

    - support pads → support
    - side → left alone
    - physical → main
    - ghost inside physical shell span → main
    - ghost strictly outside physical shell span → support
      (those samples live in the support-stock Z band; they must form the
      support-collision prism, not a main prism that never hits original stock)
    """
    sk = _shell_kind_of_region(region)
    if sk == "support":
        return "support"
    if sk == "side":
        return "side"
    if sk == "ghost" and phys_span is not None:
        h = _region_layer_h(region) if region else None
        if h is not None:
            p_lo, p_hi = float(phys_span[0]), float(phys_span[1])
            if float(h) < p_lo - 1e-9 or float(h) > p_hi + 1e-9:
                return "support"
    return "main"


# When sample max/min is within this of a physical shell face, snap the prism
# end onto that face so stock − collision does not leave a hairline slab.
DEFAULT_Z_FACE_SNAP_MM = 1.0
# Outward seal of the largest-layer prism footprint (mm) after merge so the
# cut reaches creases (kills ~0.05–0.2 mm thin walls along mountain/valley).
DEFAULT_PRISM_CREASE_SEAL_MM = 0.35


def _expand_support_z_to_phys(
    h_lo: float,
    h_hi: float,
    phys_span: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Expand support-collision Z so the prism fills the support stock band.

    Support stock tightly abuts physical: below → […, phys_lo], above →
    [phys_hi, …]. A lone support sample at one height becomes a real solid
    prism, same idea as the main ghost continuous stack.
    """
    if phys_span is None:
        return float(h_lo), float(h_hi)
    p_lo, p_hi = float(phys_span[0]), float(phys_span[1])
    lo, hi = float(h_lo), float(h_hi)
    if hi < lo:
        lo, hi = hi, lo
    # Entirely at or below the physical bottom face → extend up to phys_lo
    if hi <= p_lo + 1e-9:
        return lo, p_lo
    # Entirely at or above the physical top face → extend down to phys_hi
    if lo >= p_hi - 1e-9:
        return p_hi, hi
    return lo, hi


def _snap_stream_z_to_phys_faces(
    h_lo: float,
    h_hi: float,
    phys_span: Optional[Tuple[float, float]],
    *,
    face_snap_mm: float = DEFAULT_Z_FACE_SNAP_MM,
) -> Tuple[float, float]:
    """
    Snap prism Z ends onto physical shell faces when they nearly reach them.

    Kills super-thin residual walls at the bottom/top of the stock when the
    last ghost sample stops ~0.2 mm short of the shell height (e.g. −3.2 vs −3).
    """
    if phys_span is None:
        return float(h_lo), float(h_hi)
    p_lo, p_hi = float(phys_span[0]), float(phys_span[1])
    lo, hi = float(h_lo), float(h_hi)
    if hi < lo:
        lo, hi = hi, lo
    tol = max(float(face_snap_mm), 0.0)
    # Only when the prism already overlaps the physical band
    if hi < p_lo - 1e-12 or lo > p_hi + 1e-12:
        return lo, hi
    if lo <= p_lo + tol:
        lo = min(lo, p_lo)
    if hi >= p_hi - tol:
        hi = max(hi, p_hi)
    return lo, hi


def _region_ring_for_merge(region: dict, *, prefer_fabric: bool) -> List[List[float]]:
    """Cleaned cut/fabric ring for one layer (export priority)."""
    if prefer_fabric:
        keys = ("fabric_ring", "cut_ring", "coordinates", "clean_ring")
    else:
        keys = ("cut_ring", "fabric_ring", "coordinates", "clean_ring")
    for key in keys:
        raw = region.get(key)
        if raw is not None and len(raw) >= 3:
            return [_xy(p) for p in raw]
    return _region_export_ring(region)


def _panel_outline_xy(data: Optional[dict], panel_idx: Optional[int]) -> List[List[float]]:
    """Design-panel outline ring for a unit index."""
    if data is None or panel_idx is None:
        return []
    units = list(data.get("units") or [])
    try:
        pi = int(panel_idx)
    except (TypeError, ValueError):
        return []
    if pi < 0 or pi >= len(units):
        return []
    unit = units[pi]
    if not unit or len(unit) < 3:
        return []
    ring = [_xy(p) for p in unit]
    # Drop duplicate close
    if len(ring) >= 2:
        a, b = ring[0], ring[-1]
        if abs(float(a[0]) - float(b[0])) < 1e-12 and abs(float(a[1]) - float(b[1])) < 1e-12:
            ring = ring[:-1]
    return ring if len(ring) >= 3 else []


def _seal_merged_prism_to_creases(
    region: dict,
    data: Optional[dict],
    *,
    amount: float = DEFAULT_PRISM_CREASE_SEAL_MM,
    round_dp: int = DEFAULT_ROUND_DP,
    snap_tol: float = DEFAULT_SNAP_TOL,
) -> bool:
    """
    Slightly inflate the merged prism footprint and re-snap onto creases.

    Closes hairline gaps (~0.05–0.2 mm) between the largest-layer ring and
    mountain/valley edges so stock − collision leaves no super-thin wall.
    Clamped to the host panel outline.
    """
    d = float(amount)
    if d <= 1e-12:
        return False
    prefer_fabric = bool(region.get("fabric_ring"))
    ring = _region_ring_for_merge(region, prefer_fabric=prefer_fabric)
    if len(ring) < 3:
        return False

    panel_xy = _panel_outline_xy(data, region.get("panel"))
    sealed: Optional[List[List[float]]] = None
    try:
        from shapely.geometry import Polygon as _ShPoly
        from shapely.validation import make_valid as _make_valid
    except Exception:
        sealed = None
    else:
        try:
            g = _ShPoly(ring)
            if g.is_empty:
                return False
            if not g.is_valid:
                try:
                    g = g.buffer(0)
                except Exception:
                    g = _make_valid(g)
            g = g.buffer(d, join_style=2, mitre_limit=5.0)
            if panel_xy and len(panel_xy) >= 3:
                try:
                    panel = _ShPoly(panel_xy)
                    if not panel.is_valid:
                        panel = panel.buffer(0)
                    inter = g.intersection(panel)
                    if inter is not None and not inter.is_empty:
                        g = inter
                except Exception:
                    pass
            # Largest exterior part
            best = None
            best_a = -1.0
            gt = getattr(g, "geom_type", "")
            cands = []
            if gt == "Polygon":
                cands = [g]
            elif gt == "MultiPolygon":
                cands = list(g.geoms)
            else:
                cands = list(getattr(g, "geoms", []) or [])
            for p in cands:
                if getattr(p, "geom_type", "") != "Polygon":
                    continue
                a = float(getattr(p, "area", 0.0) or 0.0)
                if a > best_a:
                    best_a = a
                    best = p
            if best is None:
                return False
            coords = list(best.exterior.coords)
            sealed = _round_poly(
                coords[:-1] if len(coords) > 1 else coords, dp=round_dp
            )
            sealed = _dedupe_points(sealed, eps=10 ** (-int(round_dp)))
            if len(sealed) < 3:
                return False
            if _polygon_area_2d(sealed) < 0:
                sealed = list(reversed(sealed))
        except Exception:
            return False

    if not sealed:
        return False

    # Write sealed ring back
    ring_out = [list(p) for p in sealed]
    if prefer_fabric or region.get("fabric_ring"):
        region["fabric_ring"] = ring_out
        region["cut_ring"] = ring_out
        region["coordinates"] = ring_out
    else:
        region["cut_ring"] = ring_out
        region["coordinates"] = ring_out
    region["prism_crease_seal_mm"] = float(d)
    try:
        region["area"] = float(abs(_polygon_area_2d(sealed)))
        region["ribbon_area"] = float(region["area"])
    except Exception:
        pass

    # Pull sealed edges fully onto creases
    if float(snap_tol) > 0 and data is not None:
        snap_region_to_creases(
            region, data, snap_tol=float(snap_tol), rings_only=True
        )
    return True


def _pick_largest_layer(
    group: Sequence[dict],
    *,
    stream: str,
) -> Optional[dict]:
    """
    Single layer that defines the prism footprint.

    Prefer the largest-area **ghost** layer when any ghosts are in the group
    (worst-case intermediate cross-section). Else largest physical / support.
    Same rule for main and support streams.
    """
    if not group:
        return None

    def _area_of(r: dict) -> float:
        try:
            return float(r.get("area") or region_area(r) or 0.0)
        except Exception:
            return 0.0

    ghosts = [r for r in group if _shell_kind_of_region(r) == "ghost"]
    if ghosts:
        return max(ghosts, key=_area_of)
    return max(group, key=_area_of)


def _region_export_ring(region: dict) -> List[List[float]]:
    """Prefer fabrication / cut ring used by cleaned exporters & viz."""
    for key in ("fabric_ring", "cut_ring", "coordinates", "clean_ring"):
        raw = region.get(key)
        if raw is not None and len(raw) >= 3:
            return [_xy(p) for p in raw]
    c0 = region.get("c0") or []
    c1 = region.get("c1") or []
    if c0 and c1:
        n = min(len(c0), len(c1))
        if n >= 2:
            ring = [_xy(p) for p in c0[:n]]
            ring += [_xy(p) for p in reversed(c1[:n])]
            if len(ring) >= 3:
                return ring
    return []


def _region_layer_h(region: dict) -> Optional[float]:
    lh = region.get("layer_h")
    if lh is None:
        return None
    try:
        return float(lh)
    except (TypeError, ValueError):
        return None


def _union_rings_2d(
    rings: Sequence[Sequence],
    *,
    round_dp: int = DEFAULT_ROUND_DP,
) -> List[List[List[float]]]:
    """
    2D-union of closed rings ΓåÆ list of exterior rings (MultiPolygon parts).

    Uses shapely when available; otherwise returns deduped input rings.
    """
    cleaned: List[List[List[float]]] = []
    for ring in rings or []:
        pts = _dedupe_points([_xy(p) for p in ring], eps=10 ** (-int(round_dp)))
        if len(pts) < 3:
            continue
        if abs(_polygon_area_2d(pts)) < 1e-18:
            continue
        if _polygon_area_2d(pts) < 0:
            pts = list(reversed(pts))
        cleaned.append(pts)
    if not cleaned:
        return []

    try:
        from shapely.geometry import Polygon as _ShPoly
        from shapely.ops import unary_union as _uunion
        from shapely.validation import make_valid as _make_valid
    except Exception:
        return cleaned

    polys = []
    for pts in cleaned:
        try:
            g = _ShPoly(pts)
        except Exception:
            continue
        if g.is_empty:
            continue
        if not g.is_valid:
            try:
                g = g.buffer(0)
            except Exception:
                try:
                    g = _make_valid(g)
                except Exception:
                    continue
        if g is None or g.is_empty:
            continue
        if g.geom_type == "Polygon" and g.area > 1e-12:
            polys.append(g)
        elif g.geom_type == "MultiPolygon":
            for p in g.geoms:
                if p.geom_type == "Polygon" and p.area > 1e-12:
                    polys.append(p)
        else:
            try:
                for p in getattr(g, "geoms", []):
                    if p.geom_type == "Polygon" and p.area > 1e-12:
                        polys.append(p)
            except Exception:
                continue
    if not polys:
        return cleaned
    try:
        merged = _uunion(polys)
    except Exception:
        merged = polys[0]
        for p in polys[1:]:
            try:
                merged = merged.union(p)
            except Exception:
                continue
    if merged is None or getattr(merged, "is_empty", True):
        return cleaned
    if not getattr(merged, "is_valid", True):
        try:
            merged = _make_valid(merged)
        except Exception:
            try:
                merged = merged.buffer(0)
            except Exception:
                return cleaned

    out_rings: List[List[List[float]]] = []

    def _ext(poly) -> Optional[List[List[float]]]:
        try:
            coords = list(poly.exterior.coords)
        except Exception:
            return None
        pts = _round_poly(coords[:-1] if len(coords) > 1 else coords, dp=round_dp)
        pts = _dedupe_points(pts, eps=10 ** (-int(round_dp)))
        if len(pts) < 3:
            return None
        if abs(_polygon_area_2d(pts)) < 1e-18:
            return None
        if _polygon_area_2d(pts) < 0:
            pts = list(reversed(pts))
        return pts

    gt = getattr(merged, "geom_type", "")
    if gt == "Polygon":
        r = _ext(merged)
        if r:
            out_rings.append(r)
    elif gt == "MultiPolygon":
        for p in merged.geoms:
            if p.geom_type == "Polygon" and p.area > 1e-12:
                r = _ext(p)
                if r:
                    out_rings.append(r)
    else:
        for p in getattr(merged, "geoms", []) or []:
            if p.geom_type == "Polygon" and getattr(p, "area", 0) > 1e-12:
                r = _ext(p)
                if r:
                    out_rings.append(r)
    return out_rings if out_rings else cleaned


def _merge_one_panel_stream(
    pid: int,
    group: List[dict],
    *,
    stream: str,
    round_dp: int,
    phys_span: Optional[Tuple[float, float]] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Merge one panel stream into a single vertical solid prism.

    Same rules for main and support:
      1) Z span = min→max layer_h of every cleaned layer in the stream
         (support expands to abut the physical stock face)
      2) XY footprint = the **single largest cleaned layer ring only**
         (main prefers largest ghost; not a multi-layer 2D union)
      3) Extrude that ring as one solid prism over the full Z span
    """
    empty_info: Dict[str, Any] = {
        "panel": pid,
        "stream": stream,
        "n_sources": len(group),
        "n_parts": 0,
        "union_failed": True,
    }
    if not group:
        return [], empty_info

    def _area_of(r: dict) -> float:
        try:
            return float(r.get("area") or region_area(r) or 0.0)
        except Exception:
            return 0.0

    heights: List[float] = []
    for r in group:
        h = _region_layer_h(r)
        if h is not None:
            heights.append(h)
    if not heights:
        return [dict(r) for r in group], {
            **empty_info,
            "n_parts": len(group),
            "reason": "no_layer_h",
        }

    h_lo = float(min(heights))
    h_hi = float(max(heights))
    if stream == "support":
        h_lo, h_hi = _expand_support_z_to_phys(h_lo, h_hi, phys_span)
    else:
        # Main: snap ends onto phys faces when ghosts stop just short
        # (kills hairline slabs at the stock bottom/top).
        h_lo, h_hi = _snap_stream_z_to_phys_faces(h_lo, h_hi, phys_span)
    h_mid = 0.5 * (h_lo + h_hi)
    span = float(h_hi - h_lo)

    has_fabric = any(
        (r.get("fabric_ring") and len(r.get("fabric_ring") or []) >= 3)
        for r in group
    )

    # Prism basis = one largest cleaned layer (not union of every height)
    best = _pick_largest_layer(group, stream=stream)
    if best is None:
        best = max(group, key=_area_of)
    best_ring = _region_ring_for_merge(best, prefer_fabric=has_fabric)
    # Validate / normalize the single ring (may split MultiPolygon parts)
    prism_rings = _union_rings_2d([best_ring], round_dp=round_dp) if best_ring else []
    if not prism_rings and best_ring and len(best_ring) >= 3:
        prism_rings = [[_xy(p) for p in best_ring]]

    n_ghost = sum(1 for r in group if _shell_kind_of_region(r) == "ghost")
    n_phys = sum(1 for r in group if _shell_kind_of_region(r) == "physical")
    n_support = sum(1 for r in group if _shell_kind_of_region(r) == "support")
    largest_area = _area_of(best)
    best_h = _region_layer_h(best)

    if not prism_rings:
        stamped: List[dict] = []
        for r in group:
            rr = dict(r)
            rr["h_lo"] = h_lo
            rr["h_hi"] = h_hi
            rr["layer_h"] = h_mid if span > 1e-12 else (heights[0] if heights else 0.0)
            rr["stock_span_mm"] = span
            rr["layer_stack_merged"] = len(group) > 1 or span > 1e-12
            rr["merge_stream"] = stream
            rr["largest_layer_area"] = float(largest_area)
            if best_h is not None:
                rr["largest_layer_h"] = float(best_h)
            if stream == "support":
                rr["shell_kind"] = "support"
            stamped.append(rr)
        return stamped, {
            "panel": pid,
            "stream": stream,
            "n_sources": len(group),
            "n_parts": len(group),
            "h_lo": h_lo,
            "h_hi": h_hi,
            "largest_layer_area": float(largest_area),
            "largest_layer_h": best_h,
            "union_failed": True,
            "reason": "no_largest_ring",
        }

    if stream == "support":
        shell_kind = "support"
        template = best
    else:
        phys = [r for r in group if _shell_kind_of_region(r) == "physical"]
        # Geometry from largest ghost; keep physical as metadata host when present
        template = dict(best)
        shell_kind = "physical" if phys else "ghost"

    out: List[dict] = []
    for part_i, ring in enumerate(prism_rings):
        m = dict(template)
        for k in (
            "c0",
            "c1",
            "linear_c0",
            "linear_c1",
            "clean_ring",
            "clean_edges",
            "layer_idx",
        ):
            m.pop(k, None)
        m["panel"] = pid
        m["shell_kind"] = shell_kind
        m["layer_h"] = h_mid if span > 1e-12 else h_lo
        m["h_lo"] = h_lo
        m["h_hi"] = h_hi
        m["stock_span_mm"] = span
        m["layer_stack_merged"] = True
        m["merge_stream"] = stream
        m["prism_from_largest_layer"] = True
        m["n_merged_sources"] = len(group)
        m["n_merged_ghost"] = n_ghost
        m["n_merged_physical"] = n_phys
        m["n_merged_support"] = n_support
        m["largest_layer_area"] = float(largest_area)
        if best_h is not None:
            m["largest_layer_h"] = float(best_h)
        m["merged_layer_heights"] = sorted(set(round(h, 9) for h in heights))
        m["merge_part"] = part_i
        m["n_merge_parts"] = len(prism_rings)
        m["kind"] = "final_trim" if (
            m.get("fabric_ring") or m.get("cut_ring") or m.get("cut_kind")
        ) else (m.get("kind") or "sweep")
        area = abs(_polygon_area_2d(ring))
        m["area"] = float(area)
        m["ribbon_area"] = float(area)
        ring_out = [list(p) for p in ring]
        if has_fabric or best.get("fabric_ring"):
            m["fabric_ring"] = ring_out
            m["coordinates"] = ring_out
            m["cut_ring"] = ring_out
        else:
            m["cut_ring"] = ring_out
            m["coordinates"] = ring_out
            m.pop("fabric_ring", None)
        out.append(m)

    return out, {
        "panel": pid,
        "stream": stream,
        "n_sources": len(group),
        "n_parts": len(prism_rings),
        "h_lo": h_lo,
        "h_hi": h_hi,
        "n_ghost": n_ghost,
        "n_physical": n_phys,
        "n_support": n_support,
        "largest_layer_area": float(largest_area),
        "largest_layer_h": best_h,
        "prism_from_largest_layer": True,
        "union_failed": False,
    }


def merge_panel_layer_stacks(
    regions: Sequence[dict],
    *,
    enabled: bool = DEFAULT_MERGE_LAYER_STACK,
    round_dp: int = DEFAULT_ROUND_DP,
    phys_span_by_panel: Optional[Dict[int, Tuple[float, float]]] = None,
    data: Optional[dict] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Per panel: merge collision shades into continuous Z prisms.

    Two independent streams (identical geometry rules):
      main    — physical + ghost → shell_kind physical|ghost
      support — support pads     → shell_kind support
                (Z expanded to abut physical stock)

    Footprint for each stream is the **single largest cleaned layer ring**
    extruded min→max layer_h (a true prism — not a multi-layer 2D union).

    Side ribbons are left unchanged.
    """
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "n_in": len(regions or []),
        "n_out": len(regions or []),
        "n_panels_merged": 0,
        "n_sources_merged": 0,
        "n_main_streams": 0,
        "n_support_streams": 0,
        "n_support_sources": 0,
        "panels": [],
    }
    if not enabled:
        return list(regions or []), info
    if not regions:
        return [], info

    spans = phys_span_by_panel
    if spans is None:
        spans = _panel_phys_spans_from_data(data)
    spans = spans or {}

    sides: List[dict] = []
    by_panel: Dict[int, Dict[str, List[dict]]] = {}
    orphan: List[dict] = []
    n_ghost_to_support = 0

    for r in regions:
        panel = r.get("panel")
        try:
            pid = int(panel) if panel is not None else None
        except (TypeError, ValueError):
            pid = None
        phys_span = spans.get(pid) if pid is not None else None
        stream = _merge_stream_of(r, phys_span=phys_span)
        if stream == "support" and _shell_kind_of_region(r) == "ghost":
            n_ghost_to_support += 1
        if stream == "side":
            sides.append(dict(r))
            continue
        if pid is None:
            orphan.append(dict(r))
            continue
        by_panel.setdefault(pid, {}).setdefault(stream, []).append(dict(r))

    info["n_ghosts_routed_to_support"] = int(n_ghost_to_support)

    out: List[dict] = []
    out.extend(sides)
    out.extend(orphan)

    for pid in sorted(by_panel.keys()):
        streams = by_panel[pid]
        phys_span = spans.get(pid)
        for stream in ("main", "support"):
            group = streams.get(stream) or []
            if not group:
                continue
            merged, pinfo = _merge_one_panel_stream(
                pid,
                group,
                stream=stream,
                round_dp=round_dp,
                phys_span=phys_span,
            )
            out.extend(merged)
            info["n_panels_merged"] += 1
            info["n_sources_merged"] += int(pinfo.get("n_sources") or 0)
            if stream == "support":
                info["n_support_streams"] += 1
                info["n_support_sources"] += int(pinfo.get("n_sources") or 0)
            else:
                info["n_main_streams"] += 1
            info["panels"].append(pinfo)

    info["n_out"] = len(out)
    return out, info


def clean_data(
    data: dict,
    *,
    min_area: float = DEFAULT_MIN_AREA,
    drop_line_kind: bool = True,
    attach_polygon_meta: bool = True,
    straight_cut: bool = True,
    cut_tol: float = DEFAULT_CUT_TOL,
    fabric_offset: float = DEFAULT_FABRIC_OFFSET,
    group_tol: float = DEFAULT_GROUP_TOL,
    corner_merge_tol: Optional[float] = None,
    uf_respect_barriers: bool = DEFAULT_UF_RESPECT_BARRIERS,
    round_dp: int = DEFAULT_ROUND_DP,
    straight_slack: float = DEFAULT_STRAIGHT_SLACK,
    max_overshoot_ratio: float = DEFAULT_MAX_OVERSHOOT_RATIO,
    snap_tol: float = DEFAULT_SNAP_TOL,
    merge_layer_stack: bool = DEFAULT_MERGE_LAYER_STACK,
    source_path: Optional[str] = None,
) -> Tuple[dict, Dict[str, Any]]:
    """
    Return (cleaned_data, report). Deep-copies input structure.

    Pipeline (applies to physical, ghost, **and support** collisions; side left alone):
      1) filter collapsed / too-small
      2) snap dual-curve coords onto nearby creases/borders (``snap_tol``)
      3) round coords to ``round_dp`` d.p.
      4) Union-Find group messy nodes → outermost each
      5) straight containing cut (hull → slack → simplify, never undershoot)
      6) re-snap cut_ring onto creases (edge attraction; kills thin walls)
      7) optional fabric_offset from cut polygon, then re-snap fabric_ring
      8) attach clean_ring + clean_edges
      9) per panel: merge main (phys+ghost) and support streams each into
         one solid prism from the largest cleaned layer ring over
         min→max layer_h (support Z expands to abut physical stock);
         re-snap merged rings onto creases
    """
    gtol = float(group_tol if corner_merge_tol is None else corner_merge_tol)
    rdp = int(round_dp)
    stol = float(snap_tol)
    out = copy.deepcopy(data)
    regions = list(out.get("shaded_regions") or [])
    n_support_in = sum(
        1 for r in regions if _shell_kind_of_region(r) == "support"
    )
    kept, dropped_log = filter_shaded_regions(
        regions, min_area=min_area, drop_line_kind=drop_line_kind
    )
    n_support_kept = sum(
        1 for r in kept if _shell_kind_of_region(r) == "support"
    )

    # Step 1: snap samples onto nearby mountain / valley / border segments
    snap_log: List[Dict[str, Any]] = []
    n_snap_total = 0
    if stol > 0:
        for r in kept:
            sinfo = snap_region_to_creases(r, out, snap_tol=stol)
            n_snap_total += int(sinfo.get("n_snapped") or 0)
            snap_log.append({
                "panel": r.get("panel"),
                "layer_h": r.get("layer_h"),
                **sinfo,
            })

    # Step 2: quantize dual curves on every kept region
    for r in kept:
        round_region_geometry(r, dp=rdp)

    cut_log: List[Dict[str, Any]] = []
    if straight_cut:
        for r in kept:
            bars = collect_barrier_segments(out, region=r)
            info = apply_straight_containing_cut(
                r,
                tol=float(cut_tol),
                barriers=bars,
                data=out,
                group_tol=gtol,
                uf_respect_barriers=bool(uf_respect_barriers),
                round_dp=rdp,
                straight_slack=float(straight_slack),
                max_overshoot_ratio=float(max_overshoot_ratio),
            )
            cut_log.append({
                "panel": r.get("panel"),
                "layer_h": r.get("layer_h"),
                **info,
            })
            # Pull cut_ring back onto creases (simplify/UF can leave ~1ΓÇô3 mm
            # inset gaps ΓåÆ thin residual wall after original ΓêÆ collision).
            if not info.get("skipped") and stol > 0:
                sinfo = snap_region_to_creases(
                    r, out, snap_tol=stol, rings_only=True
                )
                cut_log[-1]["post_cut_snap"] = sinfo
    else:
        # Still round; no cut polygon
        for r in kept:
            round_region_geometry(r, dp=rdp)

    fabric_log: List[Dict[str, Any]] = []
    fo = float(fabric_offset)
    if fo > 1e-12:
        for r in kept:
            bars = collect_barrier_segments(out, region=r)
            finfo = apply_fabric_offset(
                r,
                offset=fo,
                barriers=bars,
                data=out,
                group_tol=gtol,
                round_dp=rdp,
                cut_tol=float(cut_tol),
            )
            fabric_log.append({
                "panel": r.get("panel"),
                "layer_h": r.get("layer_h"),
                **finfo,
            })
            if not finfo.get("skipped") and stol > 0:
                sinfo = snap_region_to_creases(
                    r, out, snap_tol=stol, rings_only=True
                )
                fabric_log[-1]["post_fabric_snap"] = sinfo

    if attach_polygon_meta:
        for r in kept:
            attach_closed_polygon_meta(r)
            # Final quantize of viz helpers
            round_region_geometry(r, dp=rdp)

    # Step 8: merge main (phys+ghost) and support streams into largest-layer
    # solid prisms. Keep a pre-merge copy so viz can still show every layer.
    layers_for_viz: List[dict] = []
    if bool(merge_layer_stack) and kept:
        for r in kept:
            layer = copy.deepcopy(r)
            layer["pre_merge_layer"] = True
            layer.pop("layer_stack_merged", None)
            layers_for_viz.append(layer)

    kept, merge_info = merge_panel_layer_stacks(
        kept,
        enabled=bool(merge_layer_stack),
        round_dp=rdp,
        data=out,
    )
    # After prism merge: seal footprint to creases (small outward inflate +
    # re-snap) so stock − collision does not leave super-thin walls.
    n_merge_snap = 0
    n_sealed = 0
    if kept:
        for r in kept:
            if not (
                r.get("layer_stack_merged")
                or r.get("prism_from_largest_layer")
                or r.get("cut_ring")
                or r.get("fabric_ring")
            ):
                continue
            if stol > 0:
                sinfo = snap_region_to_creases(
                    r, out, snap_tol=stol, rings_only=True
                )
                n_merge_snap += int(sinfo.get("n_ring_moved") or 0)
            if _seal_merged_prism_to_creases(
                r,
                out,
                amount=DEFAULT_PRISM_CREASE_SEAL_MM,
                round_dp=rdp,
                snap_tol=max(stol, DEFAULT_PRISM_CREASE_SEAL_MM * 2.0),
            ):
                n_sealed += 1
            if rdp > 0:
                round_region_geometry(r, dp=rdp)
        merge_info = dict(merge_info)
        merge_info["n_post_merge_snap_moved"] = int(n_merge_snap)
        merge_info["n_prism_crease_sealed"] = int(n_sealed)
    n_support_out = sum(
        1 for r in kept if _shell_kind_of_region(r) == "support"
    )
    if attach_polygon_meta and merge_info.get("enabled"):
        for r in kept:
            if r.get("layer_stack_merged"):
                # Re-attach edges for merged rings (dual curves dropped)
                cut = (
                    r.get("fabric_ring")
                    or r.get("cut_ring")
                    or r.get("coordinates")
                )
                if cut and len(cut) >= 3:
                    edges: List[Dict[str, Any]] = []
                    m = len(cut)
                    role = "fabric" if r.get("fabric_ring") else "cut"
                    for i in range(m):
                        p0, p1 = cut[i], cut[(i + 1) % m]
                        dx = float(p0[0]) - float(p1[0])
                        dy = float(p0[1]) - float(p1[1])
                        edges.append({
                            "role": role,
                            "p0": list(p0),
                            "p1": list(p1),
                            "length": math.sqrt(dx * dx + dy * dy),
                            "grouped": False,
                            "linear": True,
                        })
                    r["clean_edges"] = edges
                    r["clean_ring"] = [list(p) for p in cut]
                round_region_geometry(r, dp=rdp)

    # shaded_regions        = merged continuous solids (main + support streams)
    # shaded_regions_layers = pre-merge phys/ghost/support at each layer_h (viz)
    out["shaded_regions"] = kept
    if (
        layers_for_viz
        and merge_info.get("enabled")
        and int(merge_info.get("n_panels_merged") or 0) > 0
    ):
        out["shaded_regions_layers"] = layers_for_viz
        meta_tmp = out.get("export_meta")
        if not isinstance(meta_tmp, dict):
            meta_tmp = {}
            out["export_meta"] = meta_tmp
        meta_tmp["clean_n_layer_viz"] = len(layers_for_viz)
    else:
        out.pop("shaded_regions_layers", None)
    _refresh_collision_stats(out, kept)

    report = {
        "source": source_path,
        "min_area": float(min_area),
        "drop_line_kind": bool(drop_line_kind),
        "straight_cut": bool(straight_cut),
        "cut_tol": float(cut_tol),
        "straight_slack": float(straight_slack),
        "max_overshoot_ratio": float(max_overshoot_ratio),
        "fabric_offset": fo,
        "group_tol": gtol,
        "round_dp": rdp,
        "snap_tol": stol,
        "merge_layer_stack": bool(merge_layer_stack),
        "n_snapped_total": int(n_snap_total),
        "uf_respect_barriers": bool(uf_respect_barriers),
        "n_in": len(regions),
        "n_out": len(kept),
        "n_dropped": len(dropped_log),
        "n_support_in": int(n_support_in),
        "n_support_kept": int(n_support_kept),
        "n_support_out": int(n_support_out),
        "n_support_streams": int(merge_info.get("n_support_streams") or 0),
        "dropped": dropped_log,
        "snap_log": snap_log,
        "cut_log": cut_log,
        "fabric_log": fabric_log,
        "merge_log": merge_info,
    }
    meta = out.get("export_meta")
    if not isinstance(meta, dict):
        meta = {}
        out["export_meta"] = meta
    meta["cleaned"] = True
    meta["clean_min_area"] = float(min_area)
    meta["clean_n_in"] = len(regions)
    meta["clean_n_out"] = len(kept)
    meta["clean_n_dropped"] = len(dropped_log)
    meta["clean_n_support_in"] = int(n_support_in)
    meta["clean_n_support_kept"] = int(n_support_kept)
    meta["clean_n_support_out"] = int(n_support_out)
    meta["clean_n_support_streams"] = int(merge_info.get("n_support_streams") or 0)
    meta["clean_straight_cut"] = bool(straight_cut)
    meta["clean_cut_tol"] = float(cut_tol)
    meta["clean_straight_slack"] = float(straight_slack)
    meta["clean_max_overshoot_ratio"] = float(max_overshoot_ratio)
    meta["clean_cut_kind"] = "straight_containing" if straight_cut else None
    meta["clean_fabric_offset"] = fo
    meta["clean_group_tol"] = gtol
    meta["clean_snap_tol"] = stol
    meta["clean_n_snapped"] = int(n_snap_total)
    meta["clean_corner_merge_tol"] = gtol  # legacy
    meta["clean_round_dp"] = rdp
    meta["clean_uf_respect_barriers"] = bool(uf_respect_barriers)
    meta["clean_merge_layer_stack"] = bool(merge_layer_stack)
    meta["clean_n_panels_layer_merged"] = int(merge_info.get("n_panels_merged") or 0)
    meta["clean_n_sources_layer_merged"] = int(merge_info.get("n_sources_merged") or 0)
    meta["clean_n_main_streams"] = int(merge_info.get("n_main_streams") or 0)
    if out.get("shaded_regions_layers") is not None:
        meta["clean_n_layer_viz"] = len(out["shaded_regions_layers"])
    else:
        meta.pop("clean_n_layer_viz", None)
    meta["approx_type"] = "linear" if straight_cut else meta.get("approx_type")
    meta["boundary_order"] = 1 if straight_cut else meta.get("boundary_order")
    meta["curved_boundary"] = False if straight_cut else meta.get("curved_boundary")
    for k in (
        "clean_vertex_tol",
        "clean_edge_tol",
        "clean_n_grouped",
        "clean_n_snapped",
        "clean_n_removed_near_outer",
        "clean_linearize",
        "clean_linear_tol",
        "curve_order",
        "spline",
        "high_order",
    ):
        meta.pop(k, None)
    if source_path:
        meta["clean_source"] = os.path.basename(source_path)
    return out, report


def clean_file(
    input_path: str,
    output_path: str,
    *,
    min_area: float = DEFAULT_MIN_AREA,
    drop_line_kind: bool = True,
    attach_polygon_meta: bool = True,
    straight_cut: bool = True,
    cut_tol: float = DEFAULT_CUT_TOL,
    fabric_offset: float = DEFAULT_FABRIC_OFFSET,
    group_tol: float = DEFAULT_GROUP_TOL,
    corner_merge_tol: Optional[float] = None,
    uf_respect_barriers: bool = DEFAULT_UF_RESPECT_BARRIERS,
    round_dp: int = DEFAULT_ROUND_DP,
    straight_slack: float = DEFAULT_STRAIGHT_SLACK,
    max_overshoot_ratio: float = DEFAULT_MAX_OVERSHOOT_RATIO,
    snap_tol: float = DEFAULT_SNAP_TOL,
    merge_layer_stack: bool = DEFAULT_MERGE_LAYER_STACK,
) -> Dict[str, Any]:
    data = _load_json(input_path)
    cleaned, report = clean_data(
        data,
        min_area=min_area,
        drop_line_kind=drop_line_kind,
        attach_polygon_meta=attach_polygon_meta,
        straight_cut=straight_cut,
        cut_tol=cut_tol,
        fabric_offset=fabric_offset,
        group_tol=group_tol,
        corner_merge_tol=corner_merge_tol,
        uf_respect_barriers=uf_respect_barriers,
        round_dp=round_dp,
        straight_slack=straight_slack,
        max_overshoot_ratio=max_overshoot_ratio,
        snap_tol=snap_tol,
        merge_layer_stack=merge_layer_stack,
        source_path=input_path,
    )
    _write_json(output_path, cleaned)
    report["output"] = output_path
    return report


def _default_output_path(input_path: str, output_dir: str) -> str:
    return os.path.join(output_dir, _cleaned_basename(input_path))


def _visualize_module():
    """Load panel_trimming/visualize/visualize.py (not necessarily a package)."""
    viz_py = os.path.join(_PANEL_TRIM_DIR, "visualize", "visualize.py")
    if not os.path.isfile(viz_py):
        raise FileNotFoundError(f"visualize.py not found: {viz_py}")
    spec = importlib.util.spec_from_file_location(
        "panel_trimming_visualize_clean_hook", viz_py
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load visualize from {viz_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_visualize_cleaned(
    cleaned_path: str,
    *,
    quiet: bool = False,
) -> int:
    """
    Render the cleaned JSON via panel_trimming/visualize/visualize.py.

    Writes PNG under panel_trimming/visualize/output/ (same as standalone viz).
    """
    path = os.path.abspath(cleaned_path)
    if not os.path.isfile(path):
        print(f"[clean] viz skip ΓÇö missing {path}", file=sys.stderr)
        return 1
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        viz = _visualize_module()
    except Exception as exc:
        print(f"[clean] viz import failed: {exc}", file=sys.stderr)
        return 1
    if not quiet:
        print(f"[clean] visualize --name {stem}")
    try:
        return int(viz.main(["--name", stem]) or 0)
    except Exception as exc:
        print(f"[clean] viz FAILED {stem}: {exc}", file=sys.stderr)
        return 1


def _resolve_output_dir(path: Optional[str] = None) -> str:
    """
    Resolve cleaned-JSON output directory.

    Default: panel_trimming/trimmedData (same folder as trimmed exports).
    Absolute paths used as-is.
    Relative paths resolved from panel_trimming/, then project root.
    """
    if path is None or str(path).strip() == "":
        return DEFAULT_OUTPUT_DIR
    p = str(path).strip()
    if os.path.isabs(p):
        return p
    norm = p.replace("\\", "/").rstrip("/")
    if norm in ("trimmedData", "trimmed", "cleanedData", "cleaned", "output", "./output"):
        return TRIMMED_DIR
    under_panel = os.path.join(_PANEL_TRIM_DIR, p)
    under_root = os.path.join(_PROJECT_ROOT, p)
    if norm.startswith("panel_trimming/"):
        return os.path.join(_PROJECT_ROOT, p)
    if os.path.isdir(under_panel):
        return under_panel
    return under_root


def _is_cleaned_json(path: str) -> bool:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    return stem.endswith("-cleaned") or stem.endswith("_cleaned")


def _job_settings_from_cfg(job: dict, defaults: dict) -> dict:
    """Merge per-job YAML with top-level defaults (CLI applied later)."""
    out = dict(defaults)
    for k in (
        "min_area",
        "straight_cut",
        "cut_tol",
        "straight_slack",
        "max_overshoot_ratio",
        "fabric_offset",
        "group_tol",
        "corner_merge_tol",
        "uf_respect_barriers",
        "round_dp",
        "snap_tol",
        "merge_layer_stack",
        "attach_polygon_meta",
        "drop_line_kind",
        "output",
        "output_dir",
    ):
        if k in job and job[k] is not None:
            out[k] = job[k]
    # Legacy alias: corner_merge_tol ΓåÆ group_tol when group_tol omitted
    if "group_tol" not in job and job.get("corner_merge_tol") is not None:
        out["group_tol"] = job["corner_merge_tol"]
    if "output_dir" in job and job["output_dir"] is not None:
        out["output_dir"] = _resolve_output_dir(job["output_dir"])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Drop collapsed crease-like / too-small collision shaded_regions, "
            "build straight containing cuts, optional fabrication offset "
            "(barrier-clamped), write panel_trimming/trimmedData/*-cleaned.json."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Input JSON path (optional if --name / config jobs)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "YAML config (default: panel_trimming/clean/config.yml, "
            "else config.example.yml)"
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Stem or file under panel_trimming/trimmedData/ (overrides config jobs)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=None,
        help=(
            "Also drop filled ribbons with area < this "
            f"(default from config or {DEFAULT_MIN_AREA:g})."
        ),
    )
    parser.add_argument(
        "--keep-line-kind",
        action="store_true",
        help="Do not auto-drop kind=line (still subject to --min-area)",
    )
    parser.add_argument(
        "--no-polygon-meta",
        action="store_true",
        help="Do not attach clean_ring / clean_edges for viz",
    )
    parser.add_argument(
        "--no-straight-cut",
        action="store_true",
        help=(
            "Do not replace paint polygon with a straight containing cut "
            "(default: hull + simplify + expand-to-contain)"
        ),
    )
    parser.add_argument(
        "--cut-tol",
        type=float,
        default=None,
        help=(
            "Forgiving max edge deviation for straight-edge simplify "
            f"(larger ΓåÆ fewer longer lines; default {DEFAULT_CUT_TOL:g})"
        ),
    )
    parser.add_argument(
        "--straight-slack",
        type=float,
        default=None,
        help=(
            "Outward slack before simplify so straight cuts don't undershoot "
            f"(default {DEFAULT_STRAIGHT_SLACK:g}; 0 = off)"
        ),
    )
    parser.add_argument(
        "--max-overshoot-ratio",
        type=float,
        default=None,
        help=(
            "Max cut_area / hull_area allowed when expanding for straight edges "
            f"(default {DEFAULT_MAX_OVERSHOOT_RATIO:g})"
        ),
    )
    parser.add_argument(
        "--fabric-offset",
        type=float,
        default=None,
        help=(
            "Outward fabrication offset (design units / mm) of the cut; "
            "clamped so it cannot cross crease/border/panel "
            f"(default from config or {DEFAULT_FABRIC_OFFSET:g} = off)"
        ),
    )
    parser.add_argument(
        "--snap-tol",
        type=float,
        default=None,
        help=(
            "Snap dual-curve samples onto creases/borders within this distance "
            f"(mm; default {DEFAULT_SNAP_TOL:g}; 0 = off)"
        ),
    )
    parser.add_argument(
        "--group-tol",
        type=float,
        default=None,
        help=(
            "Union-Find cluster distance for messy nodes "
            f"(default from config or {DEFAULT_GROUP_TOL:g})"
        ),
    )
    parser.add_argument(
        "--corner-merge-tol",
        type=float,
        default=None,
        help="Deprecated alias for --group-tol",
    )
    parser.add_argument(
        "--round-dp",
        type=int,
        default=None,
        help=f"Round geometry to this many decimal places (default {DEFAULT_ROUND_DP})",
    )
    parser.add_argument(
        "--uf-respect-barriers",
        action="store_true",
        help="Do not UF-merge pairs whose segment crosses a crease/border",
    )
    parser.add_argument(
        "--merge-layer-stack",
        action="store_true",
        default=None,
        help=(
            "Per panel: merge phys+ghost (main) and support streams each into "
            "continuous minΓåÆmax height (default on)"
        ),
    )
    parser.add_argument(
        "--no-merge-layer-stack",
        action="store_true",
        help=(
            "Keep intermediate phys/ghost/support layers separate "
            "(no Z-stack merge)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=f"Output directory (default: config or {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Exact output file path (overrides --output-dir; single job)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clean every non-cleaned JSON in panel_trimming/trimmedData/",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less logging",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Do not auto-run panel_trimming/visualize/visualize.py after clean",
    )
    args = parser.parse_args(argv)

    cfg_path = _resolve_config_path(args.config)
    cfg: dict = {}
    if cfg_path:
        try:
            cfg = _load_yaml(cfg_path)
        except Exception as exc:
            print(f"[clean] bad config {cfg_path}: {exc}", file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"[clean] config {cfg_path}")

    _cfg_group = cfg.get("group_tol", cfg.get("corner_merge_tol", DEFAULT_GROUP_TOL))
    cfg_defaults = {
        "min_area": float(cfg.get("min_area", DEFAULT_MIN_AREA)),
        "straight_cut": bool(cfg.get("straight_cut", True)),
        "cut_tol": float(cfg.get("cut_tol", DEFAULT_CUT_TOL)),
        "straight_slack": float(cfg.get("straight_slack", DEFAULT_STRAIGHT_SLACK)),
        "max_overshoot_ratio": float(
            cfg.get("max_overshoot_ratio", DEFAULT_MAX_OVERSHOOT_RATIO)
        ),
        "fabric_offset": float(cfg.get("fabric_offset", DEFAULT_FABRIC_OFFSET)),
        "snap_tol": float(cfg.get("snap_tol", DEFAULT_SNAP_TOL)),
        "group_tol": float(_cfg_group),
        "corner_merge_tol": float(
            cfg.get("corner_merge_tol", cfg.get("group_tol", DEFAULT_GROUP_TOL))
        ),
        "uf_respect_barriers": bool(
            cfg.get("uf_respect_barriers", DEFAULT_UF_RESPECT_BARRIERS)
        ),
        "round_dp": int(cfg.get("round_dp", DEFAULT_ROUND_DP)),
        "merge_layer_stack": bool(
            cfg.get("merge_layer_stack", DEFAULT_MERGE_LAYER_STACK)
        ),
        "attach_polygon_meta": bool(cfg.get("attach_polygon_meta", True)),
        "drop_line_kind": bool(cfg.get("drop_line_kind", True)),
        "output_dir": _resolve_output_dir(cfg.get("output_dir")),
        "output": None,
    }
    # CLI overrides
    if args.min_area is not None:
        cfg_defaults["min_area"] = float(args.min_area)
    if args.cut_tol is not None:
        cfg_defaults["cut_tol"] = float(args.cut_tol)
    if args.straight_slack is not None:
        cfg_defaults["straight_slack"] = float(args.straight_slack)
    if args.max_overshoot_ratio is not None:
        cfg_defaults["max_overshoot_ratio"] = float(args.max_overshoot_ratio)
    if args.fabric_offset is not None:
        cfg_defaults["fabric_offset"] = float(args.fabric_offset)
    if args.snap_tol is not None:
        cfg_defaults["snap_tol"] = float(args.snap_tol)
    if args.group_tol is not None:
        cfg_defaults["group_tol"] = float(args.group_tol)
    elif args.corner_merge_tol is not None:
        cfg_defaults["group_tol"] = float(args.corner_merge_tol)
    if args.round_dp is not None:
        cfg_defaults["round_dp"] = int(args.round_dp)
    if args.uf_respect_barriers:
        cfg_defaults["uf_respect_barriers"] = True
    if args.no_straight_cut:
        cfg_defaults["straight_cut"] = False
    if args.no_polygon_meta:
        cfg_defaults["attach_polygon_meta"] = False
    if args.keep_line_kind:
        cfg_defaults["drop_line_kind"] = False
    if args.no_merge_layer_stack:
        cfg_defaults["merge_layer_stack"] = False
    elif args.merge_layer_stack:
        cfg_defaults["merge_layer_stack"] = True
    if args.output_dir is not None:
        cfg_defaults["output_dir"] = _resolve_output_dir(args.output_dir)
    if args.output is not None:
        cfg_defaults["output"] = args.output

    # Build job list: each entry is (input_path, settings)
    work: List[Tuple[str, dict]] = []

    if args.all:
        if not os.path.isdir(TRIMMED_DIR):
            print(f"[clean] trimmedData not found: {TRIMMED_DIR}", file=sys.stderr)
            return 1
        for fn in sorted(os.listdir(TRIMMED_DIR)):
            if not fn.lower().endswith(".json"):
                continue
            path = os.path.join(TRIMMED_DIR, fn)
            if _is_cleaned_json(path):
                continue
            work.append((path, dict(cfg_defaults)))
    elif args.path:
        work.append((_resolve_input(args.path), dict(cfg_defaults)))
    elif args.name:
        work.append((_resolve_input(args.name), dict(cfg_defaults)))
    else:
        jobs_cfg = list(cfg.get("jobs") or cfg.get("simulations") or [])
        if jobs_cfg:
            for job in jobs_cfg:
                if not isinstance(job, dict) or not job.get("name"):
                    continue
                settings = _job_settings_from_cfg(job, cfg_defaults)
                # re-apply global CLI flags that must win over job
                if args.min_area is not None:
                    settings["min_area"] = float(args.min_area)
                if args.cut_tol is not None:
                    settings["cut_tol"] = float(args.cut_tol)
                if args.straight_slack is not None:
                    settings["straight_slack"] = float(args.straight_slack)
                if args.max_overshoot_ratio is not None:
                    settings["max_overshoot_ratio"] = float(args.max_overshoot_ratio)
                if args.fabric_offset is not None:
                    settings["fabric_offset"] = float(args.fabric_offset)
                if args.snap_tol is not None:
                    settings["snap_tol"] = float(args.snap_tol)
                if args.group_tol is not None:
                    settings["group_tol"] = float(args.group_tol)
                elif args.corner_merge_tol is not None:
                    settings["group_tol"] = float(args.corner_merge_tol)
                if args.round_dp is not None:
                    settings["round_dp"] = int(args.round_dp)
                if args.uf_respect_barriers:
                    settings["uf_respect_barriers"] = True
                if args.no_straight_cut:
                    settings["straight_cut"] = False
                if args.no_polygon_meta:
                    settings["attach_polygon_meta"] = False
                if args.keep_line_kind:
                    settings["drop_line_kind"] = False
                if args.no_merge_layer_stack:
                    settings["merge_layer_stack"] = False
                elif args.merge_layer_stack:
                    settings["merge_layer_stack"] = True
                try:
                    in_path = _resolve_input(str(job["name"]))
                except FileNotFoundError as exc:
                    print(f"[clean] SKIP {job['name']}: {exc}", file=sys.stderr)
                    continue
                work.append((in_path, settings))
        else:
            default_name = "miura-thick-trimmed"
            try:
                work.append((_resolve_input(default_name), dict(cfg_defaults)))
            except FileNotFoundError:
                if os.path.isdir(TRIMMED_DIR):
                    for fn in sorted(os.listdir(TRIMMED_DIR)):
                        if not fn.lower().endswith(".json"):
                            continue
                        path = os.path.join(TRIMMED_DIR, fn)
                        if _is_cleaned_json(path):
                            continue
                        work.append((path, dict(cfg_defaults)))
                if not work:
                    print(
                        "[clean] no input; pass --name, path, --all, or config jobs",
                        file=sys.stderr,
                    )
                    return 1

    if not work:
        print("[clean] nothing to do", file=sys.stderr)
        return 1

    rc = 0
    for in_path, settings in work:
        out_dir = _resolve_output_dir(settings.get("output_dir"))
        os.makedirs(out_dir, exist_ok=True)
        if settings.get("output") and len(work) == 1:
            out_path = str(settings["output"])
        else:
            out_path = _default_output_path(in_path, out_dir)
        try:
            _gt = settings.get("group_tol", settings.get(
                "corner_merge_tol", DEFAULT_GROUP_TOL
            ))
            report = clean_file(
                in_path,
                out_path,
                min_area=float(settings.get("min_area", DEFAULT_MIN_AREA)),
                drop_line_kind=bool(settings.get("drop_line_kind", True)),
                attach_polygon_meta=bool(settings.get("attach_polygon_meta", True)),
                straight_cut=bool(settings.get("straight_cut", True)),
                cut_tol=float(settings.get("cut_tol", DEFAULT_CUT_TOL)),
                fabric_offset=float(settings.get("fabric_offset", DEFAULT_FABRIC_OFFSET)),
                group_tol=float(_gt),
                uf_respect_barriers=bool(
                    settings.get("uf_respect_barriers", DEFAULT_UF_RESPECT_BARRIERS)
                ),
                round_dp=int(settings.get("round_dp", DEFAULT_ROUND_DP)),
                straight_slack=float(
                    settings.get("straight_slack", DEFAULT_STRAIGHT_SLACK)
                ),
                max_overshoot_ratio=float(
                    settings.get("max_overshoot_ratio", DEFAULT_MAX_OVERSHOOT_RATIO)
                ),
                snap_tol=float(settings.get("snap_tol", DEFAULT_SNAP_TOL)),
                merge_layer_stack=bool(
                    settings.get("merge_layer_stack", DEFAULT_MERGE_LAYER_STACK)
                ),
            )
        except Exception as exc:
            print(f"[clean] FAILED {in_path}: {exc}", file=sys.stderr)
            rc = 1
            continue

        if not args.quiet:
            fo = float(settings.get("fabric_offset", DEFAULT_FABRIC_OFFSET))
            gt = float(settings.get("group_tol", DEFAULT_GROUP_TOL))
            st = float(settings.get("snap_tol", DEFAULT_SNAP_TOL))
            mls = bool(settings.get("merge_layer_stack", DEFAULT_MERGE_LAYER_STACK))
            mlog = report.get("merge_log") or {}
            print(
                f"[clean] {os.path.basename(in_path)}: "
                f"{report['n_in']} ΓåÆ {report['n_out']} "
                f"(dropped {report['n_dropped']}, "
                f"support {report.get('n_support_in', 0)}ΓåÆ"
                f"{report.get('n_support_out', 0)}, "
                f"min_area={float(settings.get('min_area', 0)):g}, "
                f"straight_cut={bool(settings.get('straight_cut', True))}, "
                f"cut_tol={float(settings.get('cut_tol', DEFAULT_CUT_TOL)):g}, "
                f"slack={float(settings.get('straight_slack', DEFAULT_STRAIGHT_SLACK)):g}, "
                f"group_tol={gt:g}, "
                f"round_dp={int(settings.get('round_dp', DEFAULT_ROUND_DP))}, "
                f"snap_tol={st:g}, snapped={report.get('n_snapped_total', 0)}, "
                f"fabric_offset={fo:g}, "
                f"merge_layer_stack={mls}, "
                f"panels_merged={mlog.get('n_panels_merged', 0)}, "
                f"support_streams={mlog.get('n_support_streams', 0)})"
            )
            for d in report["dropped"][:12]:
                print(
                    f"    drop panel={d.get('panel')} layer_h={d.get('layer_h')} "
                    f"kind={d.get('kind')} area={float(d.get('area') or 0):.4g} "
                    f"({d.get('reason')})"
                )
            if report["n_dropped"] > 12:
                print(f"    ... +{report['n_dropped'] - 12} more")
            for C in (report.get("cut_log") or [])[:12]:
                if C.get("skipped"):
                    print(
                        f"    cut skip panel={C.get('panel')} ({C.get('reason')})"
                    )
                else:
                    clamp = " clamp" if C.get("barrier_clamped") else ""
                    pclip = " panel_clip" if C.get("panel_clipped") else ""
                    inc = " incomplete" if C.get("containment_incomplete") else ""
                    uf = C.get("uf") or {}
                    uf_m = uf.get("n_merged", 0)
                    ov = float(C.get("overshoot_ratio") or 1.0)
                    print(
                        f"    cut panel={C.get('panel')} "
                        f"[{C.get('shape')}] "
                        f"uf_merge={uf_m} "
                        f"hull={C.get('n_hull')}ΓåÆcut={C.get('n_cut')} "
                        f"ribbon={float(C.get('ribbon_area') or 0):.4g}ΓåÆ"
                        f"cut_area={float(C.get('cut_area') or 0):.4g} "
                        f"ov={ov:.2f} "
                        f"out={C.get('n_outside', 0)} "
                        f"out_panel={C.get('n_outside_panel', 0)}"
                        f"{clamp}{pclip}{inc}"
                    )
            n_cut = len(report.get("cut_log") or [])
            if n_cut > 12:
                print(f"    ... +{n_cut - 12} more cut")
            for F in (report.get("fabric_log") or [])[:12]:
                if F.get("skipped"):
                    print(
                        f"    fabric skip panel={F.get('panel')} "
                        f"({F.get('reason')})"
                    )
                else:
                    cflag = " clamp" if F.get("clamped") else ""
                    pflag = " panel_clip" if F.get("panel_clipped") else ""
                    print(
                        f"    fabric panel={F.get('panel')} "
                        f"offset={float(F.get('fabric_offset') or 0):g} "
                        f"area {float(F.get('area_in') or 0):.4g}ΓåÆ"
                        f"{float(F.get('area_out') or 0):.4g}"
                        f"{cflag}{pflag}"
                    )
            n_fab = len(report.get("fabric_log") or [])
            if n_fab > 12:
                print(f"    ... +{n_fab - 12} more fabric")
            for M in (mlog.get("panels") or [])[:12]:
                stream = M.get("stream") or "main"
                print(
                    f"    merge panel={M.get('panel')} stream={stream} "
                    f"sources={M.get('n_sources')}ΓåÆparts={M.get('n_parts')} "
                    f"h=[{float(M.get('h_lo') or 0):g},{float(M.get('h_hi') or 0):g}] "
                    f"phys={M.get('n_physical', '?')} ghost={M.get('n_ghost', '?')} "
                    f"support={M.get('n_support', '?')}"
                    f"{' UNION_FAIL' if M.get('union_failed') else ''}"
                )
            n_m = len(mlog.get("panels") or [])
            if n_m > 12:
                print(f"    ... +{n_m - 12} more merge")
            print(f"[clean] wrote {out_path}")

        # Auto-visualize cleaned JSON (default on)
        do_viz = not bool(args.no_viz)
        if do_viz:
            vrc = run_visualize_cleaned(out_path, quiet=bool(args.quiet))
            if vrc != 0:
                rc = rc or vrc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
