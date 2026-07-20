"""
Clean trimmed collision shades: drop collapsed / free-standing line paints.

Reads  panel_trimming/trimmedData/<name>-trimmed.json  (or a path)
Writes panel_trimming/trimmedData/<name>-cleaned.json  (-trimmed → -cleaned suffix)

Drops:
  - kind == "line"  (exporter collapsed-contact flag)
  - recomputed ribbon area ~ 0  (c0/c1 reverse-trace; shoelace cancels)
  - optional: area < min_area (default 0 = off; keeps small pink fills)

Straight containing cut (default on):
  1) Convex hull of all dual-curve samples (works for concave ribbons)
  2) Simplify dense / near-convex hulls to few straight edges (RDP)
  3) Expand cut from centroid until samples are inside again, but each
     vertex ray stops at the nearest crease / border / panel edge
     → prefer greater enclosure without crossing design walls

Viz helpers:
  clean_ring   = all dual-curve sample nodes (c0 + reverse c1)
  coordinates  = straight containing cut polygon
  cut_ring     = same as coordinates
  clean_edges  = straight cut edges

Fabrication offset (optional, config / --fabric-offset):
  Outward offset of cut_ring for kerf / tool allowance. Each vertex is pushed
  along the outward normal, then clamped so it cannot cross a crease, border,
  or host-panel edge (same barrier model as cut expand).

YAML (panel_trimming/clean/config.yml):
  jobs:
    - name: miura-thick-trimmed
      fabric_offset: 0.5

Usage:
  python panel_trimming/clean/clean.py
  python panel_trimming/clean/clean.py --config panel_trimming/clean/config.yml
  python panel_trimming/clean/clean.py --name miura-thick-trimmed --fabric-offset 0.8
  python panel_trimming/clean/clean.py --name mountain-thick-trimmed --min-area 20
  python panel_trimming/clean/clean.py --name miura-thick-trimmed --cut-tol 2.0
  python panel_trimming/clean/clean.py --name miura-thick-trimmed --no-straight-cut
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# panel_trimming/clean/ → panel_trimming → project root
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
# Max perpendicular deviation when simplifying dense convex hulls (design units)
DEFAULT_CUT_TOL = 1.5
# Outward fabrication offset (design units / mm). 0 = off.
DEFAULT_FABRIC_OFFSET = 0.0
# line_features.type: mountain / valley / border — expansion may not cross these
TYPE_MOUNTAIN = 0
TYPE_VALLEY = 1
TYPE_BORDER = 2
BARRIER_TYPES = (TYPE_MOUNTAIN, TYPE_VALLEY, TYPE_BORDER)
# Stop a short epsilon before the barrier so we don't sit on top of creases
BARRIER_STOP_EPS = 1e-4
# Merge near-coincident cut corners into one outermost node (design units)
DEFAULT_CORNER_MERGE_TOL = 3.0


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
    mountain-thick-trimmed.json → mountain-thick-cleaned.json
    mountain-thick.json         → mountain-thick-cleaned.json
    mountain-thick-cleaned.json → mountain-thick-cleaned.json  (idempotent)
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
    Smallest convex set containing every input point → full red ribbon inside,
    hull area ≥ ribbon area.
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
    # On boundary?
    for i in range(n):
        ax, ay = float(poly[i][0]), float(poly[i][1])
        bx, by = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        abx, aby = bx - ax, by - ay
        apx, apy = x - ax, y - ay
        lab2 = abx * abx + aby * aby
        if lab2 < 1e-24:
            if abs(apx) <= eps and abs(apy) <= eps:
                return True
            continue
        cross = abx * apy - aby * apx
        if abs(cross) <= eps * math.sqrt(lab2):
            t = (apx * abx + apy * aby) / lab2
            if -1e-9 <= t <= 1.0 + 1e-9:
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
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(p[0]), float(p[1])
    dx, dy = bx - ax, by - ay
    lab2 = dx * dx + dy * dy
    if lab2 < 1e-24:
        return math.hypot(px - ax, py - ay)
    return abs(dx * (ay - py) - dy * (ax - px)) / math.sqrt(lab2)


def _rdp_indices_open(pts: Sequence, eps: float) -> List[int]:
    """Douglas–Peucker keep indices for an open polyline."""
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
    # but we need to close: RDP the chain rot[0]..rot[-1] then connect last→first.
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
    already sitting on a border) caps expansion — previously only s>1 was
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
        # the barrier lies between C and V (s in (0,1)) — vertex already past wall.
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
    Sutherland–Hodgman: clip subject polygon to the half-plane left of directed
    edge A→B (points with cross(B-A, P-A) >= 0 kept).
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
        # (p-a) × e + t (d × e) = 0
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
        # scale s in [0,1] along origin→target
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
            # outward normal for CCW poly = right-to-left rotate → (-ey, ex)? 
            # CCW edge A→B, interior left, outward = (ey, -ex) / el
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
    neighbour chord is ≤ max_dev (handles near-collinear hull runs on convex
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
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Straight-edged cut that contains every dual-curve sample.

    Pipeline:
      1) Convex hull of {c0 ∪ c1}  — outer enclosure for **concave** ribbons.
      2) Simplify dense / **convex** hulls (RDP + collinear drop).
      3) Expand from centroid until samples are inside, but **never past the
         nearest crease / border** along each vertex ray (incl. verts on wall).
      4) Hard-clip to host panel polygon so nothing exceeds the border.
    """
    samples: List[List[float]] = []
    for p in c0 or []:
        samples.append(_xy(p))
    for p in c1 or []:
        samples.append(_xy(p))
    ribbon = _ribbon_ring(c0, c1)
    ribbon_a = _polygon_area_2d(ribbon) if ribbon else 0.0
    bars = list(barriers or [])
    panel = [_xy(p) for p in panel_poly] if panel_poly and len(panel_poly) >= 3 else None

    hull = convex_hull(samples)
    n_hull_raw = len(hull)
    hull_a = _polygon_area_2d(hull) if len(hull) >= 3 else 0.0
    if ribbon is not None and hull_a + 1e-9 < ribbon_a:
        hull = convex_hull(ribbon)
        hull_a = _polygon_area_2d(hull)
        n_hull_raw = len(hull)

    t = float(tol)
    simplified = simplify_closed_polygon(hull, tol=t) if t > 0.0 else [list(p) for p in hull]
    simplified = drop_near_collinear(simplified, max_dev=t)
    if len(simplified) < 3:
        simplified = [list(p) for p in hull]

    # Clip simplified hull to panel first (hull can sit on border; expand must not leave)
    if panel is not None and len(simplified) >= 3:
        clipped0 = clip_polygon_to_convex(simplified, panel)
        if len(clipped0) >= 3:
            simplified = clipped0

    # Tiny ribbons: do not edge-push (can runaway along panel); keep hull only
    use_push = ribbon_a >= 5.0 or n_hull_raw >= 8
    if use_push:
        cut, n_expand, scale, exp_info = expand_polygon_to_contain(
            simplified, samples, barriers=bars, panel_poly=panel
        )
    else:
        cut = [list(p) for p in simplified]
        n_expand, scale = 0, 1.0
        exp_info = {
            "barrier_clamped": False,
            "n_barriers": len(bars),
            "skip_push_tiny": True,
        }
    if len(cut) < 3:
        cut = [list(p) for p in (clip_polygon_to_convex(hull, panel) if panel else hull)]
        n_expand, scale = 0, 1.0
        exp_info = {"barrier_clamped": False, "n_barriers": len(bars)}

    # Final panel clip (hard guarantee — cut must never leave the host panel)
    if panel is not None and len(cut) >= 3:
        clipped = clip_polygon_to_convex(cut, panel)
        if len(clipped) >= 3:
            cut = clipped
            exp_info = dict(exp_info)
            exp_info["panel_clipped"] = True

    def _count_out(poly_pts: Sequence) -> int:
        return sum(
            1 for p in samples
            if len(poly_pts) >= 3 and not _point_in_or_on_polygon(p, poly_pts, eps=1e-6)
        )

    cut_a = _polygon_area_2d(cut)
    n_out = _count_out(cut)

    # If a few samples remain outside after edge-push, re-hull cut ∪ those
    # samples (adds only the needed extremes), then panel-clip. Avoids falling
    # back to the full dense 50+ vertex hull.
    if n_out > 0:
        outside_pts = [
            p for p in samples
            if not _point_in_or_on_polygon(p, cut, eps=1e-6)
        ]
        if outside_pts:
            merged = convex_hull(list(cut) + outside_pts)
            if panel is not None:
                merged = clip_polygon_to_convex(merged, panel)
            if len(merged) >= 3:
                n_out_m = _count_out(merged)
                # Prefer merged if better containment or fewer verts than full hull
                if n_out_m < n_out or (
                    n_out_m <= n_out and len(merged) <= max(len(cut) + len(outside_pts), 8)
                ):
                    cut = merged
                    cut_a = _polygon_area_2d(cut)
                    n_out = n_out_m
                    exp_info = dict(exp_info)
                    exp_info["merged_outside_samples"] = True

    # Only fall back to full hull if still many samples outside
    if n_out > max(2, int(0.02 * max(len(samples), 1))):
        hull_clip = (
            clip_polygon_to_convex(hull, panel) if panel else [list(p) for p in hull]
        )
        if len(hull_clip) >= 3:
            n_out_h = _count_out(hull_clip)
            if n_out_h < n_out:
                cut = hull_clip
                cut_a = _polygon_area_2d(cut)
                n_out = n_out_h
                n_expand, scale = 0, 1.0
                exp_info = {
                    "barrier_clamped": bool(exp_info.get("barrier_clamped")),
                    "n_barriers": len(bars),
                    "fell_back_to_hull": True,
                    "panel_clipped": bool(panel is not None),
                }

    # Final hard clamp: every cut vertex must lie in the host panel
    if panel is not None and len(cut) >= 3:
        cut = [_clamp_point_to_convex(p, panel) for p in cut]
        cut = clip_polygon_to_convex(cut, panel) if len(cut) >= 3 else cut
        if len(cut) >= 3:
            cut_a = _polygon_area_2d(cut)
            exp_info = dict(exp_info)
            exp_info["panel_clipped"] = True

    # Collinear cleanup only if containment preserved
    if n_out == 0 and t > 0.0 and len(cut) > 4:
        simp2 = drop_near_collinear(cut, max_dev=max(t, 0.5))
        if panel is not None and len(simp2) >= 3:
            simp2 = clip_polygon_to_convex(simp2, panel)
        if len(simp2) >= 3 and _count_out(simp2) == 0:
            cut = simp2
            cut_a = _polygon_area_2d(cut)

    n_out = _count_out(cut)
    n_out_panel = 0
    if panel is not None:
        n_out_panel = sum(
            1 for p in cut
            if not _point_in_or_on_polygon(p, panel, eps=1e-3)
        )
    shape = "convex_dense" if n_hull_raw >= 12 else "concave_or_simple"

    info = {
        "n_samples": len(samples),
        "n_hull": int(n_hull_raw),
        "n_cut": len(cut),
        "n_simplified": len(simplified),
        "ribbon_area": float(ribbon_a),
        "cut_area": float(cut_a),
        "n_outside": int(n_out),
        "n_outside_panel": int(n_out_panel),
        "n_expand_iters": int(n_expand),
        "expand_scale": float(scale),
        "cut_tol": float(t),
        "shape": shape,
        "n_barriers": int(exp_info.get("n_barriers") or len(bars)),
        "barrier_clamped": bool(exp_info.get("barrier_clamped")),
        "panel_clipped": bool(exp_info.get("panel_clipped")),
        "containment_incomplete": bool(exp_info.get("containment_incomplete")),
    }
    return cut, info


def collapse_nodes_keep_outermost(
    pts: Sequence,
    *,
    tol: float = DEFAULT_CORNER_MERGE_TOL,
    origin: Optional[Sequence] = None,
) -> List[List[float]]:
    """
    Collapse near-coincident corner clusters to one outermost node each.

    Walks the closed ring and merges **consecutive** vertices within ``tol``
    into a single outermost point (farthest from ``origin`` / centroid).
    Adaptive tol for tiny polygons so they do not collapse to one point.
    """
    raw = [_xy(p) for p in pts or []]
    n = len(raw)
    if n <= 1 or float(tol) <= 0.0:
        return raw

    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    use_tol = float(tol)
    if diag > 1e-9:
        use_tol = min(use_tol, max(0.35, 0.15 * diag))
    tol2 = use_tol * use_tol

    if origin is None:
        ox = sum(p[0] for p in raw) / n
        oy = sum(p[1] for p in raw) / n
    else:
        ox, oy = float(origin[0]), float(origin[1])

    def outer_score(p: Sequence) -> float:
        return (float(p[0]) - ox) ** 2 + (float(p[1]) - oy) ** 2

    start = max(range(n), key=lambda i: outer_score(raw[i]))
    order = [(start + k) % n for k in range(n)]

    runs: List[List[int]] = []
    cur_run = [order[0]]
    for k in range(1, n):
        i_prev, i = order[k - 1], order[k]
        if _dist2(raw[i_prev], raw[i]) <= tol2:
            cur_run.append(i)
        else:
            runs.append(cur_run)
            cur_run = [i]
    runs.append(cur_run)
    if len(runs) >= 2 and _dist2(raw[runs[0][0]], raw[runs[-1][-1]]) <= tol2:
        runs[0] = runs[-1] + runs[0]
        runs.pop()

    out: List[List[float]] = []
    for run in runs:
        win = max(run, key=lambda i: outer_score(raw[i]))
        p = list(raw[win])
        if not out or _dist2(out[-1], p) > tol2:
            out.append(p)
    if len(out) >= 2 and _dist2(out[0], out[-1]) <= tol2:
        if outer_score(out[0]) >= outer_score(out[-1]):
            out = out[:-1]
        else:
            out = out[1:]
    return out if len(out) >= 3 else raw


def apply_straight_containing_cut(
    region: dict,
    *,
    tol: float = DEFAULT_CUT_TOL,
    barriers: Optional[Sequence[Tuple[Sequence, Sequence]]] = None,
    data: Optional[dict] = None,
    corner_merge_tol: float = DEFAULT_CORNER_MERGE_TOL,
) -> Dict[str, Any]:
    """
    In-place: set coordinates / cut_ring to a straight containing cut.
    Keeps original c0/c1 (sample trail + red/green first-last).

    Expansion is limited by creases / borders; result is clipped to host panel.
    Near-coincident cut corners collapse to one outermost node each.
    """
    c0 = region.get("c0") or []
    c1 = region.get("c1") or []
    if not c0 or not c1 or len(c0) < 2 or len(c1) < 2:
        return {"skipped": True, "reason": "missing dual curves"}

    bars = list(barriers) if barriers is not None else collect_barrier_segments(
        data, region=region
    )
    panel = _panel_polygon(data, region)
    cut, info = straight_containing_cut(
        c0, c1, tol=tol, barriers=bars, panel_poly=panel
    )
    if len(cut) < 3:
        return {"skipped": True, "reason": "cut < 3 verts", **info}

    n_before = len(cut)
    cut = collapse_nodes_keep_outermost(cut, tol=float(corner_merge_tol))
    if len(cut) < 3:
        cut, info = straight_containing_cut(
            c0, c1, tol=tol, barriers=bars, panel_poly=panel
        )
    else:
        info = dict(info)
        info["n_cut"] = len(cut)
        info["n_corners_merged"] = max(0, n_before - len(cut))
        info["cut_area"] = float(_polygon_area_2d(cut))

    region["coordinates"] = [list(p) for p in cut]
    region["cut_ring"] = [list(p) for p in cut]
    region["area"] = float(info.get("cut_area") or _polygon_area_2d(cut))
    region["ribbon_area"] = float(info["ribbon_area"])
    region["cut_kind"] = "straight_containing"
    region["cut_tol"] = float(tol)
    region["corner_merge_tol"] = float(corner_merge_tol)
    region["approx_type"] = "linear"
    region["boundary_order"] = 1
    region["curve_order"] = 1
    region["spline"] = False
    region["curved_boundary"] = False
    if str(region.get("kind") or "").lower() in ("line", "empty", ""):
        if float(region["area"]) > COLLAPSED_AREA_EPS:
            region["kind"] = "sweep"
    return {"skipped": False, **info}


# ---------------------------------------------------------------------------
# Fabrication offset (outward), barrier-clamped
# ---------------------------------------------------------------------------

def _edge_outward_normal(a: Sequence, b: Sequence, *, ccw: bool) -> Tuple[float, float]:
    """Unit outward normal for directed edge A→B (poly CCW ⇒ outward = right)."""
    ex = float(b[0]) - float(a[0])
    ey = float(b[1]) - float(a[1])
    el = math.hypot(ex, ey) or 1.0
    # CCW: left of edge is interior → outward = (ey, -ex) / el
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

    Positive distance = grow outward. Zero / negative → no-op copy.
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
            # nearly opposite normals — fall back to n1
            bx, by = n1[0], n1[1]
            bl = math.hypot(bx, by) or 1.0
        bx, by = bx / bl, by / bl
        # miter length so offset distance along edge normals ≈ d
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
) -> Dict[str, Any]:
    """
    In-place: offset cut_ring / coordinates outward for fabrication.

    Stores:
      fabric_ring      — offset polygon (clamped)
      fabric_offset    — requested distance
      cut_ring         — geometric cut (unchanged if already set)
      coordinates      — set to fabric_ring when offset > 0 (paint/export cut)
    """
    d = float(offset)
    if d <= 1e-12:
        region.pop("fabric_ring", None)
        region.pop("fabric_offset", None)
        return {"skipped": True, "reason": "fabric_offset<=0", "fabric_offset": d}

    poly = region.get("cut_ring") or region.get("coordinates")
    if not poly or len(poly) < 3:
        # fall back to dual-curve ribbon
        ring = _ribbon_ring(region.get("c0") or [], region.get("c1") or [])
        poly = ring
    if not poly or len(poly) < 3:
        return {"skipped": True, "reason": "no polygon to offset", "fabric_offset": d}

    bars = list(barriers) if barriers is not None else collect_barrier_segments(
        data, region=region
    )
    panel = _panel_polygon(data, region)
    fab, info = offset_polygon_outward(
        poly, d, barriers=bars, panel_poly=panel
    )
    if len(fab) < 3:
        return {"skipped": True, "reason": "offset collapsed", **info}

    # Keep geometric cut separate from fabrication outline
    if not region.get("cut_ring") and region.get("coordinates"):
        region["cut_ring"] = [list(p) for p in region["coordinates"]]
    # Group near-coincident fab corners → one outermost each
    n_before = len(fab)
    fab = collapse_nodes_keep_outermost(
        fab, tol=float(DEFAULT_CORNER_MERGE_TOL)
    )
    if len(fab) < 3:
        fab, info = offset_polygon_outward(
            poly, d, barriers=bars, panel_poly=panel
        )
    else:
        info = dict(info)
        info["n_corners_merged"] = max(0, n_before - len(fab))
        info["area_out"] = float(_polygon_area_2d(fab))
        info["n_out"] = len(fab)

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
      clean_ring  — all dual-curve sample nodes (for blue/yellow node viz)
      clean_edges — straight edges of the containing cut (hull) when present,
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
    n_side = sum(
        1 for r in regions if str(r.get("shell_kind") or "").lower() == "side"
    )
    n_phys = sum(
        1 for r in regions
        if str(r.get("shell_kind") or "").lower() not in ("ghost", "side")
    )
    stats["schema"] = stats.get("schema") or "dual_curve_v1"
    stats["n_shaded_regions"] = len(regions)
    stats["n_shaded_areas"] = len(regions)
    stats["n_sweep_paints"] = n_sweep
    stats["n_closed_polygons"] = n_sweep
    stats["n_physical_regions"] = n_phys
    stats["n_ghost_regions"] = n_ghost
    stats["n_side_regions"] = n_side
    stats["shaded_regions_key"] = "shaded_regions"


def clean_data(
    data: dict,
    *,
    min_area: float = DEFAULT_MIN_AREA,
    drop_line_kind: bool = True,
    attach_polygon_meta: bool = True,
    straight_cut: bool = True,
    cut_tol: float = DEFAULT_CUT_TOL,
    fabric_offset: float = DEFAULT_FABRIC_OFFSET,
    corner_merge_tol: float = DEFAULT_CORNER_MERGE_TOL,
    source_path: Optional[str] = None,
) -> Tuple[dict, Dict[str, Any]]:
    """
    Return (cleaned_data, report). Deep-copies input structure.

    Pipeline:
      1) filter collapsed / too-small
      2) straight containing cut (hull → simplify → expand, barrier-clamped)
      3) group near-coincident cut corners → one outermost each
      4) optional fabric_offset (outward, barrier/panel-clamped)
      5) attach clean_ring (all samples) + clean_edges (fabric or cut)
    """
    out = copy.deepcopy(data)
    regions = list(out.get("shaded_regions") or [])
    kept, dropped_log = filter_shaded_regions(
        regions, min_area=min_area, drop_line_kind=drop_line_kind
    )

    cut_log: List[Dict[str, Any]] = []
    if straight_cut:
        for r in kept:
            bars = collect_barrier_segments(out, region=r)
            info = apply_straight_containing_cut(
                r,
                tol=float(cut_tol),
                barriers=bars,
                data=out,
                corner_merge_tol=float(corner_merge_tol),
            )
            cut_log.append({
                "panel": r.get("panel"),
                "layer_h": r.get("layer_h"),
                **info,
            })

    fabric_log: List[Dict[str, Any]] = []
    fo = float(fabric_offset)
    if fo > 1e-12:
        for r in kept:
            bars = collect_barrier_segments(out, region=r)
            finfo = apply_fabric_offset(
                r, offset=fo, barriers=bars, data=out
            )
            fabric_log.append({
                "panel": r.get("panel"),
                "layer_h": r.get("layer_h"),
                **finfo,
            })

    if attach_polygon_meta:
        for r in kept:
            attach_closed_polygon_meta(r)

    out["shaded_regions"] = kept
    _refresh_collision_stats(out, kept)

    report = {
        "source": source_path,
        "min_area": float(min_area),
        "drop_line_kind": bool(drop_line_kind),
        "straight_cut": bool(straight_cut),
        "cut_tol": float(cut_tol),
        "fabric_offset": fo,
        "n_in": len(regions),
        "n_out": len(kept),
        "n_dropped": len(dropped_log),
        "dropped": dropped_log,
        "cut_log": cut_log,
        "fabric_log": fabric_log,
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
    meta["clean_straight_cut"] = bool(straight_cut)
    meta["clean_cut_tol"] = float(cut_tol)
    meta["clean_cut_kind"] = "straight_containing" if straight_cut else None
    meta["clean_fabric_offset"] = fo
    meta["clean_corner_merge_tol"] = float(corner_merge_tol)
    meta["approx_type"] = "linear" if straight_cut else meta.get("approx_type")
    meta["boundary_order"] = 1 if straight_cut else meta.get("boundary_order")
    meta["curved_boundary"] = False if straight_cut else meta.get("curved_boundary")
    for k in (
        "clean_group_tol",
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
    corner_merge_tol: float = DEFAULT_CORNER_MERGE_TOL,
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
        corner_merge_tol=corner_merge_tol,
        source_path=input_path,
    )
    _write_json(output_path, cleaned)
    report["output"] = output_path
    return report


def _default_output_path(input_path: str, output_dir: str) -> str:
    return os.path.join(output_dir, _cleaned_basename(input_path))


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
        "fabric_offset",
        "corner_merge_tol",
        "attach_polygon_meta",
        "drop_line_kind",
        "output",
        "output_dir",
    ):
        if k in job and job[k] is not None:
            out[k] = job[k]
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
            "Max edge deviation when simplifying dense/convex hull chains "
            f"(default from config or {DEFAULT_CUT_TOL:g})"
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

    cfg_defaults = {
        "min_area": float(cfg.get("min_area", DEFAULT_MIN_AREA)),
        "straight_cut": bool(cfg.get("straight_cut", True)),
        "cut_tol": float(cfg.get("cut_tol", DEFAULT_CUT_TOL)),
        "fabric_offset": float(cfg.get("fabric_offset", DEFAULT_FABRIC_OFFSET)),
        "corner_merge_tol": float(
            cfg.get("corner_merge_tol", DEFAULT_CORNER_MERGE_TOL)
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
    if args.fabric_offset is not None:
        cfg_defaults["fabric_offset"] = float(args.fabric_offset)
    if args.no_straight_cut:
        cfg_defaults["straight_cut"] = False
    if args.no_polygon_meta:
        cfg_defaults["attach_polygon_meta"] = False
    if args.keep_line_kind:
        cfg_defaults["drop_line_kind"] = False
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
                if args.fabric_offset is not None:
                    settings["fabric_offset"] = float(args.fabric_offset)
                if args.no_straight_cut:
                    settings["straight_cut"] = False
                if args.no_polygon_meta:
                    settings["attach_polygon_meta"] = False
                if args.keep_line_kind:
                    settings["drop_line_kind"] = False
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
            report = clean_file(
                in_path,
                out_path,
                min_area=float(settings.get("min_area", DEFAULT_MIN_AREA)),
                drop_line_kind=bool(settings.get("drop_line_kind", True)),
                attach_polygon_meta=bool(settings.get("attach_polygon_meta", True)),
                straight_cut=bool(settings.get("straight_cut", True)),
                cut_tol=float(settings.get("cut_tol", DEFAULT_CUT_TOL)),
                fabric_offset=float(settings.get("fabric_offset", DEFAULT_FABRIC_OFFSET)),
                corner_merge_tol=float(
                    settings.get("corner_merge_tol", DEFAULT_CORNER_MERGE_TOL)
                ),
            )
        except Exception as exc:
            print(f"[clean] FAILED {in_path}: {exc}", file=sys.stderr)
            rc = 1
            continue

        if not args.quiet:
            fo = float(settings.get("fabric_offset", DEFAULT_FABRIC_OFFSET))
            print(
                f"[clean] {os.path.basename(in_path)}: "
                f"{report['n_in']} → {report['n_out']} "
                f"(dropped {report['n_dropped']}, "
                f"min_area={float(settings.get('min_area', 0)):g}, "
                f"straight_cut={bool(settings.get('straight_cut', True))}, "
                f"cut_tol={float(settings.get('cut_tol', DEFAULT_CUT_TOL)):g}, "
                f"fabric_offset={fo:g})"
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
                    print(
                        f"    cut panel={C.get('panel')} "
                        f"[{C.get('shape')}] "
                        f"hull={C.get('n_hull')}→cut={C.get('n_cut')} "
                        f"ribbon={float(C.get('ribbon_area') or 0):.4g}→"
                        f"cut_area={float(C.get('cut_area') or 0):.4g} "
                        f"barriers={C.get('n_barriers', 0)} "
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
                        f"area {float(F.get('area_in') or 0):.4g}→"
                        f"{float(F.get('area_out') or 0):.4g}"
                        f"{cflag}{pflag}"
                    )
            n_fab = len(report.get("fabric_log") or [])
            if n_fab > 12:
                print(f"    ... +{n_fab - 12} more fabric")
            print(f"[clean] wrote {out_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
