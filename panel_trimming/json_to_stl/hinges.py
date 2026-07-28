"""
Flexible TPU crease-connector (living hinge) export.

Everything runs **after final.stl is built**. Hinges never use original
stock outlines for XY — only post-boolean residual bands:

  build_panel_meshes records final_bands on each final mesh:
    {z_lo, z_hi, rings, stream=main|support, cut}

  remapped_face(P, H) =
      union of final residual bands on design panel P that are solid at H
      and have a face endpoint at H (z_lo≈H or z_hi≈H),
      **main and support together** (trimmed heights, not original stock).

  Per crease connecting panels A and B at H:
    hinge XY = remapped_face(A,H) ∪ remapped_face(B,H)
               (only residual pieces that touch this crease)
               + local bridge across the crease gap
    Do not pull in other panels or faces at other heights.

  mountain → film on BOTTOM of H
  valley   → film on TOP of H
  face-to-face (top + bottom at H) → film straddles H

Border / side-wall edges (type=2) are skipped.
  ``hinge_width`` is strip fallback only if final remap fails.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from panel_trimming.visualize.visualize_3d import (  # noqa: E402
    TYPE_MOUNTAIN,
    TYPE_VALLEY,
    _edge_key,
    _poly_xy,
    panel_thickness_offsets,
)

DEFAULT_HINGE_THICKNESS_MM = 0.6
DEFAULT_HINGE_WIDTH_MM = 4.0
DEFAULT_HINGE_END_INSET_MM = 1.5
DEFAULT_HINGE_EMBED_MM = 0.15
DEFAULT_HINGE_ENABLED = True
DEFAULT_LAYER_MATCH_MM = 0.6


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dedupe_closed_xy(pts: Sequence) -> List[List[float]]:
    out: List[List[float]] = []
    if pts is None:
        return out
    for p in pts:
        xy = [float(p[0]), float(p[1])]
        if out:
            q = out[-1]
            if abs(xy[0] - q[0]) < 1e-12 and abs(xy[1] - q[1]) < 1e-12:
                continue
        out.append(xy)
    if len(out) >= 2:
        a, b = out[0], out[-1]
        if abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12:
            out = out[:-1]
    return out


def _poly_signed_area(poly: Sequence) -> float:
    acc = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = float(poly[i][0]), float(poly[i][1])
        x1, y1 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        acc += x0 * y1 - x1 * y0
    return 0.5 * acc


def _panels_owning_edge(
    units: Sequence,
    a: Sequence[float],
    b: Sequence[float],
) -> List[int]:
    key = _edge_key(a, b)
    owned: List[int] = []
    for pi, unit in enumerate(units):
        outer = _dedupe_closed_xy(_poly_xy(unit))
        n = len(outer)
        if n < 2:
            continue
        for j in range(n):
            if _edge_key(outer[j], outer[(j + 1) % n]) == key:
                owned.append(int(pi))
                break
    return owned


def _ring_to_polygon(pts: Sequence) -> Optional[Polygon]:
    ring = _dedupe_closed_xy(pts)
    if len(ring) < 3:
        return None
    if abs(_poly_signed_area(ring)) < 1e-18:
        return None
    if _poly_signed_area(ring) < 0:
        ring = list(reversed(ring))
    try:
        g = Polygon(ring)
    except Exception:
        return None
    if g.is_empty:
        return None
    if not g.is_valid:
        try:
            g = make_valid(g)
        except Exception:
            try:
                g = g.buffer(0)
            except Exception:
                return None
    parts = _geom_to_polygons(g)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    try:
        u = unary_union(parts)
    except Exception:
        return max(parts, key=lambda p: p.area)
    more = _geom_to_polygons(u)
    return max(more, key=lambda p: p.area) if more else None


def _unit_to_polygon(unit) -> Optional[Polygon]:
    return _ring_to_polygon(_poly_xy(unit))


def _geom_to_polygons(geom) -> List[Polygon]:
    if geom is None:
        return []
    try:
        if geom.is_empty:
            return []
    except Exception:
        return []
    try:
        gt = geom.geom_type
    except Exception:
        return []
    if gt == "Polygon":
        return [geom] if geom.area > 1e-12 else []
    if gt == "MultiPolygon":
        return [p for p in geom.geoms if p.geom_type == "Polygon" and p.area > 1e-12]
    out: List[Polygon] = []
    for g in getattr(geom, "geoms", []) or []:
        out.extend(_geom_to_polygons(g))
    return out


def _polygon_to_ring(p: Polygon) -> Optional[List[List[float]]]:
    if p is None or p.is_empty or p.area < 1e-12:
        return None
    try:
        coords = list(p.exterior.coords)
    except Exception:
        return None
    ring = _dedupe_closed_xy(coords)
    return ring if len(ring) >= 3 else None


def _polygons_to_rings(geom, *, min_area: float = 1.0) -> List[List[List[float]]]:
    rings: List[List[List[float]]] = []
    for p in _geom_to_polygons(geom):
        try:
            if float(p.area) < float(min_area):
                continue
        except Exception:
            continue
        ring = _polygon_to_ring(p)
        if ring is not None:
            rings.append(ring)
    return rings


def _panel_stock_span(
    offsets: Sequence[Sequence[float]],
    panel_id: int,
    *,
    default_thickness: float = 6.0,
) -> Optional[Tuple[float, float]]:
    try:
        ui = int(panel_id)
    except (TypeError, ValueError):
        return None
    if ui < 0 or ui >= len(offsets):
        return None
    offs = [float(h) for h in (offsets[ui] or [])]
    if len(offs) >= 2:
        return float(min(offs)), float(max(offs))
    if len(offs) == 1:
        mid = float(offs[0])
        half = 0.5 * float(default_thickness)
        return mid - half, mid + half
    return None


def _face_role_at_height(
    span: Optional[Tuple[float, float]],
    H: float,
    *,
    match_mm: float = DEFAULT_LAYER_MATCH_MM,
) -> str:
    if span is None:
        return "unknown"
    z0, z1 = float(span[0]), float(span[1])
    eps = max(float(match_mm), 1e-6)
    if abs(float(H) - z1) <= eps:
        return "top"
    if abs(float(H) - z0) <= eps:
        return "bottom"
    if z0 - eps <= float(H) <= z1 + eps:
        return "inside"
    return "outside"


# ---------------------------------------------------------------------------
# Remap panels from final.stl build bands
# ---------------------------------------------------------------------------

def _band_covers_height(
    band: Dict[str, Any],
    H: float,
    *,
    match_mm: float = DEFAULT_LAYER_MATCH_MM,
) -> bool:
    """
    True if final residual band is solid at height H.

    Strict interval test (only ~1e-3 mm float slack). Do **not** use the large
    layer_match window here — that would let a thick solid band above a face
    (e.g. main [-2.9, 3] at H=-3) steal the residual meant for the thin cut
    face band / support interface.
    """
    _ = match_mm
    eps = 1e-3
    try:
        za = float(band.get("z_lo"))
        zb = float(band.get("z_hi"))
    except (TypeError, ValueError):
        return False
    if zb < za:
        za, zb = zb, za
    return (za - eps) <= float(H) <= (zb + eps)


def _band_z_lo_hi(band: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        za = float(band.get("z_lo"))
        zb = float(band.get("z_hi"))
    except (TypeError, ValueError):
        return None
    if zb < za:
        za, zb = zb, za
    return za, zb


def _select_face_bands(
    covering: Sequence[Dict[str, Any]],
    H: float,
    face_role: str,
    *,
    match_mm: float = DEFAULT_LAYER_MATCH_MM,
) -> List[Dict[str, Any]]:
    """
    Pick final residual bands for one panel at height H.

    Keep **main and support together** — final.stl is (main − coll) ∪
    (support − coll); the face footprint at H is their combined residual
    (trimmed heights), not original stock and not main-only.

    Prefer bands with any face endpoint at H (z_lo≈H or z_hi≈H) so a thick
    interior slab does not replace a thin cut face. ``face_role`` is only
    used for hinge Z placement later, not to drop a stream here.
    """
    _ = face_role
    if not covering:
        return []
    eps = max(float(match_mm), 1e-3)
    h = float(H)

    def _any_endpoint(b: Dict[str, Any]) -> bool:
        zh = _band_z_lo_hi(b)
        if zh is None:
            return False
        za, zb = zh
        return abs(za - h) <= eps or abs(zb - h) <= eps

    preferred = [b for b in covering if _any_endpoint(b)]
    if preferred:
        return list(preferred)
    return list(covering)


def extract_final_bands_by_panel(
    final_meshes: Sequence[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Collect final residual bands from build_panel_meshes final entries.

    Each band is the exact 2D footprint extruded into final.stl for that
    Z range (main and support streams).
    """
    by_panel: Dict[int, List[Dict[str, Any]]] = {}
    for m in final_meshes or []:
        try:
            pid = int(m.get("panel"))
        except (TypeError, ValueError):
            continue
        bands = list(m.get("final_bands") or [])
        if not bands:
            # Legacy final mesh without bands: invent one full span outline
            # from vertex AABB projection (last resort)
            V = m.get("vertices")
            if V is not None:
                arr = np.asarray(V, dtype=float)
                if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 3:
                    z_lo = float(np.min(arr[:, 2]))
                    z_hi = float(np.max(arr[:, 2]))
                    # No reliable XY ring without bands — skip
                    by_panel.setdefault(pid, []).append({
                        "z_lo": z_lo,
                        "z_hi": z_hi,
                        "rings": [],
                        "stream": "unknown",
                        "cut": False,
                        "n_rings": 0,
                        "source": "legacy_no_bands",
                    })
            continue
        for b in bands:
            entry = dict(b)
            entry.setdefault("source", "final_build")
            by_panel.setdefault(pid, []).append(entry)
    return by_panel


def build_remapped_panels_from_final(
    final_meshes: Sequence[Dict[str, Any]],
    data: dict,
    *,
    heights: Optional[Sequence[float]] = None,
    match_mm: float = DEFAULT_LAYER_MATCH_MM,
    default_thickness: float = 6.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Remap each design panel at every relevant height from final residual bands.

    (design_panel, H) → union of main+support residual rings solid at H
    (endpoint-matched face plane when possible). This is post-final geometry.
    """
    units = list(data.get("units") or [])
    lines = list(data.get("lines") or [])
    feats = list(data.get("line_features") or [])
    shades = list(data.get("shaded_regions") or [])
    offsets = panel_thickness_offsets(units, data, shades)
    bands_by_panel = extract_final_bands_by_panel(final_meshes)

    # Heights to evaluate: crease heights + stock faces + band endpoints
    want_heights: List[float] = []
    if heights is not None:
        for h in heights:
            try:
                want_heights.append(float(h))
            except (TypeError, ValueError):
                pass

    for i, ln in enumerate(lines):
        if not ln or len(ln) < 2:
            continue
        ft = feats[i] if i < len(feats) else {}
        try:
            ctype = int(ft.get("type", 2))
        except (TypeError, ValueError):
            ctype = 2
        if ctype not in (TYPE_MOUNTAIN, TYPE_VALLEY):
            continue
        try:
            want_heights.append(float(ft.get("thick_panel_height")))
        except (TypeError, ValueError):
            pass

    for offs in offsets:
        for h in offs or []:
            try:
                want_heights.append(float(h))
            except (TypeError, ValueError):
                pass

    for pid, bands in bands_by_panel.items():
        for b in bands:
            for key in ("z_lo", "z_hi"):
                try:
                    want_heights.append(float(b.get(key)))
                except (TypeError, ValueError):
                    pass

    # Unique heights
    want_heights = sorted(set(round(h, 6) for h in want_heights))

    remapped: List[Dict[str, Any]] = []
    n_main = 0
    n_support = 0
    n_both = 0
    n_empty = 0

    panels = sorted(set(bands_by_panel.keys()) | set(range(len(units))))
    for pid in panels:
        span = _panel_stock_span(
            offsets, pid, default_thickness=default_thickness
        )
        bands = list(bands_by_panel.get(pid) or [])
        for H in want_heights:
            covering = [
                b for b in bands
                if _band_covers_height(b, H, match_mm=match_mm)
                and list(b.get("rings") or [])
            ]
            if not covering:
                n_empty += 1
                continue

            face_role = _face_role_at_height(span, H, match_mm=match_mm)
            selected = _select_face_bands(
                covering, H, face_role, match_mm=match_mm
            )
            if not selected:
                n_empty += 1
                continue

            # Whole face residual for this panel at H (selected bands only)
            polys: List[Polygon] = []
            streams: List[str] = []
            any_cut = False
            for b in selected:
                streams.append(str(b.get("stream") or "main"))
                if b.get("cut"):
                    any_cut = True
                for ring in b.get("rings") or []:
                    p = _ring_to_polygon(ring)
                    if p is not None:
                        polys.append(p)
            if not polys:
                n_empty += 1
                continue
            try:
                geom = unary_union(polys) if len(polys) > 1 else polys[0]
            except Exception:
                geom = polys[0]
                for p in polys[1:]:
                    try:
                        geom = geom.union(p)
                    except Exception:
                        continue
            rings = _polygons_to_rings(geom)
            if not rings:
                n_empty += 1
                continue

            stream_set = {s for s in streams if s}
            if stream_set == {"main"}:
                stream = "main"
                n_main += 1
            elif stream_set == {"support"}:
                stream = "support"
                n_support += 1
            else:
                stream = "main+support"
                n_both += 1

            # Support-only residual outside phys stock is a support face
            if stream == "support" and face_role == "outside":
                face_role = "support"

            area = float(sum(abs(_poly_signed_area(r)) for r in rings))
            remapped.append({
                "remap_id": len(remapped),
                "design_panel": int(pid),
                "panel": int(pid),
                "layer_h": float(H),
                "face_role": face_role,
                "stream": stream,
                "source": "final_main_support_bands",
                "cut": bool(any_cut),
                "n_bands": len(selected),
                "n_covering_bands": len(covering),
                "n_rings": len(rings),
                "area": area,
                "rings": rings,
                "stock_z_lo": float(span[0]) if span else None,
                "stock_z_hi": float(span[1]) if span else None,
                "band_streams": sorted(stream_set),
                # Actual residual Z span of selected bands (trimmed final heights)
                "face_z_lo": float(min(
                    float(b["z_lo"]) for b in selected
                )),
                "face_z_hi": float(max(
                    float(b["z_hi"]) for b in selected
                )),
            })

    info = {
        "n_remapped_panels": len(remapped),
        "n_design_panels": len(units),
        "n_final_panels_with_bands": len(bands_by_panel),
        "n_stream_main": int(n_main),
        "n_stream_support": int(n_support),
        "n_stream_main_support": int(n_both),
        "n_empty_heights": int(n_empty),
        "n_heights": len(want_heights),
        "layer_match_mm": float(match_mm),
        "note": (
            "Panels remapped after final.stl from residual bands at each "
            "height (main+support together, endpoint face plane). "
            "Hinge joins only the two faces that share each crease."
        ),
    }
    return remapped, info


def _lookup_remapped(
    remapped: Sequence[Dict[str, Any]],
    design_panel: int,
    H: float,
    *,
    match_mm: float = DEFAULT_LAYER_MATCH_MM,
) -> Optional[Dict[str, Any]]:
    best = None
    best_dh = float("inf")
    for rp in remapped or []:
        try:
            if int(rp.get("design_panel")) != int(design_panel):
                continue
            lh = float(rp.get("layer_h"))
        except (TypeError, ValueError):
            continue
        dh = abs(lh - float(H))
        if dh < best_dh:
            best_dh = dh
            best = rp
    if best is None or best_dh > float(match_mm):
        return None
    return best


def _span_final_face_pair(
    ring_sets: Sequence[Sequence[Sequence[Sequence[float]]]],
    *,
    crease_a: Sequence[float],
    crease_b: Sequence[float],
    end_inset: float = 0.0,
    crease_gap: float = 0.0,
    panel_outers: Optional[Sequence[Optional[Polygon]]] = None,
) -> Tuple[List[List[List[float]]], Dict[str, Any]]:
    """
    Hinge XY from post-final remapped faces of the crease's two panels.

    ``ring_sets``: one list of residual rings per crease-owning panel
    (main∪support at H). Keep pieces that touch this crease, union, and
    bridge the gap only along the crease line.
    """
    info: Dict[str, Any] = {
        "bridged": False,
        "n_input_parts": 0,
        "n_panels_with_face": 0,
        "n_touching_parts": 0,
        "close_mm": 0.0,
        "mode": "final_face_pair",
    }

    try:
        crease_line = LineString(
            [
                (float(crease_a[0]), float(crease_a[1])),
                (float(crease_b[0]), float(crease_b[1])),
            ]
        )
    except Exception:
        return [], info
    if crease_line.is_empty or crease_line.length < 1e-12:
        return [], info

    # Final residuals are crease-gap inset; allow slack to count as touching
    touch_tol = max(float(crease_gap), 0.0) + 0.75
    info["touch_tol_mm"] = float(touch_tol)

    face_polys: List[Polygon] = []
    for rings in ring_sets:
        panel_parts: List[Polygon] = []
        for ring in rings or []:
            p = _ring_to_polygon(ring)
            if p is not None:
                panel_parts.extend(_geom_to_polygons(p))
        if not panel_parts:
            continue
        info["n_panels_with_face"] = int(info["n_panels_with_face"]) + 1
        touching = []
        for p in panel_parts:
            try:
                if float(p.distance(crease_line)) <= touch_tol:
                    touching.append(p)
            except Exception:
                continue
        if touching:
            face_polys.extend(touching)
        else:
            # Residual exists but is gap-separated — keep largest piece
            face_polys.append(max(panel_parts, key=lambda q: q.area))

    info["n_touching_parts"] = len(face_polys)
    if not face_polys:
        return [], info

    try:
        span = unary_union(face_polys) if len(face_polys) > 1 else face_polys[0]
    except Exception:
        span = face_polys[0]
        for p in face_polys[1:]:
            try:
                span = span.union(p)
            except Exception:
                continue
    if span is None or getattr(span, "is_empty", True):
        return [], info

    parts = _geom_to_polygons(span)
    info["n_input_parts"] = len(parts)

    # Local bridge along crease only (main/support faces across the gap)
    if len(parts) > 1:
        max_gap = 0.0
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                try:
                    max_gap = max(max_gap, float(parts[i].distance(parts[j])))
                except Exception:
                    continue
        if max_gap > 1e-9:
            close_d = min(
                max(0.5 * max_gap + 0.15, float(crease_gap) + 0.15, 0.25),
                2.0,
            )
            info["close_mm"] = float(close_d)
            try:
                bridge = crease_line.buffer(close_d, cap_style=2, join_style=2)
                joined = unary_union([span, bridge])
                if joined is not None and not joined.is_empty:
                    span = joined
                    info["bridged"] = True
            except Exception:
                try:
                    closed = span.buffer(close_d, join_style=1).buffer(
                        -close_d, join_style=1
                    )
                    if closed is not None and not closed.is_empty:
                        span = closed
                        info["bridged"] = True
                except Exception:
                    pass

    # Uniform outline inset (not circular holes at crease vertices)
    inset = max(float(end_inset), 0.0)
    if inset > 1e-12:
        try:
            eroded = span.buffer(-inset, join_style=2, mitre_limit=5.0)
            if (
                eroded is not None
                and not eroded.is_empty
                and getattr(eroded, "area", 0) > 1.0
            ):
                # Keep only pieces that still meet the crease
                keep = []
                for p in _geom_to_polygons(eroded):
                    try:
                        if float(p.area) < 1.0:
                            continue
                        if float(p.distance(crease_line)) <= touch_tol + inset:
                            keep.append(p)
                    except Exception:
                        keep.append(p)
                if keep:
                    span = unary_union(keep) if len(keep) > 1 else keep[0]
                    info["end_inset_applied"] = True
                    info["inset_mm"] = float(inset)
        except Exception:
            pass

    # Domain: crease-owning design panels only (keeps support from other units out)
    if panel_outers:
        outs = [p for p in panel_outers if p is not None and not p.is_empty]
        if outs:
            try:
                domain = unary_union(outs) if len(outs) > 1 else outs[0]
                domain = domain.buffer(
                    max(float(crease_gap) + 0.5, 0.75), join_style=2
                )
                clipped = span.intersection(domain)
                if clipped is not None and not clipped.is_empty:
                    span = clipped
            except Exception:
                pass

    if not getattr(span, "is_valid", True):
        try:
            span = make_valid(span)
        except Exception:
            try:
                span = span.buffer(0)
            except Exception:
                return [], info

    return _polygons_to_rings(span, min_area=1.0), info


def _strip_rectangle_xy(
    a: Sequence[float],
    b: Sequence[float],
    *,
    width: float,
    end_inset: float,
) -> Optional[List[List[float]]]:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    inset = max(float(end_inset), 0.0)
    if length - 2.0 * inset < 1e-6:
        inset = max(0.0, 0.25 * length)
    if length - 2.0 * inset < 1e-6:
        return None
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    half_w = 0.5 * max(float(width), 1e-6)
    a2x, a2y = ax + ux * inset, ay + uy * inset
    b2x, b2y = bx - ux * inset, by - uy * inset
    return [
        [a2x + nx * half_w, a2y + ny * half_w],
        [b2x + nx * half_w, b2y + ny * half_w],
        [b2x - nx * half_w, b2y - ny * half_w],
        [a2x - nx * half_w, a2y - ny * half_w],
    ]


def final_z_span_by_panel(
    final_meshes: Sequence[Dict[str, Any]],
) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    for m in final_meshes or []:
        try:
            pid = int(m.get("panel"))
        except (TypeError, ValueError):
            continue
        V = m.get("vertices")
        if V is None:
            continue
        arr = np.asarray(V, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
            continue
        z = arr[:, 2]
        z_lo, z_hi = float(np.min(z)), float(np.max(z))
        if pid in out:
            prev = out[pid]
            out[pid] = (min(prev[0], z_lo), max(prev[1], z_hi))
        else:
            out[pid] = (z_lo, z_hi)
    return out


def _hinge_z_span(
    kind: str,
    H: float,
    *,
    t_film: float,
    emb: float,
    face_roles: Sequence[str],
    streams: Sequence[str],
) -> Tuple[float, float, str]:
    face = float(H)
    roles = {str(r) for r in face_roles if r}
    stream_set = {str(s) for s in streams if s}
    interface = (
        ("top" in roles and "bottom" in roles)
        or any("main+support" in str(s) for s in stream_set)
        or ("main" in stream_set and "support" in stream_set)
        or "interface" in roles
    )

    if interface:
        z_lo = face - 0.5 * float(t_film) - float(emb)
        z_hi = face + 0.5 * float(t_film) + float(emb)
        side = "interface"
    elif kind == "mountain":
        z_lo, z_hi = face - float(t_film), face + float(emb)
        side = "bottom"
    else:
        z_lo, z_hi = face - float(emb), face + float(t_film)
        side = "top"

    if z_hi < z_lo:
        z_lo, z_hi = z_hi, z_lo
    if z_hi - z_lo < 1e-9:
        z_hi = z_lo + float(t_film)
    return float(z_lo), float(z_hi), side


# ---------------------------------------------------------------------------
# Public hinge API
# ---------------------------------------------------------------------------

def collect_crease_hinges(
    data: dict,
    *,
    final_meshes: Optional[Sequence[Dict[str, Any]]] = None,
    final_z_by_panel: Optional[Dict[int, Tuple[float, float]]] = None,
    hinge_thickness: float = DEFAULT_HINGE_THICKNESS_MM,
    hinge_width: float = DEFAULT_HINGE_WIDTH_MM,
    end_inset: float = DEFAULT_HINGE_END_INSET_MM,
    embed: float = DEFAULT_HINGE_EMBED_MM,
    crease_gap: float = 0.0,
    default_thickness: float = 6.0,
    thickness: Optional[float] = None,
    height_merge_mm: float = 0.35,
    min_band_mm: float = 0.5,
    layer_match_mm: float = DEFAULT_LAYER_MATCH_MM,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    One thin film per mountain/valley crease.

    Requires final_meshes from build_panel_meshes (with final_bands).
    XY and face Z come from post-final main+support residual remap at H.
    """
    _ = final_z_by_panel, thickness, height_merge_mm, min_band_mm

    units = list(data.get("units") or [])
    lines = list(data.get("lines") or [])
    feats = list(data.get("line_features") or [])

    gap = max(float(crease_gap), 0.0)
    width = max(float(hinge_width), gap + 2.0)
    t_film = max(float(hinge_thickness), 1e-4)
    emb = max(float(embed), 0.0)
    inset = max(float(end_inset), 0.0)
    match = max(float(layer_match_mm), 0.0)
    stock_th = float(default_thickness)

    final_parts = list(final_meshes or [])
    remapped, remap_info = build_remapped_panels_from_final(
        final_parts,
        data,
        match_mm=match,
        default_thickness=stock_th,
    )

    hinges: List[Dict[str, Any]] = []
    n_skip_border = 0
    n_skip_short = 0
    n_skip_no_panel = 0
    n_skip_no_height = 0
    n_skip_no_remap = 0
    n_fallback_strip = 0
    n_from_final_remap = 0
    n_bridged = 0

    for i, ln in enumerate(lines):
        if not ln or len(ln) < 2:
            continue
        ft = feats[i] if i < len(feats) else {}
        try:
            ctype = int(ft.get("type", 2))
        except (TypeError, ValueError):
            ctype = 2
        if ctype not in (TYPE_MOUNTAIN, TYPE_VALLEY):
            n_skip_border += 1
            continue

        a, b = ln[0], ln[1]
        panels = _panels_owning_edge(units, a, b)
        if not panels:
            n_skip_no_panel += 1
            continue

        th_h = ft.get("thick_panel_height")
        try:
            H = float(th_h) if th_h is not None else None
        except (TypeError, ValueError):
            H = None
        if H is None:
            try:
                if len(a) > 2:
                    H = float(a[2])
            except (TypeError, ValueError, IndexError):
                H = None
        if H is None:
            n_skip_no_height += 1
            continue

        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        length = math.hypot(bx - ax, by - ay)
        kind = "mountain" if ctype == TYPE_MOUNTAIN else "valley"

        ring_sets: List[List[List[List[float]]]] = []
        panel_remap_info: List[Dict[str, Any]] = []
        face_roles: List[str] = []
        streams: List[str] = []
        panel_outers: List[Optional[Polygon]] = []
        any_cut = False
        remap_ids: List[int] = []

        for pid in panels:
            ui = int(pid)
            try:
                panel_outers.append(
                    _unit_to_polygon(units[ui]) if 0 <= ui < len(units) else None
                )
            except (TypeError, ValueError):
                panel_outers.append(None)

            rp = _lookup_remapped(remapped, ui, float(H), match_mm=match)
            if rp is None:
                panel_remap_info.append({
                    "design_panel": ui,
                    "layer_h": float(H),
                    "missing": True,
                })
                face_roles.append("unknown")
                streams.append("missing")
            else:
                rings_p = list(rp.get("rings") or [])
                meta = {
                    "remap_id": rp.get("remap_id"),
                    "design_panel": rp.get("design_panel"),
                    "layer_h": rp.get("layer_h"),
                    "face_role": rp.get("face_role"),
                    "stream": rp.get("stream"),
                    "source": rp.get("source"),
                    "cut": rp.get("cut"),
                    "n_bands": rp.get("n_bands"),
                    "n_rings": rp.get("n_rings"),
                    "area": rp.get("area"),
                    "band_streams": rp.get("band_streams"),
                    "face_z_lo": rp.get("face_z_lo"),
                    "face_z_hi": rp.get("face_z_hi"),
                    "missing": False,
                }
                panel_remap_info.append(meta)
                face_roles.append(str(rp.get("face_role") or "unknown"))
                streams.append(str(rp.get("stream") or "main"))
                if bool(rp.get("cut")):
                    any_cut = True
                if rp.get("remap_id") is not None:
                    remap_ids.append(int(rp["remap_id"]))
                if rings_p:
                    ring_sets.append(rings_p)

        rings: List[List[List[float]]] = []
        footprint = "final_face_pair"
        bridge_info: Dict[str, Any] = {}

        # XY from post-final remapped main+support faces of crease owners only
        if ring_sets:
            rings, bridge_info = _span_final_face_pair(
                ring_sets,
                crease_a=a,
                crease_b=b,
                end_inset=inset,
                crease_gap=gap,
                panel_outers=panel_outers,
            )
            if rings:
                n_from_final_remap += 1
                if bridge_info.get("bridged"):
                    n_bridged += 1

        if not rings:
            n_skip_no_remap += 1
            ring = _strip_rectangle_xy(a, b, width=width, end_inset=inset)
            if ring is None:
                n_skip_short += 1
                continue
            rings = [ring]
            footprint = "strip_fallback"
            n_fallback_strip += 1

        z_lo, z_hi, side = _hinge_z_span(
            kind,
            float(H),
            t_film=t_film,
            emb=emb,
            face_roles=face_roles,
            streams=streams,
        )

        primary = rings[0]
        hinges.append({
            "id": int(i),
            "layer_index": 0,
            "n_layers": 1,
            "line_index": int(i),
            "kind": kind,
            "type": int(ctype),
            "side": side,
            "panels": list(int(p) for p in panels),
            "owners": list(int(p) for p in panels),
            "remap_ids": list(remap_ids),
            "crease_height": float(H),
            "layer_h": float(H),
            "p0": [ax, ay],
            "p1": [bx, by],
            "length_mm": float(length),
            "strip_length_mm": float(length - 2.0 * inset),
            "width_mm": float(width),
            "thickness_mm": float(t_film),
            "embed_mm": float(emb),
            "end_inset_mm": float(inset),
            "face_z": float(H),
            "z_lo": float(z_lo),
            "z_hi": float(z_hi),
            "face_roles": list(face_roles),
            "streams": list(streams),
            "footprint": footprint,
            "panel_remapped": bool(footprint == "final_face_pair"),
            "panel_collision_cut": bool(any_cut),
            "bridged": bool(bridge_info.get("bridged")),
            "bridge": dict(bridge_info),
            "panel_remap": panel_remap_info,
            "ring_xy": [list(p) for p in primary],
            "segment_rings": [[list(p) for p in r] for r in rings],
        })

    n_cut = sum(1 for h in hinges if h.get("panel_collision_cut"))
    info = {
        "n_hinges": len(hinges),
        "n_design_creases": len(hinges),
        "n_layer_creases": len(hinges),
        "n_mountain": sum(1 for h in hinges if h["kind"] == "mountain"),
        "n_valley": sum(1 for h in hinges if h["kind"] == "valley"),
        "n_skip_border": int(n_skip_border),
        "n_skip_short": int(n_skip_short),
        "n_skip_no_panel": int(n_skip_no_panel),
        "n_skip_no_height": int(n_skip_no_height),
        "n_skip_no_remap": int(n_skip_no_remap),
        "n_from_final_remap": int(n_from_final_remap),
        "n_panel_collision_cut": int(n_cut),
        "n_bridged": int(n_bridged),
        "n_fallback_strip": int(n_fallback_strip),
        "hinge_thickness_mm": float(t_film),
        "hinge_width_mm": float(width),
        "end_inset_mm": float(inset),
        "embed_mm": float(emb),
        "crease_gap_mm": float(gap),
        "layer_match_mm": float(match),
        "cut_mode": "final_face_pair_at_crease",
        "panel_remap": remap_info,
        "note": (
            "After final.stl: each panel remapped at H as main∪support "
            "residual face. Hinge = those faces for the two panels the crease "
            "connects, bridged only across the gap. "
            "mountain=bottom, valley=top; interfaces straddle H."
        ),
    }
    return hinges, info


def build_hinge_meshes(
    hinges: Sequence[Dict[str, Any]],
    *,
    extrude_polygon_mesh,
    merge_meshes,
    data: Optional[dict] = None,
    boolean_panel_minus_collisions=None,
    layer_match_mm: float = DEFAULT_LAYER_MATCH_MM,
    main_slabs=None,
    support_coll_slabs=None,
    union_rings_unclipped=None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extrude each crease as a thin film over post-final remapped face pairs."""
    _ = (
        data,
        boolean_panel_minus_collisions,
        layer_match_mm,
        main_slabs,
        support_coll_slabs,
        union_rings_unclipped,
    )
    groups: Dict[str, List[Dict[str, Any]]] = {
        "hinges": [],
        "hinges_mountain": [],
        "hinges_valley": [],
    }

    for h in hinges or []:
        rings = h.get("segment_rings") or []
        if not rings:
            ring = h.get("ring_xy")
            if ring and len(ring) >= 3:
                rings = [ring]
        if not rings:
            continue
        za, zb = float(h["z_lo"]), float(h["z_hi"])
        meshes: List[Tuple[Any, Any, Any]] = []
        n_ok = 0
        for ring in rings:
            if ring is None or len(ring) < 3:
                continue
            V, F, N = extrude_polygon_mesh(ring, z_lo=za, z_hi=zb, holes=None)
            if len(V) and len(F):
                meshes.append((V, F, N))
                n_ok += 1
        if not meshes:
            continue
        V, F, N = merge_meshes(meshes)
        entry = {
            "kind": f"hinge_{h['kind']}",
            "hinge_id": h["id"],
            "layer_index": h.get("layer_index", 0),
            "n_layers": h.get("n_layers", 1),
            "layer_h": h.get("layer_h"),
            "crease_kind": h["kind"],
            "side": h["side"],
            "panels": list(h.get("panels") or []),
            "remap_ids": list(h.get("remap_ids") or []),
            "streams": list(h.get("streams") or []),
            "owners": h.get("owners"),
            "face_z": h.get("face_z"),
            "z_lo": za,
            "z_hi": zb,
            "n_pieces": int(n_ok),
            "cut": bool(h.get("panel_collision_cut")),
            "bridged": bool(h.get("bridged")),
            "cut_mode": h.get("footprint") or "final_face_pair_at_crease",
            "n_verts": int(len(V)),
            "n_faces": int(len(F)),
            "vertices": V,
            "faces": F,
            "normals": N,
        }
        groups["hinges"].append(entry)
        if h["kind"] == "mountain":
            groups["hinges_mountain"].append(entry)
        else:
            groups["hinges_valley"].append(entry)

    return groups
