"""Pure helpers for panel collision / locus paint / dual-curve export.

No Taichi, no simulator state — only NumPy / pure Python so the mixin in
``collision.py`` can stay focused on orchestration and kernels.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 3D triangle / design-plane map
# ---------------------------------------------------------------------------

def closest_point_on_triangle_3d(p, a, b, c):
    """
    Closest point to p on triangle abc (Ericson).

    Returns (q, u, v, w) with q = u*a + v*b + w*c, u+v+w = 1, u,v,w >= 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    p = np.asarray(p, dtype=float)

    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a, 1.0, 0.0, 0.0

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b, 0.0, 1.0, 0.0

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3) if abs(d1 - d3) > 1e-18 else 0.0
        return a + v * ab, 1.0 - v, v, 0.0

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c, 0.0, 0.0, 1.0

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6) if abs(d2 - d6) > 1e-18 else 0.0
        return a + w * ac, 1.0 - w, 0.0, w

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom_bc = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / denom_bc if abs(denom_bc) > 1e-18 else 0.0
        return b + w * (c - b), 0.0, 1.0 - w, w

    denom = va + vb + vc
    if abs(denom) < 1e-18:
        return a, 1.0, 0.0, 0.0
    v = vb / denom
    w = vc / denom
    u = 1.0 - v - w
    return a + ab * v + ac * w, u, v, w


def _pad3(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.size >= 3:
        return v[:3]
    out = np.zeros(3, dtype=float)
    out[: v.size] = v
    return out


def map_point_3d_to_flat_2d(p3d, verts_3d, verts_flat_xy, return_bary=False):
    """
    Map a 3D point on (or near) a triangle to flat design 2D coords.

    Affine map: deformed triangle verts_3d → rest triangle verts_flat_xy.
    Returns [x, y], or None if degenerate.
    If return_bary=True, returns ([x,y], (u,v,w)) or (None, None).
    """
    a = np.asarray(verts_3d[0], dtype=float).reshape(3)[:3]
    b = np.asarray(verts_3d[1], dtype=float).reshape(3)[:3]
    c = np.asarray(verts_3d[2], dtype=float).reshape(3)[:3]
    p = np.asarray(p3d, dtype=float).reshape(-1)[:3]
    if a.shape[0] < 3 or p.shape[0] < 3:
        a, b, c, p = _pad3(a), _pad3(b), _pad3(c), _pad3(p)

    n = np.cross(b - a, c - a)
    if float(np.linalg.norm(n)) < 1e-14:
        return (None, None) if return_bary else None

    q, u, v, w = closest_point_on_triangle_3d(p, a, b, c)
    if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(w)):
        return (None, None) if return_bary else None

    f0 = np.asarray(verts_flat_xy[0], dtype=float).reshape(-1)[:2]
    f1 = np.asarray(verts_flat_xy[1], dtype=float).reshape(-1)[:2]
    f2 = np.asarray(verts_flat_xy[2], dtype=float).reshape(-1)[:2]
    xy = u * f0 + v * f1 + w * f2
    if not np.all(np.isfinite(xy)):
        return (None, None) if return_bary else None
    out = [float(xy[0]), float(xy[1])]
    if return_bary:
        return out, (float(u), float(v), float(w))
    return out


def map_points_on_tris_batch(points, tri_kp, positions, flat_kps):
    """
    Batch map N contact points on triangles → design xy + barycentric loc.

    Parameters
    ----------
    points : (N, 3) float
    tri_kp : (N, 3) int keypoint indices
    positions : (V, 3) live verts
    flat_kps : (V, >=2) design-plane coords

    Returns
    -------
    xy : (N, 2) float64  (NaN row if degenerate)
    loc : (N, 6) float64  columns = i0,i1,i2,u,v,w
    ok  : (N,) bool
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    n = int(pts.shape[0])
    if n == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 6), dtype=np.float64),
            np.zeros(0, dtype=bool),
        )
    tk = np.asarray(tri_kp, dtype=np.intp).reshape(n, 3)
    pos = np.asarray(positions, dtype=np.float64)
    flat = np.asarray(flat_kps, dtype=np.float64)

    i0, i1, i2 = tk[:, 0], tk[:, 1], tk[:, 2]
    a, b, c = pos[i0, :3], pos[i1, :3], pos[i2, :3]
    v0 = b - a
    v1 = c - a
    v2 = pts[:, :3] - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    ok = np.abs(denom) >= 1e-18
    denom_safe = np.where(ok, denom, 1.0)
    v = (d11 * d20 - d01 * d21) / denom_safe
    w = (d00 * d21 - d01 * d20) / denom_safe
    u = 1.0 - v - w

    xy = u[:, None] * flat[i0, :2] + v[:, None] * flat[i1, :2] + w[:, None] * flat[i2, :2]
    loc = np.empty((n, 6), dtype=np.float64)
    loc[:, 0] = i0
    loc[:, 1] = i1
    loc[:, 2] = i2
    loc[:, 3] = u
    loc[:, 4] = v
    loc[:, 5] = w
    bad = ~ok | ~np.isfinite(xy).all(axis=1)
    if np.any(bad):
        xy = xy.copy()
        loc = loc.copy()
        xy[bad] = np.nan
        loc[bad] = np.nan
        ok = ~bad
    return xy, loc, ok


def eval_bary_batch(locs, positions, lift=0.0):
    """Vectorized bary eval: locs (N,6) → (N, 3) float64."""
    n = len(locs)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    L = np.asarray(locs, dtype=np.float64).reshape(n, -1)
    i0 = L[:, 0].astype(np.intp)
    i1 = L[:, 1].astype(np.intp)
    i2 = L[:, 2].astype(np.intp)
    u, v, w = L[:, 3:4], L[:, 4:5], L[:, 5:6]
    pos = np.asarray(positions)
    a, b, c = pos[i0, :3], pos[i1, :3], pos[i2, :3]
    p = u * a + v * b + w * c
    if lift != 0.0:
        nrm = np.cross(b - a, c - a)
        nn = np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        p = p + (float(lift) / nn) * nrm
    return p


def copy_loc(loc):
    """Copy barycentric handle (i0,i1,i2,u,v,w)."""
    if loc is None:
        return None
    try:
        return (
            int(loc[0]), int(loc[1]), int(loc[2]),
            float(loc[3]), float(loc[4]), float(loc[5]),
        )
    except (TypeError, ValueError, IndexError):
        return None


def seg_mid3(p0_3d, p1_3d):
    """Midpoint of a 3D segment, or None."""
    if p0_3d is None or p1_3d is None:
        return None
    try:
        return [
            0.5 * (float(p0_3d[0]) + float(p1_3d[0])),
            0.5 * (float(p0_3d[1]) + float(p1_3d[1])),
            0.5 * (float(p0_3d[2]) + float(p1_3d[2])),
        ]
    except (TypeError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Contact identity / 2D sweep samples
# ---------------------------------------------------------------------------

def contact_pair_key(entry: dict) -> Tuple[int, int, int, int, int]:
    """Stable key for one triangle–triangle contact (or unit-pair fallback)."""
    ua = int(entry["unit_a"])
    ub = int(entry["unit_b"])
    layer = int(entry["layer_idx"])
    tri_a = entry.get("tri_a", entry.get("tri_lo", -1))
    tri_b = entry.get("tri_b", entry.get("tri_hi", -1))
    try:
        tri_a = int(tri_a) if tri_a is not None else -1
        tri_b = int(tri_b) if tri_b is not None else -1
    except (TypeError, ValueError):
        tri_a, tri_b = -1, -1
    if tri_a >= 0 and tri_b >= 0:
        t0, t1 = (tri_a, tri_b) if tri_a < tri_b else (tri_b, tri_a)
        return (ua, ub, layer, t0, t1)
    return (ua, ub, layer, -1, -1)


def panel_group_key(entry: dict) -> Tuple[int, int]:
    """Two JSON panels that collide = one group."""
    pa = int(entry.get("panel_a", -1))
    pb = int(entry.get("panel_b", -1))
    return (min(pa, pb), max(pa, pb))


def order_segment_endpoints_2d(p0, p1, ref0=None, ref1=None):
    """
    Return (p0, p1) as lists; swap if needed so endpoints track ref0/ref1
    (avoids bow-tie when chaining sweep samples).
    """
    a = [float(p0[0]), float(p0[1])]
    b = [float(p1[0]), float(p1[1])]
    if ref0 is None or ref1 is None:
        return a, b
    r0x, r0y = float(ref0[0]), float(ref0[1])
    r1x, r1y = float(ref1[0]), float(ref1[1])
    cost_keep = (
        ((a[0] - r0x) ** 2 + (a[1] - r0y) ** 2) ** 0.5
        + ((b[0] - r1x) ** 2 + (b[1] - r1y) ** 2) ** 0.5
    )
    cost_swap = (
        ((a[0] - r1x) ** 2 + (a[1] - r1y) ** 2) ** 0.5
        + ((b[0] - r0x) ** 2 + (b[1] - r0y) ** 2) ** 0.5
    )
    return (b, a) if cost_swap < cost_keep else (a, b)


def segment_mid_dist_2d(a0, a1, b0, b1) -> float:
    """Distance between midpoints of two 2D segments."""
    m0x = 0.5 * (float(a0[0]) + float(a1[0]))
    m0y = 0.5 * (float(a0[1]) + float(a1[1]))
    m1x = 0.5 * (float(b0[0]) + float(b1[0]))
    m1y = 0.5 * (float(b0[1]) + float(b1[1]))
    return ((m1x - m0x) ** 2 + (m1y - m0y) ** 2) ** 0.5


def append_sweep_sample_2d(side: Optional[dict], angle: float, min_move: float = 0.15):
    """
    Append current side p0–p1 to a 2D sweep trail.

    Seeds with crease-snapped first_* when present; skips samples that barely
    moved; stabilizes endpoint order against the previous sample.
    """
    if side is None or "p0" not in side or "p1" not in side:
        return
    p0 = [float(side["p0"][0]), float(side["p0"][1])]
    p1 = [float(side["p1"][0]), float(side["p1"][1])]
    if (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 < 1e-24:
        return

    samples = side.get("sweep_samples")
    if samples is None:
        samples = []
        side["sweep_samples"] = samples

    if not samples and "first_p0" in side and "first_p1" in side:
        f0 = [float(side["first_p0"][0]), float(side["first_p0"][1])]
        f1 = [float(side["first_p1"][0]), float(side["first_p1"][1])]
        if (f1[0] - f0[0]) ** 2 + (f1[1] - f0[1]) ** 2 >= 1e-24:
            samples.append({
                "angle": float(side.get("first_folding_angle", angle)),
                "p0": f0,
                "p1": f1,
            })

    if samples:
        last = samples[-1]
        p0, p1 = order_segment_endpoints_2d(p0, p1, last["p0"], last["p1"])
        moved = segment_mid_dist_2d(last["p0"], last["p1"], p0, p1)
        end_move = max(
            ((p0[0] - last["p0"][0]) ** 2 + (p0[1] - last["p0"][1]) ** 2) ** 0.5,
            ((p1[0] - last["p1"][0]) ** 2 + (p1[1] - last["p1"][1]) ** 2) ** 0.5,
        )
        mm = float(min_move)
        if moved < mm and end_move < mm:
            if moved < 1e-9:
                return
            samples[-1] = {"angle": float(angle), "p0": p0, "p1": p1}
            return

    samples.append({"angle": float(angle), "p0": p0, "p1": p1})


# ---------------------------------------------------------------------------
# Dual-curve shaded regions
# ---------------------------------------------------------------------------

def xy2(p) -> List[float]:
    return [float(p[0]), float(p[1])]


def polygon_area_2d(pts) -> float:
    """Absolute shoelace area of a 2D ring."""
    n = len(pts) if pts is not None else 0
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x0, y0 = float(pts[i][0]), float(pts[i][1])
        x1, y1 = float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1])
        acc += x0 * y1 - x1 * y0
    return abs(acc) * 0.5


def pack_dual_curves(samples: Sequence[dict]):
    """Pack {p0,p1,angle} samples into (theta, c0, c1) float lists."""
    theta, c0, c1 = [], [], []
    prev0 = prev1 = None
    if not samples:
        return theta, c0, c1
    for s in samples:
        if s is None or "p0" not in s or "p1" not in s:
            continue
        p0, p1 = order_segment_endpoints_2d(s["p0"], s["p1"], prev0, prev1)
        a, b = xy2(p0), xy2(p1)
        theta.append(float(s.get("angle", 0.0)))
        c0.append(a)
        c1.append(b)
        prev0, prev1 = a, b
    return theta, c0, c1


def ribbon_ring_from_curves(c0, c1):
    """Closed shaded ring: walk c0 then reverse(c1). None if degenerate."""
    if not c0 or not c1 or len(c0) != len(c1) or len(c0) < 2:
        return None
    ring = list(c0)
    for q in reversed(c1):
        ring.append(q)
    cleaned = [ring[0]]
    for q in ring[1:]:
        prev = cleaned[-1]
        if abs(q[0] - prev[0]) > 1e-9 or abs(q[1] - prev[1]) > 1e-9:
            cleaned.append(q)
    if len(cleaned) < 3 or polygon_area_2d(cleaned) < 1e-12:
        return None
    return cleaned


def sweep_paint_polygon_2d(samples):
    """Ribbon polygon from raw sample list (compat helper)."""
    _, c0, c1 = pack_dual_curves(samples)
    return ribbon_ring_from_curves(c0, c1)


# ---------------------------------------------------------------------------
# Contact / side dict builders (shared by physical / ghost / side paths)
# ---------------------------------------------------------------------------

def stack_depth_fields(stock_span: Optional[Tuple[float, float]], layer_h: float) -> dict:
    """Depth of a shell height within a panel stock span (zmin, zmax)."""
    if not stock_span:
        return {
            "stock_span_mm": 0.0,
            "depth_from_top_mm": 0.0,
            "depth_from_bottom_mm": 0.0,
        }
    zmin, zmax = float(stock_span[0]), float(stock_span[1])
    stock = max(zmax - zmin, 0.0)
    lh = float(layer_h)
    return {
        "stock_span_mm": float(stock),
        "depth_from_top_mm": float(max(0.0, zmax - lh)),
        "depth_from_bottom_mm": float(max(0.0, lh - zmin)),
    }


def make_side_dict(
    unit,
    panel,
    layer_idx,
    layer_h,
    shell_kind,
    *,
    xy=None,
    loc=None,
    xy0=None,
    loc0=None,
    xy1=None,
    loc1=None,
    parent_panel=None,
    h_lo=None,
    h_hi=None,
    stock_span=None,
) -> Optional[dict]:
    """Build one side_a / side_b contact dict (point or segment form)."""
    side: Dict[str, Any] = {
        "unit": int(unit),
        "panel": int(panel),
        "layer_idx": int(layer_idx),
        "layer_h": float(layer_h),
        "shell_kind": shell_kind,
    }
    if parent_panel is not None:
        side["parent_panel"] = int(parent_panel)
    if h_lo is not None:
        side["h_lo"] = float(h_lo)
    if h_hi is not None:
        side["h_hi"] = float(h_hi)

    if xy0 is not None and xy1 is not None and loc0 is not None and loc1 is not None:
        side["p0"] = [float(xy0[0]), float(xy0[1])]
        side["p1"] = [float(xy1[0]), float(xy1[1])]
        side["loc0"] = (
            int(loc0[0]), int(loc0[1]), int(loc0[2]),
            float(loc0[3]), float(loc0[4]), float(loc0[5]),
        )
        side["loc1"] = (
            int(loc1[0]), int(loc1[1]), int(loc1[2]),
            float(loc1[3]), float(loc1[4]), float(loc1[5]),
        )
    elif xy is not None and loc is not None:
        side["p"] = [float(xy[0]), float(xy[1])]
        side["loc"] = (
            int(loc[0]), int(loc[1]), int(loc[2]),
            float(loc[3]), float(loc[4]), float(loc[5]),
        )
    else:
        return None

    side.update(stack_depth_fields(stock_span, float(layer_h)))
    return side


def make_contact_entry(
    *,
    unit_a,
    unit_b,
    panel_a,
    panel_b,
    layer_idx,
    layer_h_a,
    layer_h_b,
    layer_idx_a,
    layer_idx_b,
    tri_a,
    tri_b,
    shell_kind,
    shell_kind_a=None,
    shell_kind_b=None,
    parent_panel_a=None,
    parent_panel_b=None,
    stock_span=None,
    **extra,
) -> dict:
    """Shared metadata dict for a contact point or segment."""
    p_lo = min(int(panel_a), int(panel_b))
    p_hi = max(int(panel_a), int(panel_b))
    ent = {
        "unit_a": int(unit_a),
        "unit_b": int(unit_b),
        "panel_of_unit_a": int(panel_a),
        "panel_of_unit_b": int(panel_b),
        "panel_a": p_lo,
        "panel_b": p_hi,
        "layer_idx": int(layer_idx),
        "layer_h": float(layer_h_a),
        "layer_idx_a": int(layer_idx_a),
        "layer_h_a": float(layer_h_a),
        "layer_idx_b": int(layer_idx_b),
        "layer_h_b": float(layer_h_b),
        "tri_a": int(tri_a),
        "tri_b": int(tri_b),
        "tri_lo": int(tri_a),
        "tri_hi": int(tri_b),
        "shell_kind": shell_kind,
        "shell_kind_a": shell_kind_a or shell_kind,
        "shell_kind_b": shell_kind_b or shell_kind,
        **stack_depth_fields(stock_span, float(layer_h_a)),
    }
    if parent_panel_a is not None:
        ent["parent_panel_a"] = int(parent_panel_a)
    if parent_panel_b is not None:
        ent["parent_panel_b"] = int(parent_panel_b)
    ent.update(extra)
    return ent


# ---------------------------------------------------------------------------
# Paint trail SoA buffers
# ---------------------------------------------------------------------------

def paint_buf_new(cap=64, *, is_side=False) -> dict:
    """Allocate one paint trail buffer (physical unit or side panel)."""
    cap = int(max(cap, 16))
    buf = {
        "n": 0,
        "cap": cap,
        "is_side": bool(is_side),
        "loc0": np.zeros((cap, 6), dtype=np.float64),
        "loc1": np.zeros((cap, 6), dtype=np.float64),
        "p0": np.zeros((cap, 2), dtype=np.float64),
        "p1": np.zeros((cap, 2), dtype=np.float64),
    }
    if is_side:
        buf["mid3"] = np.zeros((cap, 3), dtype=np.float64)
    return buf


def paint_buf_grow(buf: dict, need_n: int) -> dict:
    """Grow trail buffer if need_n exceeds capacity."""
    need_n = int(need_n)
    if need_n <= int(buf["cap"]):
        return buf
    new_cap = max(need_n, int(buf["cap"]) * 2)
    new_cap = ((new_cap + 63) // 64) * 64
    n = int(buf["n"])
    nb = paint_buf_new(new_cap, is_side=bool(buf.get("is_side")))
    if n > 0:
        nb["loc0"][:n] = buf["loc0"][:n]
        nb["loc1"][:n] = buf["loc1"][:n]
        nb["p0"][:n] = buf["p0"][:n]
        nb["p1"][:n] = buf["p1"][:n]
        if "mid3" in buf and "mid3" in nb:
            nb["mid3"][:n] = buf["mid3"][:n]
        nb["n"] = n
    return nb


def paint_total_strokes(by_unit: Optional[dict]) -> int:
    by_u = by_unit or {}
    return int(sum(int(b.get("n", 0) or 0) for b in by_u.values()))


def pack_unit_locus_lines(pts0, pts1, max_link=None, side_mode=False):
    """
    Vectorized line packing for one unit trail → (M, 3) float32 interleaved
    endpoints (path along p0, path along p1, cross p0[i]–p1[i]).
    side_mode: midpoint polyline only (vertical walls).
    """
    n = int(pts0.shape[0])
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    good0 = np.isfinite(pts0).all(axis=1)
    good1 = np.isfinite(pts1).all(axis=1)
    chunks = []
    max_link_sq = (
        None if max_link is None or float(max_link) <= 0.0
        else float(max_link) * float(max_link)
    )

    def _path_segments(pts, good):
        if n < 2:
            return None
        link = good[:-1] & good[1:]
        if max_link_sq is not None:
            d = pts[1:] - pts[:-1]
            d2 = np.einsum("ij,ij->i", d, d)
            link = link & (d2 <= max_link_sq)
        n_link = int(np.count_nonzero(link))
        if n_link <= 0:
            return None
        a = pts[:-1][link]
        b = pts[1:][link]
        out = np.empty((n_link * 2, 3), dtype=np.float32)
        out[0::2] = a.astype(np.float32, copy=False)
        out[1::2] = b.astype(np.float32, copy=False)
        return out

    if side_mode:
        mid = 0.5 * (pts0 + pts1)
        good = good0 & good1
        p_mid = _path_segments(mid, good)
        if p_mid is not None:
            chunks.append(p_mid)
    else:
        p0_path = _path_segments(pts0, good0)
        if p0_path is not None:
            chunks.append(p0_path)
        p1_path = _path_segments(pts1, good1)
        if p1_path is not None:
            chunks.append(p1_path)
        both = good0 & good1
        if max_link_sq is not None and np.any(both):
            d = pts1 - pts0
            d2 = np.einsum("ij,ij->i", d, d)
            both = both & (d2 <= max_link_sq)
        n_cross = int(np.count_nonzero(both))
        if n_cross > 0:
            a = pts0[both]
            b = pts1[both]
            cross = np.empty((n_cross * 2, 3), dtype=np.float32)
            cross[0::2] = a.astype(np.float32, copy=False)
            cross[1::2] = b.astype(np.float32, copy=False)
            chunks.append(cross)

    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(chunks)
