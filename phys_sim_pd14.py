import taichi as ti
import taichi.math as tm
import json
import time, os
import yaml
import gc
from collections import defaultdict
from spatialhash import SpatialHash
from ori_sim_sys import *
from utils import triangle_intersection_contacts_3d, _triangle_aabb_overlap, _triangles_coplanar

data_type = ti.f64
numpy_data_type = np.float64
use_gpu = 0

if use_gpu:
    ti.init(arch=ti.gpu, default_fp=data_type, fast_math=False, advanced_optimization=False, kernel_profiler=True)
else:
    ti.init(arch=ti.cpu, default_fp=data_type, fast_math=False, advanced_optimization=False, cpu_max_num_threads=1) #, kernel_profiler=False, verbose=True, debug=True, gdb_trigger=True)

# ---------------------------------------------------------------------------
# Intruder classification + trim helpers (signed-distance slide test)
# ---------------------------------------------------------------------------

def _trim_panel_normal_3d(verts_3d):
    """Unit normal of a planar polygon (Newell method)."""
    n = len(verts_3d)
    normal = np.zeros(3)
    for i in range(n):
        a = verts_3d[i]
        b = verts_3d[(i + 1) % n]
        normal[0] += (a[1] - b[1]) * (a[2] + b[2])
        normal[1] += (a[2] - b[2]) * (a[0] + b[0])
        normal[2] += (a[0] - b[0]) * (a[1] + b[1])
    mag = np.linalg.norm(normal)
    if mag < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    return normal / mag


def _trim_orient_normals_toward_mid(n_A, C_A, n_B, C_B):
    """
    Flip Newell normals so each points toward the pair midpoint.

    Vertex winding is arbitrary after thick expand / PD, so raw normals can
    make the same physical dig-in look like "separation" on one face of a
    thick stack and "penetration" on the other.  Facing both toward the mid
    makes signed distances comparable across layers.
    """
    n_A = np.asarray(n_A, dtype=float).reshape(3).copy()
    n_B = np.asarray(n_B, dtype=float).reshape(3).copy()
    C_A = np.asarray(C_A, dtype=float).reshape(3)
    C_B = np.asarray(C_B, dtype=float).reshape(3)
    mid = 0.5 * (C_A + C_B)
    if float(np.dot(n_A, mid - C_A)) < 0.0:
        n_A = -n_A
    if float(np.dot(n_B, mid - C_B)) < 0.0:
        n_B = -n_B
    return n_A, n_B


def _trim_is_outer_face(panel_rec):
    """True for outer faces of a multi-layer thick panel (or any single layer)."""
    li = panel_rec.get("layer_idx")
    n = panel_rec.get("num_layers")
    if li is not None and n is not None:
        n = int(n)
        li = int(li)
        if n <= 1:
            return True
        return min(li, n - 1 - li) == 0
    # Fallback: far from design midplane
    return abs(float(panel_rec.get("layer_h", 0.0))) > 1e-9


def _trim_polygon_area_3d(verts_3d):
    n = len(verts_3d)
    if n < 3:
        return 0.0
    total = np.zeros(3)
    o = np.asarray(verts_3d[0], dtype=float)
    for i in range(1, n - 1):
        total += np.cross(
            np.asarray(verts_3d[i], dtype=float) - o,
            np.asarray(verts_3d[i + 1], dtype=float) - o,
        )
    return float(np.linalg.norm(total) * 0.5)


def _trim_outer_score(layer_h=None, layer_idx=None, num_layers=None):
    """
    Higher = more outer / thicker-face priority for intruder selection.

    - |layer_h|: distance from design midplane (outer faces sit farther out)
    - stack extremity: layer 0 and last layer of a multi-layer thick panel
      are outer; middle layers are inner
    """
    score = 0.0
    if layer_h is not None:
        score += abs(float(layer_h))
    if layer_idx is not None and num_layers is not None:
        n = int(num_layers)
        li = int(layer_idx)
        if n > 1:
            # 0 at both outer faces; grows toward the stack interior
            extremity = min(li, n - 1 - li)
            is_outer_face = extremity == 0
            # Strong preference for outer faces of a thick stack
            score += 1.0e6 if is_outer_face else 0.0
            # Mild preference for panels that sit farther from the stack mid
            score += float(n - 1 - extremity) * 1.0e-3
        elif n == 1:
            # Single-layer panel is its own outer surface
            score += 1.0e6
    return score


def _trim_prefer_outer(layer_h_A=None, layer_h_B=None,
                       layer_idx_A=None, layer_idx_B=None,
                       num_layers_A=None, num_layers_B=None,
                       eps=1e-9):
    """
    Always prioritize the outer (thicker-face) panel when outerness differs.
    Returns 'A', 'B', or None if no preference.
    """
    sA = _trim_outer_score(layer_h_A, layer_idx_A, num_layers_A)
    sB = _trim_outer_score(layer_h_B, layer_idx_B, num_layers_B)
    if sA > sB + eps:
        return "A"
    if sB > sA + eps:
        return "B"
    return None


def _trim_classify_intruder(
    n_A, C_A, p_A, n_B, C_B, p_B,
    area_A=None, area_B=None, threshold=0.0,
    layer_h_A=None, layer_h_B=None,
    layer_idx_A=None, layer_idx_B=None,
    num_layers_A=None, num_layers_B=None,
    orient_normals=True,
):
    """
    Signed distance test (slide):
      d_A->B = n_B · (C_A - p_B)
      d_B->A = n_A · (C_B - p_A)

    Returns 'A', 'B', or None.
    Requires at least one-sided penetration (d < -T). Outer / thicker-face
    preference only applies once penetration is detected (does not invent
    intruders from layer meta alone). Callers that already saw a geometric
    collision may force-pick via ``_trim_pick_intruder_for_pair(..., force=True)``.

    orient_normals: flip both normals toward the pair midpoint so thick-stack
    winding polarity does not systematically zero out one face of the stack.
    """
    n_A = np.asarray(n_A, dtype=float).reshape(3)
    n_B = np.asarray(n_B, dtype=float).reshape(3)
    C_A = np.asarray(C_A, dtype=float).reshape(3)
    C_B = np.asarray(C_B, dtype=float).reshape(3)
    p_A = np.asarray(p_A, dtype=float).reshape(3)
    p_B = np.asarray(p_B, dtype=float).reshape(3)
    if orient_normals:
        n_A, n_B = _trim_orient_normals_toward_mid(n_A, C_A, n_B, C_B)

    d_AtoB = float(np.dot(n_B, C_A - p_B))
    d_BtoA = float(np.dot(n_A, C_B - p_A))
    T = float(threshold)

    A_in_B = d_AtoB < -T
    B_in_A = d_BtoA < -T

    # No penetration → no intruder
    if not A_in_B and not B_in_A:
        return None, d_AtoB, d_BtoA

    # Penetration exists → outer / thicker face wins when outerness differs
    outer = _trim_prefer_outer(
        layer_h_A, layer_h_B, layer_idx_A, layer_idx_B, num_layers_A, num_layers_B,
    )
    if outer is not None:
        return outer, d_AtoB, d_BtoA

    if A_in_B and not B_in_A:
        return "A", d_AtoB, d_BtoA
    if B_in_A and not A_in_B:
        return "B", d_AtoB, d_BtoA
    # Both penetrate, outerness equal → smaller area
    aA = 0.0 if area_A is None else area_A
    aB = 0.0 if area_B is None else area_B
    return ("A" if aA < aB else "B"), d_AtoB, d_BtoA


def _trim_plane_plane_intersection(n1, p1, n2, p2):
    direction = np.cross(n1, n2)
    mag = np.linalg.norm(direction)
    if mag < 1e-10:
        return None
    direction = direction / mag
    d1 = float(np.dot(n1, p1))
    d2 = float(np.dot(n2, p2))
    A = np.array([n1, n2, direction], dtype=float)
    b = np.array([d1, d2, 0.0], dtype=float)
    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return point, direction


def _trim_closest_pt_on_seg_to_line(seg_a, seg_b, line_p, line_d):
    u = np.asarray(seg_b, dtype=float) - np.asarray(seg_a, dtype=float)
    v = np.asarray(line_d, dtype=float)
    w = np.asarray(seg_a, dtype=float) - np.asarray(line_p, dtype=float)
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b
    t = 0.0 if abs(denom) < 1e-12 else (b * e - c * d) / denom
    t = float(np.clip(t, 0.0, 1.0))
    return t, (np.asarray(seg_a, dtype=float) + t * u)


def _closest_point_on_triangle_3d(p, a, b, c):
    """
    Closest point to p on triangle abc (Ericson). Returns (q, u, v, w) with
    q = u*a + v*b + w*c and u+v+w = 1, u,v,w >= 0.
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


def _barycentric_2d(p, a, b, c):
    """
    Barycentric (u,v,w) of p in triangle abc (2D), with p = u*a + v*b + w*c.
    Returns (u,v,w) or None if the triangle is degenerate.
    """
    a = np.asarray(a, dtype=float).reshape(-1)[:2]
    b = np.asarray(b, dtype=float).reshape(-1)[:2]
    c = np.asarray(c, dtype=float).reshape(-1)[:2]
    p = np.asarray(p, dtype=float).reshape(-1)[:2]
    v0 = b - a
    v1 = c - a
    v2 = p - a
    den = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(den) < 1e-14:
        return None
    v = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    w = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    u = 1.0 - v - w
    return float(u), float(v), float(w)


def _map_point_flat_2d_to_3d(p_xy, verts_3d, verts_flat_xy, eps=1e-7):
    """
    Inverse of ``_map_point_3d_to_flat_2d`` for one triangle:
    design-plane point → 3D point on the deformed triangle (material map).

    Returns [x,y,z] if p_xy lies in the flat triangle (with small eps), else None.
    """
    f0 = np.asarray(verts_flat_xy[0], dtype=float).reshape(-1)[:2]
    f1 = np.asarray(verts_flat_xy[1], dtype=float).reshape(-1)[:2]
    f2 = np.asarray(verts_flat_xy[2], dtype=float).reshape(-1)[:2]
    bary = _barycentric_2d(p_xy, f0, f1, f2)
    if bary is None:
        return None
    u, v, w = bary
    if u < -eps or v < -eps or w < -eps:
        return None
    a = np.asarray(verts_3d[0], dtype=float).reshape(-1)[:3]
    b = np.asarray(verts_3d[1], dtype=float).reshape(-1)[:3]
    c = np.asarray(verts_3d[2], dtype=float).reshape(-1)[:3]
    if a.size < 3:
        def _pad3(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            out = np.zeros(3, dtype=float)
            out[: min(3, x.size)] = x[: min(3, x.size)]
            return out
        a, b, c = _pad3(a), _pad3(b), _pad3(c)
    p = u * a + v * b + w * c
    if not np.all(np.isfinite(p)):
        return None
    return [float(p[0]), float(p[1]), float(p[2])]


def _map_point_3d_to_flat_2d(p3d, verts_3d, verts_flat_xy, return_bary=False):
    """
    Map a 3D point on (or near) a triangle to flat design 2D coords.

    Exact for any point on the triangle under the standard piecewise-linear
    mesh assumption: the unique affine map sending the deformed triangle
    (verts_3d) to the rest/design triangle (verts_flat_xy).

    Steps:
      1. Closest point of p3d on the 3D triangle (handles slight off-plane /
         off-edge numerical noise from intersection tests).
      2. Barycentric (u,v,w) of that closest point.
      3. flat_xy = u*f0 + v*f1 + w*f2  (same vertex order as verts_3d).

    Returns [x, y], or None if the triangle is degenerate.
    If return_bary=True, returns ([x,y], (u,v,w)) or (None, None).
    """
    a = np.asarray(verts_3d[0], dtype=float).reshape(3)[:3]
    b = np.asarray(verts_3d[1], dtype=float).reshape(3)[:3]
    c = np.asarray(verts_3d[2], dtype=float).reshape(3)[:3]
    p = np.asarray(p3d, dtype=float).reshape(-1)[:3]
    if a.shape[0] < 3 or p.shape[0] < 3:
        # pad 2d → 3d
        def _pad3(v):
            v = np.asarray(v, dtype=float).reshape(-1)
            if v.size >= 3:
                return v[:3]
            out = np.zeros(3, dtype=float)
            out[: v.size] = v
            return out
        a, b, c, p = _pad3(a), _pad3(b), _pad3(c), _pad3(p)

    # Degenerate if edges are parallel / zero area
    n = np.cross(b - a, c - a)
    n_mag = float(np.linalg.norm(n))
    if n_mag < 1e-14:
        if return_bary:
            return None, None
        return None

    q, u, v, w = _closest_point_on_triangle_3d(p, a, b, c)
    if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(w)):
        if return_bary:
            return None, None
        return None

    f0 = np.asarray(verts_flat_xy[0], dtype=float).reshape(-1)[:2]
    f1 = np.asarray(verts_flat_xy[1], dtype=float).reshape(-1)[:2]
    f2 = np.asarray(verts_flat_xy[2], dtype=float).reshape(-1)[:2]
    xy = u * f0 + v * f1 + w * f2
    if not np.all(np.isfinite(xy)):
        if return_bary:
            return None, None
        return None
    out = [float(xy[0]), float(xy[1])]
    if return_bary:
        return out, (float(u), float(v), float(w))
    return out


def _verify_map_point_3d_to_flat_2d(tol=1e-9):
    """
    Self-check for 3D→flat mapping. Returns (ok: bool, messages: list[str]).
    Used by tests / debug; does not touch simulation state.
    """
    msgs = []
    ok = True

    def _check(name, got, expected):
        nonlocal ok
        g = np.asarray(got, dtype=float)
        e = np.asarray(expected, dtype=float)
        err = float(np.linalg.norm(g - e))
        if err > tol:
            ok = False
            msgs.append(f"FAIL {name}: err={err:g} got={g.tolist()} exp={e.tolist()}")
        else:
            msgs.append(f"ok   {name}: err={err:g}")

    verts3 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    flat = np.array([[10.0, 20.0], [30.0, 20.0], [10.0, 40.0]])
    for i, lab in enumerate("ABC"):
        out = _map_point_3d_to_flat_2d(verts3[i], verts3, flat)
        _check(f"vertex {lab}", out, flat[i])

    mid = 0.5 * (verts3[0] + verts3[1])
    _check("mid AB", _map_point_3d_to_flat_2d(mid, verts3, flat), 0.5 * (flat[0] + flat[1]))
    _check("centroid", _map_point_3d_to_flat_2d(verts3.mean(0), verts3, flat), flat.mean(0))

    # Rigid motion of the 3D triangle must not change material (flat) coords
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([5.0, -3.0, 1.0])
    verts3r = verts3 @ R.T + t
    mid_r = 0.5 * (verts3r[0] + verts3r[1])
    _check("rigid mid AB", _map_point_3d_to_flat_2d(mid_r, verts3r, flat), 0.5 * (flat[0] + flat[1]))

    # Off-plane → same as on-plane projection / closest point
    mid_off = mid + np.array([0.0, 0.0, 0.7])
    _check("off-plane mid AB", _map_point_3d_to_flat_2d(mid_off, verts3, flat), 0.5 * (flat[0] + flat[1]))

    # Known barycentric (0.2, 0.3, 0.5)
    p3 = 0.2 * verts3[0] + 0.3 * verts3[1] + 0.5 * verts3[2]
    exp = 0.2 * flat[0] + 0.3 * flat[1] + 0.5 * flat[2]
    out, bary = _map_point_3d_to_flat_2d(p3, verts3, flat, return_bary=True)
    _check("bary 0.2/0.3/0.5", out, exp)
    if abs(bary[0] - 0.2) > 1e-9 or abs(bary[1] - 0.3) > 1e-9:
        ok = False
        msgs.append(f"FAIL bary weights: {bary}")

    # Outside → clamped to closest on triangle (vertex C for far point near C side)
    outside = np.array([-1.0, -1.0, 0.0])
    out_o, bary_o = _map_point_3d_to_flat_2d(outside, verts3, flat, return_bary=True)
    if out_o is None or min(bary_o) < -1e-9:
        ok = False
        msgs.append(f"FAIL outside clamp bary={bary_o}")
    else:
        msgs.append(f"ok   outside→closest bary={bary_o}")

    # Degenerate triangle
    degen = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    if _map_point_3d_to_flat_2d([0.5, 0.0, 0.0], degen, flat) is not None:
        ok = False
        msgs.append("FAIL degenerate should return None")
    else:
        msgs.append("ok   degenerate → None")

    return ok, msgs


def _trim_clip_polygon_by_line_2d(poly_2d, pt1_2d, pt2_2d, keep_point_2d):
    pt1 = np.array(pt1_2d[:2], dtype=float)
    pt2 = np.array(pt2_2d[:2], dtype=float)
    cut_dir = pt2 - pt1
    cut_len = float(np.linalg.norm(cut_dir))
    if cut_len < 1e-10:
        return poly_2d
    normal_2d = np.array([-cut_dir[1], cut_dir[0]]) / cut_len
    keep_raw = float(np.dot(normal_2d, np.array(keep_point_2d[:2], dtype=float) - pt1))
    keep_sign = 1.0 if keep_raw >= 0 else -1.0
    n = len(poly_2d)
    verts = [np.array([float(v[0]), float(v[1])], dtype=float) for v in poly_2d]
    dists = [float(np.dot(normal_2d, v - pt1) * keep_sign) for v in verts]
    new_poly = []
    for i in range(n):
        j = (i + 1) % n
        vi, vj = verts[i], verts[j]
        di, dj = dists[i], dists[j]
        if di >= 0:
            new_poly.append([float(vi[0]), float(vi[1]), 0.0])
        if (di > 0 and dj < 0) or (di < 0 and dj > 0):
            t = di / (di - dj)
            xi = vi + t * (vj - vi)
            new_poly.append([float(xi[0]), float(xi[1]), 0.0])
    return new_poly if len(new_poly) >= 3 else poly_2d


def _trim_xy_key(pt):
    return (round(float(pt[0]), 3), round(float(pt[1]), 3))


def _trim_h_str(h):
    return str(round(float(h), 6))


def _trim_find_orig_unit(flat_xy_np, units_list):
    target = set(_trim_xy_key(pt) for pt in flat_xy_np)
    for u_idx, unit in enumerate(units_list):
        if set(_trim_xy_key(pt) for pt in unit) == target:
            return u_idx
    return -1


def _trim_build_panels(
    x_np, panel_indices_list, panel_crease_types_list,
    initial_kps_np=None, unit_layer_meta=None,
):
    panels = []
    x_np = np.asarray(x_np, dtype=float)
    init = None if initial_kps_np is None else np.asarray(initial_kps_np, dtype=float)
    meta = unit_layer_meta or {}
    for i, raw_idx in enumerate(panel_indices_list):
        idx = [j for j in raw_idx if j >= 0]
        if len(idx) < 3:
            continue
        ct = list(panel_crease_types_list[i])[: len(idx)]
        verts_3d = x_np[idx]
        m = meta.get(i, {})
        if init is not None:
            flat_xy = init[idx][:, :2]
            layer_h = float(init[idx][0][2]) if init.shape[1] > 2 else 0.0
        else:
            flat_xy = verts_3d[:, :2]
            layer_h = float(m.get("layer_h", m.get("height_z", 0.0)))
        if "layer_h" in m or "height_z" in m:
            layer_h = float(m.get("layer_h", m.get("height_z", layer_h)))
        centroid = verts_3d.mean(axis=0)
        panels.append({
            "sim_unit": i,
            "vertex_idx": idx,
            "verts_3d": verts_3d,
            "flat_xy": flat_xy,
            "layer_h": layer_h,
            "layer_idx": m.get("layer_idx"),
            "num_layers": m.get("num_layers"),
            "centroid": centroid,
            # Plane reference = centroid (more stable than verts[0])
            "plane_point": centroid,
            "normal": _trim_panel_normal_3d(verts_3d),
            "crease_types": ct,
            "area": _trim_polygon_area_3d(verts_3d),
        })
    return panels


def _trim_fold_edge_map(panels):
    fold_edge_map = {}
    for pi, p in enumerate(panels):
        n = len(p["vertex_idx"])
        for j in range(n):
            if p["crease_types"][j] not in (MOUNTAIN, VALLEY):
                continue
            v1 = p["vertex_idx"][j]
            v2 = p["vertex_idx"][(j + 1) % n]
            key = (min(v1, v2), max(v1, v2))
            fold_edge_map.setdefault(key, []).append(pi)
    return fold_edge_map


def _trim_panels_by_sim_unit(panels):
    """sim_unit id -> panel record (panels list is 1:1 with sim units that have >=3 verts)."""
    return {int(p["sim_unit"]): p for p in panels}


def _trim_pick_intruder_for_pair(pA, pB, threshold=0.0, force=False):
    """
    Pick exactly one intruder between two panel records.
    force=True: always choose one (used for real geometric collisions / outer faces).
    Always prioritizes the outer (thicker-face) panel when outerness differs.
    Normals are re-oriented toward the pair midpoint before the slide test.
    """
    pA_pt = pA.get("plane_point", pA["centroid"])
    pB_pt = pB.get("plane_point", pB["centroid"])
    who, d_AtoB, d_BtoA = _trim_classify_intruder(
        pA["normal"], pA["centroid"], pA_pt,
        pB["normal"], pB["centroid"], pB_pt,
        area_A=pA["area"], area_B=pB["area"], threshold=threshold,
        layer_h_A=pA.get("layer_h"), layer_h_B=pB.get("layer_h"),
        layer_idx_A=pA.get("layer_idx"), layer_idx_B=pB.get("layer_idx"),
        num_layers_A=pA.get("num_layers"), num_layers_B=pB.get("num_layers"),
        orient_normals=True,
    )
    if who is None and force:
        # Outer still preferred if classify returned None without layer meta
        outer = _trim_prefer_outer(
            pA.get("layer_h"), pB.get("layer_h"),
            pA.get("layer_idx"), pB.get("layer_idx"),
            pA.get("num_layers"), pB.get("num_layers"),
        )
        if outer is not None:
            who = outer
        # Geometric contact / outer-face pair: deeper intrusion (more negative) wins
        elif d_AtoB < d_BtoA:
            who = "A"
        elif d_BtoA < d_AtoB:
            who = "B"
        else:
            who = "A" if pA["area"] <= pB["area"] else "B"
    return who, d_AtoB, d_BtoA


def _trim_oriented_normals_for_pair(pA, pB):
    """Return (nA, nB) flipped to face the pair midpoint."""
    return _trim_orient_normals_toward_mid(
        pA["normal"], pA["centroid"], pB["normal"], pB["centroid"],
    )


def _trim_apply_cut_for_pair(
    intruder, other, input_json, units_by_layer,
    min_cut_len=0.5, interior_t=(0.05, 0.95),
):
    """
    Plane–plane cut of ``intruder`` by ``other``; clip design polygon.
    Returns dict with cut metadata, or None if the cut could not be placed.
    """
    n_I, n_O = _trim_oriented_normals_for_pair(intruder, other)
    p_I = intruder.get("plane_point", intruder["centroid"])
    p_O = other.get("plane_point", other["centroid"])
    res = _trim_plane_plane_intersection(n_I, p_I, n_O, p_O)
    if res is None:
        return None
    line_pt, line_dir = res

    nI = len(intruder["vertex_idx"])
    edge_dists = []
    for j in range(nI):
        seg_a = intruder["verts_3d"][j]
        seg_b = intruder["verts_3d"][(j + 1) % nI]
        t, closest = _trim_closest_pt_on_seg_to_line(seg_a, seg_b, line_pt, line_dir)
        dist = float(np.linalg.norm(np.cross(closest - line_pt, line_dir)))
        edge_dists.append((dist, t, j))
    edge_dists.sort(key=lambda x: x[0])
    if len(edge_dists) < 2:
        return None

    t_lo, t_hi = interior_t
    good = [(d, t, j) for d, t, j in edge_dists if t_lo < t < t_hi]
    best_two = good[:2] if len(good) >= 2 else edge_dists[:2]
    if best_two[0][2] == best_two[1][2]:
        candidates = [e for e in edge_dists if e[2] != best_two[0][2]]
        if not candidates:
            return None
        best_two = [best_two[0], candidates[0]]

    flat_pts = []
    for _, t, j in best_two:
        v1f = np.asarray(intruder["flat_xy"][j], dtype=float)[:2]
        v2f = np.asarray(intruder["flat_xy"][(j + 1) % nI], dtype=float)[:2]
        flat_pts.append(v1f + t * (v2f - v1f))
    pt1_2d, pt2_2d = flat_pts[0], flat_pts[1]
    if float(np.linalg.norm(pt2_2d - pt1_2d)) < float(min_cut_len):
        return None

    orig_unit_idx = _trim_find_orig_unit(intruder["flat_xy"], input_json["units"])
    if orig_unit_idx < 0:
        return None

    layer_h = float(intruder["layer_h"])
    hs = _trim_h_str(layer_h)
    if hs not in units_by_layer:
        units_by_layer[hs] = deepcopy(input_json["units"])
    orig_poly = units_by_layer[hs][orig_unit_idx]
    centroid_2d = np.array([v[:2] for v in orig_poly], dtype=float).mean(axis=0)
    units_by_layer[hs][orig_unit_idx] = _trim_clip_polygon_by_line_2d(
        orig_poly, pt1_2d, pt2_2d, centroid_2d
    )
    return {
        "trimmed_unit": orig_unit_idx,
        "trimmed_layer_h": layer_h,
        "pt1_2d": pt1_2d,
        "pt2_2d": pt2_2d,
        "layer_h": layer_h,
        "hs": hs,
    }


def classify_colliding_panel_pairs(
    x_np, panel_indices_list, colliding_unit_pairs, threshold=0.0,
    unit_layer_meta=None,
):
    """
    Signed-distance classification on unit pairs that *actually collide*
    (triangle intersection). Exactly one intruder per pair.
    Always prioritizes outer (thicker-face) panels when outerness differs.

    colliding_unit_pairs: iterable of (sim_unit_a, sim_unit_b).
    unit_layer_meta: optional dict sim_unit_id -> {layer_h, layer_idx, num_layers}.
    """
    x_np = np.asarray(x_np, dtype=float)
    # Cache polygon data per sim unit
    cache = {}
    meta = unit_layer_meta or {}

    def panel_of(uid):
        if uid in cache:
            return cache[uid]
        if uid < 0 or uid >= len(panel_indices_list):
            cache[uid] = None
            return None
        idx = [j for j in panel_indices_list[uid] if j >= 0]
        if len(idx) < 3:
            cache[uid] = None
            return None
        verts = x_np[idx]
        m = meta.get(uid, {})
        centroid = verts.mean(axis=0)
        rec = {
            "sim_unit": uid,
            "verts_3d": verts,
            "centroid": centroid,
            "plane_point": centroid,
            "normal": _trim_panel_normal_3d(verts),
            "area": _trim_polygon_area_3d(verts),
            "layer_h": m.get("layer_h", m.get("height_z", 0.0)),
            "layer_idx": m.get("layer_idx"),
            "num_layers": m.get("num_layers"),
        }
        cache[uid] = rec
        return rec

    out = []
    processed = set()
    for ua, ub in colliding_unit_pairs:
        pair_key = (min(ua, ub), max(ua, ub))
        if pair_key in processed:
            continue
        processed.add(pair_key)
        pA, pB = panel_of(ua), panel_of(ub)
        if pA is None or pB is None:
            continue
        who, d_AtoB, d_BtoA = _trim_pick_intruder_for_pair(pA, pB, threshold=threshold, force=True)
        if who is None:
            continue
        # Exactly one of the two colliding panels
        if who == "A":
            intruder_u, other_u = pA["sim_unit"], pB["sim_unit"]
        else:
            intruder_u, other_u = pB["sim_unit"], pA["sim_unit"]
        out.append({
            "sim_unit_A": pA["sim_unit"],
            "sim_unit_B": pB["sim_unit"],
            "intruder_sim_unit": intruder_u,
            "other_sim_unit": other_u,
            "d_AtoB": d_AtoB,
            "d_BtoA": d_BtoA,
        })
    return out


def classify_adjacent_panel_pairs(
    x_np, panel_indices_list, panel_crease_types_list, threshold=0.0, unit_layer_meta=None,
    force_outer_faces=True,
):
    """Signed-distance classification on fold-adjacent panel pairs (trim pipeline)."""
    panels = _trim_build_panels(
        x_np, panel_indices_list, panel_crease_types_list, unit_layer_meta=unit_layer_meta,
    )
    fold_edge_map = _trim_fold_edge_map(panels)
    out = []
    processed = set()
    for adj in fold_edge_map.values():
        if len(adj) != 2:
            continue
        pi_A, pi_B = adj[0], adj[1]
        pair_key = (min(pi_A, pi_B), max(pi_A, pi_B))
        if pair_key in processed:
            continue
        processed.add(pair_key)
        pA, pB = panels[pi_A], panels[pi_B]
        force = bool(
            force_outer_faces
            and (_trim_is_outer_face(pA) or _trim_is_outer_face(pB))
        )
        who, d_AtoB, d_BtoA = _trim_pick_intruder_for_pair(
            pA, pB, threshold=threshold, force=force,
        )
        if who is None:
            continue
        intruder_pi = pi_A if who == "A" else pi_B
        other_pi = pi_B if who == "A" else pi_A
        out.append({
            "sim_unit_A": pA["sim_unit"],
            "sim_unit_B": pB["sim_unit"],
            "intruder_sim_unit": panels[intruder_pi]["sim_unit"],
            "other_sim_unit": panels[other_pi]["sim_unit"],
            "d_AtoB": d_AtoB,
            "d_BtoA": d_BtoA,
        })
    return out


def compute_trimmed_design_from_sim(
    input_json, x_np, initial_kps_np, panel_indices_list, panel_crease_types_list,
    threshold=0.0, unit_layer_meta=None,
    colliding_unit_pairs=None,
    force_outer_fold_pairs=True,
):
    """
    Classify intruders and clip intruder polygons; returns augmented JSON dict.

    Pair sources (union, each sim-unit pair processed once):
      1) ``colliding_unit_pairs`` — real triangle-intersection contacts
         (force=True, same as GUI red markers)
      2) fold-adjacent mountain/valley edges
         (force=True on outer thick faces so −h / +h polarity does not
         zero out a whole face of the stack; force=False for interior)

    Normals are oriented toward each pair midpoint before the slide test.
    """
    result = deepcopy(input_json)
    panels = _trim_build_panels(
        x_np, panel_indices_list, panel_crease_types_list,
        initial_kps_np=initial_kps_np, unit_layer_meta=unit_layer_meta,
    )
    by_unit = _trim_panels_by_sim_unit(panels)
    fold_edge_map = _trim_fold_edge_map(panels)

    all_hs = sorted(set(_trim_h_str(p["layer_h"]) for p in panels))
    units_by_layer = {h: deepcopy(input_json["units"]) for h in all_hs}
    new_lines, new_features = [], []
    processed = set()
    classifications = []
    n_cuts = 0
    n_from_collision = 0
    n_from_fold = 0

    def _process_pair(pA, pB, *, force, source):
        nonlocal n_cuts, n_from_collision, n_from_fold
        ua, ub = int(pA["sim_unit"]), int(pB["sim_unit"])
        pair_key = (min(ua, ub), max(ua, ub))
        if pair_key in processed:
            return
        processed.add(pair_key)

        who, d_AtoB, d_BtoA = _trim_pick_intruder_for_pair(
            pA, pB, threshold=threshold, force=force,
        )
        entry = {
            "sim_unit_A": ua,
            "sim_unit_B": ub,
            "d_AtoB": d_AtoB,
            "d_BtoA": d_BtoA,
            "intruder": who,
            "layer_h_A": pA["layer_h"],
            "layer_h_B": pB["layer_h"],
            "source": source,
            "force": bool(force),
        }
        classifications.append(entry)
        if who is None:
            return

        if who == "A":
            intruder, other = pA, pB
        else:
            intruder, other = pB, pA
        entry["intruder_sim_unit"] = int(intruder["sim_unit"])
        entry["other_sim_unit"] = int(other["sim_unit"])

        cut = _trim_apply_cut_for_pair(
            intruder, other, input_json, units_by_layer,
        )
        if cut is None:
            entry["cut_failed"] = True
            return

        layer_h = cut["layer_h"]
        pt1_2d, pt2_2d = cut["pt1_2d"], cut["pt2_2d"]
        new_lines.append([
            [float(pt1_2d[0]), float(pt1_2d[1]), layer_h],
            [float(pt2_2d[0]), float(pt2_2d[1]), layer_h],
        ])
        new_features.append({
            "type": BORDER,
            "level": 0,
            "coeff": 1.0,
            "recover_level": [],
            "recover_angle": [],
            "hard": False,
            "hard_angle": math.pi,
            "hard_angle_down": -math.pi,
            "thick_panel_height": layer_h,
        })
        n_cuts += 1
        if source == "collision":
            n_from_collision += 1
        else:
            n_from_fold += 1
        entry["trimmed_unit"] = cut["trimmed_unit"]
        entry["trimmed_layer_h"] = cut["trimmed_layer_h"]

    # --- 1) Real geometric collisions (GUI red markers) ---
    for ua, ub in (colliding_unit_pairs or []):
        pA = by_unit.get(int(ua))
        pB = by_unit.get(int(ub))
        if pA is None or pB is None:
            continue
        _process_pair(pA, pB, force=True, source="collision")

    # --- 2) Fold-adjacent mountain/valley pairs ---
    for adj in fold_edge_map.values():
        if len(adj) != 2:
            continue
        pA, pB = panels[adj[0]], panels[adj[1]]
        force = bool(
            force_outer_fold_pairs
            and (_trim_is_outer_face(pA) or _trim_is_outer_face(pB))
        )
        _process_pair(pA, pB, force=force, source="fold_adjacent")

    result["units_by_layer"] = units_by_layer
    result["lines"] = result["lines"] + new_lines
    result["line_features"] = result["line_features"] + new_features
    result["trim_3d_metadata"] = {
        "n_panels": len(panels),
        "n_pairs_checked": len(processed),
        "n_cuts": n_cuts,
        "n_cuts_from_collision": n_from_collision,
        "n_cuts_from_fold_adjacent": n_from_fold,
        "n_colliding_pairs_in": len(list(colliding_unit_pairs or [])),
        "force_outer_fold_pairs": bool(force_outer_fold_pairs),
    }
    result["intruder_classifications"] = classifications
    print(
        f"[Trimmer-3D] panels={len(panels)}, pairs checked={len(processed)}, "
        f"cuts={n_cuts} (collision={n_from_collision}, fold={n_from_fold})"
    )
    return result


@ti.data_oriented
class PD_Origami_Simulator:
    def __init__(self, origami_name, use_gui=True, fast=1, pd_local_time=1, pd_global_time=1, pd_iter_time=5, damping=0.975, material_type=1, ref_target=False, verbose=False, collision_shading=False):
        self.use_gui = use_gui
        self.collision_shading = collision_shading
        self._collision_contact_count = 0
        self._collision_segment_count = 0
        self._collision_contact_max = 4096
        # Cached contact geometry for GUI text (two nodes per intersection line)
        self._collision_contact_points_list = []     # 3D sim-space [x,y,z]
        self._collision_contact_segments_list = []   # 3D (p0, p1)
        # Flat design-space 2D coords (JSON unfolded xy) + layer + unit pair
        # Live frame lists: entries include p0/p1 (or p), layer_idx, layer_h, unit_a, unit_b
        self._collision_contact_points_flat = []
        self._collision_contact_segments_flat = []
        # Per-collision tracking: each panel-pair finishes at its own time.
        # key = (min_unit, max_unit, layer_idx) -> last-seen flat entry
        self._collision_active_segments = {}   # still colliding this fold
        self._collision_active_points = {}
        self._collision_fixed_segments = {}    # finalized when that pair's contact ended
        self._collision_fixed_points = {}
        self._collision_coords_exported = False  # one-shot export at π done
        self._sweep_draw_stopped = False  # True once θ hits π — no more paint growth
        self._intruder_classifications = []
        self._intruder_by_unit_pair = {}  # (min_u, max_u) -> classification dict
        self._intruder_indice_count = 0
        # Locus paint: path of two contact nodes + the line (design xy → panel).
        self._paint_canvas = {}          # unit_id -> list[{p0,p1}]
        self._paint_line_vert_count = 0
        self._paint_n_strokes = 0
        # Starting GPU buffer size only — grows as needed (no sample/line cap)
        self._paint_max_line_verts = 8192
        self._paint_min_move = None
        # Collision / paint are expensive in pure Python — throttle hard
        self._collision_frame_i = 0
        self._collision_every_n = 4      # run broadphase every N render frames
        self._paint_dirty = False
        # Per-frame host cache of x (shared by collision / paint / intruder)
        self._frame_positions = None
        self._frame_positions_gen = -1
        self._render_gen = 0
        self._stamp_paint_needed = False
        self._crease_pairs_np = None
        self._flat_kps_np = None
        self._mesh_advanced = True
        self._collision_ran_once = False
        self.ID = 0

        self.pd_local_time = pd_local_time
        self.pd_global_time = pd_global_time
        self.pd_iter_time = pd_iter_time

        self.material_type = material_type
        self.ref_target = ref_target
        self.verbose = verbose

        self.split_origami_num = 1
        self.split_start_index = []
        self.split_connection = []
        self.split_kp_sets = []

        self.MAXIMUM_FIX_PANEL = 1
        self.sparse_solver = ti.linalg.SparseSolver(data_type, "LLT")

        self.folding_angle = 0.
        self.enable_add_folding_angle = 0.
        self.angle_protection = 1.
        self.damping = damping

        self.collision_indice = 1e-1
        self.collision_d = 1e-4      

        self.paused = False
        self.step_once = False

        # if use_gui:
        self.window = ti.ui.Window("Origami Simulation", (1600, 900), vsync=True, show_window=self.use_gui)
        self.gui = self.window.get_gui()
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((0.08, 0.09, 0.12))
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
       
        self.fast_simulation_mode = fast
        self.pd_local_time = 1
        self.pd_global_time = 1

        self.time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        self.origami_name = origami_name

        self.image_id = 0
        if not self.fast_simulation_mode and use_gui:
            try:
                os.makedirs(f"./physResult/{self.origami_name}-{self.time}")
            except:
                pass
        
        self.origami_thickness = 1.32

        self.backup_spring_cons_num = 0
        self.backup_facet_space = 0
        self.backup_maximum_kp_number = 0
        self.backup_maximum_line_indice_num = 0

    def _init_ti(self):
        """安全初始化 Taichi，避免重复初始化。
        Safely initialize Taichi to avoid re-init errors."""
        try:
            ti.init(arch=ti.cpu, default_fp=data_type, fast_math=False, advanced_optimization=False, cpu_max_num_threads=1, kernel_profiler=False, verbose=False)
        except Exception:
            pass
         
    def biasKp(self, kp, bias):
        new_kp = deepcopy(kp)
        for i in range(min(len(new_kp), len(bias))):
            new_kp[i] += bias[i]
        return new_kp
    
    def biasId(self, id, bias):
        """
        对ID施加偏移量。
        Apply bias to ID.
        
        :param id: 原始ID / Original ID
        :param bias: 偏移量 / Bias
        :return: 偏移后的ID / Biased ID
        """
        id += bias
        return id
      
    def pointInList(self, kp, tolerance=2):
        """
        检查关键点是否已存在于列表中。
        Check if a keypoint already exists in the list.
        
        :param kp: 待检查的关键点 / Keypoint to check
        :param tolerance: 距离容差 / Distance tolerance
        :return: 已存在点的索引，若不存在则返回-1 / Index of existing point, -1 if not found
        """
        for i in range(len(self.kps)):
            if distance3D(kp, self.kps[i]) < tolerance:
                return i
        return -1      

    def reconstructRoutingUsingMap(self, method, map):
        new_method = {
            "type": [],
            "id": [],
            "reverse": []
        }
        types = method["type"]
        string_number = len(types)
        ids = method["id"]
        directions = method["reverse"]
        for i in range(string_number):
            current_type_list = types[i]
            current_id_list = ids[i]
            current_direction_list = directions[i]
            new_current_type_list = []
            new_id_list = []
            new_direction_list = []
            length = len(current_type_list)
            for j in range(length):
                current_type = current_type_list[j]
                current_id = current_id_list[j]
                current_direction = current_direction_list[j]
                unit_id_map = map[current_id]
                if current_type == 'A':
                    new_current_type_list.append(current_type)
                    new_id_list.append(current_id)
                    new_direction_list.append(current_direction)
                if current_type == 'B':
                    min_z_axis = self.units[unit_id_map[0]].crease[0][START][Z]
                    min_choosed_id = unit_id_map[0]
                    max_z_axis = self.units[unit_id_map[0]].crease[0][START][Z]
                    max_choosed_id = unit_id_map[0]

                    for unit_id in unit_id_map:
                        new_z_axis = self.units[unit_id].crease[0][START][Z]
                        if new_z_axis < min_z_axis:
                            min_z_axis = new_z_axis
                            min_choosed_id = unit_id
                        if new_z_axis > max_z_axis:
                            max_z_axis = new_z_axis
                            max_choosed_id = unit_id

                    if len(unit_id_map) > 1:
                        if current_direction == -1:
                            new_current_type_list.append(current_type)
                            new_id_list.append(min_choosed_id)
                            new_direction_list.append(current_direction)
                            new_current_type_list.append(current_type)
                            new_id_list.append(max_choosed_id)
                            new_direction_list.append(-1)
                        elif current_direction == 1:
                            new_current_type_list.append(current_type)
                            new_id_list.append(max_choosed_id)
                            new_direction_list.append(current_direction)
                            new_current_type_list.append(current_type)
                            new_id_list.append(min_choosed_id)
                            new_direction_list.append(1)
                    else:
                        new_current_type_list.append(current_type)
                        new_id_list.append(unit_id_map[0])
                        new_direction_list.append(current_direction)

            new_method["type"].append(new_current_type_list)
            new_method["id"].append(new_id_list)
            new_method["reverse"].append(new_direction_list)
        return new_method
    
    def commonStart_1(self, unit_edge_max, thick_mode):
        self.unit_edge_max = unit_edge_max
                    
        # 构造折纸系统
        density = 1.24e-9
        if self.material_type == 2:
            density = 0.08e-9
        tolerance = 0.05 if thick_mode else 1.0
        self.ori_sim = OrigamiSimulationSystem(unit_edge_max, material_density=density, split_unit_list=self.split_start_index)
        
        for ele in self.units:
            self.ori_sim.addUnit(ele, ele.special, self.origami_thickness, tol=tolerance)

        self.ori_sim.mesh() #构造三角剖分

        self.unit_edge_max = self.ori_sim.unit_edge_max
        self.ori_sim.fillBlankIndices() # fill all blank indice with -1

    def backupSimulationSetting(self):
        self.backup_spring_cons_num = self.spring_cons_num
        self.backup_facet_space =self.facet_space
        self.backup_maximum_kp_number = self.maximum_kp_number
        self.backup_maximum_line_indice_num = self.maximum_line_indice_num

    def commonStart_2(self, occupy_memory=True, multiple=1.1):
        ori_sim = self.ori_sim
        self.connection_matrix = ori_sim.connection_matrix
        self.mass_list = ori_sim.mass_list
        self.crease_pairs_list = ori_sim.crease_pairs
        self.bending_pairs_list = ori_sim.bending_pairs
        self.facet_crease_pairs_list = ori_sim.facet_crease_pairs
        self.facet_bending_pairs_list = ori_sim.facet_bending_pairs
        self.spring_k = ori_sim.spring_k
        self.bending_k = ori_sim.bending_k
        self.facet_bending_k = ori_sim.face_k

        new_lines = ori_sim.getNewLines()  
        new_line_indices = ori_sim.getNewLineIndices()
        self.kps = ori_sim.kps                                           # all keypoints of origami
        self.creases = new_lines                                         # all creases of origami
        self.tri_indices = ori_sim.tri_indices                           # all triangle indices of origami
        self.kp_num = len(ori_sim.kps)                                   # total number of keypoints
        self.indices_num = len(ori_sim.tri_indices)                      # total number of triangles indices
        self.div_indices_num = int(self.indices_num / 3)                 # total_number of triangles
        self.unit_indices_num = len(ori_sim.indices)                     # total number of units
        self.line_total_indice_num = len(new_line_indices)               # total number of lines
        self.bending_pairs_num = len(ori_sim.bending_pairs)              # total number of bending pairs
        self.crease_pairs_num = len(ori_sim.crease_pairs)                # total number of crease pairs
        self.facet_bending_pairs_num = len(ori_sim.facet_bending_pairs)  # total number of facet bending pairs
        self.facet_crease_pairs_num = len(ori_sim.facet_crease_pairs)    # total number of facet crease pairs

        self.split_kp_sets_min = []  # 每个分片包含的 kp 集合（Python set)
        self.split_kp_sets_max = []  # 每个分片包含的 kp 集合（Python set

        self.facet_split_sets = ori_sim.facet_cons_id
        self.additional_kp_origami_id = list(ori_sim.new_kp_origami_id)

        for split_id in range(self.split_origami_num):
            start_unit = self.split_start_index[split_id]
            end_unit = self.split_start_index[split_id + 1] if split_id + 1 < self.split_origami_num else len(self.units)
            kp_set = set()
            for uid in range(start_unit, end_unit):
                for kp_idx in self.ori_sim.indices[uid]:
                    if kp_idx != -1:
                        kp_set.add(kp_idx)
            self.split_kp_sets_min.append(min(kp_set))
            self.split_kp_sets_max.append(max(kp_set))

        self.spring_cons_num = int(np.count_nonzero(np.array(self.connection_matrix)) // 2 + 3 * sum([len(ori_sim.indices[self.connected_unit_pairs[i][0]]) for i in range(len(self.connected_unit_pairs))]))
        # self.spring_cons_num = int(np.count_nonzero(np.array(self.connection_matrix)) // 2 + self.maximum_number_of_thick_panel_spring)
        self.bending_cons_num = len(self.crease_pairs_list)
        self.facet_bending_cons_num = len(self.facet_crease_pairs_list)
        
        self.facet_space = self.facet_bending_cons_num
        self.maximum_kp_number = int(self.kp_num)
        self.maximum_line_indice_num = int(self.line_total_indice_num)

        if self.verbose:
            print(f"# of Spring constraints: {np.count_nonzero(np.array(self.connection_matrix)) // 2} in-panel + {3 * sum([len(ori_sim.indices[self.connected_unit_pairs[i][0]]) for i in range(len(self.connected_unit_pairs))])} out-of-panel in practice\n" + \
                f"# of Bending constraints: {self.bending_cons_num}\n" + \
                f"# of Facet bending constraints: {self.facet_bending_cons_num} in practice\n" + \
                f"# of keypoints: {self.kp_num} in practice\n" + \
                f"# of line indices: {self.line_total_indice_num} in practice")
        
        self.maximum_level_number = 1
        for i in range(len(self.ori_sim.line_indices)):
            self.ori_sim.line_indices[i] = [self.ori_sim.line_indices[i][0][START], self.ori_sim.line_indices[i][0][END], self.ori_sim.line_indices[i][1], self.ori_sim.line_indices[i][2], self.ori_sim.line_indices[i][3]]

        if self.spring_cons_num > self.backup_spring_cons_num or \
              self.facet_space > self.backup_facet_space or \
                self.maximum_kp_number > self.backup_maximum_kp_number or \
                    self.maximum_line_indice_num > self.backup_maximum_line_indice_num:
            occupy_memory = True
            self.maximum_kp_number = int(self.maximum_kp_number * multiple) // 2 * 2
            self.spring_cons_num = int(self.spring_cons_num * multiple) // 2 * 2
            self.facet_space = int(self.facet_space * multiple) // 2 * 2
            self.maximum_line_indice_num = int(self.maximum_line_indice_num * multiple) // 2 * 2
            self.unit_indices_num = int(self.unit_indices_num * multiple) // 2 * 2
            if self.backup_maximum_kp_number != 0:
                return 0
        else:
            self.spring_cons_num = self.backup_spring_cons_num
            self.facet_space = self.backup_facet_space
            self.maximum_kp_number = self.backup_maximum_kp_number
            self.maximum_line_indice_num = self.backup_maximum_line_indice_num
            occupy_memory = False
        
        # ---memory--- #
        if occupy_memory:
            self.x = ti.Vector.field(3, dtype=data_type, shape=self.maximum_kp_number) #点的位置
            self.x0 = ti.Vector.field(3, dtype=data_type, shape=self.maximum_kp_number) #点的位置
            self.s = ti.Vector.field(3, dtype=data_type, shape=self.maximum_kp_number) #点的位置
            self.v = ti.Vector.field(3, dtype=data_type, shape=self.maximum_kp_number) #点的速度
            self.dv = ti.Vector.field(3, dtype=data_type, shape=self.maximum_kp_number) #点的加速度

            self.unit_indices = ti.Vector.field(self.unit_edge_max, dtype=int, shape=self.unit_indices_num) # 每个单元的索引信息

            self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.maximum_kp_number) #点的位置
            self.original_vertices = ti.Vector.field(3, dtype=data_type, shape=self.maximum_kp_number) # 原始点坐标

            self.masses = ti.field(dtype=data_type, shape=self.maximum_kp_number) # 质量信息

            self.energy = ti.field(dtype=data_type, shape=())
            self.backup_energy = ti.field(dtype=data_type, shape=())
            self.split_energy = ti.field(dtype=data_type, shape=self.split_origami_num)

            self.spring_k_param = ti.field(dtype=data_type, shape=())
            self.bending_k_param = ti.field(dtype=data_type, shape=())
            self.facet_bending_k_param = ti.field(dtype=data_type, shape=())

            self.kp_num_param = ti.field(dtype=int, shape=())
            self.kp_max_num_param = ti.field(dtype=int, shape=())
            self.line_total_indice_num_param = ti.field(dtype=int, shape=())
            self.indices_num_param = ti.field(dtype=int, shape=())
            self.split_origami_num_param = ti.field(dtype=int, shape=())

            self.spring_num_param = ti.field(dtype=int, shape=())
            self.bending_num_param = ti.field(dtype=int, shape=())
            self.facet_bending_num_param = ti.field(dtype=int, shape=())

            # === 约束拓扑索引（拆分为 3 种约束各自独立的 field，避免 base 偏移计算） ===

            # spring: 每个约束 2 个索引 → shape = 2 * spring_cons_num
            self.spring_selection = ti.field(dtype=int, shape=2 * self.spring_cons_num)
            self.spring_x_proj = ti.Vector.field(3, dtype=data_type, shape=2 * self.spring_cons_num)
            self.spring_selection_corresponding_origami_id = ti.field(dtype=int, shape=self.spring_cons_num)
            # bending: 每个约束 4 个索引 → shape = 4 * bending_cons_num
            self.bending_selection = ti.field(dtype=int, shape=4 * self.bending_cons_num)
            self.bending_x_proj = ti.Vector.field(3, dtype=data_type, shape=4 * self.bending_cons_num)
            self.bending_selection_corresponding_origami_id = ti.field(dtype=int, shape=self.bending_cons_num)
            # facet_bending: 每个约束 4 个索引 → shape = 4 * facet_bending_cons_num
            self.facet_bending_selection = ti.field(dtype=int, shape=4 * max(self.facet_space, 1))
            self.facet_bending_x_proj = ti.Vector.field(3, dtype=data_type, shape=4 * max(self.facet_space, 1))
            self.facet_bending_selection_corresponding_origami_id = ti.field(dtype=int, shape=self.facet_space)

            # === 余切权重（cotangent Laplacian 构型相关常数，initializeRunning 时计算一次） ===
            self.cotangent_vector = ti.Vector.field(4, dtype=data_type, shape=self.bending_cons_num)
            self.facet_cotangent_vector = ti.Vector.field(4, dtype=data_type, shape=max(self.facet_space, 1))
            self.cotangent_matrix = ti.Matrix.field(4, 4, dtype=data_type, shape=self.bending_cons_num)
            self.facet_cotangent_matrix = ti.Matrix.field(4, 4, dtype=data_type, shape=max(self.facet_space, 1))

            self.line_pairs = ti.field(dtype=int, shape=(self.maximum_line_indice_num, 2)) #线段索引信息，用于初始化渲染
            self.line_color = ti.Vector.field(3, dtype=data_type, shape=self.maximum_line_indice_num*2) #线段颜色，用于渲染
            self.line_vertex = ti.Vector.field(3, dtype=ti.f32, shape=self.maximum_line_indice_num*2) #线段顶点位置，用于渲染
            # Contact markers for refined collision shading (points + segments)
            self.collision_contact_points = ti.Vector.field(3, dtype=ti.f32, shape=self._collision_contact_max)
            self.collision_contact_lines = ti.Vector.field(3, dtype=ti.f32, shape=self._collision_contact_max * 2)
            # Intruder panel triangle indices (same mesh verts, painted pass)
            self.intruder_indices = ti.field(int, shape=self.maximum_indice_num)
            # Paint strokes reprojected to panel surfaces (lines only — canvas model)
            if self.collision_shading:
                n_pl = int(self._paint_max_line_verts)
                self.paint_line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_pl)

            self.indices = ti.field(int, shape=self.maximum_indice_num) #三角面索引信息

            self.bending_pairs = ti.field(dtype=int, shape=(self.bending_pairs_num, 2)) #弯曲对索引信息
            self.crease_pairs = ti.field(dtype=int, shape=(self.crease_pairs_num, 2)) #折痕对索引信息

            self.crease_folding_angle = ti.field(dtype=data_type, shape=self.crease_pairs_num) #折痕折角
            self.bending_pairs_area = ti.field(dtype=data_type, shape=(self.bending_pairs_num, 2)) #弯曲对面积信息
            self.crease_initial_length = ti.field(dtype=data_type, shape=self.crease_pairs_num) #折痕长度

            self.spring_original_length = ti.field(dtype=data_type, shape=self.spring_cons_num)

            self.crease_type = ti.field(dtype=int, shape=self.crease_pairs_num) #折痕类型信息，与折痕对一一对应
            self.crease_level = ti.field(dtype=int, shape=self.crease_pairs_num)
            self.crease_coeff = ti.field(dtype=data_type, shape=self.crease_pairs_num)

            self.recover_level_need = ti.field(dtype=bool, shape=(self.crease_pairs_num, self.maximum_level_number))
            self.recover_level = ti.field(dtype=int, shape=(self.crease_pairs_num, self.maximum_level_number))
            self.recover_angle = ti.field(dtype=float, shape=(self.crease_pairs_num, self.maximum_level_number))

            self.crease_angle = ti.field(dtype=data_type, shape=self.bending_pairs_num)
            self.backup_crease_angle = ti.field(dtype=data_type, shape=self.bending_pairs_num)
            self.backup_crease_angle_sum = ti.field(dtype=data_type, shape=())
            self.target_crease_angle = ti.field(dtype=data_type, shape=self.bending_pairs_num)
            self.previous_dir = ti.field(dtype=data_type, shape=self.bending_pairs_num)
            self.folding_angle_upper_bound = ti.field(dtype=data_type, shape=self.bending_pairs_num) #折痕折角上限，正值或0
            self.folding_angle_lower_bound = ti.field(dtype=data_type, shape=self.bending_pairs_num) #折痕折角下限，负值或0

            self.folding_angle_reach_pi = ti.field(dtype=bool, shape=())

            # 有可能所有单元都是三角形，故没有面折痕，根据特定条件初始化面折痕信息
            if self.facet_bending_pairs_num > 0:
                self.facet_bending_pairs = ti.field(dtype=int, shape=(self.facet_space, 2))
                self.facet_crease_pairs = ti.field(dtype=int, shape=(self.facet_space, 2))
                self.facet_bending_pairs_area = ti.field(dtype=data_type, shape=(self.facet_space, 2)) #弯曲对面积信息
                self.facet_crease_initial_length = ti.field(dtype=data_type, shape=self.facet_space) #折痕长度
                self.facet_bending_pairs_distance = ti.field(dtype=data_type, shape=self.facet_space) #折痕有效弯曲长度

            else:
                self.facet_bending_pairs = ti.field(dtype=int, shape=(1, 2))
                self.facet_crease_pairs = ti.field(dtype=int, shape=(1, 2))
                self.facet_bending_pairs_area = ti.field(dtype=data_type, shape=(1, 2)) #弯曲对面积信息
                self.facet_crease_initial_length = ti.field(dtype=data_type, shape=1) #折痕长度
                self.facet_bending_pairs_distance = ti.field(dtype=data_type, shape=1) #折痕有效弯曲长度
            
            self.fix_id_list = ti.field(dtype=int, shape=self.MAXIMUM_FIX_PANEL)

            self.unit_kp_num_list = ti.field(dtype=int, shape=self.unit_indices_num)
            self.unit_contributions = ti.Vector.field(self.unit_edge_max, dtype=data_type, shape=self.unit_indices_num) # 每个单元的贡献度

            self.thick_panel_additional_connection_id = ti.Vector.field(2, dtype=int, shape=self.unit_indices_num * self.unit_edge_max)

            self.sequence_level = ti.field(int, shape=2) # max, min
            self.folding_micro_step = ti.field(data_type, shape=()) # step calculated by sequence_level max and min

            self.folding_angle_param = ti.field(data_type, shape=())
            self.enable_add_folding_angle_param = ti.field(data_type, shape=())
            self.angle_protection_param = ti.field(data_type, shape=())
            self.damping_param = ti.field(data_type, shape=())

            self.collision_indice_param = ti.field(data_type, shape=())
            self.collision_d_param = ti.field(data_type, shape=())

            self.AK_field = ti.field(dtype=data_type, shape=(3 * self.maximum_kp_number, 3 * self.maximum_kp_number))

            self.AK = ti.linalg.SparseMatrixBuilder(3 * self.maximum_kp_number, 3 * self.maximum_kp_number, max_num_triplets=9 * self.maximum_kp_number ** 2, dtype=data_type)

            self.AM = ti.linalg.SparseMatrix(3 * self.maximum_kp_number, 3 * self.maximum_kp_number, dtype=data_type)

            self.b = ti.field(data_type, shape=3 * self.maximum_kp_number)
            self.b_array = ti.ndarray(data_type, 3 * self.maximum_kp_number)
            self.u0 = ti.field(data_type, shape=3 * self.maximum_kp_number) # solution

        return 1

    def start(self, filepath, unit_edge_max, thick_mode=False, occupy_memory=True):
        # 存储厚板模式标志 / Store thick mode flag
        self.thick_mode_flag = thick_mode
        self.origami_name = filepath

        with open("./descriptionData/" + filepath + ".json", 'r', encoding='utf-8') as fw:
            input_json = json.load(fw)
        self.input_json = input_json
        self.kps = []
        self.lines = []
        self.units = []

        spatialhash = SpatialHash(cell_size=1.0)

        self.contributions = []
        self.connected_unit_pairs = []

        if occupy_memory:
            self.maximum_number_of_thick_panel = 0
            self.maximum_number_of_thick_panel_spring = 0
            self.maximum_facet_crease_number = 0
            self.maximum_kp_number = 0
            self.maximum_indice_num = 0
            self.maximum_line_indice_num = 0

        unit_mapping = []

        try:
            self.split_origami_num = input_json["split_num"]
        except:
            self.split_origami_num = 1
        
        try:
            self.target = input_json["crease_angle"]
        except:
            self.target = []

        split_interval = int(len(input_json["units"]) / self.split_origami_num)
        self.split_start_index.clear()
        self.split_connection.clear()

        if not thick_mode:
            self.split_start_index = [split_interval * _ for _ in range(self.split_origami_num)]
            for i in range(len(input_json["kps"])):
                self.kps.append(input_json["kps"][i])
                spatialhash.insert(input_json["kps"][i], i)
                
            for i in range(len(input_json["lines"])):
                self.lines.append(Crease(
                    input_json["lines"][i][START], input_json["lines"][i][END], BORDER 
                ))
                self.lines[i].crease_type = input_json["line_features"][i]["type"]
                self.lines[i].level = input_json["line_features"][i]["level"]
                self.lines[i].coeff = input_json["line_features"][i]["coeff"]
                try:
                    self.lines[i].recover_level = input_json["line_features"][i]["recover_level"]
                    if type(self.lines[i].recover_level) != list:
                        self.lines[i].recover_level = []
                except:
                    self.lines[i].recover_level = []
                try:
                    self.lines[i].recover_angle = input_json["line_features"][i]["recover_angle"]
                except:
                    self.lines[i].recover_angle = []
                # try:
                #     self.lines[i].thick_panel_height = input_json["line_features"][i]["thick_panel_height"]
                # except:
                self.lines[i].thick_panel_height = 0.0
                self.lines[i].hard = input_json["line_features"][i]["hard"]
                self.lines[i].folding_angle_upper_bound = input_json["line_features"][i]["hard_angle"]
                self.lines[i].folding_angle_lower_bound = input_json["line_features"][i]["hard_angle_down"]
            for i in range(len(input_json["units"])):
                self.units.append(Unit())
                kps = deepcopy(input_json["units"][i])
                for j in range(0, -len(kps), -1):
                    crease_type = BORDER
                    hard = False
                    current_kp = deepcopy(kps[j])
                    next_kp = deepcopy(kps[j - 1])
                    for line in self.lines:
                        if (distance3D(line[START], current_kp) < 1e-3 and distance3D(line[END], next_kp) < 1e-3) or \
                            (distance3D(line[END], current_kp) < 1e-3 and distance3D(line[START], next_kp) < 1e-3):
                            crease_type = line.getType()
                            hard = line.hard
                            folding_angle_upper_bound = line.folding_angle_upper_bound
                            folding_angle_lower_bound = line.folding_angle_lower_bound
                            break
                    if occupy_memory:
                        if crease_type == BORDER:
                            self.maximum_kp_number += 1.
                            self.maximum_line_indice_num += 1.
                        else:
                            self.maximum_kp_number += .5
                            self.maximum_line_indice_num += .5
                    self.units[i].addCrease(Crease(
                        current_kp, next_kp, crease_type, hard=hard, upper=folding_angle_upper_bound, lower=folding_angle_lower_bound
                    ))
                if occupy_memory:
                    self.maximum_indice_num += 3 * (len(kps) - 2)
            
                try:
                    contribution_for_unit = deepcopy(input_json["contributions"][i])
                    new_contribution = []
                    for j in range(0, -len(contribution_for_unit), -1):
                        new_contribution.append(contribution_for_unit[j])
                    self.contributions.append(new_contribution)
                except:
                    pass
            
        else:
            for i in range(len(input_json["lines"])):
                line_start = deepcopy(input_json["lines"][i][START])

                add_height = 10.0 if input_json["line_features"][i]["type"] == 0 else -10.0
                try:
                    add_height = input_json["line_features"][i]["thick_panel_height"]
                except:
                    pass
                if abs(add_height) < 1e-2:
                    add_height = 10.0 if input_json["line_features"][i]["type"] == 0 else -10.0

                if len(line_start) == 2:
                    line_start.append(add_height)
                else:
                    line_start[Z] = add_height
                line_end = deepcopy(input_json["lines"][i][END])
                if len(line_end) == 2:
                    line_end.append(add_height)
                else:
                    line_end[Z] = add_height
                
                self.lines.append(Crease(
                    line_start, line_end, BORDER 
                ))
                self.lines[i].crease_type = input_json["line_features"][i]["type"]
                self.lines[i].level = input_json["line_features"][i]["level"]
                self.lines[i].coeff = input_json["line_features"][i]["coeff"]
                try:
                    self.lines[i].recover_level = input_json["line_features"][i]["recover_level"]
                    if type(self.lines[i].recover_level) != list:
                        self.lines[i].recover_level = []
                except:
                    self.lines[i].recover_level = []
                try:
                    self.lines[i].recover_angle = input_json["line_features"][i]["recover_angle"]
                except:
                    self.lines[i].recover_angle = []
                try:
                    self.lines[i].thick_panel_height = add_height
                except:
                    self.lines[i].thick_panel_height = 10.0 if self.lines[i].crease_type == 0 else -10.0
                self.lines[i].hard = input_json["line_features"][i]["hard"]
                self.lines[i].folding_angle_upper_bound = input_json["line_features"][i]["hard_angle"]
                self.lines[i].folding_angle_lower_bound = input_json["line_features"][i]["hard_angle_down"]

            line_dict_2D = {}
            line_dict_3D = {}

            for line in self.lines:
                key_s_2d = (round(line[START][X], 3), round(line[START][Y], 3))
                key_e_2d = (round(line[END][X], 3), round(line[END][Y], 3))
                line_dict_2D[(key_s_2d, key_e_2d)] = line
                line_dict_2D[(key_e_2d, key_s_2d)] = line  # 双向
                key_s = (round(line[START][X], 3), round(line[START][Y], 3), round(line[START][Z], 3))
                key_e = (round(line[END][X], 3), round(line[END][Y], 3), round(line[END][Z], 3))
                line_dict_3D[(key_s, key_e)] = line
                line_dict_3D[(key_e, key_s)] = line  # 双向

            for i in range(len(input_json["units"])):
                if i % split_interval == 0:
                    self.split_start_index.append(len(self.units))
                kps = deepcopy(input_json["units"][i])
                # check different height
                height_parameters = []
                current_possible_thick_panel_number = 0
                for j in range(0, -len(kps), -1):
                    crease_type = BORDER
                    hard = False
                    current_kp = deepcopy(kps[j])
                    next_kp = deepcopy(kps[j - 1])

                    key_s = (round(current_kp[X], 3), round(current_kp[Y], 3))
                    key_e = (round(next_kp[X], 3), round(next_kp[Y], 3))

                    line = line_dict_2D.get((key_s, key_e))
                    # for line in self.lines:
                    #     if (distance(line[START], current_kp) < 1e-3 and distance(line[END], next_kp) < 1e-3) or \
                    #         (distance(line[END], current_kp) < 1e-3 and distance(line[START], next_kp) < 1e-3):
                    if line is not None:
                        if occupy_memory:
                            self.maximum_number_of_thick_panel += 1
                            self.maximum_facet_crease_number += len(kps) - 3
                            current_possible_thick_panel_number += 1
                        crease_type = line.getType()
                        if (crease_type != BORDER) and line.thick_panel_height not in height_parameters:
                            height_parameters.append(line.thick_panel_height)

                if occupy_memory:
                    self.maximum_number_of_thick_panel_spring += 3 * len(kps) * (current_possible_thick_panel_number - 1)    
                    self.maximum_indice_num += 3 * (len(kps) - 2) * (current_possible_thick_panel_number)  

                height_parameters.sort()

                unit_mapping.append([len(self.units) + k for k in range(len(height_parameters))])
                if i == 0 and len(height_parameters) >= 2:
                    self.connected_unit_pairs += [[0, x] for x in range(1, len(height_parameters))]
                    self.split_connection += [i // split_interval for _ in range(1, len(height_parameters))]
                elif i > 0 and len(height_parameters) >= 2:
                    self.connected_unit_pairs += [[len(self.units), x + len(self.units)] for x in range(1, len(height_parameters))]
                    self.split_connection += [i // split_interval for _ in range(1, len(height_parameters))]
                
                for height in height_parameters:
                    self.units.append(Unit())
                    for j in range(0, -len(kps), -1):
                        crease_type = BORDER
                        hard = False
                        current_kp = deepcopy(kps[j])
                        current_kp[Z] = height
                        next_kp = deepcopy(kps[j - 1])
                        next_kp[Z] = height
                        folding_angle_upper_bound = math.pi
                        folding_angle_lower_bound = -math.pi
                        level = 0
                        coeff = 1.
                        rec_level = []
                        rec_angle = []
                        
                        key_s = (round(current_kp[X], 3), round(current_kp[Y], 3), round(current_kp[Z], 3))
                        key_e = (round(next_kp[X], 3), round(next_kp[Y], 3), round(next_kp[Z], 3))

                        line = line_dict_3D.get((key_s, key_e))
                        # for line in self.lines:
                        #     if (distance3D(line[START], current_kp) < 1e-3 and distance3D(line[END], next_kp) < 1e-3) or \
                        #         (distance3D(line[END], current_kp) < 1e-3 and distance3D(line[START], next_kp) < 1e-3):
                        if line is not None:
                            current_kp[Z] = line.thick_panel_height
                            next_kp[Z] = line.thick_panel_height
                            crease_type = line.getType()
                            hard = line.hard
                            folding_angle_upper_bound = line.folding_angle_upper_bound
                            folding_angle_lower_bound = line.folding_angle_lower_bound
                            level = line.level
                            coeff = line.coeff
                            rec_level = line.recover_level
                            rec_angle = line.recover_angle
                            # break
                        
                        if occupy_memory:
                            if crease_type == BORDER:
                                self.maximum_kp_number += 1.
                                self.maximum_line_indice_num += 1.
                            else:
                                self.maximum_kp_number += .5
                                self.maximum_line_indice_num += .5
                                
                        if spatialhash.find(current_kp, 1e-3) == -1:
                            self.kps.append(current_kp)
                            spatialhash.insert(current_kp, len(self.kps) - 1)
                        if spatialhash.find(next_kp, 1e-3) == -1:
                            self.kps.append(next_kp)
                            spatialhash.insert(next_kp, len(self.kps) - 1)
                             
                        new_crease = Crease(
                            current_kp, next_kp, crease_type, hard=hard, upper=folding_angle_upper_bound, lower=folding_angle_lower_bound
                        )
                        new_crease.level = level
                        new_crease.coeff = coeff
                        new_crease.recover_level = rec_level
                        new_crease.recover_angle = rec_angle
                        self.units[-1].addCrease(new_crease)
                    
                    try:
                        contribution_for_unit = deepcopy(input_json["contributions"][i])
                        new_contribution = []
                        for j in range(0, -len(contribution_for_unit), -1):
                            new_contribution.append(contribution_for_unit[j])
                        self.contributions.append(new_contribution)
                    except:
                        pass
                
                for ele in range(current_possible_thick_panel_number - len(height_parameters)):
                    self.maximum_kp_number += 1. * len(kps)
                    self.maximum_line_indice_num += 1. * len(kps)
            
            self.maximum_kp_number = int(self.maximum_kp_number)
            if self.verbose:
                print(f"Maximum thick panels in theory: {self.maximum_number_of_thick_panel}\n" + \
                      f"Maximum thick panel spring in theory: {self.maximum_number_of_thick_panel_spring}\n" + \
                        f"Maximum facet crease in theory: {self.maximum_facet_crease_number}\n" + \
                            f"Maximum keypoints in theory: {self.maximum_kp_number}\n" + \
                                f"Maximum indices in theory: {self.maximum_indice_num}")
                        
        if len(self.contributions) == 0:
            self.contributions.append([])

        self.maximum_line_indice_num = int(self.maximum_line_indice_num)
        
        try:
            self.fix_id = deepcopy(input_json["fix"])
            if len(self.connected_unit_pairs):
                new_fix_id = []
                for ele in self.fix_id:
                    true_unit_id_list = unit_mapping[ele]
                    new_fix_id += true_unit_id_list
                self.fix_id = new_fix_id
        except:
            self.fix_id = [-1]

        try:
            self.targets = deepcopy(input_json["crease_angle"])
        except:
            self.targets = []
            
        # calculate max length of view
        self.max_size, max_x, max_y = getMaxDistance(self.kps)
        self.total_bias = getTotalBias(self.units)
        self.unit_mapping = unit_mapping

        self.commonStart_1(unit_edge_max, thick_mode)
    
        ok = self.commonStart_2(occupy_memory=occupy_memory)

        return ok

    def getUnitMapping(self):
        return self.unit_mapping
    
    @ti.kernel
    def fill_line_vertex(self):
        for i in ti.ndrange(self.line_total_indice_num_param[None]):
            indice1 = self.line_pairs[i, 0]
            indice2 = self.line_pairs[i, 1]
            self.line_vertex[2 * i] = self.vertices[indice1]
            self.line_vertex[2 * i + 1] = self.vertices[indice2]

    @ti.func
    def calculateKpNumWithUnitId(self, unit_kps):
        """
        计算单元的有效关键点数量。
        Calculate the number of valid keypoints for a unit.
        
        :param unit_kps: 单元关键点索引列表 / Unit keypoint indices list
        :return: 有效关键点数量 / Number of valid keypoints
        """
        kp_len = 0
        for i in ti.ndrange(len(unit_kps)):
            if unit_kps[i] != -1:
                kp_len += 1
        return kp_len
    
    @ti.kernel
    def initialize(     self, 
                        numpy_indices               : ti.types.ndarray(), 
                        numpy_kps                   : ti.types.ndarray(), 
                        numpy_mass_list             : ti.types.ndarray(), 
                        numpy_tri_indices           : ti.types.ndarray(), 
                        numpy_connection_matrix     : ti.types.ndarray(), 
                        numpy_bending_pairs         : ti.types.ndarray(), 
                        numpy_crease_pairs          : ti.types.ndarray(), 
                        numpy_line_indices          : ti.types.ndarray(), 
                        numpy_facet_bending_pairs   : ti.types.ndarray(), 
                        numpy_facet_crease_pairs    : ti.types.ndarray(),
                        numpy_original_kps          : ti.types.ndarray(), 
                        numpy_tb_line               : ti.types.ndarray(),
                        numpy_contributions         : ti.types.ndarray(),
                        numpy_recover_level_need    : ti.types.ndarray(), 
                        numpy_recover_level         : ti.types.ndarray(), 
                        numpy_recover_angle         : ti.types.ndarray(), 
                        numpy_fix_id                : ti.types.ndarray(), 
                        numpy_connected_unit_id     : ti.types.ndarray(), 
                        dt                          : data_type,
                        kp_num                      : ti.i32,
                        maximum_kp_num              : ti.i32,
                        unit_indices_num            : ti.i32,
                        line_total_indice_num       : ti.i32,
                        indices_num                 : ti.i32,
                        spring_k                    : data_type, 
                        bending_k                   : data_type, 
                        facet_bending_k             : data_type,
                        spring_cons_num             : ti.i32,
                        bending_cons_num            : ti.i32,
                        facet_bending_cons_num      : ti.i32,
                        folding_angle               : data_type,
                        enable_add_folding_angle    : data_type,
                        damping                     : data_type,
                        angle_protection            : data_type,
                        collision_indice            : data_type,
                        collision_d                 : data_type,
                        split_origami_num           : ti.i32,
                        numpy_split_start_kp_id     : ti.types.ndarray(),
                        numpy_split_facet_id        : ti.types.ndarray(),
                        numpy_split_connection      : ti.types.ndarray(),
                        numpy_additional_kp_ori_id  : ti.types.ndarray(),
                        numpy_target_angle          : ti.types.ndarray(),
                        verbose                     : bool,
        ):
        self.fix_id_list.fill(-1)
        self.AK_field.fill(0.)
        self.cotangent_matrix.fill(0.)
        self.facet_cotangent_matrix.fill(0.)
        self.x.fill(0.)
        self.x0.fill(0.)
        self.s.fill(0.)
        self.v.fill(0.)
        self.dv.fill(0.)
        self.b.fill(0.)

        self.indices.fill(maximum_kp_num - 1)
        self.vertices.fill(0.)
        self.original_vertices.fill(0.)
        self.masses.fill(0.)
        self.line_pairs.fill(0)
        self.line_color.fill(0.)
        self.line_vertex.fill(0.)

        self.energy[None] = 0.
        self.backup_energy[None] = 0.
        self.backup_crease_angle_sum[None] = 0.
        self.folding_angle_reach_pi[None] = False
        self.split_energy.fill(0.)

        self.thick_panel_additional_connection_id.fill(-1)
        self.unit_indices.fill(0)
        self.unit_kp_num_list.fill(0)
        self.unit_contributions.fill(0.)

        self.split_origami_num_param[None] = split_origami_num
        self.kp_num_param[None] = kp_num
        self.kp_max_num_param[None] = maximum_kp_num
        self.spring_k_param[None] = spring_k
        self.bending_k_param[None] = bending_k
        self.facet_bending_k_param[None] = facet_bending_k

        self.spring_num_param[None] = spring_cons_num
        self.bending_num_param[None] = bending_cons_num
        self.facet_bending_num_param[None] = facet_bending_cons_num
        self.line_total_indice_num_param[None] = line_total_indice_num
        self.folding_angle_param[None] = folding_angle
        self.enable_add_folding_angle_param[None] = enable_add_folding_angle
        self.angle_protection_param[None] = angle_protection
        self.damping_param[None] = damping

        self.collision_indice_param[None] = collision_indice
        self.collision_d_param[None] = collision_d
        self.indices_num_param[None] = indices_num

        for i in ti.ndrange(min(self.MAXIMUM_FIX_PANEL, numpy_fix_id.shape[0])):
            self.fix_id_list[i] = numpy_fix_id[i]
        
        for i, j in ti.ndrange(numpy_connected_unit_id.shape[0], 2):
            self.thick_panel_additional_connection_id[i][j] = numpy_connected_unit_id[i, j]
        
        # 初始化单元索引
        for i, j in ti.ndrange(numpy_indices.shape[0], self.unit_edge_max):
            self.unit_indices[i][j] = numpy_indices[i, j]
        
        for i in ti.ndrange(numpy_indices.shape[0]):
            self.unit_kp_num_list[i] = self.calculateKpNumWithUnitId(self.unit_indices[i])

        for i, j in ti.ndrange(numpy_indices.shape[0], self.unit_edge_max):
            self.unit_contributions[i][j] = 1. / self.unit_kp_num_list[i]

        if numpy_contributions.shape[0] > 0:
            for i, j in ti.ndrange(numpy_contributions.shape[0], numpy_contributions.shape[1]):   
                self.unit_contributions[i][j] = numpy_contributions[i, j]

        # 初始化节点位置与质量
        for i in ti.ndrange(self.kp_num_param[None]):
            self.original_vertices[i] = [numpy_kps[i, X], numpy_kps[i, Y], numpy_kps[i, Z]]
            self.masses[i] = numpy_mass_list[i]

        for i in ti.ndrange((self.kp_num_param[None], self.kp_max_num_param[None])):
            self.original_vertices[i] = [0., 0., 0.]
            self.masses[i] = numpy_mass_list[0]

        # 初始化三角面索引
        for i in ti.ndrange(self.indices_num_param[None]):
            self.indices[i] = numpy_tri_indices[i]

        # # 初始化连接矩阵
        # for i, j in ti.ndrange(self.kp_num_param[None], self.kp_num_param[None]):
        #     self.connection_matrix[i, j] = numpy_connection_matrix[i, j]
        
        # 初始化弯曲对和折痕对
        for i, j in ti.ndrange(self.bending_num_param[None], 2):
            self.bending_pairs[i, j] = numpy_bending_pairs[i, j]
            self.crease_pairs[i, j] = numpy_crease_pairs[i, j]

        for i in ti.ndrange(self.bending_num_param[None]):
            # 初始化弯曲对和折痕对的面积
            cs = self.original_vertices[self.crease_pairs[i, 0]]
            ce = self.original_vertices[self.crease_pairs[i, 1]]
            p1 = self.original_vertices[self.bending_pairs[i, 0]]
            p2 = self.original_vertices[self.bending_pairs[i, 1]]
            a1 = ((ce - cs).cross(p1 - cs)).norm()
            a2 = ((p2 - cs).cross(ce - cs)).norm()
            self.bending_pairs_area[i, 0] = a1 * 0.5
            self.bending_pairs_area[i, 1] = a2 * 0.5
            self.crease_initial_length[i] = (ce - cs).norm()

        # 初始化线段对
        for i, j in ti.ndrange(self.line_total_indice_num_param[None], 2):
            self.line_pairs[i, j] = int(numpy_line_indices[i, j])

        # 初始化面折痕对
        for i, j in ti.ndrange(self.facet_bending_num_param[None], 2):
            self.facet_bending_pairs[i, j] = numpy_facet_bending_pairs[i, j]
            self.facet_crease_pairs[i, j] = numpy_facet_crease_pairs[i, j]

        for i in ti.ndrange(self.facet_bending_num_param[None]):
            # 初始化弯曲对和折痕对的面积
            cs = self.original_vertices[self.facet_crease_pairs[i, 0]]
            ce = self.original_vertices[self.facet_crease_pairs[i, 1]]
            p1 = self.original_vertices[self.facet_bending_pairs[i, 0]]
            p2 = self.original_vertices[self.facet_bending_pairs[i, 1]]
            a1 = ((ce - cs).cross(p1 - cs)).norm()
            a2 = ((p2 - cs).cross(ce - cs)).norm()
            # assert a1 > 0 and a2 > 0
            self.facet_bending_pairs_area[i, 0] = a1 * 0.5
            self.facet_bending_pairs_area[i, 1] = a2 * 0.5
            self.facet_bending_pairs_distance[i] = ((p1 - p2) - (p1 - p2).dot(ce - cs) / (ce - cs).norm_sqr() * (ce - cs)).norm()
            # print(self.facet_bending_pairs_distance[i])
            self.facet_crease_initial_length[i] = (ce - cs).norm()

        #初始化折痕折角
        for i in ti.ndrange(self.bending_num_param[None]):
            self.crease_angle[i] = 0.0
            self.crease_folding_angle[i] = 0.0
            self.previous_dir[i] = 0.0
            if numpy_target_angle.shape[0] > 1:
                self.target_crease_angle[i] = numpy_target_angle[i]
            else:
                self.target_crease_angle[i] = 1.

        # 初始化折痕类型
        for i in ti.ndrange(self.bending_num_param[None]):
            for j in ti.ndrange(self.line_total_indice_num_param[None]):
                if numpy_crease_pairs[i, 0] == int(numpy_line_indices[j, 0]) and numpy_crease_pairs[i, 1] == int(numpy_line_indices[j, 1]):
                    self.crease_type[i] = int(numpy_line_indices[j, 2])

                    if numpy_line_indices[j, 3] > tm.pi:
                        self.folding_angle_upper_bound[i] = tm.pi
                    else:
                        self.folding_angle_upper_bound[i] = numpy_line_indices[j, 3]

                    if numpy_line_indices[j, 4] < -tm.pi:
                        self.folding_angle_lower_bound[i] = -tm.pi
                    else:
                        self.folding_angle_lower_bound[i] = numpy_line_indices[j, 4]

                    break
                    
        # ti.loop_config(serialize=True)
        self.sequence_level[0] = 0
        self.sequence_level[1] = 0

        # 初始化折叠等级和系数
        for i, j in ti.ndrange(self.bending_num_param[None], numpy_tb_line.shape[0]):
            kp1 = [numpy_original_kps[numpy_crease_pairs[i, 0], X], numpy_original_kps[numpy_crease_pairs[i, 0], Y]]
            kp2 = [numpy_original_kps[numpy_crease_pairs[i, 1], X], numpy_original_kps[numpy_crease_pairs[i, 1], Y]]
            kp11 = [numpy_tb_line[j, 0], numpy_tb_line[j, 1]]
            kp22 = [numpy_tb_line[j, 2], numpy_tb_line[j, 3]]
            if (((kp1[X] - kp11[X]) ** 2 + (kp1[Y] - kp11[Y]) ** 2) <= 16. and \
                ((kp2[X] - kp22[X]) ** 2 + (kp2[Y] - kp22[Y]) ** 2) <= 16.) or \
                (((kp1[X] - kp22[X]) ** 2 + (kp1[Y] - kp22[Y]) ** 2) <= 16. and \
                ((kp2[X] - kp11[X]) ** 2 + (kp2[Y] - kp11[Y]) ** 2) <= 16.):

                self.crease_level[i] = int(numpy_tb_line[j, 4])
                self.crease_coeff[i] = numpy_tb_line[j, 5]
                
                for k in ti.static(range(self.maximum_level_number)):
                    self.recover_level_need[i, k] = numpy_recover_level_need[j, k]
                    self.recover_level[i, k] = numpy_recover_level[j, k]
                    self.recover_angle[i, k] = numpy_recover_angle[j, k]
                    if self.recover_level[i, k] > self.sequence_level[0]:
                        self.sequence_level[0] = self.recover_level[i, k]
                    if self.recover_level[i, k] < self.sequence_level[1]:
                        self.sequence_level[1] = self.recover_level[i, k]
                
            if self.crease_level[i] > self.sequence_level[0]:
                self.sequence_level[0] = self.crease_level[i]
            if self.crease_level[i] < self.sequence_level[1]:
                self.sequence_level[1] = self.crease_level[i]
                
        self.folding_micro_step[None] = tm.pi / 100.0 / (1 + self.sequence_level[0] - self.sequence_level[1])

        # 初始化渲染的线的颜色信息
        for i in ti.ndrange(self.line_total_indice_num_param[None]):
            if numpy_line_indices[i, 2] == BORDER:
                self.line_color[2 * i] = [0, 0, 0]
                self.line_color[2 * i + 1] = [0, 0, 0]
            elif numpy_line_indices[i, 2] == VALLEY:
                self.line_color[2 * i] = [0, 0.17, 0.83]
                self.line_color[2 * i + 1] = [0, 0.17, 0.83]
            elif numpy_line_indices[i, 2] == MOUNTAIN:
                self.line_color[2 * i] = [0.75, 0.2, 0.05]
                self.line_color[2 * i + 1] = [0.75, 0.2, 0.05]
            else:
                self.line_color[2 * i] = [0.5, 0.5, 0.5]
                self.line_color[2 * i + 1] = [0.5, 0.5, 0.5]
                
            # pointer = 0
            # for j in ti.ndrange(unit_indices_num):
            #     unit_ids = self.unit_indices[j]
            #     exist1 = False
            #     exist2 = False
            #     for k in ti.ndrange(self.unit_edge_max):
            #         if unit_ids[k] == numpy_line_indices[i, 0]:
            #             exist1 = True
            #         if unit_ids[k] == numpy_line_indices[i, 1]:
            #             exist2 = True
            #     if exist1 and exist2:
            #         self.line_connection_unit[i][pointer] = j
            #         pointer += 1

        for i in ti.ndrange(self.kp_num_param[None]):
            self.x[i] = self.original_vertices[i]
            self.x0[i] = self.original_vertices[i]
            self.s[i] = self.original_vertices[i]
            self.v[i] = [0., 0., 0.]
            self.dv[i] = [0., 0., 0.]
        
        # assemble cons
        # 1. spring
        current_count = 0
        current_connection_pos = 0
        spring_count = 0
        thick_count = 0

        ti.loop_config(serialize=True)
        for s in ti.ndrange(split_origami_num):
            # in-panel
            range_down = numpy_split_start_kp_id[s]
            range_up = range_down
            if s == split_origami_num - 1:
                if numpy_additional_kp_ori_id[0, 0] != -1:
                    range_up = self.kp_num_param[None] - numpy_additional_kp_ori_id.shape[0]
                else:
                    range_up = self.kp_num_param[None]
            else:
                range_up = numpy_split_start_kp_id[s + 1]

            for i in ti.ndrange((range_down, range_up)):
                for j in ti.ndrange((range_down, range_up)):
                    if i < j and numpy_connection_matrix[i, j] > 0.:
                        self.spring_selection[current_count * 2 + 0] = i
                        self.spring_selection[current_count * 2 + 1] = j
                        self.spring_x_proj[current_count * 2 + 0] = self.x[i]
                        self.spring_x_proj[current_count * 2 + 1] = self.x[j]
                        self.spring_original_length[current_count] = (self.original_vertices[j] - self.original_vertices[i]).norm()
                        self.spring_selection_corresponding_origami_id[current_count] = s
                        current_count += 1
                        spring_count += 1
            
            # additional
            if numpy_additional_kp_ori_id[0, 0] != -1:
                for k in ti.ndrange(numpy_additional_kp_ori_id.shape[0]):
                    if s == numpy_additional_kp_ori_id[k, 1]:
                        j = numpy_additional_kp_ori_id[k, 0]
                        for i in ti.ndrange(j):
                            if numpy_connection_matrix[i, j] > 0.:
                                self.spring_selection[current_count * 2 + 0] = i
                                self.spring_selection[current_count * 2 + 1] = j
                                self.spring_x_proj[current_count * 2 + 0] = self.x[i]
                                self.spring_x_proj[current_count * 2 + 1] = self.x[j]
                                self.spring_original_length[current_count] = (self.original_vertices[j] - self.original_vertices[i]).norm()
                                self.spring_selection_corresponding_origami_id[current_count] = s
                                current_count += 1
                                spring_count += 1

            # panel connection
            while current_connection_pos < numpy_connected_unit_id.shape[0] and numpy_split_connection[current_connection_pos] == s:
                index1 = numpy_connected_unit_id[current_connection_pos, 0]
                index2 = numpy_connected_unit_id[current_connection_pos, 1]
                indices1 = self.unit_indices[index1]
                indices2 = self.unit_indices[index2]
                lens = self.unit_kp_num_list[index1]
                for j in ti.ndrange(lens):
                    self.spring_selection[current_count * 2 + 0] = indices1[j]
                    self.spring_selection[current_count * 2 + 1] = indices2[j]
                    self.spring_x_proj[current_count * 2 + 0] = self.x[indices1[j]]
                    self.spring_x_proj[current_count * 2 + 1] = self.x[indices2[j]]
                    self.spring_original_length[current_count] = (self.original_vertices[indices1[j]] - self.original_vertices[indices2[j]]).norm()
                    self.spring_selection_corresponding_origami_id[current_count] = s
                    current_count += 1
                    thick_count += 1
                for j in ti.ndrange(lens):
                    self.spring_selection[current_count * 2 + 0] = indices1[j]
                    self.spring_selection[current_count * 2 + 1] = indices2[(j + 1) % lens]
                    self.spring_x_proj[current_count * 2 + 0] = self.x[indices1[j]]
                    self.spring_x_proj[current_count * 2 + 1] = self.x[indices2[(j + 1) % lens]]
                    self.spring_original_length[current_count] = (self.original_vertices[indices1[j]] - self.original_vertices[indices2[(j + 1) % lens]]).norm()
                    self.spring_selection_corresponding_origami_id[current_count] = s
                    current_count += 1
                    thick_count += 1
                for j in ti.ndrange(lens):
                    self.spring_selection[current_count * 2 + 0] = indices1[j]
                    self.spring_selection[current_count * 2 + 1] = indices2[(j - 1 + lens) % lens]
                    self.spring_x_proj[current_count * 2 + 0] = self.x[indices1[j]]
                    self.spring_x_proj[current_count * 2 + 1] = self.x[indices2[(j - 1 + lens) % lens]]
                    self.spring_original_length[current_count] = (self.original_vertices[indices1[j]] - self.original_vertices[indices2[(j - 1 + lens) % lens]]).norm()
                    self.spring_selection_corresponding_origami_id[current_count] = s
                    current_count += 1
                    thick_count += 1
                current_connection_pos += 1
            
        if (spring_count + thick_count) < self.spring_num_param[None]:
            if verbose:
                print(f"Warning: real spring count is {spring_count} + {thick_count} = {(spring_count + thick_count)}, less than {self.spring_num_param[None]}")
            self.spring_num_param[None] = spring_count + thick_count

        # 2. bending
        interval = int(self.bending_num_param[None] / split_origami_num)
        ti.loop_config(serialize=False)
        for i in ti.ndrange(self.bending_num_param[None]):
            index1 = numpy_crease_pairs[i, 0]
            index2 = numpy_crease_pairs[i, 1]
            index3 = numpy_bending_pairs[i, 0]
            index4 = numpy_bending_pairs[i, 1]
            self.bending_selection[i * 4 + 0] = index1
            self.bending_selection[i * 4 + 1] = index2
            self.bending_selection[i * 4 + 2] = index3
            self.bending_selection[i * 4 + 3] = index4
            self.bending_x_proj[i * 4 + 0] = self.x[index1]
            self.bending_x_proj[i * 4 + 1] = self.x[index2]
            self.bending_x_proj[i * 4 + 2] = self.x[index3]
            self.bending_x_proj[i * 4 + 3] = self.x[index4]
            self.bending_selection_corresponding_origami_id[i] = i // interval

        # 3. facet_bending
        ti.loop_config(serialize=False)
        for i in ti.ndrange(self.facet_bending_num_param[None]):
            index1 = numpy_facet_crease_pairs[i, 0]
            index2 = numpy_facet_crease_pairs[i, 1]
            index3 = numpy_facet_bending_pairs[i, 0]
            index4 = numpy_facet_bending_pairs[i, 1]
            self.facet_bending_selection[i * 4 + 0] = index1
            self.facet_bending_selection[i * 4 + 1] = index2
            self.facet_bending_selection[i * 4 + 2] = index3
            self.facet_bending_selection[i * 4 + 3] = index4
            self.facet_bending_x_proj[i * 4 + 0] = self.x[index1]
            self.facet_bending_x_proj[i * 4 + 1] = self.x[index2]
            self.facet_bending_x_proj[i * 4 + 2] = self.x[index3]
            self.facet_bending_x_proj[i * 4 + 3] = self.x[index4]
            self.facet_bending_selection_corresponding_origami_id[i] = numpy_split_facet_id[i]

    @ti.func
    def cotangent(self, vs, v1, v2):
        e02 = v1 - vs
        e12 = v2 - vs
        cos_alpha = tm.dot(e02, e12)
        sin_alpha = tm.cross(e02, e12).norm()
        cot_alpha = cos_alpha / sin_alpha
        return cot_alpha

    @ti.kernel
    def compute_bending_cotangent_weights(self):
        """Compute cotangent weights from initial configuration for bending constraints.
        For each bending quad (v0,v1,v2,v3) where edge v0-v1 is the shared edge:
          alpha = angle at v2 (opposite to v0-v1 from triangle v0-v2-v1)
          beta  = angle at v3 (opposite to v0-v1 from triangle v0-v1-v3)
        C matrix (4x4 cotangent Laplacian, symmetric):
          C = [[cab, -cab, -ca, -cb],
               [-cab, cab, -cb, -ca],
               [-ca, -cb, ca,  0 ],
               [-cb, -ca,  0,  cb]]
        where ca = cot(alpha), cb = cot(beta), cab = ca + cb
        """
        for i in ti.ndrange(self.bending_num_param[None]):
            index0 = self.bending_selection[i * 4 + 0]
            index1 = self.bending_selection[i * 4 + 1]
            index2 = self.bending_selection[i * 4 + 2]
            index3 = self.bending_selection[i * 4 + 3]

            # triangle 1: 
            cot_alpha_0 = self.cotangent(self.x[index0], self.x[index2], self.x[index1])
            cot_alpha_1 = self.cotangent(self.x[index1], self.x[index0], self.x[index2])
            cot_alpha_2 = self.cotangent(self.x[index2], self.x[index1], self.x[index0])
            # triangle 2: 
            cot_beta_0 = self.cotangent(self.x[index0], self.x[index1], self.x[index3])
            cot_beta_1 = self.cotangent(self.x[index1], self.x[index3], self.x[index0])
            cot_beta_3 = self.cotangent(self.x[index3], self.x[index0], self.x[index1])

            k = self.bending_k_param[None] * self.crease_initial_length[i]

            val = (cot_alpha_2 + cot_beta_3)

            self.cotangent_matrix[i][0, 0] += val * k
            self.cotangent_matrix[i][1, 1] += val * k
            self.cotangent_matrix[i][0, 1] -= val * k
            self.cotangent_matrix[i][1, 0] -= val * k

            self.cotangent_matrix[i][0, 0] += (cot_alpha_1) * k
            self.cotangent_matrix[i][2, 2] += (cot_alpha_1) * k
            self.cotangent_matrix[i][0, 2] -= (cot_alpha_1) * k
            self.cotangent_matrix[i][2, 0] -= (cot_alpha_1) * k

            self.cotangent_matrix[i][0, 0] += (cot_beta_1) * k
            self.cotangent_matrix[i][3, 3] += (cot_beta_1) * k
            self.cotangent_matrix[i][0, 3] -= (cot_beta_1) * k
            self.cotangent_matrix[i][3, 0] -= (cot_beta_1) * k

            self.cotangent_matrix[i][1, 1] += (cot_alpha_0) * k
            self.cotangent_matrix[i][2, 2] += (cot_alpha_0) * k
            self.cotangent_matrix[i][1, 2] -= (cot_alpha_0) * k
            self.cotangent_matrix[i][2, 1] -= (cot_alpha_0) * k

            self.cotangent_matrix[i][1, 1] += (cot_beta_0) * k
            self.cotangent_matrix[i][3, 3] += (cot_beta_0) * k
            self.cotangent_matrix[i][1, 3] -= (cot_beta_0) * k
            self.cotangent_matrix[i][3, 1] -= (cot_beta_0) * k

            # self.cotangent_matrix[i][0, 0] += 1e-6
            # self.cotangent_matrix[i][1, 1] += 1e-6
            # self.cotangent_matrix[i][2, 2] += 1e-6
            # self.cotangent_matrix[i][3, 3] += 1e-6

        for i in ti.ndrange(self.facet_bending_num_param[None]):
            index0 = self.facet_bending_selection[i * 4 + 0]
            index1 = self.facet_bending_selection[i * 4 + 1]
            index2 = self.facet_bending_selection[i * 4 + 2]
            index3 = self.facet_bending_selection[i * 4 + 3]

            # triangle 1: 
            cot_alpha_0 = self.cotangent(self.x[index0], self.x[index2], self.x[index1])
            cot_alpha_1 = self.cotangent(self.x[index1], self.x[index0], self.x[index2])
            cot_alpha_2 = self.cotangent(self.x[index2], self.x[index1], self.x[index0])
            # triangle 2: 
            cot_beta_0 = self.cotangent(self.x[index0], self.x[index1], self.x[index3])
            cot_beta_1 = self.cotangent(self.x[index1], self.x[index3], self.x[index0])
            cot_beta_3 = self.cotangent(self.x[index3], self.x[index0], self.x[index1])

            k = self.facet_bending_k_param[None] * self.facet_crease_initial_length[i]

            val = (cot_alpha_2 + cot_beta_3)

            self.facet_cotangent_matrix[i][0, 0] += val * k
            self.facet_cotangent_matrix[i][1, 1] += val * k
            self.facet_cotangent_matrix[i][0, 1] -= val * k
            self.facet_cotangent_matrix[i][1, 0] -= val * k

            self.facet_cotangent_matrix[i][0, 0] += (cot_alpha_1) * k
            self.facet_cotangent_matrix[i][2, 2] += (cot_alpha_1) * k
            self.facet_cotangent_matrix[i][0, 2] -= (cot_alpha_1) * k
            self.facet_cotangent_matrix[i][2, 0] -= (cot_alpha_1) * k

            self.facet_cotangent_matrix[i][0, 0] += (cot_beta_1) * k
            self.facet_cotangent_matrix[i][3, 3] += (cot_beta_1) * k
            self.facet_cotangent_matrix[i][0, 3] -= (cot_beta_1) * k
            self.facet_cotangent_matrix[i][3, 0] -= (cot_beta_1) * k

            self.facet_cotangent_matrix[i][1, 1] += (cot_alpha_0) * k
            self.facet_cotangent_matrix[i][2, 2] += (cot_alpha_0) * k
            self.facet_cotangent_matrix[i][1, 2] -= (cot_alpha_0) * k
            self.facet_cotangent_matrix[i][2, 1] -= (cot_alpha_0) * k

            self.facet_cotangent_matrix[i][1, 1] += (cot_beta_0) * k
            self.facet_cotangent_matrix[i][3, 3] += (cot_beta_0) * k
            self.facet_cotangent_matrix[i][1, 3] -= (cot_beta_0) * k
            self.facet_cotangent_matrix[i][3, 1] -= (cot_beta_0) * k

            # self.facet_cotangent_matrix[i][0, 0] += 1e-6
            # self.facet_cotangent_matrix[i][1, 1] += 1e-6
            # self.facet_cotangent_matrix[i][2, 2] += 1e-6
            # self.facet_cotangent_matrix[i][3, 3] += 1e-6
        
        # for i in ti.ndrange(self.bending_num_param[None]):
        #     for j, k in ti.ndrange(4, 4):
        #         if j == k:
        #             self.cotangent_matrix[i][j, k] = 1.
        #         else:
        #             self.cotangent_matrix[i][j, k] = 0.
        
        # for i in ti.ndrange(self.facet_bending_num_param[None]):
        #     for j, k in ti.ndrange(4, 4):
        #         if j == k:
        #             self.facet_cotangent_matrix[i][j, k] = 1.
        #         else:
        #             self.facet_cotangent_matrix[i][j, k] = 0.

    @ti.kernel
    def fill_AK_field(self, h: data_type):
        for i in ti.ndrange(self.kp_max_num_param[None]):
            self.AK_field[i * 3 + 0, i * 3 + 0] += self.masses[i] / (h ** 2)
            self.AK_field[i * 3 + 1, i * 3 + 1] += self.masses[i] / (h ** 2)
            self.AK_field[i * 3 + 2, i * 3 + 2] += self.masses[i] / (h ** 2)

        for i in ti.ndrange(self.spring_num_param[None]):
            index1 = self.spring_selection[i * 2 + 0]
            index2 = self.spring_selection[i * 2 + 1]
            for j in ti.static(range(3)):
                self.AK_field[index1 * 3 + j, index1 * 3 + j] += self.spring_k_param[None]
                self.AK_field[index2 * 3 + j, index2 * 3 + j] += self.spring_k_param[None]
                self.AK_field[index1 * 3 + j, index2 * 3 + j] -= self.spring_k_param[None]
                self.AK_field[index2 * 3 + j, index1 * 3 + j] -= self.spring_k_param[None]

        for i, j, k in ti.ndrange(self.bending_num_param[None], 4, 4):
            for l in ti.static(range(3)):
                self.AK_field[self.bending_selection[i * 4 + j] * 3 + l, self.bending_selection[i * 4 + k] * 3 + l] += self.cotangent_matrix[i][j, k]
   
        for i, j, k in ti.ndrange(self.facet_bending_num_param[None], 4, 4):
            for l in ti.static(range(3)):
                self.AK_field[self.facet_bending_selection[i * 4 + j] * 3 + l, self.facet_bending_selection[i * 4 + k] * 3 + l] += self.facet_cotangent_matrix[i][j, k]

    @ti.kernel
    def construct_hessian(self, builder: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(3 * self.kp_max_num_param[None], 3 * self.kp_max_num_param[None]):
            if self.AK_field[i, j] != 0:
                builder[i, j] += self.AK_field[i, j]

    def initializeRunning(self):
        # parameters reset
        self.dead_count = 0
        self.positive_count = 0
        self.candidate_method = True
        self.recorded_t = []
        self.recorded_max_force = []
        self.recorded_nodal_maximum_force = []
        self.recorded_folding_percent = []
        self.recorded_folding_error = []
        self.recorded_maximum_folding_percent = []
        self.recorded_minimum_folding_percent = []
        self.recorded_maximum_folding_error = []
        self.recorded_minimum_folding_error = []
        
        self.stable_state = 0
        self.past_move_indice = 0.0
        self.folding_percent = 0.0
        self.abs_folding_percent = 0.0
        self.folding_error = 0.0
        self.offset_x = 0.
        self.offset_y = 0.

        self.folding_angle = 0.
        self.enable_add_folding_angle = 0.

        # Clear per-pair contact tracking + paint locus on restart ('r')
        self._collision_active_segments = {}
        self._collision_active_points = {}
        self._collision_fixed_segments = {}
        self._collision_fixed_points = {}
        self._collision_coords_exported = False
        self._sweep_draw_stopped = False
        self._collision_contact_count = 0
        self._collision_segment_count = 0
        self._collision_contact_points_list = []
        self._collision_contact_segments_list = []
        self._collision_contact_points_flat = []
        self._collision_contact_segments_flat = []
        self._collision_stats = None
        self._intruder_classifications = []
        self._intruder_by_unit_pair = {}
        self._intruder_indice_count = 0
        self._collision_ran_once = False
        self._collision_frame_i = 0
        # Surface sweep paint (red locus on panels)
        self._paint_canvas = {}
        self._paint_line_vert_count = 0
        self._paint_n_strokes = 0
        self._paint_dirty = True
        self._stamp_paint_needed = False
        self._debug_lines = None
        self._debug_lines_key = None
        if hasattr(self, "paint_line_vertices"):
            try:
                self.paint_line_vertices.fill(0)
            except Exception:
                pass
        if hasattr(self, "collision_contact_points"):
            try:
                self.collision_contact_points.fill(0)
                self.collision_contact_lines.fill(0)
            except Exception:
                pass
        if hasattr(self, "intruder_indices"):
            try:
                self.intruder_indices.fill(0)
            except Exception:
                pass

        self.substeps = 1
        self.dt = 1. / (60. * self.substeps) #仿真的时间间隔
        self.basic_dt = self.substeps * self.dt
        self.now_t = 0.

        # if self.use_gui:
        self.camera.position(-1.1 * self.max_size, min(-1.1 * self.max_size, -400), max(1.1 * self.max_size, 400))
        self.camera.up(0, 0, 1.0)
        self.camera.lookat(0, 0, 0)
        self.camera.z_far(max(10 * self.max_size, 5000.))
        self.ITER = 10

        self.image_id = 0

        self.current_t = 0.0

        self.image_id = 0
        
        numpy_indices                       = np.array(self.ori_sim.indices, dtype=np.int32)
        for i in range(len(self.contributions)):
            if self.units[i].repaired:
                self.contributions[i] = self.units[i].getContribution()
            if len(self.contributions[i]):
                for j in range(len(self.contributions[i]), self.unit_edge_max):
                    self.contributions[i].append(0.)
                    
        numpy_fix_id = np.array(self.fix_id, dtype=np.int32)
        numpy_connected_unit_id = np.array([[-1, -1]])
        if len(self.connected_unit_pairs):
            numpy_connected_unit_id = np.array(self.connected_unit_pairs, dtype=np.int32)
        numpy_split_connection = np.array([-1])
        if len(self.split_connection):
            numpy_split_connection = np.array(self.split_connection) #每个厚板弹簧约束属于的折纸id

        numpy_contributions                 = np.array(self.contributions, dtype=numpy_data_type)
        numpy_kps                           = np.array(self.kps, dtype=numpy_data_type) - np.array(self.total_bias + [0.], dtype=numpy_data_type)
        numpy_original_kps                  = np.array(self.kps, dtype=numpy_data_type)
        numpy_mass_list                     = np.array(self.mass_list, dtype=numpy_data_type)
        numpy_tri_indices                   = np.array(self.tri_indices, dtype=np.int32)
        numpy_connection_matrix             = np.array(self.ori_sim.connection_matrix)
        numpy_bending_pairs                 = np.array(self.ori_sim.bending_pairs, dtype=np.int32)
        numpy_crease_pairs                  = np.array(self.ori_sim.crease_pairs, dtype=np.int32)
        numpy_line_indices                  = np.array(self.ori_sim.getNewLineIndices(), dtype=numpy_data_type)

        numpy_split_start_kp_id = np.array(self.split_kp_sets_min) #每个单独折纸的起始结点id
        numpy_split_facet_id = np.array(self.facet_split_sets) #每个面折痕属于的折纸id
        numpy_additional_kp_origami_id = np.array([[-1, -1]])
        if len(self.additional_kp_origami_id):
            numpy_additional_kp_origami_id = np.array(self.additional_kp_origami_id)

        if len(self.ori_sim.facet_bending_pairs) == 0:
            numpy_facet_bending_pairs           = np.array([[0, 0]], dtype=np.int32)
            numpy_facet_crease_pairs            = np.array([[0, 0]], dtype=np.int32)
        else:
            numpy_facet_bending_pairs           = np.array(self.ori_sim.facet_bending_pairs, dtype=np.int32)
            numpy_facet_crease_pairs            = np.array(self.ori_sim.facet_crease_pairs, dtype=np.int32)
        
        # construct tb_line information which contains start, end, level and coeff
        tb_line = []
        for line in self.lines:
            tb_line.append([line[START][X], line[START][Y], line[END][X], line[END][Y], line.level, line.coeff])

        numpy_tb_line                       = np.array(tb_line, dtype=numpy_data_type)

        maximum_recover_level_length = max([len(line.recover_level) for line in self.lines])
        if maximum_recover_level_length > self.maximum_level_number:
            raise NotImplementedError
        numpy_recover_level_need = np.zeros(shape=(len(self.lines), self.maximum_level_number), dtype=bool)
        numpy_recover_level = np.zeros(shape=(len(self.lines), self.maximum_level_number), dtype=int)
        numpy_recover_angle = np.zeros(shape=(len(self.lines), self.maximum_level_number), dtype=numpy_data_type)
        for i in range(len(self.lines)):
            length = len(self.lines[i].recover_level)
            for j in range(maximum_recover_level_length):
                if j < length:
                    numpy_recover_level_need[i][j] = 1
                    numpy_recover_level[i][j] = self.lines[i].recover_level[j]
                    numpy_recover_angle[i][j] = self.lines[i].recover_angle[j]
                else:
                    numpy_recover_level_need[i][j] = 0
        
        if len(self.target):
            numpy_target_angle = np.array(self.target, dtype=numpy_data_type)
        else:
            numpy_target_angle = np.array([0.])
                    
        # initialize!
        self.initialize(
            numpy_indices, 
            numpy_kps, 
            numpy_mass_list, 
            numpy_tri_indices, 
            numpy_connection_matrix, 
            numpy_bending_pairs, 
            numpy_crease_pairs, 
            numpy_line_indices, 
            numpy_facet_bending_pairs, 
            numpy_facet_crease_pairs,
            numpy_original_kps, 
            numpy_tb_line,
            numpy_contributions,
            numpy_recover_level_need, 
            numpy_recover_level, 
            numpy_recover_angle, 
            numpy_fix_id, 
            numpy_connected_unit_id, 
            self.dt,
            self.kp_num,
            self.maximum_kp_number,
            self.unit_indices_num,
            self.line_total_indice_num,
            self.indices_num,
            self.spring_k, 
            self.bending_k, 
            self.facet_bending_k,
            self.spring_cons_num,
            self.bending_cons_num,
            self.facet_bending_cons_num,
            self.folding_angle,
            self.enable_add_folding_angle,
            self.damping,
            self.angle_protection,
            self.collision_indice,
            self.collision_d,
            self.split_origami_num,
            numpy_split_start_kp_id,
            numpy_split_facet_id,
            numpy_split_connection,
            numpy_additional_kp_origami_id,
            numpy_target_angle,
            self.verbose,
        )

        # print(self.spring_selection_corresponding_origami_id, self.bending_selection_corresponding_origami_id, self.facet_bending_selection_corresponding_origami_id)
        self.compute_bending_cotangent_weights()
        self.fill_AK_field(self.dt)

        self.construct_hessian(self.AK)
        self.AM = self.AK.build() # 1 time
        self.sparse_solver.compute(self.AM)  # A 矩阵在仿真期间不变，提前分解 / A is constant, factorize once
        if self.collision_shading:
            self._build_collision_topology()

    def deal_with_key(self, key):
        self.key = ''
        if key == 'r':
            self.initializeRunning()
        elif key == 'u': 
            self.folding_angle += math.pi
            if self.folding_angle >= math.pi:
                self.folding_angle = math.pi
                if self.collision_shading:
                    self._stop_sweep_drawing_at_pi()
                if float(getattr(self, "enable_add_folding_angle", 0.0)) > 0.0:
                    self.enable_add_folding_angle = 0.0
        elif key == 'j': 
            self.folding_angle -= math.pi
            if self.folding_angle <= 0:
                self.folding_angle = 0
        elif key == 'i': 
            # Do not resume auto-fold past a sealed π fold without restart
            if getattr(self, "_sweep_draw_stopped", False):
                self.enable_add_folding_angle = 0.0
            else:
                self.enable_add_folding_angle = self.folding_micro_step[None]
        elif key == 'k': 
            self.enable_add_folding_angle = 0.0
        elif key == 'm': 
            self.enable_add_folding_angle = -self.folding_micro_step[None]
        elif key == 'p':
            self.paused = not self.paused
        elif key == ti.ui.SPACE:
            self.step_once = True
        self.key = key

    @ti.func
    def getSkewMatrix(self, x):
        """
        计算向量的斜对称矩阵（叉积矩阵）。
        Calculate the skew-symmetric matrix (cross product matrix) of a vector.
        
        :param x: 输入向量 / Input vector
        :return: 3x3斜对称矩阵 / 3x3 skew-symmetric matrix
        """
        return ti.Matrix.cols([[0., x[Z], -x[Y]], [-x[Z], 0., x[X]], [x[Y], -x[X], 0.]])
    
    @ti.func
    def getDthetaDx(self, x0, x1, x2, x3, theta):
        """
        计算折痕角度对四个顶点位置的梯度。
        Calculate the gradient of crease angle with respect to four vertex positions.
        
        :param x0: 折痕起点 / Crease start point
        :param x1: 第一面板点 / First panel point
        :param x2: 折痕终点 / Crease end point
        :param x3: 第二面板点 / Second panel point
        :param theta: 当前折痕角度 / Current crease angle
        :return: 四个梯度矩阵 (dtheta/dx0, dtheta/dx1, dtheta/dx2, dtheta/dx3) / Four gradient matrices
        """
        s1 = x1 - x0
        cr = x2 - x0
        s2 = x3 - x0

        e = -cr / cr.norm() # valley crease is positive

        v1 = cr.cross(s2)
        v2 = s1.cross(cr)

        v1_norm = v1.norm()
        v2_norm = v2.norm()

        n1 = v1 / v1_norm
        n2 = v2 / v2_norm

        proj_v1 = tm.eye(3) - n1.outer_product(n1)
        proj_v2 = tm.eye(3) - n2.outer_product(n2)

        dv1dx0 = self.getSkewMatrix(x3 - x2)
        # dv1dx1 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        dv1dx2 = self.getSkewMatrix(x0 - x3)
        dv1dx3 = self.getSkewMatrix(x2 - x0)

        dv2dx0 = self.getSkewMatrix(x2 - x1)
        dv2dx1 = self.getSkewMatrix(x0 - x2)
        dv2dx2 = self.getSkewMatrix(x1 - x0)
        # dv2dx3 = ti.Matrix.cols([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

        v1_skew = self.getSkewMatrix(v1)
        v2_skew = self.getSkewMatrix(v2)

        cos_term_x0 = (dv2dx0 @ proj_v2 @ v1_skew - dv1dx0 @ proj_v1 @ v2_skew) @ e
        cos_term_x1 = (dv2dx1 @ proj_v2 @ v1_skew) @ e
        cos_term_x2 = (dv2dx2 @ proj_v2 @ v1_skew - dv1dx2 @ proj_v1 @ v2_skew) @ e
        cos_term_x3 = (-dv1dx3 @ proj_v1 @ v2_skew) @ e

        sin_term_x0 = dv1dx0 @ proj_v1 @ v2 + dv2dx0 @ proj_v2 @ v1
        sin_term_x1 = dv2dx1 @ proj_v2 @ v1
        sin_term_x2 = dv1dx2 @ proj_v1 @ v2 + dv2dx2 @ proj_v2 @ v1
        sin_term_x3 = dv1dx3 @ proj_v1 @ v2

        k = 1. / (v1_norm * v2_norm)

        dthetadx0 = k * (tm.cos(theta) * cos_term_x0 + tm.sin(theta) * sin_term_x0)
        dthetadx1 = k * (tm.cos(theta) * cos_term_x1 + tm.sin(theta) * sin_term_x1)
        dthetadx2 = k * (tm.cos(theta) * cos_term_x2 + tm.sin(theta) * sin_term_x2)
        dthetadx3 = k * (tm.cos(theta) * cos_term_x3 + tm.sin(theta) * sin_term_x3)

        return dthetadx0, dthetadx1, dthetadx2, dthetadx3
    
    @ti.func
    def compute_signed_dihedral(self, cs, ce, p0, p1, crease_type, id):
        """
        计算有符号二面角。
        返回: (signed_theta, dir_val, norm_val, e_axis)
        signed_theta 范围: [-2π, 2π]（含翻转处理，与现有代码一致）
        """
        barrier_left = tm.pi / 36.
        collision_indice = self.collision_indice_param[None]
        collision_d = self.collision_d_param[None]

        folding_angle_upper_bound = tm.pi
        upper_barrier = folding_angle_upper_bound - barrier_left
        folding_angle_lower_bound = -tm.pi
        lower_barrier = folding_angle_lower_bound + barrier_left

        if id != -1:
            folding_angle_upper_bound = self.folding_angle_upper_bound[id]
            upper_barrier = folding_angle_upper_bound - barrier_left
            folding_angle_lower_bound = self.folding_angle_lower_bound[id]
            lower_barrier = folding_angle_lower_bound + barrier_left

        upper_barrier_maximum = collision_indice * (2 * barrier_left * tm.log(collision_d / barrier_left) - barrier_left ** 2 / collision_d) #negative
        upper_barrier_df_maximum = 2 * collision_indice * (2 * barrier_left / collision_d + barrier_left ** 2 / (2 * (collision_d ** 2)) - tm.log(collision_d / barrier_left)) #positive

        lower_barrier_maximum = -upper_barrier_maximum #positive
        lower_barrier_df_maximum = upper_barrier_df_maximum #positive

        xc = ce - cs
        f11 = p0 - cs
        f22 = p1 - cs
        n1 = xc.cross(f11)      # (ce-cs) × (p0-cs)
        n2 = f22.cross(xc)      # (p1-cs) × (ce-cs)
        n1_norm = n1.norm()
        n2_norm = n2.norm()
        multi_n1_n2 = n1_norm * n2_norm
        dir_val = n1.cross(n2).dot(xc)
        val = n1.dot(n2)
        norm_val = val / multi_n1_n2

        # clamp
        norm_val = tm.clamp(norm_val, -1., 1.)
        theta_unsigned = tm.acos(norm_val)

        if abs(theta_unsigned) >= tm.pi - barrier_left:
            self.folding_angle_reach_pi[None] = True

        barrier_force = 0.

        # signed_theta（与 getBendingForce 完全一致）
        signed_theta = 0.0
        if dir_val >= 0.:  # mountain
            if id != -1 and self.previous_dir[id] <= 0 and norm_val <= 0.5:
                signed_theta = 2. * tm.pi - theta_unsigned
                if crease_type == VALLEY:
                    self.crease_angle[id] = abs(signed_theta / tm.pi)
                else:
                    self.crease_angle[id] = -abs(signed_theta / tm.pi)
                barrier_force = upper_barrier_maximum - (signed_theta - folding_angle_upper_bound) * upper_barrier_df_maximum
            else:
                signed_theta = -theta_unsigned
                if id != -1:
                    if crease_type == VALLEY:
                        self.crease_angle[id] = -abs(signed_theta / tm.pi)
                    else:
                        self.crease_angle[id] = abs(signed_theta / tm.pi)
                    self.previous_dir[id] = dir_val      
                t11 = signed_theta - lower_barrier
                t22 = folding_angle_lower_bound - collision_d - signed_theta
                if t11 <= 0 and t11 >= -barrier_left:
                    barrier_force = collision_indice * (2 * t11 * tm.log(t22 / -(barrier_left)) - t11 ** 2 / t22)
                elif t11 < -barrier_left:
                    barrier_force = lower_barrier_maximum - (signed_theta - folding_angle_lower_bound) * lower_barrier_df_maximum
        else:  # valley
            if id != -1 and self.previous_dir[id] >= 0 and norm_val <= 0.5:
                signed_theta = theta_unsigned - 2. * tm.pi
                if crease_type == VALLEY:
                    self.crease_angle[id] = -abs(signed_theta / tm.pi)
                else:
                    self.crease_angle[id] = abs(signed_theta / tm.pi)  
                barrier_force = lower_barrier_maximum - (signed_theta - folding_angle_lower_bound) * lower_barrier_df_maximum
            else:
                signed_theta = theta_unsigned
                if id != -1:
                    if crease_type == VALLEY:
                        self.crease_angle[id] = abs(signed_theta / tm.pi)
                    else:
                        self.crease_angle[id] = -abs(signed_theta / tm.pi)
                    self.previous_dir[id] = dir_val 
                t11 = signed_theta - upper_barrier
                t22 = folding_angle_upper_bound + collision_d - signed_theta
                if t11 >= 0 and t11 <= barrier_left:
                    barrier_force = collision_indice * (2 * t11 * tm.log(t22 / barrier_left) - t11 ** 2 / t22)
                elif t11 > barrier_left:
                    barrier_force = upper_barrier_maximum - (signed_theta - folding_angle_upper_bound) * upper_barrier_df_maximum
        
        return signed_theta, barrier_force
    
    @ti.func
    def project_dihedral_momentum_conserving(self, m1, m2, m3, m4, c0, c1, p0, p1, delta_theta):
        """
        基于离散微分几何二面角梯度公式的动量守恒弯曲投影。
        同时满足线动量守恒和角动量守恒。
        
        参考：Grinspun et al., "Discrete Shells" (2003)
        """
        e = c1 - c0
        e_len = e.norm()
        
        # total_mass = m1 + m2 + m3 + m4

        # 退化保护：若折痕边长过短，直接返回原位置
        # if e_len < 1e-12:
        #     return c0, c1, p0, p1
        
        # --- 计算两个三角面法向量与高度 ---
        # T1: (c0, c1, p0)
        n1_unnorm = tm.cross(e, p0 - c0)
        n1_norm = n1_unnorm.norm()
        # T2: (c0, c1, p1)，与用户 compute_signed_dihedral 的 n2 方向一致
        n2_unnorm = tm.cross(p1 - c0, e)
        n2_norm = n2_unnorm.norm()
        
        # 退化保护：若任一面积极小，返回原位置
        # if n1_norm < 1e-12 or n2_norm < 1e-12:
        #     return c0, c1, p0, p1
        
        n1 = n1_unnorm / n1_norm
        n2 = n2_unnorm / n2_norm
        h1 = n1_norm / e_len
        h2 = n2_norm / e_len
        
        # --- 四个顶点的二面角梯度（∂θ/∂x_i）---
        # 自由顶点梯度
        g_p0 = n1 / h1
        g_p1 = n2 / h2
        
        # 折痕顶点处的余切角
        cot_c0_t1 = self.cotangent(c0, c1, p0)  # ∠p0-c0-c1
        cot_c1_t1 = self.cotangent(c1, c0, p0)  # ∠p0-c1-c0
        cot_c0_t2 = self.cotangent(c0, c1, p1)  # ∠p1-c0-c1
        cot_c1_t2 = self.cotangent(c1, c0, p1)  # ∠p1-c1-c0
        
        # 分母保护
        denom1 = cot_c0_t1 + cot_c1_t1
        denom2 = cot_c0_t2 + cot_c1_t2
        if abs(denom1) < 1e-12:
            denom1 = 1e-12
        if abs(denom2) < 1e-12:
            denom2 = 1e-12
        
        # 折痕顶点梯度（由图片公式映射）
        g_c0 = ((-cot_c1_t1 / denom1) * g_p0) + ((-cot_c1_t2 / denom2) * g_p1)
        g_c1 = ((-cot_c0_t1 / denom1) * g_p0) + ((-cot_c0_t2 / denom2) * g_p1)
        
        # 验证：四个梯度之和为零（线动量守恒的充要条件）
        # g_p0 + g_p1 + g_c0 + g_c1 == 0 （代数恒等式）
        
        # --- XPBD 拉格朗日乘子 ---
        grad_norm_sq = g_p0.norm_sqr() + g_p1.norm_sqr() + g_c0.norm_sqr() + g_c1.norm_sqr()
        # if grad_norm_sq < 1e-24:
        #     return c0, c1, p0, p1
        
        lam = delta_theta / grad_norm_sq
        
        # 单步截断保护（与现有代码一致）
        max_disp = 1.0
        lam = tm.clamp(lam, -max_disp / tm.sqrt(grad_norm_sq), max_disp / tm.sqrt(grad_norm_sq))
        
        c0_proj = c0 + lam * g_c0
        c1_proj = c1 + lam * g_c1
        p0_proj = p0 + lam * g_p0
        p1_proj = p1 + lam * g_p1
        
        return c0_proj, c1_proj, p0_proj, p1_proj

    @ti.func
    def get_proj_and_height_vector(self, point, cs, axis):
        """
        绕 axis 旋转 point（右手定则）。
        axis 必须已归一化。旋转中心为 point 在轴上的垂足。
        """
        t = (point - cs).dot(axis)
        proj_point = cs + t * axis
        r = point - proj_point
        return proj_point, r

    @ti.func
    def rodrigues_rotate(self, proj_point, r, axis, angle):
        """
        绕 axis 旋转 point（右手定则）。
        axis 必须已归一化。旋转中心为 point 在轴上的垂足。
        """
        cos_a = tm.cos(angle)
        sin_a = tm.sin(angle)
        r_rot = r * cos_a + axis.cross(r) * sin_a
        return proj_point + r_rot
    
    # @ti.func
    # def getBendingForce(self, cs, ce, p1, p2, k, L, theta, crease_type, id=-1):
    #     """
    #     计算折痕的弯曲力。
    #     Calculate the bending force of a crease.
        
    #     :param cs: 折痕起点坐标 / Crease start point coordinates
    #     :param ce: 折痕终点坐标 / Crease end point coordinates
    #     :param p1: 第一面板点坐标 / First panel point coordinates
    #     :param p2: 第二面板点坐标 / Second panel point coordinates
    #     :param k: 弯曲刚度系数 / Bending stiffness coefficient
    #     :param theta: 目标折叠角度 / Target folding angle
    #     :param crease_type: 折痕类型（山折/谷折）/ Crease type (mountain/valley)
    #     :param debug: 是否启用调试模式 / Whether to enable debug mode
    #     :param enable_dynamic_change: 是否启用动态变化 / Whether to enable dynamic change
    #     :param a1: 第一面板面积 / First panel area
    #     :param a2: 第二面板面积 / Second panel area
    #     :param L: 折痕长度 / Crease length
    #     :param id: 折痕ID / Crease ID
    #     :param d: 厚度参数 / Thickness parameter
    #     :param tsa_mode: 是否为TSA模式 / Whether in TSA mode
    #     :return: 四个顶点的力向量 (f_cs, f_ce, f_p1, f_p2) / Force vectors for four vertices
    #     """
    #     # 求折痕的信息
    #     barrier_left = tm.pi / 36.
    #     collision_indice = self.collision_indice_param[None]
    #     collision_d = self.collision_d_param[None]

    #     folding_angle_upper_bound = tm.pi
    #     upper_barrier = folding_angle_upper_bound - barrier_left
    #     folding_angle_lower_bound = -tm.pi
    #     lower_barrier = folding_angle_lower_bound + barrier_left

    #     if id != -1:
    #         folding_angle_upper_bound = self.folding_angle_upper_bound[id]
    #         upper_barrier = folding_angle_upper_bound - barrier_left
    #         folding_angle_lower_bound = self.folding_angle_lower_bound[id]
    #         lower_barrier = folding_angle_lower_bound + barrier_left

    #     upper_barrier_energy_maximum = -collision_indice * barrier_left ** 2 * tm.log(collision_d / barrier_left) #positive
    #     upper_barrier_maximum = collision_indice * (2 * barrier_left * tm.log(collision_d / barrier_left) - barrier_left ** 2 / collision_d) #negative
    #     upper_barrier_df_maximum = 2 * collision_indice * (2 * barrier_left / collision_d + barrier_left ** 2 / (2 * (collision_d ** 2)) - tm.log(collision_d / barrier_left)) #positive

    #     lower_barrier_energy_maximum = upper_barrier_energy_maximum #positive
    #     lower_barrier_maximum = -upper_barrier_maximum #positive
    #     lower_barrier_df_maximum = upper_barrier_df_maximum #positive

    #     xc = ce - cs

    #     energy = 0.0

    #     # 求单元法向量
    #     f11 = p1 - cs
    #     f22 = p2 - cs
    #     n1 = xc.cross(f11)
    #     n2 = f22.cross(xc)

    #     n1_norm = n1.norm()
    #     n2_norm = n2.norm()

    #     multi_n1_n2 = n1_norm * n2_norm

    #     dir = n1.cross(n2).dot(xc)

    #     val = n1.dot(n2)

    #     norm_val = val / multi_n1_n2

    #     current_theta = 0.0
    #     if norm_val >= 1.0:
    #         val = multi_n1_n2
    #         norm_val = 1.0
    #         current_theta = 0.0
    #     elif norm_val <= -1.0:
    #         val = -multi_n1_n2
    #         norm_val = -1.0
    #         current_theta = tm.pi
    #     else:
    #         current_theta = tm.acos(norm_val)

    #     n_value = 0.
    #     backup_n_value = 0.
    #     signed_current_theta = 0.
       
    #     # 求折叠角
    #     if dir >= 0.: #mountain
    #         if id != -1 and self.previous_dir[id] <= 0 and norm_val <= -0.5: #180~270
    #             if crease_type == VALLEY:
    #                 self.crease_angle[id] = 1.
    #             else:
    #                 self.crease_angle[id] = -1.
    #             signed_current_theta = 2. * tm.pi - current_theta
    #             n_value = theta - signed_current_theta
    #             backup_n_value = n_value
    #             n_value += upper_barrier_maximum - (signed_current_theta - folding_angle_upper_bound) * upper_barrier_df_maximum
    #             energy += upper_barrier_energy_maximum + (-2. * upper_barrier_maximum + (signed_current_theta - folding_angle_upper_bound) * upper_barrier_df_maximum) * (signed_current_theta - folding_angle_upper_bound) * 0.5
    #         else: #-180~0
    #             if id != -1:
    #                 if crease_type == VALLEY:
    #                     self.crease_angle[id] = -1.
    #                 else:
    #                     self.crease_angle[id] = 1.
    #             signed_current_theta = -current_theta        
    #             n_value = theta - signed_current_theta
    #             backup_n_value = n_value
    #             t11 = signed_current_theta - lower_barrier
    #             t22 = folding_angle_lower_bound - collision_d - signed_current_theta
    #             if id != -1:
    #                 self.previous_dir[id] = dir 
    #             if t11 <= 0 and t11 >= -barrier_left:
    #                 n_value += collision_indice * (2 * t11 * tm.log(t22 / -(barrier_left)) - t11 ** 2 / t22)
    #                 energy += -collision_indice * t11 ** 2 * tm.log(t22 / -(barrier_left))
    #             elif t11 < -barrier_left:
    #                 n_value += lower_barrier_maximum - (signed_current_theta - folding_angle_lower_bound) * lower_barrier_df_maximum
    #                 energy += lower_barrier_energy_maximum - (2. * lower_barrier_maximum - (signed_current_theta - folding_angle_lower_bound) * lower_barrier_df_maximum) * (signed_current_theta - folding_angle_lower_bound) * 0.5
    #     else:
    #         if id != -1 and self.previous_dir[id] >= 0 and norm_val <= -0.5: #-270~-180
    #             if crease_type == VALLEY:
    #                 self.crease_angle[id] = -1.
    #             else:
    #                 self.crease_angle[id] = 1.
    #             signed_current_theta = current_theta - 2. * tm.pi
    #             n_value = theta - signed_current_theta
    #             backup_n_value = n_value
    #             n_value += lower_barrier_maximum - (signed_current_theta - folding_angle_lower_bound) * lower_barrier_df_maximum
    #             energy += lower_barrier_energy_maximum - (2. * lower_barrier_maximum - (signed_current_theta - folding_angle_lower_bound) * lower_barrier_df_maximum) * (signed_current_theta - folding_angle_lower_bound) * 0.5
    #         else: #0~180
    #             if id != -1:
    #                 if crease_type == VALLEY:
    #                     self.crease_angle[id] = 1.
    #                 else:
    #                     self.crease_angle[id] = -1.
    #             signed_current_theta = current_theta
    #             n_value = theta - signed_current_theta
    #             backup_n_value = n_value
    #             t11 = signed_current_theta - upper_barrier
    #             t22 = folding_angle_upper_bound + collision_d - signed_current_theta
    #             if id != -1:  
    #                 self.previous_dir[id] = dir 
    #             if t11 >= 0 and t11 <= barrier_left:
    #                 n_value += collision_indice * (2 * t11 * tm.log(t22 / barrier_left) - t11 ** 2 / t22)
    #                 energy += -collision_indice * t11 ** 2 * tm.log(t22 / barrier_left)
    #             elif t11 > barrier_left:
    #                 n_value += upper_barrier_maximum - (signed_current_theta - folding_angle_upper_bound) * upper_barrier_df_maximum
    #                 energy += upper_barrier_energy_maximum + (-2. * upper_barrier_maximum + (signed_current_theta - folding_angle_upper_bound) * upper_barrier_df_maximum) * (signed_current_theta - folding_angle_upper_bound) * 0.5
        
    #     dqdx0, dqdx1, dqdx2, dqdx3 = self.getDthetaDx(cs, p2, ce, p1, signed_current_theta)
            
    #     # 计算折痕等效弯曲系数
    #     # k_crease = 1.

    #     # #计算力
    #     # force = (k_crease * backup_n_value + n_value - backup_n_value)
        
    #     # csf = force * dqdx0
    #     # rpf2 = force * dqdx1
    #     # cef = force * dqdx2
    #     # rpf1 = force * dqdx3

    #     # #计算能量
    #     # energy += 0.5 * k_crease * backup_n_value ** 2

    #     # return csf, cef, rpf1, rpf2, energy, dqdx0, dqdx1, dqdx2, dqdx3, abs(signed_current_theta / tm.pi)
    #     return dqdx0, dqdx2, dqdx3, dqdx1, abs(signed_current_theta / tm.pi), n_value
    
    @ti.func
    def calculateTargetAngle(self, i, theta, ref_target):
        """
        计算目标折叠角度，考虑序列折叠的进度。
        Calculate target folding angle, considering the progress of sequential folding.
        
        :param i: 折痕索引 / Crease index
        :param theta: 当前角度 / Current angle
        :return: 目标折叠角度 / Target folding angle
        """
        target_folding_angle = 0.0
        percent_low = (self.sequence_level[0] - self.crease_level[i]) / (self.sequence_level[0] - self.sequence_level[1] + 1.)
        percent_high = (self.sequence_level[0] - self.crease_level[i] + 1.) / (self.sequence_level[0] - self.sequence_level[1] + 1.)
        percent_theta = abs(theta) / tm.pi

        if percent_theta < percent_low:
            target_folding_angle = 0.0
        elif percent_theta > percent_high:
            target_folding_angle = tm.pi
        else:
            coeff = self.crease_coeff[i]
            target_folding_angle = (percent_theta - percent_low) / (percent_high - percent_low) * tm.pi
            target_folding_angle = 2. * tm.atan2(coeff * tm.tan(target_folding_angle * 0.5), 1.)

        if self.crease_type[i]:
            target_folding_angle = -target_folding_angle
        
        true_level = self.sequence_level[0]
        current_level_need_to_be_fold = self.sequence_level[0] - percent_theta * (self.sequence_level[0] - self.sequence_level[1] + 1.)
        for level in ti.ndrange((self.sequence_level[1], self.sequence_level[0])):
            if level - current_level_need_to_be_fold <= 1 and level - current_level_need_to_be_fold > 0:
                true_level = level
        
        previous_angle = 0.0 if self.crease_level[i] < true_level + 1 else tm.pi
        if self.crease_type[i]:
            previous_angle = -previous_angle
        recover_angle = previous_angle
        find_recover_level = False
        for j in ti.ndrange(self.maximum_level_number):
            if self.recover_level_need[i, j]:
                if self.recover_level[i, j] == true_level + 1:
                    previous_angle = self.recover_angle[i, j]
                elif self.recover_level[i, j] == true_level:
                    recover_angle = self.recover_angle[i, j]
                    find_recover_level = True

        if find_recover_level:
            coeff = self.crease_coeff[i]
            target_folding_angle = previous_angle + (recover_angle - previous_angle) * (true_level - current_level_need_to_be_fold)
            target_folding_angle = 2. * tm.atan2(coeff * tm.tan(target_folding_angle * 0.5), 1.)

        # print(i, self.crease_level[i], target_folding_angle)
        if ref_target:
            target_folding_angle *= self.target_crease_angle[i]
            
        return target_folding_angle
    
    @ti.kernel
    def project_spring(self):
        for i in ti.ndrange(self.spring_num_param[None]):
            x_i = self.x[self.spring_selection[2 * i + 0]]
            x_j = self.x[self.spring_selection[2 * i + 1]]

            rest_len = self.spring_original_length[i]
            
            # 计算当前长度和方向
            dx = x_j - x_i
            current_len = dx.norm()
            
            m_i = self.masses[self.spring_selection[2 * i + 0]]
            m_j = self.masses[self.spring_selection[2 * i + 1]]
            total_mass = m_i + m_j
            
            correction = (current_len - rest_len) * (dx / current_len)
            self.spring_x_proj[2 * i + 0] = x_i + (m_j / total_mass) * correction
            self.spring_x_proj[2 * i + 1] = x_j - (m_i / total_mass) * correction

            energy = 0.5 * self.spring_k_param[None] * (current_len - rest_len) ** 2

            self.energy[None] += energy
            self.split_energy[self.spring_selection_corresponding_origami_id[i]] += energy
        
            # print(self.spring_x_proj[2 * i + 0], self.spring_x_proj[2 * i + 1])
    
    @ti.kernel
    def project_bending_2(self, theta: data_type, ref_target: bool):
        for i in ti.ndrange(self.bending_num_param[None]):
            c0 = self.x[self.bending_selection[i * 4 + 0]]
            c1 = self.x[self.bending_selection[i * 4 + 1]]
            p0 = self.x[self.bending_selection[i * 4 + 2]]
            p1 = self.x[self.bending_selection[i * 4 + 3]]

            m1 = self.masses[self.bending_selection[i * 4 + 0]]
            m2 = self.masses[self.bending_selection[i * 4 + 1]]
            m3 = self.masses[self.bending_selection[i * 4 + 2]]
            m4 = self.masses[self.bending_selection[i * 4 + 3]]

            e_axis = tm.normalize(c1 - c0)

            target_angle = self.calculateTargetAngle(i, theta, ref_target)

            signed_theta, barrier_force = self.compute_signed_dihedral(c0, c1, p0, p1, self.crease_type[i], i)

            barrier_force = tm.clamp(abs(barrier_force), 0., 3.141)

            if self.crease_type[i] == MOUNTAIN:
                barrier_force = -barrier_force

            delta_theta = (target_angle - signed_theta - barrier_force)
            # delta_theta = tm.clamp(delta_theta, -0.5, 0.5)

            p0_axis_proj, r0 = self.get_proj_and_height_vector(p0, c0, e_axis)
            p1_axis_proj, r1 = self.get_proj_and_height_vector(p1, c0, e_axis)

            r0_norm = r0.norm()
            r1_norm = r1.norm()

            bonus_0 = r1_norm / r0_norm if r1_norm < r0_norm else 1.
            bonus_1 = r0_norm / r1_norm if r0_norm < r1_norm else 1.

            p0_proj = self.rodrigues_rotate(p0_axis_proj, r0, e_axis, delta_theta * m4 / (m3 + m4) * bonus_0)
            p1_proj = self.rodrigues_rotate(p1_axis_proj, r1, e_axis, -delta_theta * m4 / (m3 + m4) * bonus_1)

            # lam = (m3 * (p0_proj - p0) + m4 * (p1_proj - p1)) / (m1 + m2 + m3 + m4)
            # lam = ((p0_proj - p0) + (p1_proj - p1)) * 0.25

            self.bending_x_proj[i * 4 + 0] = c0
            self.bending_x_proj[i * 4 + 1] = c1
            self.bending_x_proj[i * 4 + 2] = p0_proj
            self.bending_x_proj[i * 4 + 3] = p1_proj
            
            energy = 0.5 * self.bending_k_param[None] * self.crease_initial_length[i] * (delta_theta) ** 2

            self.energy[None] += energy
            self.split_energy[self.bending_selection_corresponding_origami_id[i]] += energy
    
    @ti.kernel
    def project_bending_3(self, theta: data_type):
        for i in ti.ndrange(self.bending_num_param[None]):
            c0 = self.x[self.bending_selection[i * 4 + 0]]
            c1 = self.x[self.bending_selection[i * 4 + 1]]
            p0 = self.x[self.bending_selection[i * 4 + 2]]
            p1 = self.x[self.bending_selection[i * 4 + 3]]

            m1 = self.masses[self.bending_selection[i * 4 + 0]]
            m2 = self.masses[self.bending_selection[i * 4 + 1]]
            m3 = self.masses[self.bending_selection[i * 4 + 2]]
            m4 = self.masses[self.bending_selection[i * 4 + 3]]

            target_angle = self.calculateTargetAngle(i, theta)

            signed_theta, barrier_force = self.compute_signed_dihedral(c0, c1, p0, p1, self.crease_type[i], i)

            barrier_force = tm.clamp(abs(barrier_force), 0., 3.141)

            if self.crease_type[i] == MOUNTAIN:
                barrier_force = -barrier_force

            delta_theta = (target_angle - signed_theta - barrier_force)
            # delta_theta = tm.clamp(delta_theta, -0.5, 0.5)
            
            # 新写法：
            c0_proj, c1_proj, p0_proj, p1_proj = self.project_dihedral_momentum_conserving(
                m1, m2, m3, m4, c0, c1, p0, p1, delta_theta
            )

            self.bending_x_proj[i * 4 + 0] = c0_proj
            self.bending_x_proj[i * 4 + 1] = c1_proj
            self.bending_x_proj[i * 4 + 2] = p0_proj
            self.bending_x_proj[i * 4 + 3] = p1_proj
            
            energy = 0.5 * self.bending_k_param[None] * self.crease_initial_length[i] * (delta_theta) ** 2

            self.energy[None] += energy
            self.split_energy[self.bending_selection_corresponding_origami_id[i]] += energy
    
    # @ti.kernel
    # def project_facet_bending(self):
    #     for i in ti.ndrange(self.facet_bending_num_param[None]):
    #         c0 = self.x[self.facet_bending_selection[i * 4 + 0]]
    #         c1 = self.x[self.facet_bending_selection[i * 4 + 1]]
    #         p0 = self.x[self.facet_bending_selection[i * 4 + 2]]
    #         p1 = self.x[self.facet_bending_selection[i * 4 + 3]]

    #         target_folding_angle = 0.

    #         dqdx0, dqdx1, dqdx2, dqdx3, _, c_val = self.getBendingForce(c0, c1, p0, p1, self.facet_bending_k_param[None], self.facet_crease_initial_length[i], target_folding_angle, 0, -1)

    #         divider = tm.sqrt(dqdx0.norm_sqr() + dqdx1.norm_sqr() + dqdx2.norm_sqr() + dqdx3.norm_sqr())

    #         # 零保护 + 单步截断 / Zero guard + step clamping
    #         if divider < data_type(1e-12):
    #             self.facet_bending_x_proj[4 * i + 0] = c0
    #             self.facet_bending_x_proj[4 * i + 1] = c1
    #             self.facet_bending_x_proj[4 * i + 2] = p0
    #             self.facet_bending_x_proj[4 * i + 3] = p1
    #         else:
    #             lam = c_val / divider  # XPBD 拉格朗日乘子 / XPBD Lagrange multiplier

    #             # lam = tm.clamp(lam, -self.angle_protection_param[None], self.angle_protection_param[None])

    #             # 各顶点修正量 / correction per vertex
    #             dc0 = lam * (dqdx0)
    #             dc1 = lam * (dqdx1)
    #             dp0 = lam * (dqdx2)
    #             dp1 = lam * (dqdx3)

    #             self.facet_bending_x_proj[4 * i + 0] = c0 + dc0
    #             self.facet_bending_x_proj[4 * i + 1] = c1 + dc1
    #             self.facet_bending_x_proj[4 * i + 2] = p0 + dp0
    #             self.facet_bending_x_proj[4 * i + 3] = p1 + dp1
            
    #         self.energy[None] += 0.5 * self.facet_bending_k_param[None] * self.facet_crease_initial_length[i] * (c_val) ** 2

    @ti.kernel
    def project_facet_bending_2(self):
        for i in ti.ndrange(self.facet_bending_num_param[None]):
            c0 = self.x[self.facet_bending_selection[i * 4 + 0]]
            c1 = self.x[self.facet_bending_selection[i * 4 + 1]]
            p0 = self.x[self.facet_bending_selection[i * 4 + 2]]
            p1 = self.x[self.facet_bending_selection[i * 4 + 3]]

            m1 = self.masses[self.facet_bending_selection[i * 4 + 0]]
            m2 = self.masses[self.facet_bending_selection[i * 4 + 1]]
            m3 = self.masses[self.facet_bending_selection[i * 4 + 2]]
            m4 = self.masses[self.facet_bending_selection[i * 4 + 3]]

            e_axis = tm.normalize(c1 - c0)

            signed_theta, _ = self.compute_signed_dihedral(c0, c1, p0, p1, 0., -1)

            delta_theta = -signed_theta

            p0_axis_proj, r0 = self.get_proj_and_height_vector(p0, c0, e_axis)
            p1_axis_proj, r1 = self.get_proj_and_height_vector(p1, c0, e_axis)

            r0_norm = r0.norm()
            r1_norm = r1.norm()

            bonus_0 = r1_norm / r0_norm if r1_norm < r0_norm else 1.
            bonus_1 = r0_norm / r1_norm if r0_norm < r1_norm else 1.

            p0_proj = self.rodrigues_rotate(p0_axis_proj, r0, e_axis, delta_theta * m4 / (m3 + m4) * bonus_0)
            p1_proj = self.rodrigues_rotate(p1_axis_proj, r1, e_axis, -delta_theta * m4 / (m3 + m4) * bonus_1)

            # lam = (m3 * (p0_proj - p0) + m4 * (p1_proj - p1)) / (m1 + m2 + m3 + m4)
            # lam = ((p0_proj - p0) + (p1_proj - p1)) * 0.25

            self.facet_bending_x_proj[i * 4 + 0] = c0
            self.facet_bending_x_proj[i * 4 + 1] = c1
            self.facet_bending_x_proj[i * 4 + 2] = p0_proj
            self.facet_bending_x_proj[i * 4 + 3] = p1_proj
            
            energy = 0.5 * self.facet_bending_k_param[None] * self.facet_crease_initial_length[i] * (delta_theta) ** 2

            self.energy[None] += energy
            self.split_energy[self.facet_bending_selection_corresponding_origami_id[i]] += energy

    @ti.kernel
    def project_facet_bending_3(self):
        for i in ti.ndrange(self.facet_bending_num_param[None]):
            c0 = self.x[self.facet_bending_selection[i * 4 + 0]]
            c1 = self.x[self.facet_bending_selection[i * 4 + 1]]
            p0 = self.x[self.facet_bending_selection[i * 4 + 2]]
            p1 = self.x[self.facet_bending_selection[i * 4 + 3]]

            m1 = self.masses[self.facet_bending_selection[i * 4 + 0]]
            m2 = self.masses[self.facet_bending_selection[i * 4 + 1]]
            m3 = self.masses[self.facet_bending_selection[i * 4 + 2]]
            m4 = self.masses[self.facet_bending_selection[i * 4 + 3]]

            signed_theta, _ = self.compute_signed_dihedral(c0, c1, p0, p1, 0., -1)

            delta_theta = -signed_theta

            # 新写法：
            c0_proj, c1_proj, p0_proj, p1_proj = self.project_dihedral_momentum_conserving(
                m1, m2, m3, m4, c0, c1, p0, p1, delta_theta
            )

            self.facet_bending_x_proj[i * 4 + 0] = c0_proj
            self.facet_bending_x_proj[i * 4 + 1] = c1_proj
            self.facet_bending_x_proj[i * 4 + 2] = p0_proj
            self.facet_bending_x_proj[i * 4 + 3] = p1_proj
            
            energy = 0.5 * self.facet_bending_k_param[None] * self.facet_crease_initial_length[i] * (delta_theta) ** 2

            self.energy[None] += energy
            self.split_energy[self.facet_bending_selection_corresponding_origami_id[i]] += energy

    # @ti.kernel
    # def get_u0_norm(self, dx_array: ti.types.ndarray()) -> data_type:
    #     ret = 0.
    #     for i in ti.ndrange(3 * self.kp_num_param[0]):
    #         ret += dx_array[i] ** 2
    #     return tm.sqrt(ret)
    
    @ti.kernel
    def fill_b(self, h: data_type):
        for i in ti.ndrange(self.kp_num_param[None]):
            self.b[i * 3 + 0] = -self.masses[i] * (self.x[i][0] - self.s[i][0]) / (h ** 2)
            self.b[i * 3 + 1] = -self.masses[i] * (self.x[i][1] - self.s[i][1]) / (h ** 2)
            self.b[i * 3 + 2] = -self.masses[i] * (self.x[i][2] - self.s[i][2]) / (h ** 2)
        
        # spring
        for i in ti.ndrange(self.spring_num_param[None]):
            idx1 = self.spring_selection[i * 2 + 0]
            idx2 = self.spring_selection[i * 2 + 1]
            x1 = self.x[idx1]
            x2 = self.x[idx2]
            p1 = self.spring_x_proj[i * 2 + 0]
            p2 = self.spring_x_proj[i * 2 + 1]
            for j in ti.static(range(3)):
                self.b[idx1 * 3 + j] += -self.spring_k_param[None] * ((x1[j] - x2[j]) - (p1[j] - p2[j]))
            for j in ti.static(range(3)):
                self.b[idx2 * 3 + j] += self.spring_k_param[None] * ((x1[j] - x2[j]) - (p1[j] - p2[j]))

        # bending —— 各结点贡献乘以自身质量，保证动量守恒
        # Multiply each vertex contribution by its own mass to ensure momentum conservation
        # 修正原理：b 向量中弯曲项为 -k * m_i * (x_i - p_i)，
        # 对应 Hessian 对角线为 k * m_i，保证同一约束对各结点的力之和为零
        for i in ti.ndrange(self.bending_num_param[None]):
            idx1 = self.bending_selection[i * 4 + 0]
            idx2 = self.bending_selection[i * 4 + 1]
            idx3 = self.bending_selection[i * 4 + 2]
            idx4 = self.bending_selection[i * 4 + 3]
            x1 = self.x[idx1]
            x2 = self.x[idx2]
            x3 = self.x[idx3]
            x4 = self.x[idx4]
            p1 = self.bending_x_proj[i * 4 + 0]
            p2 = self.bending_x_proj[i * 4 + 1]
            p3 = self.bending_x_proj[i * 4 + 2]
            p4 = self.bending_x_proj[i * 4 + 3]
            f1 = -(self.cotangent_matrix[i][0, 0] * (x1 - p1) + self.cotangent_matrix[i][0, 1] * (x2 - p2) + self.cotangent_matrix[i][0, 2] * (x3 - p3) + self.cotangent_matrix[i][0, 3] * (x4 - p4))
            f2 = -(self.cotangent_matrix[i][1, 0] * (x1 - p1) + self.cotangent_matrix[i][1, 1] * (x2 - p2) + self.cotangent_matrix[i][1, 2] * (x3 - p3) + self.cotangent_matrix[i][1, 3] * (x4 - p4))
            f3 = -(self.cotangent_matrix[i][2, 0] * (x1 - p1) + self.cotangent_matrix[i][2, 1] * (x2 - p2) + self.cotangent_matrix[i][2, 2] * (x3 - p3) + self.cotangent_matrix[i][2, 3] * (x4 - p4))
            f4 = -(self.cotangent_matrix[i][3, 0] * (x1 - p1) + self.cotangent_matrix[i][3, 1] * (x2 - p2) + self.cotangent_matrix[i][3, 2] * (x3 - p3) + self.cotangent_matrix[i][3, 3] * (x4 - p4))
            for j in ti.static(range(3)):
                self.b[idx1 * 3 + j] += f1[j]
            for j in ti.static(range(3)):
                self.b[idx2 * 3 + j] += f2[j]
            for j in ti.static(range(3)):
                self.b[idx3 * 3 + j] += f3[j]
            for j in ti.static(range(3)):
                self.b[idx4 * 3 + j] += f4[j]

        # facet_bending —— 同上，各结点贡献乘以自身质量
        for i in ti.ndrange(self.facet_bending_num_param[None]):
            idx1 = self.facet_bending_selection[i * 4 + 0]
            idx2 = self.facet_bending_selection[i * 4 + 1]
            idx3 = self.facet_bending_selection[i * 4 + 2]
            idx4 = self.facet_bending_selection[i * 4 + 3]
            x1 = self.x[idx1]
            x2 = self.x[idx2]
            x3 = self.x[idx3]
            x4 = self.x[idx4]
            p1 = self.facet_bending_x_proj[i * 4 + 0]
            p2 = self.facet_bending_x_proj[i * 4 + 1]
            p3 = self.facet_bending_x_proj[i * 4 + 2]
            p4 = self.facet_bending_x_proj[i * 4 + 3]
            f1 = -(self.facet_cotangent_matrix[i][0, 0] * (x1 - p1) + self.facet_cotangent_matrix[i][0, 1] * (x2 - p2) + self.facet_cotangent_matrix[i][0, 2] * (x3 - p3) + self.facet_cotangent_matrix[i][0, 3] * (x4 - p4))
            f2 = -(self.facet_cotangent_matrix[i][1, 0] * (x1 - p1) + self.facet_cotangent_matrix[i][1, 1] * (x2 - p2) + self.facet_cotangent_matrix[i][1, 2] * (x3 - p3) + self.facet_cotangent_matrix[i][1, 3] * (x4 - p4))
            f3 = -(self.facet_cotangent_matrix[i][2, 0] * (x1 - p1) + self.facet_cotangent_matrix[i][2, 1] * (x2 - p2) + self.facet_cotangent_matrix[i][2, 2] * (x3 - p3) + self.facet_cotangent_matrix[i][2, 3] * (x4 - p4))
            f4 = -(self.facet_cotangent_matrix[i][3, 0] * (x1 - p1) + self.facet_cotangent_matrix[i][3, 1] * (x2 - p2) + self.facet_cotangent_matrix[i][3, 2] * (x3 - p3) + self.facet_cotangent_matrix[i][3, 3] * (x4 - p4))

            for j in ti.static(range(3)):
                self.b[idx1 * 3 + j] += f1[j]
            for j in ti.static(range(3)):
                self.b[idx2 * 3 + j] += f2[j]
            for j in ti.static(range(3)):
                self.b[idx3 * 3 + j] += f3[j]
            for j in ti.static(range(3)):
                self.b[idx4 * 3 + j] += f4[j]

    @ti.kernel
    def fill_b_ndarray(self, b: ti.types.ndarray(), h: data_type):
        for i in ti.ndrange(self.kp_num_param[None]):
            b[i * 3 + 0] = -self.masses[i] * (self.x[i][0] - self.s[i][0]) / (h ** 2)
            b[i * 3 + 1] = -self.masses[i] * (self.x[i][1] - self.s[i][1]) / (h ** 2)
            b[i * 3 + 2] = -self.masses[i] * (self.x[i][2] - self.s[i][2]) / (h ** 2)
        
        # spring
        for i in ti.ndrange(self.spring_num_param[None]):
            idx1 = self.spring_selection[i * 2 + 0]
            idx2 = self.spring_selection[i * 2 + 1]
            x1 = self.x[idx1]
            x2 = self.x[idx2]
            p1 = self.spring_x_proj[i * 2 + 0]
            p2 = self.spring_x_proj[i * 2 + 1]
            for j in ti.static(range(3)):
                b[idx1 * 3 + j] += -self.spring_k_param[None] * ((x1[j] - x2[j]) - (p1[j] - p2[j]))
            for j in ti.static(range(3)):
                b[idx2 * 3 + j] += self.spring_k_param[None] * ((x1[j] - x2[j]) - (p1[j] - p2[j]))
        
        # bending —— 各结点贡献乘以自身质量，保证动量守恒
        # Multiply each vertex contribution by its own mass to ensure momentum conservation
        # 修正原理：b 向量中弯曲项为 -k * m_i * (x_i - p_i)，
        # 对应 Hessian 对角线为 k * m_i，保证同一约束对各结点的力之和为零
        for i in ti.ndrange(self.bending_num_param[None]):
            idx1 = self.bending_selection[i * 4 + 0]
            idx2 = self.bending_selection[i * 4 + 1]
            idx3 = self.bending_selection[i * 4 + 2]
            idx4 = self.bending_selection[i * 4 + 3]
            x1 = self.x[idx1]
            x2 = self.x[idx2]
            x3 = self.x[idx3]
            x4 = self.x[idx4]
            p1 = self.bending_x_proj[i * 4 + 0]
            p2 = self.bending_x_proj[i * 4 + 1]
            p3 = self.bending_x_proj[i * 4 + 2]
            p4 = self.bending_x_proj[i * 4 + 3]
            f1 = -(self.cotangent_matrix[i][0, 0] * (x1 - p1) + self.cotangent_matrix[i][0, 1] * (x2 - p2) + self.cotangent_matrix[i][0, 2] * (x3 - p3) + self.cotangent_matrix[i][0, 3] * (x4 - p4))
            f2 = -(self.cotangent_matrix[i][1, 0] * (x1 - p1) + self.cotangent_matrix[i][1, 1] * (x2 - p2) + self.cotangent_matrix[i][1, 2] * (x3 - p3) + self.cotangent_matrix[i][1, 3] * (x4 - p4))
            f3 = -(self.cotangent_matrix[i][2, 0] * (x1 - p1) + self.cotangent_matrix[i][2, 1] * (x2 - p2) + self.cotangent_matrix[i][2, 2] * (x3 - p3) + self.cotangent_matrix[i][2, 3] * (x4 - p4))
            f4 = -(self.cotangent_matrix[i][3, 0] * (x1 - p1) + self.cotangent_matrix[i][3, 1] * (x2 - p2) + self.cotangent_matrix[i][3, 2] * (x3 - p3) + self.cotangent_matrix[i][3, 3] * (x4 - p4))
            for j in ti.static(range(3)):
                b[idx1 * 3 + j] += f1[j]
            for j in ti.static(range(3)):
                b[idx2 * 3 + j] += f2[j]
            for j in ti.static(range(3)):
                b[idx3 * 3 + j] += f3[j]
            for j in ti.static(range(3)):
                b[idx4 * 3 + j] += f4[j]

        # facet_bending —— 同上，各结点贡献乘以自身质量
        for i in ti.ndrange(self.facet_bending_num_param[None]):
            idx1 = self.facet_bending_selection[i * 4 + 0]
            idx2 = self.facet_bending_selection[i * 4 + 1]
            idx3 = self.facet_bending_selection[i * 4 + 2]
            idx4 = self.facet_bending_selection[i * 4 + 3]
            x1 = self.x[idx1]
            x2 = self.x[idx2]
            x3 = self.x[idx3]
            x4 = self.x[idx4]
            p1 = self.facet_bending_x_proj[i * 4 + 0]
            p2 = self.facet_bending_x_proj[i * 4 + 1]
            p3 = self.facet_bending_x_proj[i * 4 + 2]
            p4 = self.facet_bending_x_proj[i * 4 + 3]
            f1 = -(self.facet_cotangent_matrix[i][0, 0] * (x1 - p1) + self.facet_cotangent_matrix[i][0, 1] * (x2 - p2) + self.facet_cotangent_matrix[i][0, 2] * (x3 - p3) + self.facet_cotangent_matrix[i][0, 3] * (x4 - p4))
            f2 = -(self.facet_cotangent_matrix[i][1, 0] * (x1 - p1) + self.facet_cotangent_matrix[i][1, 1] * (x2 - p2) + self.facet_cotangent_matrix[i][1, 2] * (x3 - p3) + self.facet_cotangent_matrix[i][1, 3] * (x4 - p4))
            f3 = -(self.facet_cotangent_matrix[i][2, 0] * (x1 - p1) + self.facet_cotangent_matrix[i][2, 1] * (x2 - p2) + self.facet_cotangent_matrix[i][2, 2] * (x3 - p3) + self.facet_cotangent_matrix[i][2, 3] * (x4 - p4))
            f4 = -(self.facet_cotangent_matrix[i][3, 0] * (x1 - p1) + self.facet_cotangent_matrix[i][3, 1] * (x2 - p2) + self.facet_cotangent_matrix[i][3, 2] * (x3 - p3) + self.facet_cotangent_matrix[i][3, 3] * (x4 - p4))

            for j in ti.static(range(3)):
                b[idx1 * 3 + j] += f1[j]
            for j in ti.static(range(3)):
                b[idx2 * 3 + j] += f2[j]
            for j in ti.static(range(3)):
                b[idx3 * 3 + j] += f3[j]
            for j in ti.static(range(3)):
                b[idx4 * 3 + j] += f4[j]
    
    def update_folding_target(self):
        self.folding_angle += self.enable_add_folding_angle
        if self.folding_angle >= 3.1415:
            self.folding_angle = 3.1415
            # Hit π this step → freeze paint/export immediately (not next render)
            if self.collision_shading:
                self._stop_sweep_drawing_at_pi()
            # Stop auto-fold so θ does not keep “driving” past clamp
            if float(getattr(self, "enable_add_folding_angle", 0.0)) > 0.0:
                self.enable_add_folding_angle = 0.0
        if self.folding_angle <= 0:
            self.folding_angle = 0

    @ti.kernel
    def forward(self, dt: data_type):
        for i in ti.ndrange(self.kp_num_param[None]):
            self.x0[i] = self.x[i]
            self.x[i] += self.v[i] * dt
            self.s[i] = self.x0[i] + self.v[i] * dt
    
    @ti.kernel
    def update_vel(self, dt: data_type):
        for i in ti.ndrange(self.kp_num_param[None]):
            self.v[i] = ((self.x[i] - self.x0[i]) / dt) * self.damping_param[None]

    @ti.kernel
    def clearEnergy(self):
        self.energy[None] = 0.0
        for i in ti.ndrange(self.split_origami_num_param[None]):
            self.split_energy[i] = 0.0

    def local_step(self, theta):
        self.project_spring()
        self.project_bending_2(theta, self.ref_target)
        self.project_facet_bending_2()

    @ti.kernel
    def update_x(self):
        for i in ti.ndrange(self.kp_num_param[None]):
            self.x[i][0] = self.x[i][0] + self.u0[i * 3 + 0]
            self.x[i][1] = self.x[i][1] + self.u0[i * 3 + 1]
            self.x[i][2] = self.x[i][2] + self.u0[i * 3 + 2]
    
    @ti.kernel
    def update_vertices(self):
        for i in ti.ndrange(self.kp_num_param[None]):
            self.vertices[i] = ti.cast(self.x[i], ti.f32)
    
    def global_step(self):
        if use_gpu:
            self.fill_b_ndarray(self.b_array, self.dt)
            dx_array = self.sparse_solver.solve(self.b_array)
            self.u0.from_numpy(dx_array.to_numpy())
        else:
            self.fill_b(self.dt)
            dx = self.sparse_solver.solve(self.b)
            self.u0.from_numpy(dx)
        self.update_x()

    def _cache_frame_positions(self, force=False):
        """Host copy of x once per render/export frame (shared by hot paths)."""
        gen = int(getattr(self, "_render_gen", 0))
        if (
            not force
            and self._frame_positions is not None
            and int(getattr(self, "_frame_positions_gen", -1)) == gen
        ):
            return self._frame_positions
        pos = self.x.to_numpy()[: self.kp_num]
        self._frame_positions = pos
        self._frame_positions_gen = gen
        return pos

    def outputFigure(self):
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.5, 0.5, 0.5))
        self.scene.point_light(pos=(0., 0., 2 * self.max_size), color=(0.8, 0.8, 0.8))

        self.update_vertices()
        self._render_gen = int(getattr(self, "_render_gen", 0)) + 1
        if self.collision_shading:
            self._cache_frame_positions(force=True)
            self._maybe_detect_panel_collisions()
        self._render_panel_meshes(self.scene)

        self.fill_line_vertex()
        self.scene.lines(vertices=self.line_vertex,
                    width=2,
                    per_vertex_color=self.line_color)

        self.canvas.scene(self.scene)
        try:
            folder = f'./physResult/cdf-' + self.origami_name
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.window.save_image(f'./physResult/cdf-' + self.origami_name + "/" + str(self.ID).zfill(8) + '.png')
            print(f"Picture ID {str(self.ID).zfill(8)} is saved.")
        except:
            pass

    @ti.kernel
    def calculateCreaseAngleSum(self) -> data_type:
        ret = 0.
        for i in ti.ndrange(self.crease_pairs_num):
            ret += abs(self.crease_angle[i])
        return ret

    def stop(self):
        # if self.energy[None] < self.backup_energy[None] and \
        #     abs(self.energy[None] - self.backup_energy[None]) < 1e-3 * self.energy[None] \
        #         and self.folding_angle > 3.14 and self.folding_angle_reach_pi[None]:
        #     return True
        # self.backup_energy[None] = self.energy[None]
        angle_sum = self.calculateCreaseAngleSum()
        if abs(angle_sum - self.backup_crease_angle_sum[None]) < 1e-3 * angle_sum and self.folding_angle > 3.14:
            # Seal every pair's final coords (including still-active) and export once
            if self.collision_shading and not self._collision_coords_exported:
                self._seal_and_export_fixed_flat_contacts()
            return True
        self.backup_crease_angle_sum[None] = angle_sum
        return False

    def step(self):
        if self.use_gui:
            if self.window.get_event(ti.ui.PRESS):
                self.deal_with_key(self.window.event.key)

        self._mesh_advanced = False
        for _ in range(self.substeps):
            if not self.paused or self.step_once:
                # print("---begin---")
                self.update_folding_target()
                self.forward(self.dt)
                for i in range(self.pd_iter_time):
                    self.clearEnergy()
                    self.local_step(self.folding_angle)
                    self.global_step()
                self.update_vel(self.dt)
                # print("---end---")
                self.step_once = False
                self._mesh_advanced = True
            self.current_t += self.dt

    def reward(self):
        reward_list = np.zeros(self.split_origami_num)
        individual_crease_num = self.crease_pairs_num // self.split_origami_num

        if 0:
            for i in range(self.split_origami_num):
                reward_list[i] = self.split_energy[i]
                start_index = individual_crease_num * i
                end_index = individual_crease_num * (i + 1)
                avg_folding_percent = 0.
                for j in range(start_index, end_index):
                    avg_folding_percent += self.crease_angle[j]
                avg_folding_percent /= individual_crease_num
                avg_folding_percent = max(1e-6, avg_folding_percent)
                reward_list[i] /= avg_folding_percent
        else:
            for i in range(self.split_origami_num):
                start_index = individual_crease_num * i
                end_index = individual_crease_num * (i + 1)
                avg_folding_percent = 0.
                for j in range(start_index, end_index):
                    avg_folding_percent += self.crease_angle[j]
                avg_folding_percent /= individual_crease_num
                reward_list[i] = 1. - avg_folding_percent

        return reward_list
    
    def render(self):
        scene  = self.scene
        camera = self.camera
        
        camera.track_user_inputs(self.window,
                                    movement_speed=1,
                                    hold_key=ti.ui.RMB)

        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        self.scene.point_light(pos=(0., 0., 2 * self.max_size), color=(0.8, 0.8, 0.8))

        self.update_vertices()
        self._render_gen = int(getattr(self, "_render_gen", 0)) + 1
        if self.collision_shading:
            self._cache_frame_positions(force=True)
            self._maybe_detect_panel_collisions()
        self._render_panel_meshes(scene)

        self.fill_line_vertex()
        self.scene.lines(vertices=self.line_vertex,
                    width=2,
                    per_vertex_color=self.line_color)

        # Main control panel (sliders only — collision debug is a separate window)
        self.gui.text(f"System time: {round(self.current_t, 3)}s")
        self.gui.text(f"Energy: {round(self.energy[None], 3)}")

        # for i in range(self.split_origami_num):
        #     self.gui.text(f"Sub-energy: {round(self.split_energy[i], 3)}")

        self.folding_angle = self.gui.slider_float('Folding angle', self.folding_angle, 0, 3.135)
        if (
            self.collision_shading
            and float(self.folding_angle) >= 3.1415 - 1e-6
        ):
            self._stop_sweep_drawing_at_pi()

        self.spring_k = self.gui.slider_float('Spring k', self.spring_k, 40., 5000.)
        self.bending_k = self.gui.slider_float('Crease k', self.bending_k, 0.01, 1.)
        self.facet_bending_k = self.gui.slider_float('Facet k', self.facet_bending_k, 1., 200.)

        self.gui.text("If the above stiffnesses are modified, press 'r' to restart. ")

        # Must draw every frame — skipping frames makes ImGui sub_window flicker
        if self.collision_shading:
            self._render_collision_debug_window()

        self.canvas.scene(scene)
        if not self.fast_simulation_mode:
            try:
                self.window.save_image(f'./physResult/' + self.origami_name + '-' + self.time + "/" + str(self.image_id).zfill(8) + '.png')
                print(f"Picture ID {str(self.image_id).zfill(8)} is saved.")
                self.image_id += 1
            except:
                pass
        self.window.show()

    def appendCreaseInfo(self):
        self.input_json["crease_angle"] = [
            max(min(self.crease_angle[i], 1.), -1.) for i in range(self.crease_pairs_num)
        ]
        self.input_json["crease_info"] = [
            [self.kps[self.crease_pairs[i, 0]], self.kps[self.crease_pairs[i, 1]]] for i in range(self.crease_pairs_num)
        ]
        with open("./descriptionData/" + self.origami_name + ".json", 'w', encoding='utf-8') as fw:
            json.dump(self.input_json, fw, indent=4)

    def run(self):
        self.initializeRunning()
        # GUI: hold angle (use slider / i=start, k=stop, m=reverse).
        # Headless: auto-drive fold target each step (same as phy_sim_target).
        if self.use_gui:
            self.enable_add_folding_angle = 0.0
        else:
            self.enable_add_folding_angle = 0.105
        while self.window.running:
            self.step()
            # ti.profiler.print_kernel_profiler_info()  # 看每个kernel的执行时间、线程数
            # ti.profiler.clear_kernel_profiler_info()
            if self.use_gui:
                self.render()
            if self.stop():
                self.backupSimulationSetting()
                break 
        if not self.ref_target:
            self.appendCreaseInfo()


    """
    core idea of PANEL TRIMMING:
    1. map each panel in each layer as a uniquely identifiable object
    2. apply AABB collision detection to each panel object
    3. classify intruders based on penetration depth and prioritize thicker panels
    4. sweeping visualization
    5. coordinates export
    """

    # 1. panel mapping

    def _panel_layer_mapping(self):
        """Return JSON panel index -> list of simulator unit ids (one entry per layer)."""
        if getattr(self, "unit_mapping", None):
            return self.unit_mapping
        if hasattr(self, "input_json") and "units" in self.input_json:
            n_panels = len(self.input_json["units"])
        else:
            n_panels = self.unit_indices_num
        return [[i] for i in range(n_panels)]

    def _unit_layer_height_z(self, sim_unit_id):
        kps = self.units[sim_unit_id].getSeqPoint()
        if not kps:
            return 0.0
        zs = [kp[Z] if len(kp) > 2 else 0.0 for kp in kps]
        return sum(zs) / len(zs)

    def _unit_kp_indices(self, sim_unit_id):
        if not hasattr(self, "ori_sim"):
            return []
        return [idx for idx in self.ori_sim.indices[sim_unit_id] if idx != -1]

    def _build_unit_layer_meta(self):
        """
        sim_unit_id -> {layer_h, layer_idx, num_layers} for outer/thicker
        intruder prioritization.
        """
        meta = {}
        try:
            registry = self.get_panel_layer_registry()
        except Exception:
            registry = []
        for entry in registry:
            uid = entry["sim_unit_id"]
            meta[uid] = {
                "layer_h": float(entry.get("height_z", 0.0)),
                "height_z": float(entry.get("height_z", 0.0)),
                "layer_idx": int(entry.get("layer_idx", 0)),
                "num_layers": int(entry.get("num_layers", 1)),
            }
        return meta

    def get_panel_layer_registry(self):
        """
        List every (panel, layer) pair with stable identifiers for downstream use
        (rendering, constraints, analysis, etc.).

        Each entry contains:
          - panel_idx:    index in the original JSON "units" list
          - layer_idx:    0 = lowest-Z layer within that panel (thick mode)
          - sim_unit_id:  index into self.units / self.unit_indices / self.ori_sim.indices
          - height_z:     mean vertex Z of the layer (mm)
          - num_layers:   how many layers this panel was expanded into
          - kp_indices:   merged-mesh keypoint indices owned by this unit
          - thick_mode:   whether thick-panel expansion was used at startup
        """
        mapping = self._panel_layer_mapping()
        thick_mode = bool(getattr(self, "thick_mode_flag", False))
        registry = []
        for panel_idx, unit_ids in enumerate(mapping):
            num_layers = len(unit_ids)
            for layer_idx, sim_unit_id in enumerate(unit_ids):
                registry.append({
                    "panel_idx": panel_idx,
                    "layer_idx": layer_idx,
                    "sim_unit_id": sim_unit_id,
                    "height_z": self._unit_layer_height_z(sim_unit_id),
                    "num_layers": num_layers,
                    "kp_indices": self._unit_kp_indices(sim_unit_id),
                    "thick_mode": thick_mode,
                })
        return registry

    def panel_layer_to_unit(self, panel_idx, layer_idx=0):
        """
        Forward lookup: (panel_idx, layer_idx) -> simulator unit id.

        :param panel_idx: index in the original JSON "units" list
        :param layer_idx: layer within that panel (0 = lowest Z in thick mode)
        :return: sim_unit_id
        """
        mapping = self._panel_layer_mapping()
        if panel_idx < 0 or panel_idx >= len(mapping):
            raise IndexError(
                f"panel_idx {panel_idx} out of range [0, {len(mapping)})"
            )
        unit_ids = mapping[panel_idx]
        if layer_idx < 0 or layer_idx >= len(unit_ids):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {len(unit_ids)}) "
                f"for panel {panel_idx}"
            )
        return unit_ids[layer_idx]

    def unit_to_panel_layer(self, sim_unit_id):
        """
        Reverse lookup: simulator unit id -> (panel_idx, layer_idx) metadata.

        :param sim_unit_id: index into self.units
        :return: dict with panel_idx, layer_idx, height_z, num_layers, kp_indices, thick_mode
        """
        mapping = self._panel_layer_mapping()
        for panel_idx, unit_ids in enumerate(mapping):
            for layer_idx, unit_id in enumerate(unit_ids):
                if unit_id == sim_unit_id:
                    return {
                        "panel_idx": panel_idx,
                        "layer_idx": layer_idx,
                        "sim_unit_id": sim_unit_id,
                        "height_z": self._unit_layer_height_z(sim_unit_id),
                        "num_layers": len(unit_ids),
                        "kp_indices": self._unit_kp_indices(sim_unit_id),
                        "thick_mode": bool(getattr(self, "thick_mode_flag", False)),
                    }
        raise ValueError(f"sim_unit_id {sim_unit_id} does not belong to any panel/layer")

    # 2. collision detection

    # =========================================================================
    # Panel contact detection / shading (refined)
    # Pipeline when collision_shading=True:
    #   initializeRunning -> _build_collision_topology
    #   render/outputFigure -> detect_panel_collisions -> _render_panel_meshes
    # Order is strict:
    #   1) triangle intersection → red contact points/segments (3D GUI)
    #   2) map contact line to design xy → stamp paper canvas (sparse strokes)
    #   3) each frame: affine-map canvas strokes onto current panel pose → 3D lines
    #   4) classify intruders → light orange panel tint
    # Paint is a design-space canvas reprojected onto panels (not mesh-triangle paint).
    # =========================================================================

    def _unit_triangle_indices_flat(self, sim_unit_id):
        refs = self.ori_sim.tri_indices_ref
        tri_start = refs[sim_unit_id]
        tri_end = refs[sim_unit_id + 1] if sim_unit_id + 1 < len(refs) else len(self.ori_sim.tri_indices) // 3
        return np.array(self.ori_sim.tri_indices[3 * tri_start:3 * tri_end], dtype=np.int32)

    def _build_collision_topology(self):
        """Precompute triangle-to-unit and panel-index maps for collision detection."""
        refs = self.ori_sim.tri_indices_ref
        tri_indices = self.ori_sim.tri_indices
        num_tris = len(tri_indices) // 3

        tri_unit_ids = np.zeros(num_tris, dtype=np.int32)
        for unit_id, tri_start in enumerate(refs):
            tri_end = refs[unit_id + 1] if unit_id + 1 < len(refs) else num_tris
            for tri_idx in range(tri_start, tri_end):
                tri_unit_ids[tri_idx] = unit_id

        unit_panel_idx = np.full(self.unit_indices_num, -1, dtype=np.int32)
        unit_layer_idx = np.full(self.unit_indices_num, -1, dtype=np.int32)
        for entry in self.get_panel_layer_registry():
            unit_panel_idx[entry["sim_unit_id"]] = entry["panel_idx"]
            unit_layer_idx[entry["sim_unit_id"]] = entry["layer_idx"]

        tri_kp_indices = np.zeros((num_tris, 3), dtype=np.int32)
        for tri_idx in range(num_tris):
            base = 3 * tri_idx
            tri_kp_indices[tri_idx] = tri_indices[base:base + 3]

        self._collision_tri_unit_ids = tri_unit_ids
        self._collision_unit_panel_idx = unit_panel_idx
        self._collision_unit_layer_idx = unit_layer_idx
        self._collision_tri_kp_indices = tri_kp_indices
        self._collision_num_tris = num_tris
        self._collision_spatial_cell_size = max(self.max_size / 20.0, 1.0)
        # Particle radius scales with model size for visibility without covering panels
        self._collision_point_radius = max(self.max_size * 0.008, 0.15)
        # Per-unit outer/thickness meta for intruder prioritization
        self._collision_unit_layer_meta = self._build_unit_layer_meta()

    def _stop_sweep_drawing_at_pi(self):
        """
        Immediately freeze locus paint and seal export when θ hits π.

        Called from update_folding_target / slider / 'u' so drawing stops on the
        same step the clamp happens — not delayed until the next collision frame.
        """
        if getattr(self, "_sweep_draw_stopped", False):
            return
        self._sweep_draw_stopped = True
        # No further canvas stamps or trail growth / reproject
        self._paint_dirty = False
        self._stamp_paint_needed = False
        # Drop live red contact markers (locus strokes stay frozen as last drawn)
        self._collision_contact_count = 0
        self._collision_segment_count = 0
        self._collision_contact_points_list = []
        self._collision_contact_segments_list = []
        self._collision_contact_points_flat = []
        self._collision_contact_segments_flat = []
        if getattr(self, "_collision_coords_exported", False):
            return
        angle = float(getattr(self, "folding_angle", 3.1415))
        for ent in (getattr(self, "_collision_active_segments", None) or {}).values():
            for sk in ("side_a", "side_b"):
                side = ent.get(sk)
                if side is not None and "p0" in side:
                    try:
                        self._append_sweep_sample_2d(side, angle, min_move=0.0)
                    except Exception:
                        pass
        try:
            self._seal_and_export_fixed_flat_contacts(reason="fold_pi")
        except Exception as exc:
            if getattr(self, "verbose", False):
                print(f"[Contact] seal@π failed: {exc}")
            self._collision_coords_exported = True

    def _maybe_detect_panel_collisions(self):
        """
        Throttle collision broadphase / paint. Full triangle tests every frame
        freeze the GUI; run every N frames (still tracks contacts while folding).

        When the mesh did not advance this frame (paused), skip entirely — same
        contacts, same sample density when it was last running.
        """
        self._collision_frame_i = int(getattr(self, "_collision_frame_i", 0)) + 1
        # Drawing sealed at π → no more collision / paint work
        if getattr(self, "_sweep_draw_stopped", False) or (
            float(getattr(self, "folding_angle", 0.0)) >= 3.1415 - 1e-6
            and getattr(self, "_collision_coords_exported", False)
        ):
            return
        # Hit π but not sealed yet (e.g. headless jumped) → freeze now
        at_pi = float(getattr(self, "folding_angle", 0.0)) >= 3.1415 - 1e-6
        if at_pi:
            self._stop_sweep_drawing_at_pi()
            return
        if (
            not getattr(self, "_mesh_advanced", True)
            and getattr(self, "_collision_ran_once", False)
        ):
            return
        n = max(1, int(getattr(self, "_collision_every_n", 4)))
        # While folding: sample every frame so the locus trail is not sparse
        folding = abs(float(getattr(self, "enable_add_folding_angle", 0.0))) > 1e-12
        if folding:
            n = 1
        if (self._collision_frame_i % n) != 0 and self._collision_frame_i > 1:
            return
        self.detect_panel_collisions()

    def detect_panel_collisions(self):
        """Find inter-panel contact geometry and classify intruder panels."""
        if not hasattr(self, "_collision_tri_kp_indices"):
            return
        if not hasattr(self, "collision_contact_points"):
            return

        positions = self._cache_frame_positions()
        tri_kp_indices = self._collision_tri_kp_indices
        tri_unit_ids = self._collision_tri_unit_ids
        unit_panel_idx = self._collision_unit_panel_idx
        unit_layer_idx = self._collision_unit_layer_idx
        num_tris = self._collision_num_tris

        # Flat design keypoints (JSON unfolded layout) for 2D coordinate readout
        flat_kps = getattr(self, "_flat_kps_np", None)
        if flat_kps is None:
            flat_kps = np.asarray(self.kps, dtype=float)
            self._flat_kps_np = flat_kps
        if flat_kps.ndim != 2 or flat_kps.shape[0] < self.kp_num:
            flat_kps = positions  # fallback (should not happen after init)

        unit_layer_meta = getattr(self, "_collision_unit_layer_meta", None) or {}

        tri_coords = positions[tri_kp_indices]
        tri_aabb_min = tri_coords.min(axis=1)
        tri_aabb_max = tri_coords.max(axis=1)
        cell_size = self._collision_spatial_cell_size
        inv_cell = 1.0 / cell_size
        # Vectorized cell ranges (one floor per tri, not per axis call)
        min_cells = np.floor(tri_aabb_min * inv_cell).astype(np.int32)
        max_cells = np.floor(tri_aabb_max * inv_cell).astype(np.int32)
        grid = defaultdict(list)
        for tri_idx in range(num_tris):
            mnx, mny, mnz = int(min_cells[tri_idx, 0]), int(min_cells[tri_idx, 1]), int(min_cells[tri_idx, 2])
            mxx, mxy, mxz = int(max_cells[tri_idx, 0]), int(max_cells[tri_idx, 1]), int(max_cells[tri_idx, 2])
            for ix in range(mnx, mxx + 1):
                for iy in range(mny, mxy + 1):
                    for iz in range(mnz, mxz + 1):
                        grid[(ix, iy, iz)].append(tri_idx)

        checked_pairs = set()
        contact_points = []
        contact_segments = []  # list of (p0, p1) in 3D
        contact_points_flat = []
        contact_segments_flat = []  # flat JSON 2D + layer
        colliding_unit_pairs = set()  # (sim_unit_a, sim_unit_b) with real intersection
        max_pts = self._collision_contact_max
        max_segs = self._collision_contact_max
        aabb_eps = 1e-9

        def _flat_of(p3d, tri_idx):
            """
            Map 3D contact → design xy + material barycentric on that triangle.
            loc = (i0,i1,i2,u,v,w) re-evaluates on current mesh to stay on panel.
            """
            kp = tri_kp_indices[tri_idx]
            i0, i1, i2 = int(kp[0]), int(kp[1]), int(kp[2])
            xy, bary = _map_point_3d_to_flat_2d(
                p3d, positions[kp], flat_kps[kp][:, :2], return_bary=True
            )
            if xy is None or bary is None:
                return None
            return {
                "p": [float(xy[0]), float(xy[1])],
                "loc": (
                    i0, i1, i2,
                    float(bary[0]), float(bary[1]), float(bary[2]),
                ),
            }

        def _layer_of(unit_id, tri_idx):
            """Layer index + design height (JSON thick_panel_height / units z)."""
            meta = unit_layer_meta.get(int(unit_id), {})
            layer_idx = meta.get("layer_idx")
            if layer_idx is None:
                layer_idx = int(unit_layer_idx[unit_id]) if unit_id < len(unit_layer_idx) else -1
            layer_h = meta.get("layer_h", meta.get("height_z"))
            if layer_h is None:
                kp = tri_kp_indices[tri_idx]
                if flat_kps.shape[1] > 2:
                    layer_h = float(flat_kps[kp[0]][2])
                else:
                    layer_h = 0.0
            return int(layer_idx), float(layer_h)

        def _share_vert(ta, tb):
            """True if triangles share any keypoint (no set/list alloc)."""
            a0 = int(tri_kp_indices[ta, 0])
            a1 = int(tri_kp_indices[ta, 1])
            a2 = int(tri_kp_indices[ta, 2])
            b0 = int(tri_kp_indices[tb, 0])
            b1 = int(tri_kp_indices[tb, 1])
            b2 = int(tri_kp_indices[tb, 2])
            return (
                a0 == b0 or a0 == b1 or a0 == b2
                or a1 == b0 or a1 == b1 or a1 == b2
                or a2 == b0 or a2 == b1 or a2 == b2
            )

        def _aabb_overlap_idx(ta, tb):
            """Precomputed AABB overlap (avoids min/max over coord lists)."""
            if tri_aabb_max[ta, 0] < tri_aabb_min[tb, 0] - aabb_eps:
                return False
            if tri_aabb_max[tb, 0] < tri_aabb_min[ta, 0] - aabb_eps:
                return False
            if tri_aabb_max[ta, 1] < tri_aabb_min[tb, 1] - aabb_eps:
                return False
            if tri_aabb_max[tb, 1] < tri_aabb_min[ta, 1] - aabb_eps:
                return False
            if tri_aabb_max[ta, 2] < tri_aabb_min[tb, 2] - aabb_eps:
                return False
            if tri_aabb_max[tb, 2] < tri_aabb_min[ta, 2] - aabb_eps:
                return False
            return True

        for tri_list in grid.values():
            n_cell = len(tri_list)
            if n_cell < 2:
                continue
            for i in range(n_cell):
                tri_a = tri_list[i]
                unit_a = int(tri_unit_ids[tri_a])
                panel_a = int(unit_panel_idx[unit_a])
                layer_a = int(unit_layer_idx[unit_a])
                if panel_a < 0:
                    continue
                coords_a = tri_coords[tri_a]  # ndarray view; no .tolist()

                for j in range(i + 1, n_cell):
                    if len(contact_points) >= max_pts and len(contact_segments) >= max_segs:
                        break

                    tri_b = tri_list[j]
                    pair_key = (tri_a, tri_b) if tri_a < tri_b else (tri_b, tri_a)
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)

                    unit_b = int(tri_unit_ids[tri_b])
                    panel_b = int(unit_panel_idx[unit_b])
                    layer_b = int(unit_layer_idx[unit_b])
                    # Only different panels on the same thickness layer; skip shared verts
                    if panel_b < 0 or panel_a == panel_b or layer_a != layer_b:
                        continue
                    if unit_a == unit_b:
                        continue
                    if _share_vert(tri_a, tri_b):
                        continue
                    if not _aabb_overlap_idx(tri_a, tri_b):
                        continue

                    coords_b = tri_coords[tri_b]
                    # Coplanar pairs are treated as non-colliding (stacked/adjacent faces)
                    if _triangles_coplanar(coords_a, coords_b, eps=1e-2):
                        continue

                    pts = triangle_intersection_contacts_3d(coords_a, coords_b)
                    if not pts:
                        continue

                    # Record contact geometry first (red markers). Only then count the
                    # unit pair for intruder classification / orange paint.
                    # Flat 2D: map the same 3D endpoints onto BOTH panels' design xy
                    # → one segment (two nodes) per panel; locked when off-crease.
                    # layer_a == layer_b by the same-layer filter above.
                    recorded = False
                    layer_idx, layer_h = _layer_of(unit_a, tri_a)
                    # Keep unit↔panel correspondence when sorting unit ids
                    if unit_a < unit_b:
                        u_lo, u_hi = int(unit_a), int(unit_b)
                        p_of_ulo, p_of_uhi = int(panel_a), int(panel_b)
                        tri_lo, tri_hi = tri_a, tri_b
                    else:
                        u_lo, u_hi = int(unit_b), int(unit_a)
                        p_of_ulo, p_of_uhi = int(panel_b), int(panel_a)
                        tri_lo, tri_hi = tri_b, tri_a
                    p_lo, p_hi = (p_of_ulo, p_of_uhi) if p_of_ulo <= p_of_uhi else (p_of_uhi, p_of_ulo)
                    pair_meta = {
                        "unit_a": u_lo,
                        "unit_b": u_hi,
                        "panel_of_unit_a": p_of_ulo,
                        "panel_of_unit_b": p_of_uhi,
                        # Group id: the two JSON panels that collide together
                        "panel_a": p_lo,
                        "panel_b": p_hi,
                        "layer_idx": layer_idx,
                        "layer_h": layer_h,
                        # Triangle identity → one log slot per red contact line/point
                        "tri_a": int(tri_lo),
                        "tri_b": int(tri_hi),
                        "tri_lo": int(tri_lo),
                        "tri_hi": int(tri_hi),
                    }

                    def _side_flat(p3d0, p3d1=None):
                        """
                        Map 3D endpoint(s) onto both panels' design xy + bary loc.
                        loc sticks the point to the hit triangle for on-panel redraw.
                        """
                        a0 = _flat_of(p3d0, tri_lo)
                        b0 = _flat_of(p3d0, tri_hi)
                        if p3d1 is None:
                            if a0 is None and b0 is None:
                                return None
                            out = {}
                            if a0 is not None:
                                out["side_a"] = {
                                    "unit": u_lo, "panel": p_of_ulo,
                                    "p": a0["p"], "loc": a0["loc"],
                                }
                                out["p"] = a0["p"]
                            if b0 is not None:
                                out["side_b"] = {
                                    "unit": u_hi, "panel": p_of_uhi,
                                    "p": b0["p"], "loc": b0["loc"],
                                }
                                if "p" not in out:
                                    out["p"] = b0["p"]
                            return out
                        a1 = _flat_of(p3d1, tri_lo)
                        b1 = _flat_of(p3d1, tri_hi)
                        out = {}
                        if a0 is not None and a1 is not None:
                            out["side_a"] = {
                                "unit": u_lo, "panel": p_of_ulo,
                                "p0": a0["p"], "p1": a1["p"],
                                "loc0": a0["loc"], "loc1": a1["loc"],
                            }
                            out["p0"], out["p1"] = a0["p"], a1["p"]
                        if b0 is not None and b1 is not None:
                            out["side_b"] = {
                                "unit": u_hi, "panel": p_of_uhi,
                                "p0": b0["p"], "p1": b1["p"],
                                "loc0": b0["loc"], "loc1": b1["loc"],
                            }
                            if "p0" not in out:
                                out["p0"], out["p1"] = b0["p"], b1["p"]
                        return out if out else None

                    if len(pts) == 1:
                        if len(contact_points) < max_pts:
                            contact_points.append(pts[0])
                            ent = dict(pair_meta)
                            both = _side_flat(pts[0])
                            if both is not None:
                                ent.update(both)
                            p0 = pts[0]
                            ent["p_3d"] = [float(p0[0]), float(p0[1]), float(p0[2])]
                            # Always log (match every red particle)
                            contact_points_flat.append(ent)
                            recorded = True
                    else:
                        # Sort along principal direction → one contact segment (endpoints)
                        arr = np.asarray(pts, dtype=float)
                        direction = arr[-1] - arr[0]
                        if np.linalg.norm(direction) < 1e-12:
                            direction = arr.max(axis=0) - arr.min(axis=0)
                        if np.linalg.norm(direction) < 1e-12:
                            if len(contact_points) < max_pts:
                                contact_points.append(arr[0].tolist())
                                ent = dict(pair_meta)
                                both = _side_flat(arr[0])
                                if both is not None:
                                    ent.update(both)
                                p0 = arr[0]
                                ent["p_3d"] = [
                                    float(p0[0]), float(p0[1]), float(p0[2])
                                ]
                                contact_points_flat.append(ent)
                                recorded = True
                        else:
                            direction = direction / np.linalg.norm(direction)
                            order = np.argsort(arr @ direction)
                            p0 = arr[order[0]].tolist()
                            p1 = arr[order[-1]].tolist()
                            if len(contact_segments) < max_segs:
                                contact_segments.append((p0, p1))
                                ent = dict(pair_meta)
                                both = _side_flat(p0, p1)
                                if both is not None:
                                    ent.update(both)
                                # 3D endpoints for crease-proximity tests + logging
                                ent["p0_3d"] = [float(p0[0]), float(p0[1]), float(p0[2])]
                                ent["p1_3d"] = [float(p1[0]), float(p1[1]), float(p1[2])]
                                # Always log (match every red segment in the GUI)
                                contact_segments_flat.append(ent)
                                recorded = True
                            # Also mark endpoints as points for visibility (3D only)
                            if len(contact_points) < max_pts:
                                contact_points.append(p0)
                                recorded = True
                            if len(contact_points) < max_pts:
                                contact_points.append(p1)
                                recorded = True

                    # Gate: orange paint only for pairs that produced red markers
                    if recorded:
                        colliding_unit_pairs.add(
                            (min(unit_a, unit_b), max(unit_a, unit_b))
                        )

        self._collision_contact_count = len(contact_points)
        self._collision_segment_count = len(contact_segments)
        # Keep Python-side copies so the GUI can print node coordinates
        self._collision_contact_points_list = list(contact_points)
        self._collision_contact_segments_list = list(contact_segments)
        self._collision_contact_points_flat = list(contact_points_flat)
        self._collision_contact_segments_flat = list(contact_segments_flat)

        # Per-pair: update last-seen; finalize each pair when its contact ends;
        # export the full accumulated set once π fold is done.
        self._track_per_pair_flat_contacts()
        # Stamp locus canvas only when contacts refreshed and fold not sealed
        if not getattr(self, "_collision_coords_exported", False):
            try:
                self._stamp_paint_canvas_from_contacts()
                self._stamp_paint_needed = False
            except Exception:
                pass

        # Upload to Taichi fields (zero unused slots)
        pt_buf = np.zeros((max_pts, 3), dtype=np.float32)
        if contact_points:
            pt_buf[: len(contact_points)] = np.asarray(contact_points, dtype=np.float32)
        self.collision_contact_points.from_numpy(pt_buf)

        ln_buf = np.zeros((max_segs * 2, 3), dtype=np.float32)
        if contact_segments:
            seg_arr = np.asarray(contact_segments, dtype=np.float32).reshape(-1, 3)
            ln_buf[: seg_arr.shape[0]] = seg_arr
        self.collision_contact_lines.from_numpy(ln_buf)

        # Intruder logic starts only when red contact geometry exists.
        # No red points/segments → clear orange paint.
        has_red_markers = (
            self._collision_contact_count > 0 or self._collision_segment_count > 0
        )
        if has_red_markers and colliding_unit_pairs:
            self._update_intruder_classification(positions, colliding_unit_pairs)
        else:
            self._update_intruder_classification(positions, set())
        self._collision_ran_once = True

    # 3. intruder classification

    @staticmethod
    def _contact_pair_key(entry):
        """
        Stable key for one red contact the GUI would draw.

        Prefer triangle–triangle identity so *every* concurrent segment between
        the same unit pair is tracked (old key collapsed all of them to one).
        Fallback keeps unit/layer when tri ids are missing.
        """
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
        # Last resort: one slot per unit pair (legacy behaviour)
        return (ua, ub, layer, -1, -1)

    # 4. sweeping visualization

    @staticmethod
    def _copy_sweep_samples(samples):
        """Deep-copy a list of 2D line samples {angle, p0, p1}."""
        if not samples:
            return []
        out = []
        for s in samples:
            if s is None or "p0" not in s or "p1" not in s:
                continue
            out.append({
                "angle": float(s.get("angle", 0.0)),
                "p0": [float(s["p0"][0]), float(s["p0"][1])],
                "p1": [float(s["p1"][0]), float(s["p1"][1])],
            })
        return out

    @staticmethod
    def _copy_loc(loc):
        """Copy material barycentric handle (i0,i1,i2,u,v,w)."""
        if loc is None:
            return None
        try:
            return (
                int(loc[0]), int(loc[1]), int(loc[2]),
                float(loc[3]), float(loc[4]), float(loc[5]),
            )
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _copy_side_seg(side):
        if not side:
            return None
        out = {
            "unit": int(side["unit"]),
            "panel": int(side["panel"]),
            "p0": [float(side["p0"][0]), float(side["p0"][1])],
            "p1": [float(side["p1"][0]), float(side["p1"][1])],
        }
        # Material handles from the hit triangle (for on-panel locus redraw)
        if side.get("loc0") is not None:
            out["loc0"] = PD_Origami_Simulator._copy_loc(side["loc0"])
        if side.get("loc1") is not None:
            out["loc1"] = PD_Origami_Simulator._copy_loc(side["loc1"])
        # first_* = paper coords snapped onto a real crease (not noisy first contact)
        if "first_p0" in side and "first_p1" in side:
            out["first_p0"] = [float(side["first_p0"][0]), float(side["first_p0"][1])]
            out["first_p1"] = [float(side["first_p1"][0]), float(side["first_p1"][1])]
            out["first_folding_angle"] = float(side.get("first_folding_angle", 0.0))
        # 2D sweep trail: successive positions of the two-node contact line
        if side.get("sweep_samples"):
            out["sweep_samples"] = PD_Origami_Simulator._copy_sweep_samples(
                side["sweep_samples"]
            )
        return out

    @staticmethod
    def _copy_side_pt(side):
        if not side:
            return None
        out = {
            "unit": int(side["unit"]),
            "panel": int(side["panel"]),
            "p": [float(side["p"][0]), float(side["p"][1])],
        }
        if "first_p" in side:
            out["first_p"] = [float(side["first_p"][0]), float(side["first_p"][1])]
            out["first_folding_angle"] = float(side.get("first_folding_angle", 0.0))
        return out

    @classmethod
    def _copy_segment_entry(cls, s):
        """Copy segment: live endpoints + optional crease-snapped first + lock meta."""
        side_a = s.get("side_a")
        side_b = s.get("side_b")
        if side_a is None and "p0" in s:
            side_a = {
                "unit": int(s["unit_a"]),
                "panel": int(s.get("panel_of_unit_a", s.get("panel_a", -1))),
                "p0": s["p0"], "p1": s["p1"],
            }
            if "first_p0" in s:
                side_a["first_p0"] = s["first_p0"]
                side_a["first_p1"] = s["first_p1"]
                side_a["first_folding_angle"] = s.get("first_folding_angle", 0.0)
        out = {
            "layer_idx": int(s["layer_idx"]),
            "layer_h": float(s["layer_h"]),
            "unit_a": int(s["unit_a"]),
            "unit_b": int(s["unit_b"]),
            "panel_a": int(s.get("panel_a", -1)),
            "panel_b": int(s.get("panel_b", -1)),
            "panel_of_unit_a": int(s.get("panel_of_unit_a", s.get("panel_a", -1))),
            "panel_of_unit_b": int(s.get("panel_of_unit_b", s.get("panel_b", -1))),
            "folding_angle": float(s.get("folding_angle", 0.0)),
            "intruder_sim_unit": s.get("intruder_sim_unit"),
            "other_sim_unit": s.get("other_sim_unit"),
            "intruder_panel": s.get("intruder_panel"),
            "other_panel": s.get("other_panel"),
            "d_AtoB": s.get("d_AtoB"),
            "d_BtoA": s.get("d_BtoA"),
            "side_a": cls._copy_side_seg(side_a),
            "side_b": cls._copy_side_seg(side_b),
            "coords_locked": bool(s.get("coords_locked", False)),
            "fixed_reason": s.get("fixed_reason"),
            "first_locked": bool(s.get("first_locked", False)),
        }
        for tk in ("tri_a", "tri_b", "tri_lo", "tri_hi"):
            if s.get(tk) is not None:
                out[tk] = int(s[tk])
        if s.get("first_kp0") is not None:
            out["first_kp0"] = int(s["first_kp0"])
            out["first_kp1"] = int(s["first_kp1"])
        if s.get("p0_3d") is not None:
            out["p0_3d"] = [float(x) for x in s["p0_3d"][:3]]
        if s.get("p1_3d") is not None:
            out["p1_3d"] = [float(x) for x in s["p1_3d"][:3]]
        if out["side_a"] is not None:
            out["p0"] = list(out["side_a"]["p0"])
            out["p1"] = list(out["side_a"]["p1"])
            if "first_p0" in out["side_a"]:
                out["first_p0"] = list(out["side_a"]["first_p0"])
                out["first_p1"] = list(out["side_a"]["first_p1"])
                out["first_folding_angle"] = float(
                    out["side_a"].get("first_folding_angle", 0.0)
                )
        elif out["side_b"] is not None:
            out["p0"] = list(out["side_b"]["p0"])
            out["p1"] = list(out["side_b"]["p1"])
            if "first_p0" in out["side_b"]:
                out["first_p0"] = list(out["side_b"]["first_p0"])
                out["first_p1"] = list(out["side_b"]["first_p1"])
                out["first_folding_angle"] = float(
                    out["side_b"].get("first_folding_angle", 0.0)
                )
        elif "p0" in s:
            out["p0"] = [float(s["p0"][0]), float(s["p0"][1])]
            out["p1"] = [float(s["p1"][0]), float(s["p1"][1])]
        return out

    @staticmethod
    def _order_last_to_first(ent):
        """
        Order live/locked last endpoints so
          first_p0 → first_p1 → last_p1 → last_p0
        is a simple quad (no bow-tie): last_p0 near first_p0, last_p1 near first_p1.
        """
        f0 = ent.get("first_p0")
        f1 = ent.get("first_p1")
        if f0 is None or f1 is None:
            return ent
        f0 = np.asarray(f0, dtype=float)
        f1 = np.asarray(f1, dtype=float)

        def _order_side(side):
            if side is None or "p0" not in side:
                return
            a = np.asarray(side["p0"], dtype=float)
            b = np.asarray(side["p1"], dtype=float)
            # pairing A: a↔f0, b↔f1  vs  B: a↔f1, b↔f0
            cost_a = float(np.linalg.norm(a - f0) + np.linalg.norm(b - f1))
            cost_b = float(np.linalg.norm(a - f1) + np.linalg.norm(b - f0))
            if cost_b < cost_a:
                side["p0"], side["p1"] = list(b), list(a)
                if side.get("loc0") is not None and side.get("loc1") is not None:
                    side["loc0"], side["loc1"] = side["loc1"], side["loc0"]

        for sk in ("side_a", "side_b"):
            _order_side(ent.get(sk))
        if ent.get("side_a"):
            ent["p0"] = list(ent["side_a"]["p0"])
            ent["p1"] = list(ent["side_a"]["p1"])
        return ent

    @staticmethod
    def _order_segment_endpoints_2d(p0, p1, ref0=None, ref1=None):
        """
        Return (p0, p1) as lists, swapped if needed so p0 tracks ref0 and p1 tracks ref1
        (avoids bow-tie when chaining successive contact lines into a sweep ribbon).
        Pure Python (no numpy) — hot path during sweep sample append.
        """
        a = [float(p0[0]), float(p0[1])]
        b = [float(p1[0]), float(p1[1])]
        if ref0 is None or ref1 is None:
            return a, b
        r0x, r0y = float(ref0[0]), float(ref0[1])
        r1x, r1y = float(ref1[0]), float(ref1[1])
        dx0 = a[0] - r0x
        dy0 = a[1] - r0y
        dx1 = b[0] - r1x
        dy1 = b[1] - r1y
        cost_keep = (dx0 * dx0 + dy0 * dy0) ** 0.5 + (dx1 * dx1 + dy1 * dy1) ** 0.5
        sx0 = a[0] - r1x
        sy0 = a[1] - r1y
        sx1 = b[0] - r0x
        sy1 = b[1] - r0y
        cost_swap = (sx0 * sx0 + sy0 * sy0) ** 0.5 + (sx1 * sx1 + sy1 * sy1) ** 0.5
        if cost_swap < cost_keep:
            return b, a
        return a, b

    @staticmethod
    def _segment_mid_dist_2d(a0, a1, b0, b1):
        """Distance between midpoints of two 2D segments (pure Python)."""
        m0x = 0.5 * (float(a0[0]) + float(a1[0]))
        m0y = 0.5 * (float(a0[1]) + float(a1[1]))
        m1x = 0.5 * (float(b0[0]) + float(b1[0]))
        m1y = 0.5 * (float(b0[1]) + float(b1[1]))
        dx = m1x - m0x
        dy = m1y - m0y
        return (dx * dx + dy * dy) ** 0.5

    @classmethod
    def _maybe_seed_sweep_with_first(cls, side):
        """Prepend crease-snapped first line if trail exists but was not seeded yet."""
        if side is None or "first_p0" not in side or "first_p1" not in side:
            return
        samples = side.get("sweep_samples")
        if not samples:
            return
        f0 = [float(side["first_p0"][0]), float(side["first_p0"][1])]
        f1 = [float(side["first_p1"][0]), float(side["first_p1"][1])]
        if float(np.hypot(f1[0] - f0[0], f1[1] - f0[1])) < 1e-12:
            return
        s0 = samples[0]
        # Already starts at first (or very near)
        d = cls._segment_mid_dist_2d(s0["p0"], s0["p1"], f0, f1)
        if d < 1e-6:
            return
        samples.insert(0, {
            "angle": float(side.get("first_folding_angle", s0.get("angle", 0.0))),
            "p0": f0,
            "p1": f1,
        })

    @classmethod
    def _append_sweep_sample_2d(cls, side, angle, min_move=0.15, max_samples=None):
        """
        Append current side p0–p1 to a 2D sweep trail (paint stroke on the design panel).

        - Seeds the trail with crease-snapped first_* when present (so paint starts on hinge).
        - Skips samples that barely moved (min_move) to avoid pure duplicates.
        - Endpoint order is stabilized against the previous sample (no twist).
        - **No sample count cap** (max_samples ignored; kept for call-site compat).
        """
        if side is None or "p0" not in side or "p1" not in side:
            return
        p0 = [float(side["p0"][0]), float(side["p0"][1])]
        p1 = [float(side["p1"][0]), float(side["p1"][1])]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if (dx * dx + dy * dy) < 1e-24:
            return

        samples = side.get("sweep_samples")
        if samples is None:
            samples = []
            side["sweep_samples"] = samples

        # Seed with crease first line once, before the live contact stroke
        if not samples and "first_p0" in side and "first_p1" in side:
            f0 = [float(side["first_p0"][0]), float(side["first_p0"][1])]
            f1 = [float(side["first_p1"][0]), float(side["first_p1"][1])]
            fdx = f1[0] - f0[0]
            fdy = f1[1] - f0[1]
            if (fdx * fdx + fdy * fdy) >= 1e-24:
                samples.append({
                    "angle": float(side.get("first_folding_angle", angle)),
                    "p0": f0,
                    "p1": f1,
                })

        if samples:
            last = samples[-1]
            p0, p1 = cls._order_segment_endpoints_2d(p0, p1, last["p0"], last["p1"])
            moved = cls._segment_mid_dist_2d(last["p0"], last["p1"], p0, p1)
            e0x = p0[0] - last["p0"][0]
            e0y = p0[1] - last["p0"][1]
            e1x = p1[0] - last["p1"][0]
            e1y = p1[1] - last["p1"][1]
            end_move = max((e0x * e0x + e0y * e0y) ** 0.5, (e1x * e1x + e1y * e1y) ** 0.5)
            # Same pose → skip (but allow last lock rewrite via force path elsewhere)
            mm = float(min_move)
            if moved < mm and end_move < mm:
                # Still refresh last sample angle/position lightly if tiny drift
                if moved < 1e-9:
                    return
                samples[-1] = {"angle": float(angle), "p0": p0, "p1": p1}
                return

        samples.append({"angle": float(angle), "p0": p0, "p1": p1})
        # No thinning / max_samples cap — keep every sample

    @classmethod
    def sweep_paint_polygon_2d(cls, samples):
        """
        Build the 2D paint region swept by successive two-node contact lines.

        samples: list of {p0, p1} in design xy.
        Returns ordered closed ring [p0_0..p0_n, p1_n..p1_0] or None if < 2 samples
        (or a single-sample degenerate band with zero area).
        Purely 2D — no 3D geometry.
        """
        if not samples:
            return None
        # Stabilize endpoint pairing along the whole trail
        ordered = []
        prev0 = prev1 = None
        for s in samples:
            if s is None or "p0" not in s or "p1" not in s:
                continue
            p0, p1 = cls._order_segment_endpoints_2d(s["p0"], s["p1"], prev0, prev1)
            ordered.append((p0, p1))
            prev0, prev1 = p0, p1
        if not ordered:
            return None
        if len(ordered) == 1:
            # Single line: no area yet — return None (caller may use first/last triangle)
            return None
        trail0 = [p0 for p0, _ in ordered]
        trail1 = [p1 for _, p1 in ordered]
        poly = [[float(p[0]), float(p[1])] for p in trail0]
        poly += [[float(p[0]), float(p[1])] for p in reversed(trail1)]
        # Drop near-duplicate consecutive verts
        cleaned = [poly[0]]
        for q in poly[1:]:
            prev = cleaned[-1]
            if abs(q[0] - prev[0]) > 1e-9 or abs(q[1] - prev[1]) > 1e-9:
                cleaned.append(q)
        if len(cleaned) < 3:
            return None
        area = cls._polygon_area_2d(cleaned)
        if area < 1e-12:
            return None
        return cleaned

    @staticmethod
    def _polygon_area_2d(pts):
        """Absolute shoelace area of a 2D ring (list of [x,y])."""
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

    @staticmethod
    def closed_polygon_from_first_last(ent, side_key="side_a"):
        """
        Build a closed 4-gon from JSON crease first + locked last on one side:
          [first_p0, first_p1, last_p1, last_p0]
        Returns list of [x,y] or None. (Legacy / debug; shading uses the triangle.)
        """
        side = ent.get(side_key) or ent
        f0 = side.get("first_p0", ent.get("first_p0"))
        f1 = side.get("first_p1", ent.get("first_p1"))
        p0 = side.get("p0", ent.get("p0"))
        p1 = side.get("p1", ent.get("p1"))
        if f0 is None or f1 is None or p0 is None or p1 is None:
            return None
        return [
            [float(f0[0]), float(f0[1])],
            [float(f1[0]), float(f1[1])],
            [float(p1[0]), float(p1[1])],
            [float(p0[0]), float(p0[1])],
        ]

    @classmethod
    def closed_triangle_max_area_from_first_last(cls, ent, side_key="side_a"):
        """
        Closed triangle for trim shading / JSON export.

        Always uses both first crease endpoints. From the two last endpoints,
        keeps the apex that forms the *larger-area* triangle:

          T0 = [first_p0, first_p1, last_p0]
          T1 = [first_p0, first_p1, last_p1]

        Returns dict:
          triangle      list of 3 [x,y]
          area          float
          last_index    0 or 1 (which last endpoint was chosen)
          last_point    [x,y] apex
          first_p0/p1   the fixed first edge
        or None if inputs are incomplete / both areas ~0.
        """
        side = ent.get(side_key) or ent
        f0 = side.get("first_p0", ent.get("first_p0"))
        f1 = side.get("first_p1", ent.get("first_p1"))
        p0 = side.get("p0", ent.get("p0"))
        p1 = side.get("p1", ent.get("p1"))
        if f0 is None or f1 is None or p0 is None or p1 is None:
            return None
        f0 = [float(f0[0]), float(f0[1])]
        f1 = [float(f1[0]), float(f1[1])]
        lasts = [
            [float(p0[0]), float(p0[1])],
            [float(p1[0]), float(p1[1])],
        ]
        best = None
        best_area = -1.0
        for idx, apex in enumerate(lasts):
            tri = [f0, f1, apex]
            area = cls._polygon_area_2d(tri)
            if area > best_area:
                best_area = area
                best = {
                    "triangle": tri,
                    "area": float(area),
                    "last_index": int(idx),
                    "last_point": list(apex),
                    "first_p0": list(f0),
                    "first_p1": list(f1),
                }
        if best is None or best_area < 1e-18:
            return None
        return best

    @classmethod
    def _copy_point_entry(cls, e):
        side_a = e.get("side_a")
        side_b = e.get("side_b")
        if side_a is None and "p" in e:
            side_a = {
                "unit": int(e["unit_a"]),
                "panel": int(e.get("panel_of_unit_a", e.get("panel_a", -1))),
                "p": e["p"],
            }
        out = {
            "layer_idx": int(e["layer_idx"]),
            "layer_h": float(e["layer_h"]),
            "unit_a": int(e["unit_a"]),
            "unit_b": int(e["unit_b"]),
            "panel_a": int(e.get("panel_a", -1)),
            "panel_b": int(e.get("panel_b", -1)),
            "panel_of_unit_a": int(e.get("panel_of_unit_a", e.get("panel_a", -1))),
            "panel_of_unit_b": int(e.get("panel_of_unit_b", e.get("panel_b", -1))),
            "folding_angle": float(e.get("folding_angle", 0.0)),
            "intruder_sim_unit": e.get("intruder_sim_unit"),
            "other_sim_unit": e.get("other_sim_unit"),
            "intruder_panel": e.get("intruder_panel"),
            "other_panel": e.get("other_panel"),
            "d_AtoB": e.get("d_AtoB"),
            "d_BtoA": e.get("d_BtoA"),
            "side_a": cls._copy_side_pt(side_a),
            "side_b": cls._copy_side_pt(side_b),
        }
        for tk in ("tri_a", "tri_b", "tri_lo", "tri_hi"):
            if e.get(tk) is not None:
                out[tk] = int(e[tk])
        if e.get("p_3d") is not None:
            out["p_3d"] = [float(x) for x in e["p_3d"][:3]]
        if out["side_a"] is not None:
            out["p"] = list(out["side_a"]["p"])
        elif out["side_b"] is not None:
            out["p"] = list(out["side_b"]["p"])
        elif "p" in e:
            out["p"] = [float(e["p"][0]), float(e["p"][1])]
        return out

    def _stamp_intruder_on_entry(self, ent):
        """
        Attach classified intruder / host to a contact entry.
        Uses latest live classification; keeps previous stamp if none this frame.
        """
        ua, ub = int(ent["unit_a"]), int(ent["unit_b"])
        pair = (min(ua, ub), max(ua, ub))
        cls_map = getattr(self, "_intruder_by_unit_pair", None) or {}
        cls = cls_map.get(pair)
        if cls is None:
            return ent  # keep any prior stamp

        intr_u = int(cls["intruder_sim_unit"])
        oth_u = int(cls["other_sim_unit"])
        # Map sim unit → JSON panel using this entry's unit↔panel fields
        if intr_u == ua:
            intr_p = int(ent.get("panel_of_unit_a", ent.get("panel_a", -1)))
            oth_p = int(ent.get("panel_of_unit_b", ent.get("panel_b", -1)))
        elif intr_u == ub:
            intr_p = int(ent.get("panel_of_unit_b", ent.get("panel_b", -1)))
            oth_p = int(ent.get("panel_of_unit_a", ent.get("panel_a", -1)))
        else:
            # Classification units don't match entry (shouldn't happen)
            intr_p = oth_p = -1

        ent["intruder_sim_unit"] = intr_u
        ent["other_sim_unit"] = oth_u
        ent["intruder_panel"] = intr_p
        ent["other_panel"] = oth_p
        ent["d_AtoB"] = float(cls.get("d_AtoB", 0.0))
        ent["d_BtoA"] = float(cls.get("d_BtoA", 0.0))
        return ent

    @staticmethod
    def _segment_length_xy(s):
        if "p0" in s and "p1" in s and s["p0"] is not None and s["p1"] is not None:
            p0, p1 = s["p0"], s["p1"]
            return float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        # Fall back to 3D length when flat map was partial / missing
        if s.get("p0_3d") is not None and s.get("p1_3d") is not None:
            a = np.asarray(s["p0_3d"], dtype=float)
            b = np.asarray(s["p1_3d"], dtype=float)
            return float(np.linalg.norm(b - a))
        return 0.0

    @staticmethod
    def _point_segment_distance_3d(p, a, b):
        """Distance from point p to segment ab in 3D (pure Python, hot path)."""
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        ax, ay, az = float(a[0]), float(a[1]), float(a[2])
        bx, by, bz = float(b[0]), float(b[1]), float(b[2])
        abx, aby, abz = bx - ax, by - ay, bz - az
        lab2 = abx * abx + aby * aby + abz * abz
        if lab2 < 1e-18:
            dx, dy, dz = px - ax, py - ay, pz - az
            return (dx * dx + dy * dy + dz * dz) ** 0.5
        t = ((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / lab2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        dx = px - (ax + t * abx)
        dy = py - (ay + t * aby)
        dz = pz - (az + t * abz)
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def _get_crease_pairs_np(self):
        """Cached crease endpoint indices (host array)."""
        cached = getattr(self, "_crease_pairs_np", None)
        if cached is not None:
            return cached
        n = int(getattr(self, "crease_pairs_num", 0) or 0)
        if n <= 0 or not hasattr(self, "crease_pairs"):
            self._crease_pairs_np = np.zeros((0, 2), dtype=np.int32)
            return self._crease_pairs_np
        try:
            pairs = self.crease_pairs.to_numpy() if hasattr(self.crease_pairs, "to_numpy") else None
        except Exception:
            pairs = None
        if pairs is None:
            pairs = np.array(
                [[int(self.crease_pairs[i, 0]), int(self.crease_pairs[i, 1])] for i in range(n)],
                dtype=np.int32,
            )
        else:
            pairs = np.asarray(pairs[:n], dtype=np.int32)
        self._crease_pairs_np = pairs
        return pairs

    def _fold_crease_records(self, positions):
        """
        Mountain/valley creases with both live 3D and design (paper) flat ends.

        Each record:
          a3d, b3d  — current sim positions
          a_flat, b_flat — design xy from self.kps (true paper line)
        """
        recs = []
        pairs = self._get_crease_pairs_np()
        if pairs is None or len(pairs) == 0:
            return recs
        flat_kps = getattr(self, "_flat_kps_np", None)
        if flat_kps is None:
            flat_kps = np.asarray(self.kps, dtype=float)
            self._flat_kps_np = flat_kps
        n_pos = len(positions)
        n_flat = len(flat_kps)
        for i0, i1 in pairs:
            i0 = int(i0)
            i1 = int(i1)
            if i0 < 0 or i1 < 0 or i0 >= n_pos or i1 >= n_pos:
                continue
            if i0 >= n_flat or i1 >= n_flat:
                continue
            recs.append({
                "a3d": positions[i0][:3],
                "b3d": positions[i1][:3],
                "a_flat": flat_kps[i0][:2],
                "b_flat": flat_kps[i1][:2],
                "i0": i0,
                "i1": i1,
            })
        return recs

    def _fold_crease_segments_3d(self, positions):
        """3D crease segments only (compatibility helper)."""
        return [(r["a3d"], r["b3d"]) for r in self._fold_crease_records(positions)]

    def _unit_kp_set(self, sim_unit_id):
        """Set of keypoint indices owned by a sim unit."""
        if not hasattr(self, "ori_sim") or sim_unit_id is None:
            return set()
        try:
            raw = self.ori_sim.indices[int(sim_unit_id)]
        except Exception:
            return set()
        return {int(j) for j in raw if int(j) >= 0}

    def _shared_crease_record(self, unit_a, unit_b, crease_recs):
        """
        Shared mountain/valley crease between two units: both crease
        endpoints belong to both panels (the common fold edge).
        """
        va = self._unit_kp_set(unit_a)
        vb = self._unit_kp_set(unit_b)
        if not va or not vb:
            return None
        for r in crease_recs:
            i0, i1 = r["i0"], r["i1"]
            if i0 in va and i1 in va and i0 in vb and i1 in vb:
                return r
        return None

    def _nearest_crease_vertex_json(self, p3d, crease_recs):
        """
        Snap p3d to the nearest *actual JSON keypoint* that is a crease end
        (not an interpolated point on the segment).

        Returns (kp_index, flat_xy, dist3d) or (None, None, inf).
        """
        if not crease_recs:
            return None, None, float("inf")
        p = np.asarray(p3d, dtype=float).reshape(3)
        best_i = None
        best_f = None
        best_d = float("inf")
        seen = set()
        for r in crease_recs:
            for key, flat_key in (("i0", "a_flat"), ("i1", "b_flat")):
                ii = r[key]
                if ii in seen:
                    continue
                seen.add(ii)
                q = r["a3d"] if key == "i0" else r["b3d"]
                d = float(np.linalg.norm(p - q))
                if d < best_d:
                    best_d = d
                    best_i = ii
                    best_f = [float(r[flat_key][0]), float(r[flat_key][1])]
        return best_i, best_f, best_d

    def _json_crease_first_pair(self, p0_3d, p1_3d, crease_recs, unit_a=None, unit_b=None):
        """
        first endpoints = real JSON keypoint coordinates on creases.

        Priority:
          1) Shared fold edge between unit_a and unit_b → exact crease kps
          2) Else nearest crease *vertex* (JSON kp) for each contact node

        Returns (f0, f1, d0, d1, kp0, kp1) or Nones.
        """
        # 1) Shared crease between the two panels — true fold edge on the paper
        shared = None
        if unit_a is not None and unit_b is not None:
            shared = self._shared_crease_record(unit_a, unit_b, crease_recs)
        if shared is not None:
            f0 = [float(shared["a_flat"][0]), float(shared["a_flat"][1])]
            f1 = [float(shared["b_flat"][0]), float(shared["b_flat"][1])]
            # Distance of contact segment mid to that crease (for "near" test)
            mid = 0.5 * (
                np.asarray(p0_3d, dtype=float)[:3]
                + np.asarray(p1_3d, dtype=float)[:3]
            )
            d_mid = self._point_segment_distance_3d(
                mid, shared["a3d"], shared["b3d"]
            )
            return f0, f1, d_mid, d_mid, int(shared["i0"]), int(shared["i1"])

        # 2) Snap each contact node to nearest crease vertex (defined in JSON)
        i0, f0, d0 = self._nearest_crease_vertex_json(p0_3d, crease_recs)
        i1, f1, d1 = self._nearest_crease_vertex_json(p1_3d, crease_recs)
        if f0 is None or f1 is None:
            return None, None, float("inf"), float("inf"), None, None

        # If both snapped to the same kp, use that kp's full crease endpoints
        if i0 is not None and i0 == i1:
            for r in crease_recs:
                if r["i0"] == i0 or r["i1"] == i0:
                    f0 = [float(r["a_flat"][0]), float(r["a_flat"][1])]
                    f1 = [float(r["b_flat"][0]), float(r["b_flat"][1])]
                    i0, i1 = int(r["i0"]), int(r["i1"])
                    break

        if float(np.hypot(f1[0] - f0[0], f1[1] - f0[1])) < 1e-9:
            return None, None, d0, d1, None, None
        return f0, f1, d0, d1, i0, i1

    def _endpoints_off_creases(self, p0_3d, p1_3d, crease_segs, tol):
        """
        True when BOTH contact nodes are farther than tol from every crease.
        Segment must still exist (caller guarantees both endpoints present).
        """
        if p0_3d is None or p1_3d is None or not crease_segs:
            return False
        def _off(p):
            for a, b in crease_segs:
                if self._point_segment_distance_3d(p, a, b) <= tol:
                    return False
            return True
        return _off(p0_3d) and _off(p1_3d)

    def _track_per_pair_flat_contacts(self):
        """
        Track every red contact (one slot per triangle–triangle pair).

        - **first**: actual JSON crease *keypoint* coords (shared fold edge
          endpoints, or nearest crease vertices) — discrete design vertices
          so first+last can form a closed polygon on the paper.
        - **last (p0/p1)**: live contact flat map; locked when both nodes leave
          the creases (or contact ends).
        - **sweep (2D paint)**: successive p0–p1 samples on the design plane
          while the contact line is live. The ribbon those samples sweep out
          is the 2D panel paint region (no 3D paint geometry) and the
          **exported shaded-area coordinates**.

        Multiple concurrent segments between the same unit pair are all kept
        (keyed by tri_a/tri_b), matching the GUI red lines.
        """
        # Drawing frozen at π — never record more samples
        if getattr(self, "_sweep_draw_stopped", False):
            return
        # Recover from a premature empty seal (exported=True with no fixed data)
        if self._collision_coords_exported:
            has_fixed = bool(self._collision_fixed_segments) or bool(
                self._collision_fixed_points
            )
            if has_fixed:
                return
            # Empty seal — reopen tracking until we have real contacts or hit π
            if float(getattr(self, "folding_angle", 0.0)) < 3.1415 - 1e-6:
                self._collision_coords_exported = False
            else:
                return

        angle = float(self.folding_angle)
        positions = self._cache_frame_positions()
        crease_recs = self._fold_crease_records(positions)
        crease_segs = [(r["a3d"], r["b3d"]) for r in crease_recs]
        crease_tol = max(float(getattr(self, "max_size", 100.0)) * 0.012, 0.35)
        # Min mid-point travel (design units) before recording another sweep sample
        # Dense trail (smaller threshold → less sparse ribbon / locus)
        sweep_min_move = max(float(getattr(self, "max_size", 100.0)) * 3e-4, 0.012)

        # One entry per triangle–triangle contact (same as each red GUI segment/point).
        # Do NOT collapse multiple contacts that share the same unit pair.
        segs_now = {}
        for s in self._collision_contact_segments_flat:
            s = dict(s)
            s["folding_angle"] = angle
            key = self._contact_pair_key(s)
            segs_now[key] = self._copy_segment_entry(s)

        pts_now = {}
        for e in self._collision_contact_points_flat:
            key = self._contact_pair_key(e)
            # Same tri-pair already produced a segment this frame → skip lone point
            if key in segs_now:
                continue
            e = dict(e)
            e["folding_angle"] = angle
            pts_now[key] = self._copy_point_entry(e)

        def _carry_intruder(ent, prev):
            if prev is None:
                return
            for fld in (
                "intruder_sim_unit", "other_sim_unit",
                "intruder_panel", "other_panel", "d_AtoB", "d_BtoA",
            ):
                if ent.get(fld) is None and prev.get(fld) is not None:
                    ent[fld] = prev[fld]

        def _carry_sweep(ent, prev):
            """Bring prior 2D sweep trail onto the live entry (fresh copy has none).

            Share the list by reference while the contact is live (append in place).
            Deep-copy only when locking / exporting via _copy_segment_entry.
            """
            if prev is None:
                return
            for sk in ("side_a", "side_b"):
                side = ent.get(sk)
                pside = prev.get(sk)
                if side is None or pside is None:
                    continue
                if side.get("sweep_samples"):
                    continue  # already has trail
                samples = pside.get("sweep_samples")
                if samples:
                    side["sweep_samples"] = samples  # shared mutable trail

        def _record_sweep(ent, force_last=False):
            """Append current 2D contact line to the paint sweep on each panel side."""
            # force_last: min_move=0 so the terminal pose is always recorded
            min_move = 0.0 if force_last else sweep_min_move
            for sk in ("side_a", "side_b"):
                side = ent.get(sk)
                if side is None:
                    continue
                self._append_sweep_sample_2d(side, angle, min_move=min_move)

        def _is_locked(key):
            fin = self._collision_fixed_segments.get(key)
            return bool(fin and fin.get("coords_locked"))

        def _apply_first_on_creases(ent, prev, p0_3d, p1_3d):
            """
            Set first_* to *actual JSON keypoint coords* on creases
            (shared fold edge endpoints, or nearest crease vertices).

            These are discrete design vertices so first+last can form a
            closed polygon with real pattern corners.
            """
            def _copy_first_from(src):
                ent["first_locked"] = bool(src.get("first_locked", False))
                if "first_kp0" in src:
                    ent["first_kp0"] = int(src["first_kp0"])
                    ent["first_kp1"] = int(src["first_kp1"])
                for sk in ("side_a", "side_b"):
                    side = ent.get(sk)
                    ps = src.get(sk)
                    if side is None or ps is None:
                        continue
                    if "first_p0" in ps:
                        side["first_p0"] = list(ps["first_p0"])
                        side["first_p1"] = list(ps["first_p1"])
                        side["first_folding_angle"] = float(
                            ps.get("first_folding_angle", angle)
                        )
                if "first_p0" in src:
                    ent["first_p0"] = list(src["first_p0"])
                    ent["first_p1"] = list(src["first_p1"])
                    ent["first_folding_angle"] = float(
                        src.get("first_folding_angle", angle)
                    )

            # Keep already-locked first (JSON vertices stay fixed)
            if prev and prev.get("first_locked") and prev.get("first_p0") is not None:
                _copy_first_from(prev)
                ent["first_locked"] = True
                return

            if p0_3d is None or p1_3d is None or not crease_recs:
                if prev and prev.get("first_p0") is not None:
                    _copy_first_from(prev)
                return

            f0, f1, d0, d1, kp0, kp1 = self._json_crease_first_pair(
                p0_3d, p1_3d, crease_recs,
                unit_a=ent.get("unit_a"), unit_b=ent.get("unit_b"),
            )
            if f0 is None:
                if prev and prev.get("first_p0") is not None:
                    _copy_first_from(prev)
                return

            near = (d0 <= crease_tol * 2.0) and (d1 <= crease_tol * 2.0)
            # If far and we already have a first, keep it (wait for a near frame)
            if not near and prev is not None and prev.get("first_p0") is not None:
                _copy_first_from(prev)
                return

            # Write exact JSON crease-vertex coords on both sides
            for sk in ("side_a", "side_b"):
                side = ent.get(sk)
                if side is None:
                    continue
                side["first_p0"] = list(f0)
                side["first_p1"] = list(f1)
                side["first_folding_angle"] = float(angle)
            ent["first_p0"] = list(f0)
            ent["first_p1"] = list(f1)
            ent["first_folding_angle"] = float(angle)
            if kp0 is not None:
                ent["first_kp0"] = int(kp0)
                ent["first_kp1"] = int(kp1)
            # Lock once contact is near the hinge (reliable shared crease)
            if near:
                ent["first_locked"] = True

        for key, ent in segs_now.items():
            already = _is_locked(key)
            prev = self._collision_active_segments.get(key)

            if already:
                locked = self._collision_fixed_segments[key]
                _carry_intruder(ent, locked)
                _carry_sweep(ent, locked)
                # Keep first from locked snapshot on live display entry
                _apply_first_on_creases(ent, locked, ent.get("p0_3d"), ent.get("p1_3d"))
                self._stamp_intruder_on_entry(ent)
                # Locked: paint trail is frozen (no more 2D samples)
                self._collision_active_segments[key] = ent
                continue

            if ent.get("side_a") and "p0" in ent["side_a"]:
                sa = ent["side_a"]
                ent["p0"], ent["p1"] = list(sa["p0"]), list(sa["p1"])
            elif ent.get("side_b") and "p0" in ent["side_b"]:
                sb = ent["side_b"]
                ent["p0"], ent["p1"] = list(sb["p0"]), list(sb["p1"])
            if prev is not None:
                if ent.get("p0_3d") is None and prev.get("p0_3d") is not None:
                    ent["p0_3d"] = list(prev["p0_3d"])
                    ent["p1_3d"] = list(prev["p1_3d"])
            _carry_intruder(ent, prev)
            _carry_sweep(ent, prev)
            self._stamp_intruder_on_entry(ent)

            p0_3d = ent.get("p0_3d")
            p1_3d = ent.get("p1_3d")
            _apply_first_on_creases(ent, prev, p0_3d, p1_3d)

            # 2D paint: record the mapped two-node line on each panel's design xy
            _record_sweep(ent, force_last=False)

            # Lock last (trim) when BOTH nodes left creases (segment still live)
            min_len = max(float(getattr(self, "max_size", 100.0)) * 0.002, 0.05)
            long_enough = self._segment_length_xy(ent) >= min_len
            if (
                long_enough
                and p0_3d is not None
                and p1_3d is not None
                and self._endpoints_off_creases(p0_3d, p1_3d, crease_segs, crease_tol)
            ):
                # Ensure first exists even if we never got a near-crease frame
                if not ent.get("first_p0") and crease_recs:
                    f0, f1, _, _, kp0, kp1 = self._json_crease_first_pair(
                        p0_3d, p1_3d, crease_recs,
                        unit_a=ent.get("unit_a"), unit_b=ent.get("unit_b"),
                    )
                    if f0 is not None:
                        for sk in ("side_a", "side_b"):
                            side = ent.get(sk)
                            if side is None:
                                continue
                            side["first_p0"] = list(f0)
                            side["first_p1"] = list(f1)
                            side["first_folding_angle"] = float(angle)
                        ent["first_p0"] = list(f0)
                        ent["first_p1"] = list(f1)
                        ent["first_folding_angle"] = float(angle)
                        if kp0 is not None:
                            ent["first_kp0"] = int(kp0)
                            ent["first_kp1"] = int(kp1)
                        ent["first_locked"] = True
                # Order last endpoints for a simple closed quad with first
                if ent.get("first_p0") is not None:
                    ent = self._order_last_to_first(ent)
                    for sk in ("side_a", "side_b"):
                        self._maybe_seed_sweep_with_first(ent.get(sk))
                # Final paint sample at lock position
                _record_sweep(ent, force_last=True)
                ent["coords_locked"] = True
                ent["fixed_reason"] = "off_crease"
                snap = self._copy_segment_entry(ent)
                snap["coords_locked"] = True
                snap["fixed_reason"] = "off_crease"
                if "first_kp0" in ent:
                    snap["first_kp0"] = ent["first_kp0"]
                    snap["first_kp1"] = ent["first_kp1"]
                self._collision_fixed_segments[key] = snap
            else:
                self._collision_fixed_segments.pop(key, None)

            self._collision_active_segments[key] = ent

        for key, ent in pts_now.items():
            if key in self._collision_fixed_segments and self._collision_fixed_segments[key].get("coords_locked"):
                self._collision_active_points.pop(key, None)
                continue
            prev = self._collision_active_points.get(key)
            if ent.get("side_a"):
                ent["p"] = list(ent["side_a"]["p"])
            _carry_intruder(ent, prev)
            self._stamp_intruder_on_entry(ent)
            self._collision_active_points[key] = ent
            self._collision_fixed_points.pop(key, None)

        # Contact ended → lock if not already locked off-crease
        for key in list(self._collision_active_segments.keys()):
            if key not in segs_now:
                fin = self._collision_active_segments.pop(key)
                self._stamp_intruder_on_entry(fin)
                if not (self._collision_fixed_segments.get(key) or {}).get("coords_locked"):
                    # Seal last paint sample on the final known 2D line
                    for sk in ("side_a", "side_b"):
                        side = fin.get(sk)
                        if side is not None and "p0" in side:
                            self._append_sweep_sample_2d(
                                side, float(fin.get("folding_angle", angle)),
                                min_move=0.0,
                            )
                    fin["coords_locked"] = True
                    fin["fixed_reason"] = fin.get("fixed_reason") or "contact_ended"
                    self._collision_fixed_segments[key] = fin
        for key in list(self._collision_active_points.keys()):
            if key not in pts_now and key not in self._collision_active_segments:
                if key in self._collision_fixed_segments:
                    self._collision_active_points.pop(key, None)
                else:
                    fin = self._collision_active_points.pop(key)
                    self._stamp_intruder_on_entry(fin)
                    fin["coords_locked"] = True
                    fin["fixed_reason"] = "contact_ended"
                    self._collision_fixed_points[key] = fin

        for key in list(self._collision_fixed_points.keys()):
            if key in self._collision_fixed_segments:
                del self._collision_fixed_points[key]

        # Freeze sweeping immediately at π (also handled in update_folding_target;
        # keep this as a safety net if track runs without that path).
        if angle >= 3.1415 - 1e-6:
            self._stop_sweep_drawing_at_pi()

    # 5. coordinates export

    def _seal_and_export_fixed_flat_contacts(self, reason="fold_complete"):
        """Move any remaining actives into fixed, then export once.

        reason: fixed_reason tag for still-live contacts (e.g. fold_pi).
        After a successful export, sweep sample recording and paint stamping stop.

        Exported coordinates for each panel side are the **shaded sweep region**
        (ribbon of successive two-node contact lines in design xy).
        """
        if self._collision_coords_exported:
            # Allow recovery only when previous seal stored nothing
            if self._collision_fixed_segments or self._collision_fixed_points:
                return
            self._collision_coords_exported = False
        seal_reason = reason or "fold_complete"
        for key, ent in list(self._collision_active_segments.items()):
            existing = self._collision_fixed_segments.get(key)
            if existing and existing.get("coords_locked"):
                # Prefer the longer sweep trail if active has more samples
                for sk in ("side_a", "side_b"):
                    es = (ent.get(sk) or {}).get("sweep_samples") or []
                    xs = (existing.get(sk) or {}).get("sweep_samples") or []
                    if len(es) > len(xs) and existing.get(sk) is not None:
                        existing[sk]["sweep_samples"] = self._copy_sweep_samples(es)
                continue  # keep off_crease (or earlier) lock
            # Snapshot so later shared-list mutations cannot grow sealed trails
            ent = self._copy_segment_entry(ent)
            ent["coords_locked"] = True
            ent["fixed_reason"] = ent.get("fixed_reason") or seal_reason
            self._collision_fixed_segments[key] = ent
        self._collision_active_segments.clear()
        for key, ent in list(self._collision_active_points.items()):
            if key in self._collision_fixed_segments:
                continue
            ent = dict(ent)
            ent["coords_locked"] = True
            ent["fixed_reason"] = ent.get("fixed_reason") or seal_reason
            self._collision_fixed_points.setdefault(key, ent)
        self._collision_active_points.clear()
        for key in list(self._collision_fixed_points.keys()):
            if key in self._collision_fixed_segments:
                del self._collision_fixed_points[key]

        # Last resort at π: harvest this-frame flat contacts if tracker is empty
        if (
            seal_reason == "fold_pi"
            and not self._collision_fixed_segments
            and getattr(self, "_collision_contact_segments_flat", None)
        ):
            for s in self._collision_contact_segments_flat:
                try:
                    key = self._contact_pair_key(s)
                except Exception:
                    continue
                if key in self._collision_fixed_segments:
                    continue
                ent = self._copy_segment_entry(s)
                ent["coords_locked"] = True
                ent["fixed_reason"] = "fold_pi"
                self._collision_fixed_segments[key] = ent

        if self._collision_fixed_segments or self._collision_fixed_points:
            self._export_all_fixed_flat_contacts()
        elif seal_reason == "fold_pi":
            # Truly nothing to export this fold — stop thrashing
            self._collision_coords_exported = True
            self._collision_stats = self.build_collision_stats(include_live=False)
            print(
                "[Contact] Seal @π with no fixed contacts "
                f"(GUI segs={getattr(self, '_collision_segment_count', 0)} "
                f"pts={getattr(self, '_collision_contact_count', 0)})."
            )
        # else: leave open so later frames can still accumulate

    def build_collision_stats(self, include_live=False):
        """
        Structured dump of every tracked contact (for JSON / debugging).
        One entry per red triangle–triangle contact the tracker kept.

        **Exported coordinates** for each panel side = the **shaded area**:
        the 2D sweep ribbon of the two-node contact line (design xy).
        Falls back to first/last triangle only when fewer than two samples.
        """
        groups = self.get_collision_groups(include_live=include_live)
        closed_polygons = []
        shaded_areas = []
        segments_out = []
        points_out = []
        n_sweep = 0
        for g in groups:
            for seg in g["segments"]:
                segments_out.append(seg)
                for side_key in ("side_a", "side_b"):
                    side = seg.get(side_key) or {}
                    if "p0" not in side and "first_p0" not in side:
                        continue
                    sweep_samples = self._copy_sweep_samples(
                        side.get("sweep_samples") or []
                    )
                    # Also accept bare [p0,p1] samples if stored as lists
                    if not sweep_samples and side.get("sweep_samples"):
                        sweep_samples = side.get("sweep_samples") or []
                    sweep_poly = self.sweep_paint_polygon_2d(sweep_samples)
                    tri_info = self.closed_triangle_max_area_from_first_last(
                        seg, side_key
                    )
                    poly4 = self.closed_polygon_from_first_last(seg, side_key)

                    # Primary: swept shaded ribbon (matches viz paint)
                    paint_poly = sweep_poly
                    paint_kind = "sweep"
                    paint_area = (
                        self._polygon_area_2d(sweep_poly) if sweep_poly else 0.0
                    )
                    if paint_poly is None and tri_info is not None:
                        paint_poly = tri_info["triangle"]
                        paint_kind = "triangle"
                        paint_area = float(tri_info["area"])
                    if paint_poly is None and poly4 is not None:
                        paint_poly = poly4
                        paint_kind = "quad"
                        paint_area = self._polygon_area_2d(poly4)
                    if paint_poly is None:
                        continue
                    if paint_kind == "sweep":
                        n_sweep += 1

                    # Coordinate list as plain [[x,y], ...] for consumers
                    coordinates = [
                        [float(q[0]), float(q[1])] for q in paint_poly
                    ]

                    entry = {
                        "panel": side.get("panel", seg.get("panel_a")),
                        "unit": side.get("unit"),
                        "side": side_key,
                        "intruder_panel": seg.get(
                            "intruder_panel", g.get("intruder_panel")
                        ),
                        "other_panel": seg.get(
                            "other_panel", g.get("other_panel")
                        ),
                        "unit_a": seg.get("unit_a"),
                        "unit_b": seg.get("unit_b"),
                        "layer_idx": seg.get("layer_idx"),
                        "layer_h": seg.get("layer_h"),
                        "tri_a": seg.get("tri_a"),
                        "tri_b": seg.get("tri_b"),
                        # === primary export: shaded region coordinates ===
                        "coordinates": coordinates,
                        "shaded_polygon": coordinates,
                        "shaded_area": float(paint_area),
                        "shaded_kind": paint_kind,
                        "n_vertices": len(coordinates),
                        # aliases used by visualizer / older consumers
                        "paint_kind": paint_kind,
                        "paint_polygon": coordinates,
                        "paint_area": float(paint_area),
                        "sweep_samples": sweep_samples,
                        "sweep_n_samples": len(sweep_samples),
                        "sweep_polygon": sweep_poly,
                        # Max-area triangle fallback (first edge + better last apex)
                        "closed_triangle": (
                            tri_info["triangle"] if tri_info else None
                        ),
                        "triangle_area": (
                            float(tri_info["area"]) if tri_info else None
                        ),
                        "triangle_last_index": (
                            tri_info["last_index"] if tri_info else None
                        ),
                        "triangle_last_point": (
                            tri_info["last_point"] if tri_info else None
                        ),
                        "first_p0": (
                            tri_info["first_p0"]
                            if tri_info
                            else side.get("first_p0", seg.get("first_p0"))
                        ),
                        "first_p1": (
                            tri_info["first_p1"]
                            if tri_info
                            else side.get("first_p1", seg.get("first_p1"))
                        ),
                        # last two-node line (trim lock / live tip)
                        "last_p0": list(side["p0"]) if "p0" in side else None,
                        "last_p1": list(side["p1"]) if "p1" in side else None,
                        # Legacy 4-gon (first + both last ends)
                        "polygon": poly4,
                        "fixed_reason": seg.get("fixed_reason"),
                        "folding_angle": seg.get("folding_angle"),
                    }
                    closed_polygons.append(entry)
                    shaded_areas.append({
                        "panel": entry["panel"],
                        "unit": entry["unit"],
                        "side": side_key,
                        "layer_idx": entry["layer_idx"],
                        "layer_h": entry["layer_h"],
                        "kind": paint_kind,
                        "area": float(paint_area),
                        "coordinates": coordinates,
                        "n_vertices": len(coordinates),
                        "sweep_n_samples": len(sweep_samples),
                        # Full sample list for complete export / logging
                        "sweep_samples": sweep_samples,
                        "first_p0": entry.get("first_p0"),
                        "first_p1": entry.get("first_p1"),
                        "last_p0": entry.get("last_p0"),
                        "last_p1": entry.get("last_p1"),
                        "fixed_reason": entry.get("fixed_reason"),
                        "folding_angle": entry.get("folding_angle"),
                        "intruder_panel": entry["intruder_panel"],
                        "other_panel": entry["other_panel"],
                        "unit_a": entry.get("unit_a"),
                        "unit_b": entry.get("unit_b"),
                        "tri_a": entry.get("tri_a"),
                        "tri_b": entry.get("tri_b"),
                    })
            for ent in g["points"]:
                points_out.append(ent)
        return {
            "n_groups": len(groups),
            "n_segments": len(segments_out),
            "n_points": len(points_out),
            "n_closed_polygons": len(closed_polygons),
            "n_closed_triangles": len(closed_polygons),
            "n_sweep_paints": n_sweep,
            "n_shaded_areas": len(shaded_areas),
            "groups": groups,
            "segments": segments_out,
            "points": points_out,
            "closed_polygons": closed_polygons,
            # Flat list of shaded regions = primary exported coordinates
            "shaded_areas": shaded_areas,
            "folding_angle": float(getattr(self, "folding_angle", 0.0)),
            # Live frame raw counts (what the GUI draws this frame)
            "gui_contact_points": int(getattr(self, "_collision_contact_count", 0)),
            "gui_contact_segments": int(getattr(self, "_collision_segment_count", 0)),
        }

    @staticmethod
    def _fmt_xy(q, prec=6):
        return f"({float(q[0]):.{prec}f}, {float(q[1]):.{prec}f})"

    @classmethod
    def _log_xy_ring(cls, pts, indent="        ", label="coordinates", prec=6):
        """Print every ring vertex (no truncation). One point per line for clarity."""
        if not pts:
            print(f"{indent}{label}: (empty)")
            return
        print(f"{indent}{label}  n={len(pts)}  (closed ring):")
        for i, q in enumerate(pts):
            print(f"{indent}  [{i:4d}] {cls._fmt_xy(q, prec)}")
        # close marker repeats first vertex for readability
        print(f"{indent}  [close] {cls._fmt_xy(pts[0], prec)}")

    @classmethod
    def _log_sweep_samples(cls, samples, indent="        ", prec=6):
        """Print every two-node sweep sample (no omission)."""
        if not samples:
            print(f"{indent}sweep_samples: (none)")
            return
        print(f"{indent}sweep_samples  n={len(samples)}:")
        for si, s in enumerate(samples):
            if not s or "p0" not in s or "p1" not in s:
                print(f"{indent}  sample[{si}]: (invalid)")
                continue
            a, b = s["p0"], s["p1"]
            ang = float(s.get("angle", 0.0))
            print(
                f"{indent}  sample[{si:4d}] @θ={ang:.6f}: "
                f"{cls._fmt_xy(a, prec)} -- {cls._fmt_xy(b, prec)}"
            )

    def _export_all_fixed_flat_contacts(self):
        """
        Complete log of locked flat trim coords.

        Primary payload = shaded sweep polygon coordinates (every vertex) and
        every sweep sample stroke. Nothing is truncated or omitted.
        Also writes a full text dump next to descriptionData for offline review.
        """
        groups = self.get_collision_groups(include_live=False)
        n_segs = sum(len(g["segments"]) for g in groups)
        n_pts = sum(len(g["points"]) for g in groups)
        stats = self.build_collision_stats(include_live=False)
        self._collision_stats = stats
        self._collision_coords_exported = True
        n_unit_pairs = len({
            (int(s["unit_a"]), int(s["unit_b"]), int(s["layer_idx"]))
            for s in stats.get("segments", [])
        })
        n_sweep = int(stats.get("n_sweep_paints", 0) or 0)
        n_paint = int(stats.get("n_closed_polygons", 0) or 0)
        n_shaded = int(stats.get("n_shaded_areas", n_paint) or 0)

        lines_out = []

        def _p(msg=""):
            print(msg)
            lines_out.append(str(msg))

        _p(
            f"[Contact] Exported SHADED areas (sweep ribbons) as coordinates: "
            f"fold angle={float(self.folding_angle):.6f}, "
            f"groups={len(groups)}, lines={n_segs}, lone_pts={n_pts}, "
            f"unit_pairs={n_unit_pairs}, shaded={n_shaded} "
            f"(sweep={n_sweep}); "
            f"GUI last frame segs={self._collision_segment_count} "
            f"pts={self._collision_contact_count}"
        )
        _p(
            f"[Contact] Full dump: every shaded vertex + every sweep sample "
            f"(no truncation)."
        )

        # ---- 1) Every shaded region (primary export) ----
        for si, sh in enumerate(stats.get("shaded_areas") or []):
            coords = sh.get("coordinates") or []
            samples = sh.get("sweep_samples") or []
            _p(
                f"  shaded[{si}] panel={sh.get('panel')} unit={sh.get('unit')} "
                f"side={sh.get('side')} L={sh.get('layer_idx')} "
                f"h={sh.get('layer_h')!s} kind={sh.get('kind')} "
                f"A={float(sh.get('area', 0)):.6f} "
                f"n_verts={len(coords)} n_samples={len(samples)} "
                f"units={sh.get('unit_a')}-{sh.get('unit_b')} "
                f"tris={sh.get('tri_a')}-{sh.get('tri_b')} "
                f"intr={sh.get('intruder_panel')} host={sh.get('other_panel')} "
                f"lock={sh.get('fixed_reason')} @θ={sh.get('folding_angle')}"
            )
            if sh.get("first_p0") is not None and sh.get("first_p1") is not None:
                _p(
                    f"    first: {self._fmt_xy(sh['first_p0'])} -- "
                    f"{self._fmt_xy(sh['first_p1'])}"
                )
            if sh.get("last_p0") is not None and sh.get("last_p1") is not None:
                _p(
                    f"    last:  {self._fmt_xy(sh['last_p0'])} -- "
                    f"{self._fmt_xy(sh['last_p1'])}"
                )
            # Capture ring + samples into lines_out via temporary print capture
            # Use direct helpers that also append
            if coords:
                _p(f"    coordinates  n={len(coords)}  (closed ring):")
                for i, q in enumerate(coords):
                    _p(f"      [{i:4d}] {self._fmt_xy(q)}")
                _p(f"      [close] {self._fmt_xy(coords[0])}")
            else:
                _p("    coordinates: (empty)")
            if samples:
                _p(f"    sweep_samples  n={len(samples)}:")
                for j, s in enumerate(samples):
                    if not s or "p0" not in s or "p1" not in s:
                        _p(f"      sample[{j:4d}]: (invalid)")
                        continue
                    _p(
                        f"      sample[{j:4d}] @θ={float(s.get('angle', 0)):.6f}: "
                        f"{self._fmt_xy(s['p0'])} -- {self._fmt_xy(s['p1'])}"
                    )
            else:
                _p("    sweep_samples: (none)")

        # ---- 2) Per-group / per-line detail (complete) ----
        for gi, g in enumerate(groups):
            intr_p = g.get("intruder_panel")
            oth_p = g.get("other_panel")
            if intr_p is not None and oth_p is not None:
                who = (
                    f"INTRUDER panel {intr_p}  into  host panel {oth_p}  "
                    f"(units {g.get('intruder_sim_unit')}→{g.get('other_sim_unit')})"
                )
            else:
                who = "INTRUDER: unknown (no classification)"
            _p(
                f"  === group {gi}: panels {g['panel_a']}-{g['panel_b']} | {who} "
                f"({len(g['segments'])} lines, {len(g['points'])} pts) ==="
            )
            for i, seg in enumerate(g["segments"]):
                s_intr = seg.get("intruder_panel", intr_p)
                s_oth = seg.get("other_panel", oth_p)
                reason = seg.get("fixed_reason") or "?"
                tri_tag = ""
                if seg.get("tri_a") is not None and seg.get("tri_b") is not None:
                    tri_tag = f" tris={seg['tri_a']}-{seg['tri_b']}"
                _p(
                    f"    line {i}  INTRUDER panel {s_intr} / host panel {s_oth}  "
                    f"units {seg['unit_a']}-{seg['unit_b']} "
                    f"L{seg['layer_idx']} h={seg['layer_h']:+.6g}{tri_tag}  "
                    f"lock={reason} @θ={float(seg.get('folding_angle', 0)):.6f}"
                )
                if seg.get("d_AtoB") is not None:
                    _p(
                        f"      dAB={seg['d_AtoB']:+.6f} dBA={seg['d_BtoA']:+.6f}  "
                        f"intr_unit={seg.get('intruder_sim_unit')} "
                        f"host_unit={seg.get('other_sim_unit')}"
                    )
                if seg.get("p0_3d") is not None and seg.get("p1_3d") is not None:
                    a3, b3 = seg["p0_3d"], seg["p1_3d"]
                    _p(
                        f"      3d (GUI red segment): "
                        f"({float(a3[0]):.6f},{float(a3[1]):.6f},{float(a3[2]):.6f}) -- "
                        f"({float(b3[0]):.6f},{float(b3[1]):.6f},{float(b3[2]):.6f})"
                    )
                n_sides_printed = 0
                for side_key, label in (("side_a", "panel-A"), ("side_b", "panel-B")):
                    side = seg.get(side_key) or {}
                    if "p0" not in side:
                        continue
                    n_sides_printed += 1
                    p0, p1 = side["p0"], side["p1"]
                    f0 = side.get("first_p0", p0)
                    f1 = side.get("first_p1", p1)
                    role = ""
                    if side.get("panel") == s_intr:
                        role = " [INTRUDER]"
                    elif side.get("panel") == s_oth:
                        role = " [HOST]"
                    _p(
                        f"      {label} p{side.get('panel')} u{side.get('unit')}{role}"
                    )
                    kp0 = seg.get("first_kp0")
                    kp1 = seg.get("first_kp1")
                    kp_tag = f" kps={kp0}-{kp1}" if kp0 is not None else ""
                    _p(
                        f"        first(JSON crease verts){kp_tag} @θ="
                        f"{float(side.get('first_folding_angle', seg.get('first_folding_angle', 0))):.6f}: "
                        f"{self._fmt_xy(f0)} -- {self._fmt_xy(f1)}"
                    )
                    _p(
                        f"        last (trim lock) @θ={float(seg.get('folding_angle', 0)):.6f}: "
                        f"{self._fmt_xy(p0)} -- {self._fmt_xy(p1)}"
                    )
                    samples = side.get("sweep_samples") or []
                    shaded = self.sweep_paint_polygon_2d(samples)
                    n_samp = len(samples)
                    if shaded is not None:
                        area = self._polygon_area_2d(shaded)
                        _p(
                            f"        SHADED area (export coords, kind=sweep, "
                            f"n_samples={n_samp}, n_verts={len(shaded)}, "
                            f"A={area:.6f}):"
                        )
                        _p(f"        coordinates  n={len(shaded)}  (closed ring):")
                        for vi, q in enumerate(shaded):
                            _p(f"          [{vi:4d}] {self._fmt_xy(q)}")
                        _p(f"          [close] {self._fmt_xy(shaded[0])}")
                    else:
                        tri_info = self.closed_triangle_max_area_from_first_last(
                            seg, side_key
                        )
                        if tri_info is not None:
                            tri = tri_info["triangle"]
                            _p(
                                f"        SHADED area (fallback tri, "
                                f"n_samples={n_samp}, A={float(tri_info['area']):.6f}):"
                            )
                            _p(f"        coordinates  n={len(tri)}  (closed ring):")
                            for vi, q in enumerate(tri):
                                _p(f"          [{vi:4d}] {self._fmt_xy(q)}")
                            _p(f"          [close] {self._fmt_xy(tri[0])}")
                        else:
                            _p(
                                f"        SHADED area: none "
                                f"(n_samples={n_samp})"
                            )
                    # Always log full sample list (never omit)
                    if samples:
                        _p(f"        sweep_samples  n={n_samp}:")
                        for j, s in enumerate(samples):
                            if not s or "p0" not in s or "p1" not in s:
                                _p(f"          sample[{j:4d}]: (invalid)")
                                continue
                            _p(
                                f"          sample[{j:4d}] @θ="
                                f"{float(s.get('angle', 0)):.6f}: "
                                f"{self._fmt_xy(s['p0'])} -- {self._fmt_xy(s['p1'])}"
                            )
                    else:
                        _p("        sweep_samples: (none)")
                if n_sides_printed == 0:
                    _p("      (no flat map for this segment — 3d only)")
            for i, ent in enumerate(g["points"]):
                _p(
                    f"    pt {i}  INTRUDER panel {ent.get('intruder_panel', '?')} / "
                    f"host panel {ent.get('other_panel', '?')}  "
                    f"units {ent['unit_a']}-{ent['unit_b']} "
                    f"L{ent['layer_idx']} h={ent['layer_h']:+.6g}"
                )
                for side_key, label in (("side_a", "panel-A"), ("side_b", "panel-B")):
                    side = ent.get(side_key) or {}
                    if "p" not in side:
                        continue
                    p = side["p"]
                    _p(
                        f"      {label} p{side.get('panel')} u{side.get('unit')}: "
                        f"{self._fmt_xy(p)}"
                    )

        # ---- 3) Write complete log file (console may scroll away) ----
        try:
            out_dir = os.path.join(".", "descriptionData")
            os.makedirs(out_dir, exist_ok=True)
            name = getattr(self, "origami_name", "origami") or "origami"
            log_path = os.path.join(out_dir, f"{name}-contact-export.log")
            with open(log_path, "w", encoding="utf-8") as fw:
                fw.write("\n".join(lines_out))
                fw.write("\n")
            _p(f"[Contact] Complete log written to {log_path}")
            # Also dump structured JSON of shaded areas (full coordinates + samples)
            json_path = os.path.join(out_dir, f"{name}-shaded-areas.json")
            payload = {
                "origami_name": name,
                "folding_angle": float(self.folding_angle),
                "n_shaded_areas": n_shaded,
                "n_segments": n_segs,
                "n_points": n_pts,
                "shaded_areas": stats.get("shaded_areas") or [],
                "closed_polygons": stats.get("closed_polygons") or [],
            }
            with open(json_path, "w", encoding="utf-8") as fw:
                json.dump(payload, fw, indent=2)
            _p(f"[Contact] Shaded-area JSON written to {json_path}")
        except Exception as exc:
            _p(f"[Contact] Failed to write export log/json: {exc}")

    def get_fixed_flat_segments(self):
        """All finalized segment cut lines (sorted by panel group, layer, units)."""
        items = list(self._collision_fixed_segments.values())
        items.sort(key=lambda s: (
            s.get("panel_a", -1), s.get("panel_b", -1),
            s["layer_idx"], s["unit_a"], s["unit_b"],
        ))
        return items

    def get_fixed_flat_points(self):
        """All finalized lone contact points (no segment for that pair)."""
        items = list(self._collision_fixed_points.values())
        items.sort(key=lambda e: (
            e.get("panel_a", -1), e.get("panel_b", -1),
            e["layer_idx"], e["unit_a"], e["unit_b"],
        ))
        return items

    @staticmethod
    def _panel_group_key(entry):
        """Two JSON panels that collide = one group."""
        pa = int(entry.get("panel_a", -1))
        pb = int(entry.get("panel_b", -1))
        return (min(pa, pb), max(pa, pb))

    def get_collision_groups(self, include_live=True):
        """
        Group collision entries by the two JSON panels that collide.

        Returns list of:
          {
            "panel_a", "panel_b",          # sorted panel indices (group id)
            "intruder_panel", "other_panel",
            "intruder_sim_unit", "other_sim_unit",
            "segments": [...],             # locked flat cut lines (off-crease)
            "points": [...],               # lone points
            "status": "done"|"live"|"mixed"
          }
        """
        buckets = {}

        def _ensure(pa, pb):
            key = (pa, pb)
            if key not in buckets:
                buckets[key] = {
                    "panel_a": pa,
                    "panel_b": pb,
                    "segments": [],
                    "points": [],
                    "_has_done": False,
                    "_has_live": False,
                    "_intr_votes": {},  # panel -> count
                    "_host_votes": {},
                    "_intr_u_votes": {},
                    "_host_u_votes": {},
                }
            return buckets[key]

        def _vote(g, entry):
            ip = entry.get("intruder_panel")
            op = entry.get("other_panel")
            iu = entry.get("intruder_sim_unit")
            ou = entry.get("other_sim_unit")
            if ip is not None and ip >= 0:
                g["_intr_votes"][ip] = g["_intr_votes"].get(ip, 0) + 1
            if op is not None and op >= 0:
                g["_host_votes"][op] = g["_host_votes"].get(op, 0) + 1
            if iu is not None:
                g["_intr_u_votes"][iu] = g["_intr_u_votes"].get(iu, 0) + 1
            if ou is not None:
                g["_host_u_votes"][ou] = g["_host_u_votes"].get(ou, 0) + 1

        for seg in self.get_fixed_flat_segments():
            pa, pb = self._panel_group_key(seg)
            g = _ensure(pa, pb)
            g["segments"].append(seg)
            g["_has_done"] = True
            _vote(g, seg)
        for ent in self.get_fixed_flat_points():
            pa, pb = self._panel_group_key(ent)
            g = _ensure(pa, pb)
            g["points"].append(ent)
            g["_has_done"] = True
            _vote(g, ent)

        if include_live:
            for seg in self._collision_active_segments.values():
                pa, pb = self._panel_group_key(seg)
                g = _ensure(pa, pb)
                g["segments"].append(seg)
                g["_has_live"] = True
                _vote(g, seg)
            for ent in self._collision_active_points.values():
                pa, pb = self._panel_group_key(ent)
                g = _ensure(pa, pb)
                g["points"].append(ent)
                g["_has_live"] = True
                _vote(g, ent)

        def _winner(votes):
            if not votes:
                return None
            return max(votes.items(), key=lambda kv: kv[1])[0]

        groups = []
        for key in sorted(buckets.keys()):
            g = buckets[key]
            if g["_has_done"] and g["_has_live"]:
                g["status"] = "mixed"
            elif g["_has_live"]:
                g["status"] = "live"
            else:
                g["status"] = "done"
            g["intruder_panel"] = _winner(g["_intr_votes"])
            g["other_panel"] = _winner(g["_host_votes"])
            g["intruder_sim_unit"] = _winner(g["_intr_u_votes"])
            g["other_sim_unit"] = _winner(g["_host_u_votes"])
            # Sort contacts inside group by layer, units, triangle pair
            g["segments"].sort(key=lambda s: (
                s["layer_idx"], s["unit_a"], s["unit_b"],
                int(s.get("tri_a", -1)), int(s.get("tri_b", -1)),
            ))
            g["points"].sort(key=lambda e: (
                e["layer_idx"], e["unit_a"], e["unit_b"],
                int(e.get("tri_a", -1)), int(e.get("tri_b", -1)),
            ))
            for k in ("_has_done", "_has_live", "_intr_votes", "_host_votes",
                      "_intr_u_votes", "_host_u_votes"):
                del g[k]
            groups.append(g)
        return groups

    # -----------------------------------------------------------------
    # Design-space paint canvas → 3D panel (affine paper map, not mesh tris)
    #
    # Model:
    #   • paper is a canvas in flat design xy
    #   • contact line stamps sparse strokes onto that canvas
    #   • every frame, strokes are reprojected onto the *current* panel
    #     pose with a least-squares affine map from panel keypoints
    #     (planar paper embedding — independent of triangulation density)
    # -----------------------------------------------------------------

    def _iter_live_sweep_sides(self):
        """Yield (side_dict, is_intruder, ent) for active + fixed contacts."""
        pairs = []
        for ent in (getattr(self, "_collision_active_segments", None) or {}).values():
            pairs.append(ent)
        for ent in (getattr(self, "_collision_fixed_segments", None) or {}).values():
            pairs.append(ent)
        seen = set()
        for ent in pairs:
            key = self._contact_pair_key(ent)
            if key in seen:
                continue
            seen.add(key)
            ip = ent.get("intruder_panel")
            for sk in ("side_a", "side_b"):
                side = ent.get(sk)
                if not side or "p0" not in side:
                    continue
                is_intr = (
                    ip is not None
                    and side.get("panel") is not None
                    and int(side["panel"]) == int(ip)
                )
                yield side, is_intr, ent

    def _paint_min_move_dist(self):
        # Match export sweep density so the on-panel locus is not sparse
        if self._paint_min_move is None:
            self._paint_min_move = max(
                float(getattr(self, "max_size", 100.0)) * 3e-4, 0.012
            )
        return float(self._paint_min_move)

    def _unit_mesh_tris_cache(self, positions, flat_kps):
        """
        Per-unit list of (i0,i1,i2, flat_tri[3,2]) for material mapping.
        Built once per paint rebuild.
        """
        cache = {}
        if not hasattr(self, "ori_sim"):
            return cache
        refs = self.ori_sim.tri_indices_ref
        tri_all = self.ori_sim.tri_indices
        n_tris = len(tri_all) // 3
        for uid, t0 in enumerate(refs):
            t1 = int(refs[uid + 1]) if uid + 1 < len(refs) else n_tris
            tris = []
            for ti in range(int(t0), t1):
                base = 3 * ti
                i0 = int(tri_all[base])
                i1 = int(tri_all[base + 1])
                i2 = int(tri_all[base + 2])
                if max(i0, i1, i2) >= len(positions) or max(i0, i1, i2) >= len(flat_kps):
                    continue
                vf = np.asarray(
                    [flat_kps[i0][:2], flat_kps[i1][:2], flat_kps[i2][:2]],
                    dtype=float,
                )
                tris.append((i0, i1, i2, vf))
            cache[uid] = tris
        return cache

    def _locate_xy_on_unit(self, xy, unit_tris, eps=0.02):
        """
        Design xy → (i0,i1,i2,u,v,w) on that unit's mesh.
        Material barycentric so eval on current x sticks to the panel surface.
        """
        if not unit_tris:
            return None
        p = np.asarray(xy, dtype=float).reshape(-1)[:2]
        best = None
        best_pen = float("inf")
        for i0, i1, i2, vf in unit_tris:
            bary = _barycentric_2d(p, vf[0], vf[1], vf[2])
            if bary is None:
                continue
            u, v, w = bary
            # inside (slightly relaxed)
            if u >= -eps and v >= -eps and w >= -eps:
                return (i0, i1, i2, u, v, w)
            # how far outside
            pen = max(0.0, -u) + max(0.0, -v) + max(0.0, -w)
            if pen < best_pen:
                best_pen = pen
                # clamp to triangle
                u2, v2, w2 = max(u, 0.0), max(v, 0.0), max(w, 0.0)
                s = u2 + v2 + w2
                if s < 1e-14:
                    continue
                best = (i0, i1, i2, u2 / s, v2 / s, w2 / s)
        return best

    @staticmethod
    def _eval_bary_on_positions(loc, positions, lift=0.0):
        """Evaluate material barycentric on current vertex positions (+ tiny lift)."""
        if loc is None:
            return None
        i0, i1, i2, u, v, w = loc
        a = np.asarray(positions[i0], dtype=float)[:3]
        b = np.asarray(positions[i1], dtype=float)[:3]
        c = np.asarray(positions[i2], dtype=float)[:3]
        p = u * a + v * b + w * c
        if lift != 0.0:
            n = np.cross(b - a, c - a)
            nn = float(np.linalg.norm(n))
            if nn > 1e-12:
                p = p + (float(lift) / nn) * n
        if not np.all(np.isfinite(p)):
            return None
        return [float(p[0]), float(p[1]), float(p[2])]

    @staticmethod
    def _eval_bary_batch(locs, positions, lift=0.0):
        """
        Vectorized bary eval: locs is list/array of (i0,i1,i2,u,v,w).
        Returns (N, 3) float64; rows with bad locs are NaN.
        """
        n = len(locs)
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)
        L = np.asarray(locs, dtype=np.float64).reshape(n, -1)
        i0 = L[:, 0].astype(np.intp)
        i1 = L[:, 1].astype(np.intp)
        i2 = L[:, 2].astype(np.intp)
        u = L[:, 3:4]
        v = L[:, 4:5]
        w = L[:, 5:6]
        pos = np.asarray(positions)
        a = pos[i0, :3]
        b = pos[i1, :3]
        c = pos[i2, :3]
        p = u * a + v * b + w * c
        if lift != 0.0:
            nrm = np.cross(b - a, c - a)
            nn = np.linalg.norm(nrm, axis=1, keepdims=True)
            nn = np.maximum(nn, 1e-12)
            p = p + (float(lift) / nn) * nrm
        return p

    def _stamp_paint_canvas_from_contacts(self):
        """
        Record locus using **barycentric locs from the collision hit triangle**
        (attached in detect_panel_collisions). No re-locate — those locs are
        already on the panel mesh; redraw just re-evaluates them on current x.
        """
        if not getattr(self, "collision_shading", False):
            return
        # Fold sealed at π → no more locus stamps
        if getattr(self, "_sweep_draw_stopped", False) or getattr(
            self, "_collision_coords_exported", False
        ):
            return
        if not hasattr(self, "_paint_canvas") or self._paint_canvas is None:
            self._paint_canvas = {}

        active = getattr(self, "_collision_active_segments", None) or {}
        if not active:
            return

        min_move = self._paint_min_move_dist()
        dirty = False
        copy_loc = self._copy_loc
        order2 = self._order_segment_endpoints_2d
        mid_dist = self._segment_mid_dist_2d

        for ent in active.values():
            # Stamp BOTH sides (each panel gets its own material locus)
            for sk in ("side_a", "side_b"):
                side = ent.get(sk)
                if not side or "p0" not in side:
                    continue
                loc0 = copy_loc(side.get("loc0"))
                loc1 = copy_loc(side.get("loc1"))
                if loc0 is None or loc1 is None:
                    continue
                uid = side.get("unit")
                if uid is None:
                    continue
                uid = int(uid)
                p0 = [float(side["p0"][0]), float(side["p0"][1])]
                p1 = [float(side["p1"][0]), float(side["p1"][1])]

                trail = self._paint_canvas.setdefault(uid, [])
                if trail:
                    last = trail[-1]
                    p0o, p1o = order2(p0, p1, last["p0"], last["p1"])
                    # If endpoints swapped for trail order, swap locs too
                    if p0o[0] != p0[0] or p0o[1] != p0[1]:
                        loc0, loc1 = loc1, loc0
                    p0, p1 = p0o, p1o
                    moved = mid_dist(last["p0"], last["p1"], p0, p1)
                    if moved < min_move:
                        # refresh tip locs so last sample tracks live contact
                        trail[-1] = {
                            "p0": p0, "p1": p1, "loc0": loc0, "loc1": loc1,
                        }
                        dirty = True
                        continue

                trail.append({
                    "p0": p0, "p1": p1,
                    "loc0": loc0, "loc1": loc1,
                })
                dirty = True
                # No max sample cap — keep full trail
        if dirty:
            self._paint_dirty = True

    def _ensure_paint_line_capacity(self, n_verts):
        """Grow Taichi paint line field if needed (no hard cap on stroke count)."""
        n_verts = int(max(0, n_verts))
        cur = int(getattr(self, "_paint_max_line_verts", 0) or 0)
        if hasattr(self, "paint_line_vertices") and n_verts <= cur and cur > 0:
            return
        need = max(n_verts, 8192)
        if cur > 0:
            while need < n_verts:
                need *= 2
            if need < cur * 2 and n_verts > cur:
                need = cur * 2
        # round up to multiple of 1024 for fewer reallocs
        need = max(need, n_verts)
        need = ((need + 1023) // 1024) * 1024
        self._paint_max_line_verts = need
        self.paint_line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=need)

    def _update_surface_sweep_paint(self):
        """
        Draw locus glued to panels: evaluate stored barycentrics on current x.
        Paths move with the paper — not free-floating 3D polylines.

        Fast path: when sim is paused and the canvas was not restamped, reuse
        the last GPU line buffer (camera-only motion still works).
        """
        if not hasattr(self, "paint_line_vertices") and not getattr(
            self, "collision_shading", False
        ):
            self._paint_line_vert_count = 0
            self._paint_n_strokes = 0
            return

        # At π: keep the last uploaded strokes; do not grow or reproject
        if getattr(self, "_sweep_draw_stopped", False):
            return

        canvas = getattr(self, "_paint_canvas", None) or {}
        if not canvas:
            if self._paint_line_vert_count != 0:
                self._paint_line_vert_count = 0
                self._paint_n_strokes = 0
            return

        # Reuse previous line verts when mesh is frozen and canvas unchanged
        paused = bool(getattr(self, "paused", False)) and not bool(
            getattr(self, "step_once", False)
        )
        if (
            paused
            and not getattr(self, "_paint_dirty", False)
            and int(getattr(self, "_paint_line_vert_count", 0) or 0) >= 2
        ):
            return

        positions = self._cache_frame_positions()
        # tiny offset so lines win depth buffer without looking off-surface
        lift = max(float(getattr(self, "max_size", 100.0)) * 5e-5, 0.005)

        # Batch all bary evals, then pack line segments
        all_loc0 = []
        all_loc1 = []
        unit_ranges = []  # (start, count) into all_loc*
        for _uid, strokes in canvas.items():
            if not strokes:
                continue
            start = len(all_loc0)
            for s in strokes:
                loc0 = s.get("loc0")
                loc1 = s.get("loc1")
                if loc0 is None or loc1 is None:
                    continue
                all_loc0.append(loc0)
                all_loc1.append(loc1)
            n = len(all_loc0) - start
            if n > 0:
                unit_ranges.append((start, n))

        if not all_loc0:
            self._paint_line_vert_count = 0
            self._paint_n_strokes = 0
            self._paint_dirty = False
            return

        pts0_all = self._eval_bary_batch(all_loc0, positions, lift=lift)
        pts1_all = self._eval_bary_batch(all_loc1, positions, lift=lift)

        # Upper bound: 2 chains of (n-1) segs + n cross bars → ~ (4n-2) verts/unit
        n_total = len(all_loc0)
        est_verts = max(4 * n_total * 2, 64)
        self._ensure_paint_line_capacity(est_verts)
        max_lv = int(self._paint_max_line_verts)
        ln_buf = np.zeros((max_lv, 3), dtype=np.float32)
        li = 0

        for start, n in unit_ranges:
            pts0 = pts0_all[start : start + n]
            pts1 = pts1_all[start : start + n]
            # drop non-finite rows
            good0 = np.isfinite(pts0).all(axis=1)
            good1 = np.isfinite(pts1).all(axis=1)
            # locus of each contact node (on surface)
            for pts, good in ((pts0, good0), (pts1, good1)):
                prev = None
                for i in range(n):
                    if not good[i]:
                        prev = None
                        continue
                    cur = pts[i]
                    if prev is not None:
                        if li + 2 > max_lv:
                            self._ensure_paint_line_capacity(li + 2 + 4096)
                            max_lv = int(self._paint_max_line_verts)
                            new_buf = np.zeros((max_lv, 3), dtype=np.float32)
                            new_buf[:li] = ln_buf[:li]
                            ln_buf = new_buf
                        ln_buf[li] = prev
                        li += 1
                        ln_buf[li] = cur
                        li += 1
                    prev = cur
            # Contact line samples on the panel (every sample)
            for i in range(n):
                if not (good0[i] and good1[i]):
                    continue
                if li + 2 > max_lv:
                    self._ensure_paint_line_capacity(li + 2 + 4096)
                    max_lv = int(self._paint_max_line_verts)
                    new_buf = np.zeros((max_lv, 3), dtype=np.float32)
                    new_buf[:li] = ln_buf[:li]
                    ln_buf = new_buf
                ln_buf[li] = pts0[i]
                li += 1
                ln_buf[li] = pts1[i]
                li += 1

        self._paint_line_vert_count = li
        self._paint_n_strokes = li // 2
        self._paint_dirty = False
        if li > 0:
            self._ensure_paint_line_capacity(li)
            # from_numpy needs full field shape
            max_lv = int(self._paint_max_line_verts)
            if ln_buf.shape[0] != max_lv:
                full = np.zeros((max_lv, 3), dtype=np.float32)
                full[:li] = ln_buf[:li]
                ln_buf = full
            self.paint_line_vertices.from_numpy(ln_buf)

    def _build_collision_debug_lines(self):
        """Build debug text lines (can be cached; window must still draw every frame)."""
        groups = self.get_collision_groups(include_live=True)
        n_fixed = len(self._collision_fixed_segments) + len(self._collision_fixed_points)
        n_live = len(self._collision_active_segments) + len(self._collision_active_points)
        n_intr = len(self._intruder_classifications)
        n_paint = int(sum(len(v) for v in (getattr(self, "_paint_canvas", None) or {}).values()))
        n_strokes = int(getattr(self, "_paint_n_strokes", 0) or 0)
        lines = [
            f"GUI red: {self._collision_contact_count} pts, "
            f"{self._collision_segment_count} segs",
            f"Locus: {n_paint} samples → {n_strokes} segs "
            f"(hit-tri bary → panel)",
            f"Tracked: {n_fixed} fixed + {n_live} live "
            f"(all tri-pairs) | groups={len(groups)}"
            + ("  [SEALED@π]" if self._collision_coords_exported else ""),
            "Paper canvas stamps → affine map onto 3D panel (not mesh tris)",
        ]
        shown_groups = 0
        max_groups = 5
        for g in groups:
            if shown_groups >= max_groups:
                break
            ip, op = g.get("intruder_panel"), g.get("other_panel")
            if ip is not None and op is not None:
                lines.append(
                    f"-- panels {g['panel_a']}-{g['panel_b']} [{g['status']}]"
                )
                lines.append(f"   INTRUDER p{ip} into HOST p{op}")
            else:
                lines.append(
                    f"-- panels {g['panel_a']}-{g['panel_b']} [{g['status']}] "
                    f"INTRUDER=?"
                )
            for seg in g["segments"][:1]:
                key = self._contact_pair_key(seg)
                tag = "live" if key in self._collision_active_segments else "done"
                reason = seg.get("fixed_reason") or ""
                lock_tag = f" {reason}" if reason else ""
                lines.append(
                    f"  [{tag}{lock_tag}] L{seg['layer_idx']} "
                    f"h={seg['layer_h']:+.4g} "
                    f"u{seg['unit_a']}-{seg['unit_b']}"
                )
                for side_key in ("side_a", "side_b"):
                    side = seg.get(side_key) or {}
                    if "p0" not in side:
                        continue
                    p0, p1 = side["p0"], side["p1"]
                    f0 = side.get("first_p0", p0)
                    f1 = side.get("first_p1", p1)
                    role = "INTR" if side.get("panel") == ip else (
                        "HOST" if side.get("panel") == op else "?"
                    )
                    lines.append(f"   p{side.get('panel')} [{role}]:")
                    lines.append(
                        f"    1st JSON ({f0[0]:.1f},{f0[1]:.1f})--"
                        f"({f1[0]:.1f},{f1[1]:.1f})"
                    )
                    lines.append(
                        f"    last ({p0[0]:.1f},{p0[1]:.1f})--"
                        f"({p1[0]:.1f},{p1[1]:.1f})"
                    )
            extra_segs = len(g["segments"]) - min(1, len(g["segments"]))
            if extra_segs > 0:
                lines.append(f"  ... +{extra_segs} more lines (console)")
            shown_groups += 1
        if len(groups) > max_groups:
            lines.append(f"... +{len(groups) - max_groups} more groups (console)")

        lines.append(f"Live intruders: {n_intr}")
        for cls in self._intruder_classifications[:4]:
            lines.append(
                f"  INTR u{cls['intruder_sim_unit']} -> "
                f"HOST u{cls['other_sim_unit']}  "
                f"(dAB={cls['d_AtoB']:+.3f})"
            )
        return lines

    def _render_collision_debug_window(self):
        """
        Separate ImGui sub-window for collision / cut-node debug text.

        The sub_window is drawn *every* frame (skipping frames causes flicker).
        Text content is rebuilt only when collision/paint state changes.
        """
        cache_key = (
            int(getattr(self, "_collision_frame_i", 0)),
            int(self._collision_contact_count),
            int(self._collision_segment_count),
            int(getattr(self, "_paint_n_strokes", 0) or 0),
            len(self._collision_active_segments),
            len(self._collision_fixed_segments),
            bool(self._collision_coords_exported),
            len(self._intruder_classifications),
        )
        if (
            getattr(self, "_debug_lines_key", None) != cache_key
            or not getattr(self, "_debug_lines", None)
        ):
            self._debug_lines = self._build_collision_debug_lines()
            self._debug_lines_key = cache_key

        # Right side of the main window (normalized 0–1 coords)
        with self.gui.sub_window(
            "Collision debug", x=0.62, y=0.02, width=0.36, height=0.70
        ) as dbg:
            for line in self._debug_lines:
                dbg.text(line)

    def _update_intruder_classification(self, positions=None, colliding_unit_pairs=None):
        """
        Intruder classification + orange paint — collision-gated.

        Only runs when ``colliding_unit_pairs`` is non-empty (unit pairs that
        already produced red contact markers in ``detect_panel_collisions``).
        Empty / None → clear classifications and orange mesh pass.
        Paints exactly one intruder per colliding pair (never both).
        """
        if not hasattr(self, "ori_sim"):
            self._intruder_classifications = []
            self._intruder_by_unit_pair = {}
            self._upload_intruder_panel_indices(set())
            return
        if positions is None:
            positions = self._cache_frame_positions()
        # Hard gate: no red-line collision pairs → no orange panels
        if not colliding_unit_pairs:
            self._intruder_classifications = []
            self._intruder_by_unit_pair = {}
            self._upload_intruder_panel_indices(set())
            return
        unit_layer_meta = getattr(self, "_collision_unit_layer_meta", None)
        if not unit_layer_meta:
            unit_layer_meta = self._build_unit_layer_meta()
            self._collision_unit_layer_meta = unit_layer_meta
        try:
            self._intruder_classifications = classify_colliding_panel_pairs(
                positions,
                self.ori_sim.indices,
                colliding_unit_pairs,
                unit_layer_meta=unit_layer_meta,
            )
        except Exception as exc:
            if self.verbose:
                print(f"[Intruder] classification failed: {exc}")
            self._intruder_classifications = []

        # Fast lookup: (min_unit, max_unit) -> classification (for stamping contacts)
        pair_map = {}
        for cls in self._intruder_classifications:
            ua = int(cls["sim_unit_A"])
            ub = int(cls["sim_unit_B"])
            pair_map[(min(ua, ub), max(ua, ub))] = cls
        self._intruder_by_unit_pair = pair_map

        # Classification runs after contact tracking each frame — re-stamp actives now
        for ent in self._collision_active_segments.values():
            self._stamp_intruder_on_entry(ent)
        for ent in self._collision_active_points.values():
            self._stamp_intruder_on_entry(ent)

        # Exactly one intruder per colliding pair (union if multiple pairs)
        intruder_units = {
            cls["intruder_sim_unit"] for cls in self._intruder_classifications
        }
        self._upload_intruder_panel_indices(intruder_units)

    def _upload_intruder_panel_indices(self, intruder_units):
        """Fill self.intruder_indices with all triangles of intruder sim units."""
        if not hasattr(self, "intruder_indices"):
            self._intruder_indice_count = 0
            return
        if not intruder_units or not hasattr(self, "ori_sim"):
            self._intruder_indice_count = 0
            self.intruder_indices.fill(0)
            return

        refs = self.ori_sim.tri_indices_ref
        tri_indices = self.ori_sim.tri_indices
        num_tris = len(tri_indices) // 3
        parts = []
        for uid in sorted(intruder_units):
            if uid < 0 or uid >= len(refs):
                continue
            tri_start = refs[uid]
            tri_end = refs[uid + 1] if uid + 1 < len(refs) else num_tris
            if tri_end > tri_start:
                parts.append(tri_indices[3 * tri_start: 3 * tri_end])

        if not parts:
            self._intruder_indice_count = 0
            self.intruder_indices.fill(0)
            return

        flat = np.concatenate(parts).astype(np.int32)
        max_n = self.intruder_indices.shape[0]
        n = min(len(flat), max_n)
        buf = np.zeros(max_n, dtype=np.int32)
        buf[:n] = flat[:n]
        self.intruder_indices.from_numpy(buf)
        self._intruder_indice_count = n

    def _collect_colliding_unit_pairs_for_trim(self):
        """
        Union of unit pairs that produced real triangle contacts during the run
        (GUI red markers) — used to drive trim cuts beyond fold-adjacent only.
        """
        pairs = set()

        def _add(ua, ub):
            try:
                ua, ub = int(ua), int(ub)
            except (TypeError, ValueError):
                return
            if ua == ub or ua < 0 or ub < 0:
                return
            pairs.add((min(ua, ub), max(ua, ub)))

        # Live / last frame flat contacts
        for lst_name in (
            "_collision_contact_segments_flat",
            "_collision_contact_points_flat",
        ):
            for ent in getattr(self, lst_name, None) or []:
                if "unit_a" in ent and "unit_b" in ent:
                    _add(ent["unit_a"], ent["unit_b"])

        # Tracked active + fixed (whole fold history)
        for dct_name in (
            "_collision_active_segments",
            "_collision_fixed_segments",
            "_collision_active_points",
            "_collision_fixed_points",
        ):
            dct = getattr(self, dct_name, None) or {}
            for ent in dct.values():
                if "unit_a" in ent and "unit_b" in ent:
                    _add(ent["unit_a"], ent["unit_b"])

        # Latest orange-paint classifications
        for cls in getattr(self, "_intruder_classifications", None) or []:
            if "sim_unit_A" in cls and "sim_unit_B" in cls:
                _add(cls["sim_unit_A"], cls["sim_unit_B"])
            if cls.get("intruder_sim_unit") is not None and cls.get("other_sim_unit") is not None:
                _add(cls["intruder_sim_unit"], cls["other_sim_unit"])

        return pairs

    def save_trimmed_design(self, out_path=None):
        """
        Classify intruders from current 3D state, place plane-plane cut lines,
        and write <origami_name>-trimmed.json (or out_path).

        Uses collision contact pairs (force=True) ∪ fold-adjacent pairs
        (force on outer thick faces) with oriented normals.
        """
        if not hasattr(self, "ori_sim") or not hasattr(self, "input_json"):
            print("[Trimmer] Simulator not initialized; cannot trim.")
            return None

        self._render_gen = int(getattr(self, "_render_gen", 0)) + 1
        x_np = self._cache_frame_positions(force=True)
        initial_kps_np = np.asarray(self.kps, dtype=float)
        self._flat_kps_np = initial_kps_np
        if initial_kps_np.ndim != 2 or initial_kps_np.shape[1] < 3:
            if initial_kps_np.ndim == 2 and initial_kps_np.shape[1] == 2:
                z = np.zeros((initial_kps_np.shape[0], 1), dtype=float)
                initial_kps_np = np.hstack([initial_kps_np, z])
            else:
                print("[Trimmer] Invalid initial keypoints; cannot trim.")
                return None

        unit_layer_meta = getattr(self, "_collision_unit_layer_meta", None)
        if not unit_layer_meta:
            unit_layer_meta = self._build_unit_layer_meta()
            self._collision_unit_layer_meta = unit_layer_meta

        # Refresh contacts at export time when collision shading was enabled
        if getattr(self, "collision_shading", False):
            try:
                if not hasattr(self, "_collision_tri_kp_indices"):
                    self._build_collision_topology()
                self.detect_panel_collisions()
            except Exception as exc:
                print(f"[Trimmer] detect_panel_collisions at export failed: {exc}")

        colliding_pairs = self._collect_colliding_unit_pairs_for_trim()
        print(
            f"[Trimmer] colliding unit pairs for cuts: {len(colliding_pairs)}"
        )

        result = compute_trimmed_design_from_sim(
            self.input_json,
            x_np,
            initial_kps_np,
            self.ori_sim.indices,
            self.ori_sim.indices_crease_type,
            unit_layer_meta=unit_layer_meta,
            colliding_unit_pairs=colliding_pairs,
            force_outer_fold_pairs=True,
        )
        # Do NOT repaint orange from fold-adjacent trim classifications.
        # Orange paint is collision-gated (red markers) and updated only by
        # detect_panel_collisions → _update_intruder_classification each frame.

        # Attach every tracked contact (matches red GUI contacts over the fold)
        stats = getattr(self, "_collision_stats", None)
        try:
            if not getattr(self, "_collision_coords_exported", False):
                self._seal_and_export_fixed_flat_contacts()
            stats = getattr(self, "_collision_stats", None)
            if stats is None:
                stats = self.build_collision_stats(include_live=False)
                self._collision_stats = stats
        except Exception as exc:
            print(f"[Trimmer] collision_stats attach skipped: {exc}")
            stats = None
        if stats is not None:
            result["collision_stats"] = stats

        if out_path is None:
            out_path = os.path.join(
                "./descriptionData", f"{self.origami_name}-trimmed.json"
            )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(result, fw, indent=4)

        meta = result.get("trim_3d_metadata", {})
        stats_out = result.get("collision_stats") or {}
        n_poly = len(stats_out.get("closed_polygons") or [])
        n_tri = int(stats_out.get("n_closed_triangles", n_poly))
        print(
            f"[Trimmer] Saved {out_path}  "
            f"(pairs={meta.get('n_pairs_checked', '?')}, cuts={meta.get('n_cuts', '?')}, "
            f"closed_triangles={n_tri})"
        )
        return result

    def _render_panel_meshes(self, scene):
        """
        Draw base mesh + surface sweep paint (design-xy mapped onto panels)
        + red contact markers.
        """
        scene.mesh(
            self.vertices,
            indices=self.indices,
            color=(0.80, 0.82, 0.93),
            two_sided=True,
        )

        n_pts = self._collision_contact_count
        n_segs = self._collision_segment_count
        has_red_markers = n_pts > 0 or n_segs > 0

        # Skip full-panel orange overlay (extra mesh pass) — locus lines are enough
        # Design-locus strokes on panel surfaces
        if self.collision_shading and hasattr(self, "paint_line_vertices"):
            try:
                self._update_surface_sweep_paint()
            except Exception:
                pass
            n_pl = int(getattr(self, "_paint_line_vert_count", 0) or 0)
            if n_pl >= 2:
                scene.lines(
                    self.paint_line_vertices,
                    width=2.5,
                    color=(0.95, 0.18, 0.10),
                    vertex_count=n_pl,
                )

        if n_segs > 0:
            scene.lines(
                self.collision_contact_lines,
                width=4.0,
                color=(0.95, 0.15, 0.10),
                vertex_count=n_segs * 2,
            )

        if n_pts > 0:
            radius = getattr(self, "_collision_point_radius", 0.3)
            scene.particles(
                self.collision_contact_points,
                radius=radius,
                color=(0.95, 0.15, 0.10),
                index_count=n_pts,
            )


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yml")
    example_path = os.path.join(base_dir, "config.example.yml")

    # load config.yml with fallback to config.example.yml
    if os.path.exists(config_path):
        config_file_used = config_path
    elif os.path.exists(example_path):
        config_file_used = example_path
        print("config.yml not found. Falling back to config.example.yml")
    else:
        print("Neither config.yml nor config.example.yml was found.")
        exit(1)

    with open(config_file_used, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    simulations = config.get("simulations", [])
    if not simulations:
        print("No simulations defined in config file.")
        exit(0)

    for sim in simulations:
        name = sim["name"]

        print(f"\n{'='*60}")
        print(f"Starting simulation: {name}")
        print(f"{'='*60}")

        ori = PD_Origami_Simulator(
            origami_name=name,
            use_gui=sim.get("use_gui", True),
            fast=sim.get("fast", True),
            material_type=sim.get("material_type", 1),
            ref_target=sim.get("ref_target", False),
            damping=sim.get("damping", 0.975),
            pd_local_time=sim.get("pd_local_time", 1),
            pd_global_time=sim.get("pd_global_time", 1),
            pd_iter_time=sim.get("pd_iter_time", 5),
            verbose=sim.get("verbose", False),
            # Collision / intruder shading is off by default; use panel-trimming/ runner to enable it.
            collision_shading=sim.get("collision_shading", False),
        )

        ori.start(
            filepath=name,
            unit_edge_max=sim.get("unit_edge_max", 4),
            thick_mode=sim.get("thick_mode", False),
        )

        ori.run()