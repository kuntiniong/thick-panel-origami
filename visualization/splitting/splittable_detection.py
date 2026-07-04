from typing import Dict, List, Tuple

import numpy as np


def _build_crease_topology(original_data: Dict) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, List[int]]]:
    """
    Build a vertex-to-lines adjacency map for topological analysis.

    Each line's two 2-D endpoints are matched to the nearest kp within a
    0.5 mm tolerance. All line types (valley, mountain, border) are included
    so that sector angles capture the full angular neighbourhood at each vertex.
    """
    kps = original_data.get("kps", [])
    lines = original_data.get("lines", [])
    if not kps or not lines:
        return {}, {}

    kp_arr = np.array([[float(kp[0]), float(kp[1])] for kp in kps])
    tol = 0.5  # mm

    line_kp_map: Dict[int, Tuple[int, int]] = {}
    vertex_to_lines: Dict[int, List[int]] = {}

    for li, line in enumerate(lines):
        pts: List[int] = []
        for ep in (0, 1):
            x, y = float(line[ep][0]), float(line[ep][1])
            dists = np.hypot(kp_arr[:, 0] - x, kp_arr[:, 1] - y)
            idx = int(np.argmin(dists))
            pts.append(idx if dists[idx] < tol else -1)
        line_kp_map[li] = (pts[0], pts[1])
        for k in pts:
            if k >= 0:
                vertex_to_lines.setdefault(k, []).append(li)

    return line_kp_map, vertex_to_lines


def _sorted_incident_creases(
    original_data: Dict,
    vertex_idx: int,
    line_kp_map: Dict[int, Tuple[int, int]],
    vertex_to_lines: Dict[int, List[int]],
    include_border: bool = True,
) -> List[Tuple[int, float]]:
    """
    Return (line_idx, angle_rad) for all lines incident to vertex_idx,
    sorted counter-clockwise (increasing angle in [-pi, pi]).
    """
    kps = original_data.get("kps", [])
    features = original_data.get("line_features", [])
    vx = float(kps[vertex_idx][0])
    vy = float(kps[vertex_idx][1])

    result: List[Tuple[int, float]] = []
    for li in vertex_to_lines.get(vertex_idx, []):
        if not include_border:
            if li < len(features) and features[li].get("type", 2) == 2:
                continue
        k0, k1 = line_kp_map[li]
        other_k = k1 if k0 == vertex_idx else k0
        if other_k < 0:
            continue
        ox = float(kps[other_k][0])
        oy = float(kps[other_k][1])
        result.append((li, float(np.arctan2(oy - vy, ox - vx))))

    result.sort(key=lambda x: x[1])
    return result


def _adjacent_sector_angles(
    original_data: Dict,
    vertex_idx: int,
    shared_line_idx: int,
    line_kp_map: Dict[int, Tuple[int, int]],
    vertex_to_lines: Dict[int, List[int]],
) -> Tuple[float, float, int, int]:
    """
    Compute the two sector angles immediately flanking shared_line_idx
    at vertex_idx.

    Folding-range analysis (Peng & Chirikjian 2023, near Eq. 6):
        valid range for shifting angle delta:
            (pi - max(alpha, beta)) < delta < (pi - min(alpha, beta))
        range width = |alpha - beta|
    """
    sorted_lines = _sorted_incident_creases(
        original_data, vertex_idx, line_kp_map, vertex_to_lines, include_border=True
    )
    n = len(sorted_lines)
    if n < 2:
        return 0.0, 0.0, -1, -1

    pos = next((i for i, (li, _) in enumerate(sorted_lines) if li == shared_line_idx), -1)
    if pos == -1:
        return 0.0, 0.0, -1, -1

    curr_ang = sorted_lines[pos][1]
    next_pos = (pos + 1) % n
    prev_pos = (pos - 1) % n

    next_ang = sorted_lines[next_pos][1]
    prev_ang = sorted_lines[prev_pos][1]

    alpha = (next_ang - curr_ang) % (2.0 * np.pi)  # CCW sector
    beta = (curr_ang - prev_ang) % (2.0 * np.pi)   # CW sector

    return alpha, beta, sorted_lines[next_pos][0], sorted_lines[prev_pos][0]


def detect_splittable_creases_for_pattern(
    original_data: Dict,
    crease_info: List[Dict],
    splittable_min_range: float,
) -> Tuple[set, Dict[int, Dict], List[Tuple[int, int, float]]]:
    """
    Detect double-hinge splittable creases using the current practical proxy.

    Note:
    This remains the existing degree/angle/MV-route heuristic that approximates
    the paper's Type 1/2/3 behavior on available data structures. It does not
    perform full symmetry-group verification from Peng & Chirikjian 2023.
    """
    line_kp_map, vertex_to_lines = _build_crease_topology(original_data)
    features = original_data.get("line_features", [])

    double_hinge_creases = set()
    splittable_info: Dict[int, Dict] = {}
    splittable_pairs: List[Tuple[int, int, float]] = []

    li_to_ci: Dict[int, int] = {crease["line_index"]: ci for ci, crease in enumerate(crease_info)}

    for ci, crease in enumerate(crease_info):
        li = crease["line_index"]
        crease_mv_type = crease["type"]  # 0=valley, 1=mountain

        k0, k1 = line_kp_map.get(li, (-1, -1))
        if k0 < 0 or k1 < 0:
            splittable_info[ci] = {"is_splittable": False, "reason": "unmapped"}
            continue

        alpha0, beta0, ccw0, cw0 = _adjacent_sector_angles(
            original_data, k0, li, line_kp_map, vertex_to_lines
        )
        alpha1, beta1, ccw1, cw1 = _adjacent_sector_angles(
            original_data, k1, li, line_kp_map, vertex_to_lines
        )

        range0 = abs(alpha0 - beta0)
        range1 = abs(alpha1 - beta1)
        geo_ok = (range0 > splittable_min_range and range1 > splittable_min_range)

        def mv_degree(v_idx: int) -> int:
            return sum(
                1
                for l in vertex_to_lines.get(v_idx, [])
                if l != li and l < len(features) and features[l].get("type", 2) in (0, 1)
            )

        deg0 = mv_degree(k0)
        deg1 = mv_degree(k1)

        if deg0 == 2 and deg1 == 2:
            unit_type = 1
        elif (deg0 == 2) != (deg1 == 2):
            unit_type = 2
        elif deg0 >= 3 and deg1 >= 3:
            unit_type = 3
        else:
            unit_type = 0

        def mv_neighbours_alternate(ccw_li: int, cw_li: int) -> bool:
            # A valid alternating route exists if at least one adjacent crease
            # has the opposite M/V type.
            for adj_li in (ccw_li, cw_li):
                if adj_li < 0:
                    continue
                if adj_li < len(features):
                    t = features[adj_li].get("type", 2)
                    if t in (0, 1) and t != crease_mv_type:
                        return True
            return False

        mv_ok = (mv_neighbours_alternate(ccw0, cw0) and mv_neighbours_alternate(ccw1, cw1))
        is_splittable = (unit_type in (1, 2, 3) and geo_ok and mv_ok)

        splittable_info[ci] = {
            "is_splittable": is_splittable,
            "unit_type": unit_type,
            "geo_ok": geo_ok,
            "mv_ok": mv_ok,
            "alpha0": alpha0,
            "beta0": beta0,
            "alpha1": alpha1,
            "beta1": beta1,
            "range_width_v0": range0,
            "range_width_v1": range1,
            "folding_range_width": min(range0, range1),
            "deg_v0": deg0,
            "deg_v1": deg1,
        }

        if is_splittable:
            double_hinge_creases.add(ci)
            range_w = min(range0, range1)
            extra_gap = (range_w / np.pi) * 4.0

            for ccw_li, cw_li in ((ccw0, cw0), (ccw1, cw1)):
                for adj_li in (ccw_li, cw_li):
                    adj_ci = li_to_ci.get(adj_li, -1)
                    if adj_ci < 0:
                        continue
                    adj_type = crease_info[adj_ci]["type"]
                    if adj_type != crease_mv_type:
                        splittable_pairs.append((ci, adj_ci, extra_gap))

    return double_hinge_creases, splittable_info, splittable_pairs
