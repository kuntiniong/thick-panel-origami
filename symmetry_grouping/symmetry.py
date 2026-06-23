"""PCA-based symmetry detection and crease line pairing."""

from typing import List, Optional, Tuple

import numpy as np


def _to_xy(point):
    return point[0], point[1]


def collect_2d_points(data: dict) -> np.ndarray:
    """Gather unique 2D coordinates from keypoints and line endpoints."""
    points = []
    for kp in data.get("kps", []):
        points.append(_to_xy(kp))
    for line in data.get("lines", []):
        points.append(_to_xy(line[0]))
        points.append(_to_xy(line[1]))

    if not points:
        return np.empty((0, 2))

    arr = np.asarray(points, dtype=float)
    _, unique_idx = np.unique(np.round(arr, 6), axis=0, return_index=True)
    return arr[np.sort(unique_idx)]


def compute_pca(points: np.ndarray) -> dict:
    """PCA on 2D points via SVD."""
    if len(points) < 2:
        raise ValueError("Need at least 2 points for PCA")

    centroid = points.mean(axis=0)
    centered = points - centroid
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singular_values ** 2) / max(len(points) - 1, 1)
    return {
        "centroid": centroid,
        "components": vt,
        "eigenvalues": eigenvalues,
    }


def reflect_across_line(
    points: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Mirror points across the infinite line through origin along unit direction."""
    d = direction / np.linalg.norm(direction)
    offsets = points - origin
    projections = np.outer(offsets @ d, d)
    on_line = origin + projections
    return 2.0 * on_line - points


def _symmetry_error(points: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    reflected = reflect_across_line(points, origin, direction)
    diffs = reflected[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    return float(distances.min(axis=1).mean())


def find_symmetry_line(points: np.ndarray, pca: dict) -> dict:
    """Pick the PCA axis with lowest reflection error as the symmetry line."""
    centroid = pca["centroid"]
    pc1, pc2 = pca["components"][0], pca["components"][1]

    best = None
    for label, direction, eigenvalue in [
        ("PC1", pc1, pca["eigenvalues"][0]),
        ("PC2", pc2, pca["eigenvalues"][1]),
    ]:
        error = _symmetry_error(points, centroid, direction)
        entry = {
            "label": label,
            "centroid": centroid,
            "direction": direction / np.linalg.norm(direction),
            "eigenvalue": float(eigenvalue),
            "symmetry_error": error,
        }
        if best is None or error < best["symmetry_error"]:
            best = entry

    return best


def _line_segment_xy(line) -> np.ndarray:
    return np.array([_to_xy(line[0]), _to_xy(line[1])], dtype=float)


def _segment_match_distance(seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    forward = np.linalg.norm(seg_a[0] - seg_b[0]) + np.linalg.norm(seg_a[1] - seg_b[1])
    reverse = np.linalg.norm(seg_a[0] - seg_b[1]) + np.linalg.norm(seg_a[1] - seg_b[0])
    return float(min(forward, reverse))


def _adaptive_match_tolerance(points: np.ndarray) -> float:
    if len(points) == 0:
        return 1.0
    span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return max(span * 1e-3, 0.5)


def build_symmetry_groups(
    data: dict,
    symmetry: dict,
    tolerance: Optional[float] = None,
) -> List[List[int]]:
    """
    Pair crease lines that are mirror images across the symmetry line.

    Returns line-index pairs into data["lines"]. On-axis lines are [i, i].
    """
    lines = data.get("lines", [])
    if not lines or symmetry is None:
        return []

    points = collect_2d_points(data)
    tol = tolerance if tolerance is not None else _adaptive_match_tolerance(points)
    origin = symmetry["centroid"]
    direction = symmetry["direction"]

    segments = [_line_segment_xy(line) for line in lines]
    reflected = [reflect_across_line(seg, origin, direction) for seg in segments]

    partners = {}
    for i, ref_seg in enumerate(reflected):
        best_j = None
        best_dist = float("inf")
        for j, seg in enumerate(segments):
            dist = _segment_match_distance(ref_seg, seg)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None and best_dist <= 2.0 * tol:
            partners[i] = best_j

    assigned = set()
    symmetry_groups = []
    for i in range(len(lines)):
        if i in assigned:
            continue
        j = partners.get(i)
        if j is None:
            symmetry_groups.append([i, i])
            assigned.add(i)
            continue

        if partners.get(j) == i:
            symmetry_groups.append(sorted([i, j]))
            assigned.add(i)
            assigned.add(j)
        else:
            symmetry_groups.append([i, i])
            assigned.add(i)

    return symmetry_groups


def detect_symmetry_line(data: dict) -> Optional[dict]:
    points = collect_2d_points(data)
    if len(points) < 2:
        return None
    pca = compute_pca(points)
    return find_symmetry_line(points, pca)


def detect_line_symmetry_groups(data: dict) -> List[List[int]]:
    symmetry = detect_symmetry_line(data)
    if symmetry is None:
        return []
    return build_symmetry_groups(data, symmetry)


def map_line_groups_to_crease_groups(
    line_groups: List[List[int]],
    crease_line_indices: List[int],
) -> List[List[int]]:
    """
    Convert line-index symmetry pairs to crease-array indices.

    Only valley/mountain creases appear in crease_line_indices. Border lines are
    skipped. Groups with fewer than two creases are omitted (on-axis lines).
    """
    line_to_crease = {
        line_idx: crease_idx for crease_idx, line_idx in enumerate(crease_line_indices)
    }

    crease_groups = []
    for pair in line_groups:
        crease_indices = []
        for line_idx in pair:
            if line_idx in line_to_crease:
                crease_indices.append(line_to_crease[line_idx])
        unique = list(dict.fromkeys(crease_indices))
        if len(unique) >= 2:
            crease_groups.append(unique)

    return crease_groups


def detect_crease_symmetry_groups(
    data: dict,
    crease_line_indices: List[int],
) -> Tuple[List[List[int]], List[List[int]], Optional[dict]]:
    """
    Detect line symmetry groups and map them to crease optimizer indices.

    Returns (line_groups, crease_groups, symmetry_line).
    """
    symmetry = detect_symmetry_line(data)
    if symmetry is None:
        return [], [], None

    line_groups = build_symmetry_groups(data, symmetry)
    crease_groups = map_line_groups_to_crease_groups(line_groups, crease_line_indices)
    return line_groups, crease_groups, symmetry