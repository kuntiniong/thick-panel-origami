"""
Origami description JSON visualizer with PCA symmetry detection.

Plots the 2D crease pattern (units, lines, keypoints) from description
JSON files listed in config.yml.
Uses PCA on keypoints/line endpoints to find the dominant symmetry line,
pairs mirror-symmetric crease lines into symmetry_groups, and colors each
pair with the same hue in the 2D plot.

Usage:
    python pca_visualize.py
    python pca_visualize.py --config config.yml
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from symmetry_grouping.symmetry import (
    build_symmetry_groups,
    collect_2d_points,
    detect_symmetry_line,
    reflect_across_line,
)

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 11

LINE_TYPE_LABELS = {
    0: "Valley",
    1: "Mountain",
    2: "Border",
}

LINE_TYPE_COLORS = {
    0: "#d62728",
    1: "#1f77b4",
    2: "#2ca02c",
}

UNIT_COLORS = ["#fff2cc", "#d5e8d4", "#dae8fc", "#f8cecc"]
NEUTRAL_PANEL_COLOR = "#f5f5f5"

SYMMETRY_GROUP_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
]


def load_config(base_dir: str, config_arg: str = None) -> Tuple[dict, str]:
    if config_arg:
        config_path = os.path.abspath(config_arg)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config_file_used = config_path
    else:
        config_path = os.path.join(base_dir, "config.yml")
        example_path = os.path.join(base_dir, "config.example.yml")
        if os.path.exists(config_path):
            config_file_used = config_path
        elif os.path.exists(example_path):
            config_file_used = example_path
            print("config.yml not found. Falling back to config.example.yml")
        else:
            raise FileNotFoundError(
                "No config.yml or config.example.yml found in symmetry_grouping/"
            )

    with open(config_file_used, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config, config_file_used


def resolve_json_path(name: str, base_dir: str) -> str:
    project_root = os.path.dirname(base_dir)
    return os.path.join(project_root, "descriptionData", f"{name}.json")


def load_description(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_xy(point):
    return point[0], point[1]


_reflect_across_line = reflect_across_line


def _symmetry_line_segment(
    centroid: np.ndarray,
    direction: np.ndarray,
    points: np.ndarray,
    margin: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extend the symmetry line to span the point cloud bounds."""
    d = direction / np.linalg.norm(direction)
    t_values = (points - centroid) @ d
    t_min = float(t_values.min()) - margin
    t_max = float(t_values.max()) + margin
    p0 = centroid + t_min * d
    p1 = centroid + t_max * d
    return p0, p1


def _line_color_from_groups(
    line_idx: int,
    symmetry_groups: List[List[int]],
) -> str:
    for group_idx, pair in enumerate(symmetry_groups):
        if line_idx in pair:
            return SYMMETRY_GROUP_COLORS[group_idx % len(SYMMETRY_GROUP_COLORS)]
    return "#333333"


def plot_symmetry_line(ax, symmetry: dict, points: np.ndarray):
    """Overlay the PCA-derived symmetry line on the 2D crease pattern."""
    p0, p1 = _symmetry_line_segment(
        symmetry["centroid"],
        symmetry["direction"],
        points,
    )
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color="#9467bd",
        linewidth=2.5,
        linestyle="--",
        zorder=6,
        label="Symmetry (PCA)",
    )
    ax.scatter(
        [symmetry["centroid"][0]],
        [symmetry["centroid"][1]],
        s=70,
        c="#9467bd",
        marker="x",
        linewidths=2.0,
        zorder=7,
    )


def plot_2d_pattern(
    ax,
    data: dict,
    title: str,
    symmetry: Optional[dict] = None,
    symmetry_groups: Optional[List[List[int]]] = None,
):
    units = data.get("units", [])
    lines = data.get("lines", [])
    kps = data.get("kps", [])
    line_features = data.get("line_features", [])

    for i, unit in enumerate(units):
        corners = [_to_xy(p) for p in unit]
        panel_color = (
            NEUTRAL_PANEL_COLOR
            if symmetry_groups
            else UNIT_COLORS[i % len(UNIT_COLORS)]
        )
        patch = Polygon(
            corners,
            closed=True,
            facecolor=panel_color,
            edgecolor="none",
            alpha=0.55,
            zorder=1,
        )
        ax.add_patch(patch)
        ax.text(
            np.mean([c[0] for c in corners]),
            np.mean([c[1] for c in corners]),
            str(i),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#333333",
            zorder=2,
        )

    segments = []
    colors = []
    for i, line in enumerate(lines):
        p0, p1 = line[0], line[1]
        segments.append([_to_xy(p0), _to_xy(p1)])
        if symmetry_groups:
            colors.append(_line_color_from_groups(i, symmetry_groups))
        else:
            line_type = 2
            if i < len(line_features):
                line_type = line_features[i].get("type", 2)
            colors.append(LINE_TYPE_COLORS.get(line_type, "#333333"))

    if segments:
        lc = LineCollection(
            segments,
            colors=colors,
            linewidths=2.0,
            capstyle="round",
            zorder=3,
        )
        ax.add_collection(lc)

    if kps:
        xs = [kp[0] for kp in kps]
        ys = [kp[1] for kp in kps]
        ax.scatter(xs, ys, s=36, c="#111111", zorder=4, edgecolors="white", linewidths=0.6)
        for idx, (x, y) in enumerate(zip(xs, ys)):
            ax.annotate(
                str(idx),
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color="#555555",
                zorder=5,
            )

    if lines or kps or units:
        all_x = []
        all_y = []
        for kp in kps:
            all_x.append(kp[0])
            all_y.append(kp[1])
        for line in lines:
            all_x.extend([line[0][0], line[1][0]])
            all_y.extend([line[0][1], line[1][1]])
        margin = 8
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    if symmetry is not None:
        points = collect_2d_points(data)
        if len(points) > 0:
            plot_symmetry_line(ax, symmetry, points)

    legend_handles = []
    if symmetry_groups:
        for group_idx, pair in enumerate(symmetry_groups):
            if pair[0] == pair[1]:
                label = f"Group {group_idx}: line {pair[0]} (on axis)"
            else:
                label = f"Group {group_idx}: lines {pair[0]}, {pair[1]}"
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=SYMMETRY_GROUP_COLORS[group_idx % len(SYMMETRY_GROUP_COLORS)],
                    linewidth=2.5,
                    label=label,
                )
            )
    else:
        legend_handles = [
            plt.Line2D([0], [0], color=LINE_TYPE_COLORS[t], linewidth=2.5, label=label)
            for t, label in LINE_TYPE_LABELS.items()
        ]
    if symmetry is not None:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color="#9467bd",
                linewidth=2.5,
                linestyle="--",
                label="Symmetry (PCA)",
            )
        )
    if symmetry_groups and len(symmetry_groups) > 12:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            framealpha=0.9,
            fontsize=6,
        )
    else:
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=8)


def compute_symmetry_from_data(data: dict) -> Optional[dict]:
    return detect_symmetry_line(data)


def compute_symmetry_groups_from_data(
    data: dict,
    symmetry: Optional[dict],
) -> List[List[int]]:
    if symmetry is None:
        return []
    return build_symmetry_groups(data, symmetry)


def plot_description(
    data: dict,
    name: str,
    output_path: str,
    symmetry: Optional[dict] = None,
    symmetry_groups: Optional[List[List[int]]] = None,
):
    fig_w = 10 if symmetry_groups and len(symmetry_groups) > 12 else 8
    fig, ax_2d = plt.subplots(figsize=(fig_w, 7), dpi=150)
    plot_2d_pattern(
        ax_2d,
        data,
        f"{name} — 2D Crease Pattern",
        symmetry=symmetry,
        symmetry_groups=symmetry_groups,
    )

    fig.suptitle(f"Origami Description: {name}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 0.85, 1] if symmetry_groups and len(symmetry_groups) > 12 else None)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Figure saved to: {output_path}")
    plt.close(fig)

    return fig


def run_visualization(sim: dict, base_dir: str, output_override: str = None):
    name = sim["name"]
    json_path = resolve_json_path(name, base_dir)

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return 1

    print(f"Loading: {json_path}")
    data = load_description(json_path)

    n_kps = len(data.get("kps", []))
    n_lines = len(data.get("lines", []))
    n_units = len(data.get("units", []))
    print(f"  Keypoints: {n_kps}, Lines: {n_lines}, Units: {n_units}")

    output_path = output_override or sim.get("output")
    if not output_path:
        output_path = os.path.join(base_dir, "output", f"{name}_visualization.png")
    elif not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)

    symmetry = compute_symmetry_from_data(data)
    symmetry_groups = compute_symmetry_groups_from_data(data, symmetry)
    if symmetry is not None:
        c = symmetry["centroid"]
        d = symmetry["direction"]
        angle = np.degrees(np.arctan2(d[1], d[0]))
        print(
            f"  Symmetry line ({symmetry['label']}): "
            f"centroid=({c[0]:.2f}, {c[1]:.2f}), "
            f"angle={angle:.1f}°, error={symmetry['symmetry_error']:.4f}"
        )
        print(f"  Line symmetry groups ({len(symmetry_groups)}):")
        for group_idx, pair in enumerate(symmetry_groups):
            if pair[0] == pair[1]:
                print(f"    [{group_idx}] line {pair[0]} (on symmetry axis)")
            else:
                print(f"    [{group_idx}] lines {pair[0]} <-> {pair[1]}")
    else:
        print("  Symmetry line: not enough points for PCA")

    plot_description(
        data,
        name,
        output_path=output_path,
        symmetry=symmetry,
        symmetry_groups=symmetry_groups,
    )
    return 0


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Visualize origami description JSON files from config.yml"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file path (default: symmetry_grouping/config.yml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Override output image path for all visualizations",
    )
    args = parser.parse_args()

    try:
        config, config_file_used = load_config(base_dir, args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Using config: {config_file_used}")

    simulations = config.get("simulations", [])
    if not simulations:
        print("No simulations defined in config file.")
        return 0

    exit_code = 0

    for sim in simulations:
        name = sim.get("name")
        if not name:
            print("Skipping entry without a name.")
            exit_code = 1
            continue

        print(f"\n{'=' * 60}")
        print(f"Visualizing: {name}")
        print(f"{'=' * 60}")

        result = run_visualization(sim, base_dir, output_override=args.output)
        if result != 0:
            exit_code = result

    return exit_code


if __name__ == "__main__":
    sys.exit(main())