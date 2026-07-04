"""
Splittable-crease visualizer for thick-panel origami.

For each pattern in config.yml, this script runs the same detector used by
visualization/splitting/splittable_detection.py and draws a single result view.

Usage:
    python visualize.py
    python visualize.py --config config.yml
    python visualize.py --config config.yml -o output/custom.png
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Polygon

from visualization.splitting.splittable_detection import detect_splittable_creases_for_pattern

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 10

BORDER_COLOR = "#aaaaaa"
PANEL_COLOR = "#f5f5f5"
PANEL_EDGE = "#cccccc"
KP_COLOR = "#222222"
NON_SPLIT_COLOR = "#274c9a"
SPLIT_COLOR = "#ff7f0e"


def _parse_creases(data: dict) -> List[Dict]:
    """Return crease dicts (type 0 or 1) with line_index."""
    creases = []
    for i, feat in enumerate(data.get("line_features", [])):
        t = feat.get("type", 2)
        if t in (0, 1):
            creases.append({"index": i, "type": t, "line_index": i})
    return creases


def _bounds(data: dict, margin: float = 8.0) -> Tuple[float, float, float, float]:
    all_x, all_y = [], []
    for kp in data.get("kps", []):
        all_x.append(kp[0])
        all_y.append(kp[1])
    for line in data.get("lines", []):
        all_x += [line[0][0], line[1][0]]
        all_y += [line[0][1], line[1][1]]

    if not all_x:
        return 0.0, 1.0, 0.0, 1.0

    return (
        min(all_x) - margin,
        max(all_x) + margin,
        min(all_y) - margin,
        max(all_y) + margin,
    )


def _compute_split_results(
    data: dict,
    creases: List[Dict],
    splittable_min_range: float,
):
    split_set, split_info, _ = detect_splittable_creases_for_pattern(
        original_data=data,
        crease_info=creases,
        splittable_min_range=float(splittable_min_range),
    )
    return split_set, split_info


def _plot_variant(
    ax,
    data: dict,
    creases: List[Dict],
    split_set: set,
    split_info: Dict[int, Dict],
    splittable_min_range: float,
    min_thickness: float,
    max_offset: float,
    splittable_height_boost: float,
    show_values: bool,
):
    lines_data = data.get("lines", [])
    line_features = data.get("line_features", [])
    kps = data.get("kps", [])

    for unit in data.get("units", []):
        corners = [(p[0], p[1]) for p in unit]
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor=PANEL_COLOR,
                edgecolor=PANEL_EDGE,
                alpha=0.5,
                zorder=1,
            )
        )

    for i, feat in enumerate(line_features):
        if feat.get("type", 2) == 2 and i < len(lines_data):
            p0, p1 = lines_data[i][0], lines_data[i][1]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=BORDER_COLOR,
                linewidth=1.5,
                zorder=2,
            )

    for ci, info in enumerate(creases):
        li = info["line_index"]
        if li >= len(lines_data):
            continue

        p0, p1 = lines_data[li][0], lines_data[li][1]

        is_split = ci in split_set
        color = SPLIT_COLOR if is_split else NON_SPLIT_COLOR
        lw = 4.2 if is_split else 2.4
        alpha = 0.98 if is_split else 0.85

        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=color,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=3,
        )

        if show_values and is_split:
            sinfo = split_info.get(ci, {})
            width = float(sinfo.get("folding_range_width", 0.0))
            boost_mm = (
                splittable_height_boost
                * (width / np.pi)
                * (max_offset - min_thickness)
            )
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            ax.text(
                mx,
                my,
                f"c{ci} | w={width:.2f} | +{boost_mm:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                zorder=5,
            )

    if kps:
        xs = [kp[0] for kp in kps]
        ys = [kp[1] for kp in kps]
        ax.scatter(xs, ys, s=25, c=KP_COLOR, zorder=6, edgecolors="white", linewidths=0.5)

    split_count = len(split_set)
    total_count = len(creases)

    x0, x1, y0, y1 = _bounds(data)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title(
        (
            f"splittable_min_range={splittable_min_range:g} rad\\n"
            f"splittable: {split_count}/{total_count}"
        ),
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)


def plot_splitting(
    data: dict,
    name: str,
    splittable_min_range: float,
    min_thickness: float,
    max_offset: float,
    splittable_height_boost: float,
    output_path: str,
    show_values: bool,
):
    creases = _parse_creases(data)
    if not creases:
        print(f"  No valley/mountain creases found in {name}, skipping.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.6), dpi=150)

    split_set, split_info = _compute_split_results(
        data=data,
        creases=creases,
        splittable_min_range=splittable_min_range,
    )

    _plot_variant(
        ax=ax,
        data=data,
        creases=creases,
        split_set=split_set,
        split_info=split_info,
        splittable_min_range=splittable_min_range,
        min_thickness=min_thickness,
        max_offset=max_offset,
        splittable_height_boost=splittable_height_boost,
        show_values=show_values,
    )

    fig.suptitle(
        (
            f"Splittable crease detection — {name}\\n"
            f"(boost={splittable_height_boost:g}, min_thickness={min_thickness:g}mm, max_offset={max_offset:g}mm)"
        ),
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved -> {output_path}")
    plt.close(fig)


def load_config(base_dir: str, config_arg: Optional[str]) -> Tuple[dict, str]:
    if config_arg:
        path = os.path.abspath(config_arg)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        config_file = path
    else:
        primary = os.path.join(base_dir, "config.yml")
        fallback = os.path.join(base_dir, "config.example.yml")
        if os.path.exists(primary):
            config_file = primary
        elif os.path.exists(fallback):
            config_file = fallback
            print("config.yml not found, falling back to config.example.yml")
        else:
            raise FileNotFoundError("No config.yml found in splitting/")

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}, config_file


def resolve_json_path(name: str, base_dir: str) -> str:
    project_root = os.path.dirname(os.path.dirname(base_dir))
    return os.path.join(project_root, "descriptionData", f"{name}.json")


def run_simulation(sim: dict, base_dir: str, output_override: Optional[str]) -> int:
    name = sim["name"]
    json_path = resolve_json_path(name, base_dir)
    if not os.path.exists(json_path):
        print(f"  Error: JSON not found: {json_path}")
        return 1

    print(f"Loading: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    show_values = bool(sim.get("show_values", True))
    min_thickness = float(sim.get("min_thickness", 2.0))
    max_offset = float(sim.get("max_offset", 25.0))
    splittable_height_boost = float(sim.get("splittable_height_boost", 1.5))
    splittable_min_range = float(sim.get("splittable_min_range", 0.2))

    output_path = output_override or sim.get("output")
    if not output_path:
        output_path = os.path.join(base_dir, "output", f"{name}_splitting.png")
    elif not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)

    creases = _parse_creases(data)
    split_set, _ = _compute_split_results(data, creases, splittable_min_range)
    print(f"  Creases: {len(creases)}")
    print(f"    splittable_min_range={splittable_min_range:g} -> {len(split_set)}/{len(creases)}")

    plot_splitting(
        data=data,
        name=name,
        splittable_min_range=splittable_min_range,
        min_thickness=min_thickness,
        max_offset=max_offset,
        splittable_height_boost=splittable_height_boost,
        output_path=output_path,
        show_values=show_values,
    )

    return 0


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Visualize splittable-crease detection on origami crease patterns."
    )
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Override output image path for all simulations.",
    )
    args = parser.parse_args()

    try:
        config, config_file = load_config(base_dir, args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Using config: {config_file}")

    simulations = config.get("simulations", [])
    if not simulations:
        print("No simulations defined in config.")
        return 0

    exit_code = 0
    for sim in simulations:
        if not sim.get("name"):
            print("Skipping entry with no 'name'.")
            continue
        code = run_simulation(sim, base_dir, args.output)
        if code:
            exit_code = code

    return exit_code


if __name__ == "__main__":
    sys.exit(main() or 0)
