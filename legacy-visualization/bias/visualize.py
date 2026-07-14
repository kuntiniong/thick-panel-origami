"""
Border-proximity bias visualizer for thick-panel origami.

For each origami pattern in config.yml, renders the 2D crease pattern with
creases coloured by their computed initial-mean magnitude under different
(max_offset, horiz_bias weight) settings.  Each variant produces one
subplot so you can compare the effect of different weights side by side.

Usage:
    python visualize.py
    python visualize.py --config config.yml
    python visualize.py --config config.yml -o output/my_plot.png
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import yaml
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.colorbar import ColorbarBase

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 10

BORDER_COLOR = "#aaaaaa"
PANEL_COLOR = "#f5f5f5"
PANEL_EDGE = "#cccccc"
KP_COLOR = "#222222"


# ---------------------------------------------------------------------------
# Geometry helpers (self-contained, no framework import needed)
# ---------------------------------------------------------------------------

def _parse_creases(data: dict) -> List[Dict]:
    """Return crease dicts (type 0 or 1) with line_index."""
    creases = []
    for i, feat in enumerate(data.get("line_features", [])):
        t = feat.get("type", 2)
        if t in (0, 1):
            creases.append({"index": i, "type": t, "line_index": i})
    return creases


def _horizontal_mask(
    data: dict,
    creases: List[Dict],
    angle_threshold_deg: float = 10.0,
) -> np.ndarray:
    """Return boolean array – True for creases within angle_threshold_deg of horizontal."""
    lines = data.get("lines", [])
    sin_threshold = np.sin(np.deg2rad(angle_threshold_deg))
    mask = np.zeros(len(creases), dtype=bool)
    for i, info in enumerate(creases):
        idx = info["line_index"]
        if idx < len(lines):
            p0, p1 = lines[idx][0], lines[idx][1]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            length = np.hypot(dx, dy)
            if length > 0 and abs(dy) / length < sin_threshold:
                mask[i] = True
    return mask


def _compute_bias(
    data: dict,
    creases: List[Dict],
    max_bias_mm: float,
) -> np.ndarray:
    """
    All horizontal creases get max_bias_mm uniformly; diagonal creases get 0.
    """
    if max_bias_mm <= 0.0:
        return np.zeros(len(creases))
    return np.where(_horizontal_mask(data, creases), max_bias_mm, 0.0)


def _initial_mean_magnitudes(
    data: dict,
    creases: List[Dict],
    min_thickness: float,
    max_offset: float,
    horiz_bias_weight: float,
) -> np.ndarray:
    """Return the absolute initial mean magnitude for each crease."""
    max_bias_mm = horiz_bias_weight * (max_offset - min_thickness)
    bias = _compute_bias(data, creases, max_bias_mm)
    return np.clip(min_thickness + bias, min_thickness, max_offset)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bounds(data: dict, margin: float = 8.0) -> Tuple[float, float, float, float]:
    all_x, all_y = [], []
    for kp in data.get("kps", []):
        all_x.append(kp[0]); all_y.append(kp[1])
    for line in data.get("lines", []):
        all_x += [line[0][0], line[1][0]]
        all_y += [line[0][1], line[1][1]]
    return (
        min(all_x) - margin, max(all_x) + margin,
        min(all_y) - margin, max(all_y) + margin,
    )


def plot_bias_variant(
    ax,
    data: dict,
    creases: List[Dict],
    magnitudes: np.ndarray,
    min_thickness: float,
    max_offset: float,
    horiz_bias_weight: float,
    title: str,
    cmap,
    norm,
    show_values: bool = True,
):
    lines_data = data.get("lines", [])
    line_features = data.get("line_features", [])
    kps = data.get("kps", [])

    # panels
    for unit in data.get("units", []):
        corners = [(p[0], p[1]) for p in unit]
        ax.add_patch(Polygon(
            corners, closed=True,
            facecolor=PANEL_COLOR, edgecolor=PANEL_EDGE,
            alpha=0.5, zorder=1,
        ))

    # border lines
    for i, feat in enumerate(line_features):
        if feat.get("type", 2) == 2 and i < len(lines_data):
            p0, p1 = lines_data[i][0], lines_data[i][1]
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]],
                color=BORDER_COLOR, linewidth=1.5, zorder=2,
            )

    # crease index lookup for magnitude
    crease_index_map = {info["line_index"]: idx for idx, info in enumerate(creases)}

    # crease lines coloured by magnitude
    for i, info in enumerate(creases):
        idx = info["line_index"]
        if idx >= len(lines_data):
            continue
        p0 = lines_data[idx][0]
        p1 = lines_data[idx][1]
        mag = magnitudes[i]
        color = cmap(norm(mag))

        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]],
            color=color, linewidth=3.0, solid_capstyle="round", zorder=3,
        )

        if show_values:
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            ax.text(
                mx, my, f"{mag:.1f}",
                ha="center", va="center", fontsize=7,
                color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                zorder=5,
            )

    # keypoints
    if kps:
        xs = [kp[0] for kp in kps]
        ys = [kp[1] for kp in kps]
        ax.scatter(xs, ys, s=25, c=KP_COLOR, zorder=6, edgecolors="white", linewidths=0.5)

    x0, x1, y0, y1 = _bounds(data)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)


def plot_origami_bias(
    data: dict,
    name: str,
    variants: List[Dict],
    min_thickness: float,
    output_path: str,
    show_values: bool = True,
):
    n = len(variants)
    creases = _parse_creases(data)
    if not creases:
        print(f"  No valley/mountain creases found in {name}, skipping.")
        return

    # pre-compute all magnitudes to establish global colour range
    all_mags = []
    variant_mags = []
    for v in variants:
        mags = _initial_mean_magnitudes(
            data, creases,
            min_thickness=min_thickness,
            max_offset=v["max_offset"],
            horiz_bias_weight=v["horiz_bias"],
        )
        variant_mags.append(mags)
        all_mags.extend(mags.tolist())

    global_min = min(all_mags)
    global_max = max(all_mags)
    if global_max == global_min:
        global_max = global_min + 1.0

    cmap = matplotlib.colormaps["plasma"]
    norm = mcolors.Normalize(vmin=global_min, vmax=global_max)

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig_w = ncols * 5.5 + 1.2  # extra for colourbar
    fig_h = nrows * 5.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=150,
                              squeeze=False)

    for i, (v, mags) in enumerate(zip(variants, variant_mags)):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        max_bias_mm = v["horiz_bias"] * (v["max_offset"] - min_thickness)
        title = (
            f"bias={v['horiz_bias']:g}  max_offset={v['max_offset']:g}mm\n"
            f"→ max added {max_bias_mm:.1f}mm  |  range [{mags.min():.1f}, {mags.max():.1f}]mm"
        )
        plot_bias_variant(
            ax, data, creases, mags,
            min_thickness=min_thickness,
            max_offset=v["max_offset"],
            horiz_bias_weight=v["horiz_bias"],
            title=title,
            cmap=cmap,
            norm=norm,
            show_values=show_values,
        )

    # hide unused axes
    for i in range(len(variants), nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    # shared colourbar on the right
    fig.suptitle(
        f"Border-proximity bias — {name}\n"
        f"(min_thickness={min_thickness}mm, creases={len(creases)})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.70])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("Initial mean magnitude (mm)", fontsize=10)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Config / entry-point
# ---------------------------------------------------------------------------

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
            raise FileNotFoundError("No config.yml found in bias/")

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}, config_file


def resolve_json_path(name: str, base_dir: str) -> str:
    project_root = os.path.dirname(os.path.dirname(base_dir))
    return os.path.join(project_root, "descriptionData", f"{name}.json")


def run_simulation(sim: dict, base_dir: str, output_override: Optional[str]):
    name = sim["name"]
    json_path = resolve_json_path(name, base_dir)
    if not os.path.exists(json_path):
        print(f"  Error: JSON not found: {json_path}")
        return 1

    print(f"Loading: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    min_thickness = float(sim.get("min_thickness", 2.0))
    show_values = bool(sim.get("show_values", True))

    variants = sim.get("variants", [])
    if not variants:
        print(f"  No variants defined for {name}, skipping.")
        return 0

    output_path = output_override or sim.get("output")
    if not output_path:
        output_path = os.path.join(base_dir, "output", f"{name}_bias.png")
    elif not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)

    creases = _parse_creases(data)
    print(f"  Creases: {len(creases)}, Variants: {len(variants)}")
    for v in variants:
        max_bias_mm = v["horiz_bias"] * (v["max_offset"] - min_thickness)
        print(f"    bias={v['horiz_bias']:g}  max_offset={v['max_offset']:g}mm  \u2192 +{max_bias_mm:.2f}mm max")

    plot_origami_bias(
        data, name, variants,
        min_thickness=min_thickness,
        output_path=output_path,
        show_values=show_values,
    )
    return 0


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Visualize border-proximity bias on origami crease patterns."
    )
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    parser.add_argument("-o", "--output", default=None,
                        help="Override output image path for all simulations.")
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
