"""
Manual splitting selector GUI for thick-panel origami.

Open a 2D crease plot, click creases to mark them as "thicker", then export
weights and paste-ready manual offsets for optimization/manual mode.

Usage:
    python visualize.py
    python visualize.py --config config.yml
    python visualize.py --config config.yml --simulation 0
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, Slider

BORDER_COLOR = "#aaaaaa"
PANEL_COLOR = "#f5f5f5"
PANEL_EDGE = "#cccccc"
KP_COLOR = "#222222"
UNSELECTED_COLOR = "#254e9a"
SELECTED_COLOR = "#d9480f"


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


def _resolve_json_path(name: str, base_dir: str) -> str:
    project_root = os.path.dirname(os.path.dirname(base_dir))
    return os.path.join(project_root, "descriptionData", f"{name}.json")


def _snap_value(value: float, min_thickness: float, max_offset: float, step: float) -> float:
    value = max(min_thickness, min(max_offset, float(value)))
    if step <= 0:
        return value
    n_steps = round((value - min_thickness) / step)
    snapped = min_thickness + n_steps * step
    return max(min_thickness, min(max_offset, snapped))


def _build_unsigned_magnitudes(
    creases: List[Dict],
    selected: Set[int],
    min_thickness: float,
    thick_thickness: float,
) -> List[float]:
    return [float(thick_thickness if i in selected else min_thickness) for i in range(len(creases))]


def _build_signed_manual_offsets(
    creases: List[Dict],
    unsigned_magnitudes: List[float],
) -> List[float]:
    offsets: List[float] = []
    for i, crease in enumerate(creases):
        mag = abs(float(unsigned_magnitudes[i]))
        offsets.append(mag if crease["type"] == 0 else -mag)
    return offsets


def _format_offsets_snippet(offsets: List[float]) -> str:
    lines = ["framework:", "  initial_offsets:"]
    for value in offsets:
        lines.append(f"    - {value:g}")
    return "\n".join(lines)


class ManualSplitGUI:
    def __init__(
        self,
        data: dict,
        name: str,
        output_dir: str,
        min_thickness: float,
        thick_thickness: float,
        max_offset: float,
        discrete_step: float,
        show_labels: bool,
    ) -> None:
        self.data = data
        self.name = name
        self.output_dir = output_dir
        self.min_thickness = float(min_thickness)
        self.max_offset = float(max_offset)
        self.discrete_step = float(discrete_step)
        self.show_labels = bool(show_labels)

        self.thick_thickness = _snap_value(
            float(thick_thickness), self.min_thickness, self.max_offset, self.discrete_step
        )

        self.creases = _parse_creases(data)
        self.selected: Set[int] = set()

        self.fig = None
        self.ax = None
        self.slider = None
        self.status_text = None
        self.hint_text = None

        self.crease_lines: Dict[int, Line2D] = {}

    def _update_line_style(self, ci: int) -> None:
        line = self.crease_lines.get(ci)
        if line is None:
            return
        if ci in self.selected:
            line.set_color(SELECTED_COLOR)
            line.set_linewidth(4.6)
            line.set_alpha(1.0)
        else:
            line.set_color(UNSELECTED_COLOR)
            line.set_linewidth(2.6)
            line.set_alpha(0.85)

    def _update_status(self) -> None:
        unsigned = _build_unsigned_magnitudes(
            self.creases,
            self.selected,
            self.min_thickness,
            self.thick_thickness,
        )
        n_selected = len(self.selected)
        if self.status_text is not None:
            self.status_text.set_text(
                f"selected={n_selected}/{len(self.creases)} | thick={self.thick_thickness:g} mm | "
                f"min={min(unsigned):g} max={max(unsigned):g}"
            )

    def _on_pick(self, event) -> None:
        artist = event.artist
        ci = None
        for idx, line in self.crease_lines.items():
            if artist is line:
                ci = idx
                break
        if ci is None:
            return

        if ci in self.selected:
            self.selected.remove(ci)
        else:
            self.selected.add(ci)

        self._update_line_style(ci)
        self._update_status()
        self.fig.canvas.draw_idle()

    def _on_slider_change(self, value: float) -> None:
        snapped = _snap_value(value, self.min_thickness, self.max_offset, self.discrete_step)
        if abs(snapped - self.thick_thickness) > 1e-9:
            self.thick_thickness = snapped
        self._update_status()
        self.fig.canvas.draw_idle()

    def _export(self) -> Tuple[str, str]:
        unsigned = _build_unsigned_magnitudes(
            self.creases,
            self.selected,
            self.min_thickness,
            self.thick_thickness,
        )
        offsets = _build_signed_manual_offsets(self.creases, unsigned)
        weights = [1 if i in self.selected else 0 for i in range(len(self.creases))]

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.output_dir, f"{self.name}_manual_selection_{stamp}.json")
        txt_path = os.path.join(self.output_dir, f"{self.name}_manual_offsets_{stamp}.txt")

        payload = {
            "name": self.name,
            "selected_creases": sorted(self.selected),
            "weights": weights,
            "unsigned_magnitudes_mm": unsigned,
            "manual_offsets": offsets,
            "note": "Paste manual_offsets into optimization framework.initial_offsets",
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        snippet = _format_offsets_snippet(offsets)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(snippet + "\n")

        print("\n=== Exported manual selection ===")
        print(f"JSON: {json_path}")
        print(f"Snippet: {txt_path}")
        print("weights:")
        print(weights)
        print("\nPaste this into optimization config:")
        print(snippet)
        print("=================================\n")

        return json_path, txt_path

    def _on_export_click(self, _event) -> None:
        json_path, txt_path = self._export()
        if self.hint_text is not None:
            self.hint_text.set_text(f"Saved: {os.path.basename(json_path)} and {os.path.basename(txt_path)}")
        self.fig.canvas.draw_idle()

    def _on_clear_click(self, _event) -> None:
        self.selected.clear()
        for ci in self.crease_lines:
            self._update_line_style(ci)
        self._update_status()
        if self.hint_text is not None:
            self.hint_text.set_text("Selection cleared")
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        if not self.creases:
            raise ValueError("No valley/mountain creases found in the selected pattern")

        self.fig = plt.figure(figsize=(11.5, 7.2), dpi=120)
        self.ax = self.fig.add_axes([0.05, 0.14, 0.75, 0.80])

        lines_data = self.data.get("lines", [])
        line_features = self.data.get("line_features", [])

        for unit in self.data.get("units", []):
            corners = [(p[0], p[1]) for p in unit]
            self.ax.add_patch(
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
                self.ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    color=BORDER_COLOR,
                    linewidth=1.3,
                    zorder=2,
                )

        for ci, info in enumerate(self.creases):
            li = info["line_index"]
            if li >= len(lines_data):
                continue
            p0, p1 = lines_data[li][0], lines_data[li][1]
            line = self.ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=UNSELECTED_COLOR,
                linewidth=2.6,
                alpha=0.85,
                zorder=3,
                picker=5,
                solid_capstyle="round",
            )[0]
            self.crease_lines[ci] = line

            if self.show_labels:
                mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
                mv = "V" if info["type"] == 0 else "M"
                self.ax.text(
                    mx,
                    my,
                    f"c{ci}:{mv}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#222222",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
                    zorder=4,
                )

        kps = self.data.get("kps", [])
        if kps:
            xs = [kp[0] for kp in kps]
            ys = [kp[1] for kp in kps]
            self.ax.scatter(xs, ys, s=20, c=KP_COLOR, zorder=5, edgecolors="white", linewidths=0.5)

        x0, x1, y0, y1 = _bounds(self.data)
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (mm)")
        self.ax.set_ylabel("y (mm)")
        self.ax.set_title(f"Manual thick-crease selector - {self.name}", fontsize=12, fontweight="bold")
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

        self.fig.canvas.mpl_connect("pick_event", self._on_pick)

        slider_ax = self.fig.add_axes([0.83, 0.62, 0.13, 0.05])
        self.slider = Slider(
            ax=slider_ax,
            label="thick mm",
            valmin=self.min_thickness,
            valmax=self.max_offset,
            valinit=self.thick_thickness,
            valstep=self.discrete_step if self.discrete_step > 0 else None,
            orientation="horizontal",
        )
        self.slider.on_changed(self._on_slider_change)

        export_ax = self.fig.add_axes([0.83, 0.51, 0.13, 0.06])
        export_btn = Button(export_ax, "Export")
        export_btn.on_clicked(self._on_export_click)

        clear_ax = self.fig.add_axes([0.83, 0.42, 0.13, 0.06])
        clear_btn = Button(clear_ax, "Clear")
        clear_btn.on_clicked(self._on_clear_click)

        self.status_text = self.fig.text(0.83, 0.34, "", fontsize=9, ha="left", va="top")
        self.hint_text = self.fig.text(
            0.83,
            0.24,
            "Click creases to toggle.\nPress Export when done.",
            fontsize=9,
            ha="left",
            va="top",
        )
        self._update_status()

        plt.show()


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
            raise FileNotFoundError("No config.yml found in splitting-manual/")

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}, config_file


def run_simulation(sim: dict, base_dir: str) -> int:
    name = sim["name"]
    json_path = _resolve_json_path(name, base_dir)
    if not os.path.exists(json_path):
        print(f"Error: JSON not found: {json_path}")
        return 1

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = sim.get("output_dir", "output")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    gui = ManualSplitGUI(
        data=data,
        name=name,
        output_dir=output_dir,
        min_thickness=float(sim.get("min_thickness", 2.0)),
        thick_thickness=float(sim.get("thick_thickness", 12.0)),
        max_offset=float(sim.get("max_offset", 25.0)),
        discrete_step=float(sim.get("discrete_step", 1.0)),
        show_labels=bool(sim.get("show_labels", True)),
    )
    gui.show()
    return 0


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Interactive manual selector for thicker creases."
    )
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    parser.add_argument(
        "--simulation",
        type=int,
        default=0,
        help="Simulation index in config.simulations (default: 0)",
    )
    args = parser.parse_args()

    try:
        config, config_file = load_config(base_dir, args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Using config: {config_file}")
    sims = config.get("simulations", [])
    if not sims:
        print("No simulations defined in config.")
        return 1

    if args.simulation < 0 or args.simulation >= len(sims):
        print(f"Invalid --simulation index {args.simulation}. Range: 0..{len(sims) - 1}")
        return 1

    sim = sims[args.simulation]
    if not sim.get("name"):
        print("Selected simulation has no 'name'.")
        return 1

    return run_simulation(sim, base_dir)


if __name__ == "__main__":
    sys.exit(main() or 0)
