"""
Collision-shading runner for phys_sim_pd14.

  headless: true  → use_gui=False (run() auto-drives θ 0→π)
  headless: false → use_gui=True  (interactive GUI)

When θ hits π, phys_sim_pd14 auto-writes:
  panel_trimming/trimmedData/<name>-trimmed.json
  (source descriptionData JSON is left alone; dual-curve shaded_regions appended)

Usage:
  python panel-trimming/run_panel_trimming.py
  python panel-trimming/run_panel_trimming.py --name mountain-thick
  python panel-trimming/run_panel_trimming.py --name mountain-thick --headless
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

os.chdir(_ROOT_DIR)


def _load_config(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _is_headless(sim: dict, cli_headless: Optional[bool]) -> bool:
    if cli_headless is not None:
        return bool(cli_headless)
    return bool(sim.get("headless", False))


def run_simulations(
    simulations: list[dict],
    *,
    cli_headless: Optional[bool] = None,
) -> None:
    from phys_sim_pd14 import PD_Origami_Simulator

    if not simulations:
        print("No simulations defined in config.")
        return

    for sim in simulations:
        name = sim["name"]
        headless = _is_headless(sim, cli_headless)
        use_gui = not headless

        print(f"\n{'=' * 60}")
        print(
            f"[collision-shading] Starting: {name}  "
            f"(headless={headless}, collision_shading=True)"
        )
        print(f"{'=' * 60}")

        # thick designs need thick_mode; default True when name contains "thick"
        thick_mode = sim.get("thick_mode")
        if thick_mode is None:
            thick_mode = "thick" in str(name).lower()

        # Collision-only ghost Z shells: one every X mm between physical heights
        # (0 = off). Not a fixed count between shells.
        thick_ghost_spacing_mm = float(sim.get("thick_ghost_spacing_mm", 0.0) or 0.0)
        # Legacy alias: thick_intermediate_layers was a count; ignore if new key set.
        if thick_ghost_spacing_mm <= 0 and sim.get("thick_intermediate_layers") is not None:
            # Old configs used integer count; treat as mm spacing of 1.0 if count>0
            # so old files still get *some* ghosts. Prefer migrating to spacing_mm.
            legacy_n = int(sim.get("thick_intermediate_layers") or 0)
            if legacy_n > 0:
                print(
                    "[collision-shading] WARNING: thick_intermediate_layers is deprecated; "
                    "use thick_ghost_spacing_mm (mm). "
                    f"Interpreting legacy count={legacy_n} as spacing=1.0 mm."
                )
                thick_ghost_spacing_mm = 1.0

        # Vertical side panels (unique indices, collision-only). Default on.
        thick_side_panels = bool(sim.get("thick_side_panels", True))

        ori = PD_Origami_Simulator(
            origami_name=name,
            use_gui=use_gui,
            fast=sim.get("fast", True),
            material_type=sim.get("material_type", 1),
            ref_target=sim.get("ref_target", False),
            damping=sim.get("damping", 0.975),
            pd_local_time=sim.get("pd_local_time", 1),
            pd_global_time=sim.get("pd_global_time", 1),
            pd_iter_time=sim.get("pd_iter_time", 5),
            verbose=sim.get("verbose", False),
            collision_shading=True,
            thick_ghost_spacing_mm=thick_ghost_spacing_mm,
            thick_side_panels=thick_side_panels,
        )

        ori.start(
            filepath=name,
            unit_edge_max=sim.get("unit_edge_max", 4),
            thick_mode=bool(thick_mode),
        )

        # headless: auto 0→π → export; GUI: manual fold, export when θ hits π
        mode = "headless auto-fold" if headless else "GUI manual fold"
        print(
            f"[collision-shading] thick_mode={bool(thick_mode)}; "
            f"thick_ghost_spacing_mm={thick_ghost_spacing_mm:g} (ghost); "
            f"thick_side_panels={thick_side_panels}; "
            f"{mode}; export panel_trimming/trimmedData/{name}-trimmed.json at π"
        )
        ori.run()

        out = getattr(ori, "_trimmed_json_path", None)
        if out and os.path.isfile(out):
            print(f"[collision-shading] Done. Trimmed JSON: {out}")
        elif getattr(ori, "_trimmed_json_exported", False):
            print(f"[collision-shading] Done. Export flag set: {out}")
        else:
            print(
                f"[collision-shading] WARNING: no trimmed JSON written "
                f"(θ={float(getattr(ori, 'folding_angle', 0.0)):.4f}). "
                f"Expected panel_trimming/trimmedData/{name}-trimmed.json"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run phys_sim_pd14 with collision_shading. "
            "headless=true: no GUI (auto-folds 0→π)."
        )
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_THIS_DIR, "config.yml"),
    )
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="true = no GUI; false = GUI. Default: config headless flag.",
    )
    args = parser.parse_args(argv)

    simulations: list[dict] = []
    if os.path.isfile(args.config):
        simulations = list((_load_config(args.config).get("simulations") or []))
    elif not args.name:
        print(f"Config not found: {args.config}")
        return 1

    if args.name:
        matched = [s for s in simulations if s.get("name") == args.name]
        if matched:
            simulations = matched
        else:
            simulations = [{
                "name": args.name,
                "headless": bool(args.headless) if args.headless is not None else False,
                "thick_mode": True,
                "unit_edge_max": 4,
                "fast": True,
            }]
            print(f"[collision-shading] No config entry for '{args.name}'; using defaults.")

    run_simulations(simulations, cli_headless=args.headless)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
