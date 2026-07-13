"""
Collision-shading runner for phys_sim_pd14 (no trimmed-JSON export).

  headless: true  → use_gui=False (run() auto-drives θ 0→π)
  headless: false → use_gui=True  (interactive GUI)

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
        )

        ori.start(
            filepath=name,
            unit_edge_max=sim.get("unit_edge_max", 4),
            thick_mode=sim.get("thick_mode", False),
        )

        # use_gui → hold angle; headless → auto-fold until stop()
        ori.run()


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
