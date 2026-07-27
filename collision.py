"""Panel collision, locus paint, and trimmed-JSON export for PD_Origami_Simulator.

Mixed into PD_Origami_Simulator so @ti.kernel / @ti.func stay on the same
@ti.data_oriented instance that owns the Taichi fields (self.x, coll_hit_*, …).

Architecture (collision_shading=True)
-------------------------------------
0. Side + ghost + support panel builders
1. Index every panel by layer (main → side → ghost ranks)
2. Collision logic
   main & ghost & support → AABB + tri-tri
   side                   → OBB  + tri-tri
3. Sweeping visualization (main, ghost, support, side)  [host]
4. JSON export helper (dual-curve shaded_regions)              [host]

Pure geometry helpers: collision_util.
"""
import json
import os

import numpy as np
import taichi as ti

from collision_util import (
    append_sweep_sample_2d,
    closest_point_on_triangle_3d,
    contact_pair_key,
    copy_loc,
    eval_bary_batch,
    make_side_dict,
    map_points_on_tris_batch,
    order_segment_endpoints_2d,
    pack_dual_curves,
    pack_unit_locus_lines,
    paint_buf_grow,
    paint_buf_new,
    paint_total_strokes,
    panel_group_key,
    polygon_area_2d,
    ribbon_ring_from_curves,
    seg_mid3,
    segment_mid_dist_2d,
    stack_depth_fields,
    sweep_paint_polygon_2d,
    xy2,
)
from utils import (
    Z,
    triangle_intersection_contacts_3d,
)

# Must match phys_sim_pd14.data_type (ti.init happens in the simulator module).
data_type = ti.f64
numpy_data_type = np.float64


@ti.data_oriented
class CollisionMixin:
    """
    Stages 0-4 in one class. phys_sim_pd14 entry points:
      _build_collision_topology, _maybe_detect_panel_collisions,
      detect_panel_collisions, _render_panel_meshes,
      _stop_sweep_drawing_at_pi, _seal_fixed_flat_contacts,
      export_trimmed_json, build_shaded_regions, get_collision_groups
    """


    # ==================================================================
    # 0. Side + ghost builders
    # 1. Index panels by layer (main → side → ghost)
    # ==================================================================

    """Stage 0 (builders) + stage 1 (unified index)."""

    # ------------------------------------------------------------------
    # Panel / layer registry (feeds stage 1 index)
    # ------------------------------------------------------------------

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
        layer metadata for collision paint.
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

    # ------------------------------------------------------------------
    # Stage 1: index main / side / ghost into collision tables
    # ------------------------------------------------------------------

    def _unit_triangle_indices_flat(self, sim_unit_id):
        refs = self.ori_sim.tri_indices_ref
        tri_start = refs[sim_unit_id]
        tri_end = refs[sim_unit_id + 1] if sim_unit_id + 1 < len(refs) else len(self.ori_sim.tri_indices) // 3
        return np.array(self.ori_sim.tri_indices[3 * tri_start:3 * tri_end], dtype=np.int32)

    def _build_collision_topology(self):
        """Stage 0+1 entry: build side/ghost geometry, index all shells, upload Taichi tables."""
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
        # Per-unit layer meta for collision paint
        self._collision_unit_layer_meta = self._build_unit_layer_meta()

        # Stage 0 builders (host geometry → Taichi buffers where applicable)
        # Index order (stage 1): main physical, then side, then ghost ranks.
        if getattr(self, "_flat_kps_np", None) is None:
            self._flat_kps_np = np.asarray(self.kps, dtype=numpy_data_type)
        # Vertical side panels between consecutive physical heights.
        self._build_side_collision_panels()
        # Ghost intermediate shells between physical heights (before support).
        self._build_ghost_collision_shells()
        # Support shells: pad every design panel to the full global height set
        # so each layer has the same panel count (collision-only, like ghosts).
        self._build_support_collision_shells()
        # Ghosts in the gaps between physical and support heights (was missing:
        # ghosts only ran physical→physical, so support extensions had no
        # intermediate collision samples).
        self._build_ghosts_across_support_gaps()
        # Pack draw topology for GUI (green support meshes; ghosts are not drawn).
        self._upload_support_panel_draw_buffers()
        # Dense unit→layer tables after ghost remaps coll layer ranks.
        self._build_unit_layer_arrays()

        # Upload topology into Taichi fields for the contact kernel
        if hasattr(self, "coll_tri_kp"):
            max_tris = int(self._coll_max_tris)
            if num_tris > max_tris:
                print(
                    f"[Contact] num_tris={num_tris} exceeds coll buffer {max_tris}; "
                    "clamping (rebuild with larger mesh budget)."
                )
                num_tris = max_tris
                self._collision_num_tris = num_tris
            tri_panel = np.full(num_tris, -1, dtype=np.int32)
            # Match by full collision-stack layer_idx (physical + ghost ranks).
            # Ghosts are host-side only; physical tris use remapped ranks so the
            # same ordinal still means "same depth sample" across panels.
            coll_layer_map = getattr(self, "_unit_coll_layer_idx", {}) or {}
            unit_layer_meta = getattr(self, "_collision_unit_layer_meta", None) or {}
            tri_layer = np.full(num_tris, -1, dtype=np.int32)
            tri_h = np.zeros(num_tris, dtype=numpy_data_type)
            for t in range(num_tris):
                u = int(tri_unit_ids[t])
                if 0 <= u < len(unit_panel_idx):
                    tri_panel[t] = int(unit_panel_idx[u])
                    if u in coll_layer_map:
                        tri_layer[t] = int(coll_layer_map[u])
                    else:
                        tri_layer[t] = int(unit_layer_idx[u])
                    meta = unit_layer_meta.get(u, {})
                    lh = meta.get("layer_h", meta.get("height_z"))
                    if lh is not None:
                        tri_h[t] = float(lh)
                    else:
                        tri_h[t] = float(self._unit_layer_height_z(u))
            kp_buf = np.zeros((max_tris, 3), dtype=np.int32)
            unit_buf = np.zeros(max_tris, dtype=np.int32)
            panel_buf = np.full(max_tris, -1, dtype=np.int32)
            layer_buf = np.full(max_tris, -1, dtype=np.int32)
            h_buf = np.zeros(max_tris, dtype=numpy_data_type)
            kp_buf[:num_tris] = tri_kp_indices[:num_tris]
            unit_buf[:num_tris] = tri_unit_ids[:num_tris]
            panel_buf[:num_tris] = tri_panel
            layer_buf[:num_tris] = tri_layer
            h_buf[:num_tris] = tri_h
            self.coll_n_tris[None] = int(num_tris)
            self.coll_tri_kp.from_numpy(kp_buf)
            self.coll_tri_unit.from_numpy(unit_buf)
            self.coll_tri_panel.from_numpy(panel_buf)
            self.coll_tri_layer.from_numpy(layer_buf)
            if hasattr(self, "coll_tri_h"):
                self.coll_tri_h.from_numpy(h_buf)

    def _stack_depth_fields(self, panel_idx, layer_h):
        """Depth of a shell height within the panel's physical stock span."""
        span = (getattr(self, "_panel_stock_span", {}) or {}).get(int(panel_idx))
        return stack_depth_fields(span, layer_h)

    def _build_unit_layer_arrays(self):
        """Pack unit layer_idx / layer_h into dense arrays for O(1) hit map."""
        n = int(getattr(self, "unit_indices_num", 0) or 0)
        if n <= 0:
            self._unit_layer_idx_arr = np.zeros(0, dtype=np.int32)
            self._unit_layer_h_arr = np.zeros(0, dtype=numpy_data_type)
            return
        meta = getattr(self, "_collision_unit_layer_meta", None) or {}
        unit_layer_idx = getattr(self, "_collision_unit_layer_idx", None)
        coll_map = getattr(self, "_unit_coll_layer_idx", None) or {}
        idx_arr = np.full(n, -1, dtype=np.int32)
        h_arr = np.zeros(n, dtype=numpy_data_type)
        for u in range(n):
            m = meta.get(u, {})
            if u in coll_map:
                idx_arr[u] = int(coll_map[u])
            elif m.get("layer_idx") is not None:
                idx_arr[u] = int(m["layer_idx"])
            elif unit_layer_idx is not None and u < len(unit_layer_idx):
                idx_arr[u] = int(unit_layer_idx[u])
            lh = m.get("layer_h", m.get("height_z"))
            if lh is not None:
                h_arr[u] = float(lh)
            else:
                try:
                    h_arr[u] = float(self._unit_layer_height_z(u))
                except Exception:
                    h_arr[u] = 0.0
        self._unit_layer_idx_arr = idx_arr
        self._unit_layer_h_arr = h_arr

    def _side_dict_from_map(
        self, unit, panel, layer_idx, layer_h, shell_kind,
        xy=None, loc=None, xy0=None, loc0=None, xy1=None, loc1=None,
        parent_panel=None, h_lo=None, h_hi=None, depth_panel=None,
    ):
        """Build one side_a/side_b dict from batch-mapped xy/loc."""
        dp = int(depth_panel) if depth_panel is not None else int(panel)
        span = (getattr(self, "_panel_stock_span", {}) or {}).get(dp)
        return make_side_dict(
            unit, panel, layer_idx, layer_h, shell_kind,
            xy=xy, loc=loc, xy0=xy0, loc0=loc0, xy1=xy1, loc1=loc1,
            parent_panel=parent_panel, h_lo=h_lo, h_hi=h_hi,
            stock_span=span,
        )

    def _match_unit_kp_correspondence(self, unit_lo, unit_hi, flat_kps):
        """Pair kp indices of two physical shells by design-xy proximity."""
        kps_lo = list(self._unit_kp_indices(unit_lo))
        kps_hi = list(self._unit_kp_indices(unit_hi))
        if len(kps_lo) < 3 or len(kps_hi) < 3 or len(kps_lo) != len(kps_hi):
            # Fall back to equal-length prefix if counts differ slightly
            n = min(len(kps_lo), len(kps_hi))
            if n < 3:
                return None, None
            kps_lo, kps_hi = kps_lo[:n], kps_hi[:n]
        flat = np.asarray(flat_kps, dtype=float)
        matched_hi = []
        used = set()
        for g_lo in kps_lo:
            xy_lo = flat[int(g_lo), :2]
            best_j, best_d = -1, 1e300
            for j, g_hi in enumerate(kps_hi):
                if j in used:
                    continue
                dxy = flat[int(g_hi), :2] - xy_lo
                d = float(dxy[0] * dxy[0] + dxy[1] * dxy[1])
                if d < best_d:
                    best_d, best_j = d, j
            if best_j < 0:
                return None, None
            used.add(best_j)
            matched_hi.append(int(kps_hi[best_j]))
        return [int(k) for k in kps_lo], matched_hi

    def _local_tris_for_unit(self, unit_id, kp_list):
        """Map unit triangle global kps → local indices into kp_list."""
        g2local = {int(g): i for i, g in enumerate(kp_list)}
        tri_arr = self._unit_triangle_indices_flat(unit_id)
        local_tris = []
        for t in range(0, len(tri_arr), 3):
            try:
                i0 = g2local[int(tri_arr[t])]
                i1 = g2local[int(tri_arr[t + 1])]
                i2 = g2local[int(tri_arr[t + 2])]
            except KeyError:
                continue
            local_tris.append((i0, i1, i2))
        return local_tris

    def _build_ghost_collision_shells(self):
        """
        Build collision-only intermediate Z shells between physical thick heights.

        Spacing is controlled by ``thick_ghost_spacing_mm``: one ghost every
        X mm along Z between consecutive physical shells (not a fixed count).
        Ghosts have no mass/springs/creases. Positions each frame are linear
        blends of the two bounding physical shell vertices.
        """
        self._collision_ghost_shells = []
        self._panel_stock_span = {}
        self._unit_coll_layer_idx = {}

        spacing = float(getattr(self, "thick_ghost_spacing_mm", 0.0) or 0.0)
        mapping = self._panel_layer_mapping()
        flat_kps = getattr(self, "_flat_kps_np", None)
        if flat_kps is None:
            flat_kps = np.asarray(self.kps, dtype=numpy_data_type)
            self._flat_kps_np = flat_kps

        n_ghost = 0
        for panel_idx, unit_ids in enumerate(mapping):
            if not unit_ids:
                continue
            phys = []
            for phys_lid, uid in enumerate(unit_ids):
                h = float(self._unit_layer_height_z(uid))
                phys.append({"unit": int(uid), "h": h, "phys_layer_idx": int(phys_lid)})
            phys.sort(key=lambda x: x["h"])
            zs = [p["h"] for p in phys]
            self._panel_stock_span[int(panel_idx)] = (
                (float(min(zs)), float(max(zs))) if zs else (0.0, 0.0)
            )

            stack = []
            for i, p in enumerate(phys):
                stack.append({
                    "kind": "physical",
                    "h": p["h"],
                    "unit": p["unit"],
                })
                if spacing <= 1e-12 or i + 1 >= len(phys):
                    continue
                h0, h1 = float(p["h"]), float(phys[i + 1]["h"])
                u0, u1 = p["unit"], phys[i + 1]["unit"]
                dh = h1 - h0
                if abs(dh) < 1e-12:
                    continue
                # One ghost every ``spacing`` mm strictly between h0 and h1
                # (do not duplicate the physical endpoints).
                h = h0 + spacing
                # Guard against floating-point landing on h1
                while h < h1 - 1e-9:
                    alpha = (h - h0) / dh
                    if alpha <= 1e-12 or alpha >= 1.0 - 1e-12:
                        h += spacing
                        continue
                    stack.append({
                        "kind": "ghost",
                        "h": float(h),
                        "alpha": float(alpha),
                        "unit_lo": int(u0),
                        "unit_hi": int(u1),
                        "panel_idx": int(panel_idx),
                    })
                    h += spacing

            for coll_rank, item in enumerate(stack):
                if item["kind"] == "physical":
                    self._unit_coll_layer_idx[int(item["unit"])] = int(coll_rank)
                    continue
                kp_lo, kp_hi = self._match_unit_kp_correspondence(
                    item["unit_lo"], item["unit_hi"], flat_kps
                )
                if kp_lo is None:
                    continue
                local_tris = self._local_tris_for_unit(item["unit_lo"], kp_lo)
                if not local_tris:
                    continue
                ghost = {
                    "ghost_id": n_ghost,
                    "panel_idx": int(item["panel_idx"]),
                    "layer_h": float(item["h"]),
                    "layer_idx": int(coll_rank),
                    "alpha": float(item["alpha"]),
                    "unit_lo": int(item["unit_lo"]),
                    "unit_hi": int(item["unit_hi"]),
                    "kp_lo": kp_lo,
                    "kp_hi": kp_hi,
                    "local_tris": local_tris,
                    "shell_kind": "ghost",
                }
                self._collision_ghost_shells.append(ghost)
                n_ghost += 1

        if spacing > 1e-12 or getattr(self, "verbose", False):
            print(
                f"[Contact] ghost shells: every {spacing:g} mm → "
                f"{n_ghost} collision-only shell(s) (no PD); "
                f"physical units unchanged."
            )

    def _support_blend_pair(self, phys, target_h):
        """
        Choose two physical shells + blend weight so support sits at target_h.

        phys: sorted list of {unit, h} for one design panel.
        Uses linear interpolation between bracketing shells, or extrapolation
        past the ends (e.g. miura P2 at h=3 from shells at -9 and -3).

        Returns (unit_lo, unit_hi, h_lo, h_hi, alpha) or None.
        """
        if not phys:
            return None
        th = float(target_h)
        if len(phys) == 1:
            # Single shell: degenerate blend (live mesh == that shell).
            # Caller may still offset; alpha=0, same unit twice.
            u = int(phys[0]["unit"])
            h = float(phys[0]["h"])
            return u, u, h, h, 0.0

        # Extrapolate below / above the panel's own stack, else interpolate.
        if th <= float(phys[0]["h"]) + 1e-12:
            lo, hi = phys[0], phys[1]
        elif th >= float(phys[-1]["h"]) - 1e-12:
            lo, hi = phys[-2], phys[-1]
        else:
            lo = phys[0]
            hi = phys[-1]
            for p in phys:
                if float(p["h"]) <= th + 1e-12:
                    lo = p
                if float(p["h"]) >= th - 1e-12:
                    hi = p
                    break

        h0 = float(lo["h"])
        h1 = float(hi["h"])
        u0 = int(lo["unit"])
        u1 = int(hi["unit"])
        if abs(h1 - h0) < 1e-12:
            return u0, u0, h0, h0, 0.0
        # Unclamped alpha: 0 at lo, 1 at hi, >1 / <0 outside the span
        # so missing layers (3 and -9 on miura) land at the right stack depth.
        alpha = (th - h0) / (h1 - h0)
        return u0, u1, h0, h1, float(alpha)

    def _build_support_collision_shells(self):
        """
        Pad every design panel to the full set of global physical heights.

        Example (miura-thick):
          global heights = {-9, -3, 3}
          h=-3 : all four panels have physical shells
          h=3  : only P0,P1 physical → support for P2,P3 (green)
          h=-9 : only P2,P3 physical → support for P0,P1 (green)

        Support shells are collision-only (no PD) and drawn green in the GUI.
        Live geometry blends/extrapolates the panel's own physical shells so
        the mesh sits at the target height, not on top of a neighbor layer.
        """
        self._collision_support_shells = []
        if not bool(getattr(self, "thick_support_panels", True)):
            return
        if not bool(getattr(self, "thick_mode_flag", False)):
            return

        mapping = self._panel_layer_mapping()
        if not mapping:
            return
        flat_kps = getattr(self, "_flat_kps_np", None)
        if flat_kps is None:
            flat_kps = np.asarray(self.kps, dtype=numpy_data_type)
            self._flat_kps_np = flat_kps

        # Per-panel physical shells sorted by height
        per_panel = []
        all_heights = []
        for panel_idx, unit_ids in enumerate(mapping):
            phys = []
            for uid in unit_ids or []:
                h = float(self._unit_layer_height_z(int(uid)))
                phys.append({"unit": int(uid), "h": h})
                all_heights.append(h)
            phys.sort(key=lambda x: x["h"])
            per_panel.append(phys)

        if not all_heights:
            return

        # Unique global physical heights — every layer must host all panels
        height_keys = sorted({round(float(h), 6) for h in all_heights})
        if len(height_keys) < 1:
            return

        # Per height: which design panels are already physical
        panels_at_h = {hk: set() for hk in height_keys}
        for panel_idx, phys in enumerate(per_panel):
            for p in phys:
                panels_at_h[round(float(p["h"]), 6)].add(int(panel_idx))

        n_support = 0
        n_panels = len(mapping)
        for panel_idx, phys in enumerate(per_panel):
            if not phys:
                continue
            present = {round(float(p["h"]), 6) for p in phys}
            for gi, hk in enumerate(height_keys):
                if hk in present:
                    continue
                target_h = float(hk)
                blend = self._support_blend_pair(phys, target_h)
                if blend is None:
                    continue
                u0, u1, h0, h1, alpha = blend

                kp_lo, kp_hi = self._match_unit_kp_correspondence(u0, u1, flat_kps)
                if kp_lo is None:
                    kps = list(self._unit_kp_indices(u0))
                    if len(kps) < 3:
                        continue
                    kp_lo = kps
                    kp_hi = list(self._unit_kp_indices(u1)) if u1 != u0 else list(kps)
                    if len(kp_hi) != len(kp_lo):
                        n = min(len(kp_lo), len(kp_hi))
                        if n < 3:
                            continue
                        kp_lo, kp_hi = kp_lo[:n], kp_hi[:n]
                local_tris = self._local_tris_for_unit(u0, kp_lo)
                if not local_tris and u1 != u0:
                    local_tris = self._local_tris_for_unit(u1, kp_hi)
                if not local_tris:
                    continue

                support = {
                    "support_id": n_support,
                    "panel_idx": int(panel_idx),
                    "layer_h": float(target_h),
                    "layer_idx": int(gi),
                    "alpha": float(alpha),
                    "unit_lo": int(u0),
                    "unit_hi": int(u1),
                    "h_lo": float(h0),
                    "h_hi": float(h1),
                    "kp_lo": kp_lo,
                    "kp_hi": kp_hi,
                    "local_tris": local_tris,
                    "shell_kind": "support",
                }
                self._collision_support_shells.append(support)
                n_support += 1

        # Always log: which heights were incomplete and how many supports filled
        if n_support > 0 or getattr(self, "verbose", False):
            bits = []
            for hk in height_keys:
                have = sorted(panels_at_h.get(hk) or [])
                missing = [p for p in range(n_panels) if p not in (panels_at_h.get(hk) or set())]
                if missing:
                    bits.append(
                        f"h={hk:g}: phys={have} +support P{missing}"
                    )
            detail = "; ".join(bits) if bits else "all layers already full"
            print(
                f"[Contact] support shells: {n_support} green collision-only "
                f"shell(s) across heights {height_keys} "
                f"({detail})"
            )

    def _build_ghosts_across_support_gaps(self):
        """
        Fill intermediate ghost shells between physical and support heights.

        ``_build_ghost_collision_shells`` only samples between consecutive
        **physical** shells. After support pads extend a panel's stack
        (e.g. P0 phys at -3/3 + support at -9), the interval [-9, -3] had
        no ghosts — so intermediate collision with other panels was missing.

        For each panel, take the sorted set of physical + support heights and
        insert ghosts every ``thick_ghost_spacing_mm`` between consecutive
        samples that are not already covered.
        """
        spacing = float(getattr(self, "thick_ghost_spacing_mm", 0.0) or 0.0)
        if spacing <= 1e-12:
            return
        if not bool(getattr(self, "thick_support_panels", True)):
            return
        supports = getattr(self, "_collision_support_shells", None) or []
        if not supports:
            return

        mapping = self._panel_layer_mapping()
        flat_kps = getattr(self, "_flat_kps_np", None)
        if flat_kps is None:
            flat_kps = np.asarray(self.kps, dtype=numpy_data_type)
            self._flat_kps_np = flat_kps

        # Existing ghost heights per panel (avoid duplicates)
        existing_h: Dict[int, List[float]] = {}
        for g in getattr(self, "_collision_ghost_shells", None) or []:
            pi = int(g["panel_idx"])
            existing_h.setdefault(pi, []).append(float(g["layer_h"]))

        support_h: Dict[int, List[float]] = {}
        for s in supports:
            pi = int(s["panel_idx"])
            support_h.setdefault(pi, []).append(float(s["layer_h"]))

        n_new = 0
        n_ghost = len(getattr(self, "_collision_ghost_shells", None) or [])

        for panel_idx, unit_ids in enumerate(mapping):
            if not unit_ids:
                continue
            phys = []
            for uid in unit_ids or []:
                h = float(self._unit_layer_height_z(int(uid)))
                phys.append({"unit": int(uid), "h": h})
            phys.sort(key=lambda x: x["h"])
            if not phys:
                continue

            # Full stack endpoints: physical + support for this panel
            h_set = {round(float(p["h"]), 6) for p in phys}
            for h in support_h.get(panel_idx, []):
                h_set.add(round(float(h), 6))
            heights = sorted(float(h) for h in h_set)
            if len(heights) < 2:
                continue

            # Extend stock span to include support (depth fields / viz)
            span = self._panel_stock_span.get(int(panel_idx), (heights[0], heights[-1]))
            self._panel_stock_span[int(panel_idx)] = (
                float(min(span[0], heights[0])),
                float(max(span[1], heights[-1])),
            )

            have = existing_h.get(panel_idx, [])

            def _already(h_val: float) -> bool:
                for eh in have:
                    if abs(float(eh) - float(h_val)) < 0.5 * spacing:
                        return True
                # also skip physical/support endpoints
                for ep in heights:
                    if abs(float(ep) - float(h_val)) < 1e-9:
                        return True
                return False

            for i in range(len(heights) - 1):
                h0, h1 = float(heights[i]), float(heights[i + 1])
                dh = h1 - h0
                if abs(dh) < 1e-12:
                    continue
                # Only fill gaps that involve support (new intervals beyond
                # pure physical-physical, which already have ghosts).
                # Physical-physical interval: both endpoints are physical.
                phys_hs = {round(float(p["h"]), 6) for p in phys}
                both_phys = (
                    round(h0, 6) in phys_hs and round(h1, 6) in phys_hs
                )
                if both_phys:
                    continue

                h = h0 + spacing
                while h < h1 - 1e-9:
                    if _already(h):
                        h += spacing
                        continue
                    blend = self._support_blend_pair(phys, h)
                    if blend is None:
                        h += spacing
                        continue
                    u0, u1, _ha, _hb, alpha = blend
                    if abs(alpha) < 1e-12 and int(u0) == int(u1):
                        h += spacing
                        continue
                    kp_lo, kp_hi = self._match_unit_kp_correspondence(
                        u0, u1, flat_kps
                    )
                    if kp_lo is None:
                        kps = list(self._unit_kp_indices(u0))
                        if len(kps) < 3:
                            h += spacing
                            continue
                        kp_lo = kps
                        kp_hi = (
                            list(self._unit_kp_indices(u1))
                            if int(u1) != int(u0)
                            else list(kps)
                        )
                        n = min(len(kp_lo), len(kp_hi))
                        if n < 3:
                            h += spacing
                            continue
                        kp_lo, kp_hi = kp_lo[:n], kp_hi[:n]
                    local_tris = self._local_tris_for_unit(u0, kp_lo)
                    if not local_tris and int(u1) != int(u0):
                        local_tris = self._local_tris_for_unit(u1, kp_hi)
                    if not local_tris:
                        h += spacing
                        continue
                    ghost = {
                        "ghost_id": n_ghost,
                        "panel_idx": int(panel_idx),
                        "layer_h": float(h),
                        # Rank by absolute height among global samples later
                        "layer_idx": -1,
                        "alpha": float(alpha),
                        "unit_lo": int(u0),
                        "unit_hi": int(u1),
                        "kp_lo": kp_lo,
                        "kp_hi": kp_hi,
                        "local_tris": local_tris,
                        "shell_kind": "ghost",
                        "support_gap_ghost": True,
                    }
                    self._collision_ghost_shells.append(ghost)
                    have.append(float(h))
                    n_ghost += 1
                    n_new += 1
                    h += spacing

        if n_new > 0 or getattr(self, "verbose", False):
            print(
                f"[Contact] support-gap ghosts: +{n_new} intermediate shell(s) "
                f"between physical and support heights "
                f"(spacing={spacing:g} mm)."
            )

    def _support_shell_live_verts(self, support, positions):
        """
        Place support shell in 3D by blending/extrapolating physical shell verts.

        verts = (1-alpha)*pos_lo + alpha*pos_hi
        alpha may be outside [0,1] so missing layers (e.g. h=3 from -9/-3)
        continue the stack rather than sitting on the nearest physical panel.
        """
        a = float(support.get("alpha", 0.0))
        lo = np.asarray(positions[support["kp_lo"]], dtype=float)
        hi = np.asarray(positions[support["kp_hi"]], dtype=float)
        if int(support["unit_lo"]) == int(support["unit_hi"]):
            # True single-shell panel: cannot invent thickness from one sample.
            return lo.copy()
        return (1.0 - a) * lo + a * hi

    def _upload_support_panel_draw_buffers(self):
        """
        Pack support shell triangle topology into Taichi draw buffers.

        Vertex positions are filled every frame from live physical kp blends
        (see ``_update_support_panel_draw_verts``). Ghost intermediate shells
        are intentionally not drawn — only support pads.
        """
        self._support_draw_meta = []
        self._support_panel_vert_count = 0
        self._support_panel_index_count = 0
        self._support_panel_edge_vert_count = 0
        if not hasattr(self, "support_panel_verts"):
            return
        supports = getattr(self, "_collision_support_shells", None) or []
        if not supports:
            return

        max_v = int(getattr(self, "_support_max_verts", 0) or 0)
        max_i = int(getattr(self, "_support_max_idx", 0) or 0)
        max_e = int(getattr(self, "_support_max_edge_v", 0) or 0)
        if max_v < 3 or max_i < 3:
            return

        idx_list = []
        edge_pairs = []  # packed local vert index pairs for outline edges
        meta = []
        v_off = 0
        n_skipped = 0
        for s in supports:
            kp_lo = list(s.get("kp_lo") or [])
            n_v = len(kp_lo)
            if n_v < 3:
                n_skipped += 1
                continue
            if v_off + n_v > max_v:
                n_skipped += 1
                continue
            local_tris = list(s.get("local_tris") or [])
            added_tri = 0
            for i0, i1, i2 in local_tris:
                if len(idx_list) + 3 > max_i:
                    break
                if (
                    0 <= int(i0) < n_v
                    and 0 <= int(i1) < n_v
                    and 0 <= int(i2) < n_v
                ):
                    idx_list.extend([
                        v_off + int(i0),
                        v_off + int(i1),
                        v_off + int(i2),
                    ])
                    added_tri += 1
            if added_tri == 0:
                n_skipped += 1
                continue
            # Outline edges in kp ring order (unit boundary)
            for i in range(n_v):
                a = v_off + i
                b = v_off + ((i + 1) % n_v)
                if len(edge_pairs) * 2 + 2 <= max_e:
                    edge_pairs.append((a, b))
            meta.append({
                "v_off": int(v_off),
                "n_v": int(n_v),
                "kp_lo": [int(k) for k in kp_lo],
                "kp_hi": [int(k) for k in (s.get("kp_hi") or kp_lo)],
                "alpha": float(s.get("alpha", 0.0)),
                "unit_lo": int(s.get("unit_lo", -1)),
                "unit_hi": int(s.get("unit_hi", -1)),
            })
            v_off += n_v

        self._support_draw_meta = meta
        self._support_panel_vert_count = int(v_off)
        self._support_panel_index_count = int(len(idx_list))
        self._support_edge_pairs = edge_pairs  # packed local vert pairs
        self._support_panel_edge_vert_count = int(len(edge_pairs) * 2)

        idx_buf = np.zeros(max_i, dtype=np.int32)
        if idx_list:
            idx_buf[: len(idx_list)] = np.asarray(idx_list, dtype=np.int32)
        self.support_panel_indices.from_numpy(idx_buf)
        # Zero verts until first frame update
        self.support_panel_verts.fill(0)
        self.support_panel_edge_verts.fill(0)

        if meta:
            print(
                f"[Contact] support draw (GUI green): {len(meta)} shell(s)  "
                f"verts={v_off}  tris={len(idx_list)//3}"
                + (f"  skipped={n_skipped}" if n_skipped else "")
            )

    def _update_support_panel_draw_verts(self, positions=None):
        """Fill support mesh + edge verts from live physical kp blends/extrapolation."""
        meta = getattr(self, "_support_draw_meta", None) or []
        if not meta or not hasattr(self, "support_panel_verts"):
            return
        if positions is None:
            positions = self._cache_frame_positions()
        positions = np.asarray(positions, dtype=float)
        max_v = int(getattr(self, "_support_max_verts", 0) or 0)
        max_e = int(getattr(self, "_support_max_edge_v", 0) or 0)
        n_v = int(getattr(self, "_support_panel_vert_count", 0) or 0)
        if n_v < 3 or max_v < 3:
            return

        V = np.zeros((max_v, 3), dtype=np.float32)
        for m in meta:
            off = int(m["v_off"])
            nv = int(m["n_v"])
            try:
                shell = {
                    "alpha": float(m["alpha"]),
                    "kp_lo": m["kp_lo"],
                    "kp_hi": m["kp_hi"],
                    "unit_lo": m["unit_lo"],
                    "unit_hi": m["unit_hi"],
                }
                verts = self._support_shell_live_verts(shell, positions)
                V[off : off + nv] = np.asarray(verts, dtype=np.float32)
            except Exception:
                continue
        self.support_panel_verts.from_numpy(V)

        # Edge line verts (pairs of packed mesh verts)
        edge_pairs = getattr(self, "_support_edge_pairs", None) or []
        n_e = min(len(edge_pairs), max_e // 2)
        if n_e > 0 and hasattr(self, "support_panel_edge_verts"):
            E = np.zeros((max_e, 3), dtype=np.float32)
            for i, (ia, ib) in enumerate(edge_pairs[:n_e]):
                E[2 * i] = V[int(ia)]
                E[2 * i + 1] = V[int(ib)]
            self.support_panel_edge_verts.from_numpy(E)
            self._support_panel_edge_vert_count = int(n_e * 2)

    def _unit_outline_kp_ordered(self, unit_id):
        """Boundary keypoint indices of a sim unit in outline order."""
        if not hasattr(self, "ori_sim"):
            return []
        raw = self.ori_sim.indices[int(unit_id)]
        return [int(idx) for idx in raw if int(idx) != -1]

    def _build_side_collision_panels(self):
        """
        Build collision-only vertical side panels between consecutive physical
        layer heights of each thick design panel.

        Each band (panel, h_lo → h_hi) becomes one side panel with:
          - a unique panel_idx (after all design panels)
          - edge quads linking corresponding outline verts of the two shells
          - no mass / springs / PD (host-side narrowphase only)

        Closes the open gap between stacked thickness layers so side faces
        participate in contact detection and export/visualization.
        """
        self._collision_side_panels = []
        self._side_panel_index_count = 0
        self._side_panel_edge_vert_count = 0
        self._side_panel_edge_kp_pairs = None
        if not bool(getattr(self, "thick_mode_flag", False)):
            return
        if not bool(getattr(self, "collision_shading", False)):
            return
        if not bool(getattr(self, "thick_side_panels", True)):
            if getattr(self, "verbose", False):
                print("[Contact] side panels: disabled (thick_side_panels=false).")
            # Clear any prior Taichi counts if fields exist
            if hasattr(self, "side_n_tris"):
                self.side_n_tris[None] = 0
                self.side_n_edges[None] = 0
                self.side_hit_count[None] = 0
            return

        mapping = self._panel_layer_mapping()
        n_design = len(mapping)
        flat_kps = getattr(self, "_flat_kps_np", None)
        if flat_kps is None:
            flat_kps = np.asarray(self.kps, dtype=numpy_data_type)
            self._flat_kps_np = flat_kps

        next_panel_idx = int(n_design)
        n_side = 0
        for panel_idx, unit_ids in enumerate(mapping):
            if not unit_ids or len(unit_ids) < 2:
                continue
            phys = []
            for phys_lid, uid in enumerate(unit_ids):
                h = float(self._unit_layer_height_z(uid))
                phys.append({
                    "unit": int(uid),
                    "h": h,
                    "phys_layer_idx": int(phys_lid),
                })
            phys.sort(key=lambda x: x["h"])

            for i in range(len(phys) - 1):
                p0, p1 = phys[i], phys[i + 1]
                h0, h1 = float(p0["h"]), float(p1["h"])
                if abs(h1 - h0) < 1e-9:
                    continue
                u0, u1 = int(p0["unit"]), int(p1["unit"])
                outline_lo = self._unit_outline_kp_ordered(u0)
                if len(outline_lo) < 3:
                    continue
                # Match by design-xy so shared/merged mesh verts still pair correctly
                # even when outline index order differs between layers.
                matched_lo, matched_hi = self._match_unit_kp_correspondence(
                    u0, u1, flat_kps
                )
                if matched_lo is None:
                    continue
                # Reorder matched pairs into outline_lo order for a closed band.
                hi_of = {
                    int(a): int(b) for a, b in zip(matched_lo, matched_hi)
                }
                kp_lo = []
                kp_hi = []
                for g in outline_lo:
                    g = int(g)
                    if g not in hi_of:
                        # nearest matched lo by design xy
                        xy = flat_kps[g, :2]
                        best_a, best_d = None, 1e300
                        for a in matched_lo:
                            dxy = flat_kps[int(a), :2] - xy
                            d = float(dxy[0] * dxy[0] + dxy[1] * dxy[1])
                            if d < best_d:
                                best_d, best_a = d, int(a)
                        if best_a is None or best_a not in hi_of:
                            continue
                        g = best_a
                    kp_lo.append(g)
                    kp_hi.append(hi_of[g])
                n_e = len(kp_lo)
                if n_e < 3 or len(kp_hi) != n_e:
                    continue

                edges = []
                local_tris = []  # flat list of (edge_i, i0, i1, i2) for narrowphase
                for j in range(n_e):
                    j1 = (j + 1) % n_e
                    # Quad corners: lo[j], lo[j1], hi[j1], hi[j]
                    corners = [
                        int(kp_lo[j]),
                        int(kp_lo[j1]),
                        int(kp_hi[j1]),
                        int(kp_hi[j]),
                    ]
                    # Skip degenerate (zero-area) edges in design xy
                    f0 = flat_kps[corners[0], :2]
                    f1 = flat_kps[corners[1], :2]
                    if float(np.dot(f1 - f0, f1 - f0)) < 1e-16:
                        continue
                    edges.append({
                        "kp": corners,
                        "edge_idx": int(j),
                    })
                    ei = len(edges) - 1
                    # Two triangles of the vertical (or near-vertical) wall
                    local_tris.append((ei, 0, 1, 2))
                    local_tris.append((ei, 0, 2, 3))

                if not edges or not local_tris:
                    continue

                # Design-xy footprint: thin strip along the parent panel outline
                # (for export/viz association). Use parent outline at mid height.
                outline_xy = []
                for g in kp_lo:
                    outline_xy.append([
                        float(flat_kps[int(g), 0]),
                        float(flat_kps[int(g), 1]),
                    ])

                side = {
                    "side_id": int(n_side),
                    "panel_idx": int(next_panel_idx),
                    "parent_panel_idx": int(panel_idx),
                    "h_lo": float(h0),
                    "h_hi": float(h1),
                    "layer_h": float(0.5 * (h0 + h1)),
                    "layer_idx": int(i),  # band ordinal within parent
                    "unit_lo": u0,
                    "unit_hi": u1,
                    "edges": edges,
                    "local_tris": local_tris,
                    "outline_xy": outline_xy,
                    "shell_kind": "side",
                }
                self._collision_side_panels.append(side)
                next_panel_idx += 1
                n_side += 1

        if n_side > 0 or getattr(self, "verbose", False):
            print(
                f"[Contact] side panels: {n_side} collision-only vertical "
                f"band(s) (unique panel indices "
                f"{n_design}..{next_panel_idx - 1 if n_side else n_design - 1}; "
                f"no PD)."
            )
        self._upload_side_panel_taichi_buffers()

    def _upload_side_panel_taichi_buffers(self):
        """
        Upload side-panel topology into Taichi fields (once after build).

        Per-frame: kernels read live ``self.x`` for contact + edge lines.
        Faces draw via ``side_panel_indices`` into live ``self.vertices``.
        """
        sides = getattr(self, "_collision_side_panels", None) or []
        if not hasattr(self, "side_n_tris"):
            # collision_shading buffers not allocated
            self._side_panel_index_count = 0
            self._side_panel_edge_vert_count = 0
            return

        max_tris = int(getattr(self, "_side_max_tris", 0) or 0)
        max_edges = int(getattr(self, "_side_max_edges", 0) or 0)
        max_idx = int(self.side_panel_indices.shape[0]) if hasattr(self, "side_panel_indices") else 0

        tri_kp = []
        tri_panel = []
        tri_parent = []
        tri_unit = []
        tri_layer = []
        tri_layer_h = []
        tri_h_lo = []
        tri_h_hi = []
        tri_id = []
        edge_pairs = []
        idx_flat = []
        # frozenset({k0,k1,k2}) → (panel, side_id, edge_j)  — both tris of a
        # wall quad share one face key so paint doesn't jump the mesh diagonal.
        face_from_kps = {}

        stopped = False
        for s in sides:
            if stopped:
                break
            panel = int(s["panel_idx"])
            parent = int(s["parent_panel_idx"])
            unit = int(s.get("unit_lo", -1))
            layer = int(s.get("layer_idx", -1))
            lh = float(s["layer_h"])
            h_lo = float(s["h_lo"])
            h_hi = float(s["h_hi"])
            sid = int(s["side_id"])
            edges = s.get("edges") or []
            for ej, e in enumerate(edges):
                if stopped:
                    break
                kp = e.get("kp") or []
                if len(kp) < 4:
                    continue
                k0, k1, k2, k3 = int(kp[0]), int(kp[1]), int(kp[2]), int(kp[3])
                face_key = (panel, sid, int(ej))
                # two tris of the vertical quad
                for a, b, c in ((k0, k1, k2), (k0, k2, k3)):
                    local_i = len(tri_kp)
                    if max_tris > 0 and local_i >= max_tris:
                        stopped = True
                        break
                    tri_kp.append((a, b, c))
                    tri_panel.append(panel)
                    tri_parent.append(parent)
                    tri_unit.append(unit)
                    tri_layer.append(layer)
                    tri_layer_h.append(lh)
                    tri_h_lo.append(h_lo)
                    tri_h_hi.append(h_hi)
                    tri_id.append(sid * 4096 + local_i)
                    idx_flat.extend([a, b, c])
                    face_from_kps[frozenset((a, b, c))] = face_key
                if stopped:
                    break
                edge_pairs.extend([(k0, k1), (k1, k2), (k2, k3), (k3, k0)])

        n_tris = len(tri_kp)
        n_edges = len(edge_pairs)
        n_idx = len(idx_flat)
        if stopped:
            print(
                f"[Contact] side tris clamped to {max_tris}; "
                "increase mesh budget if needed."
            )
        if max_edges > 0 and n_edges > max_edges:
            edge_pairs = edge_pairs[:max_edges]
            n_edges = max_edges
        if max_idx > 0 and n_idx > max_idx:
            idx_flat = idx_flat[: max_idx - (max_idx % 3)]
            n_idx = len(idx_flat)
            n_tris = n_idx // 3

        self._side_panel_index_count = int(n_idx)
        self._side_panel_edge_vert_count = int(n_edges * 2)
        # Host mirror of tri kp for flat design-xy mapping after kernel hits
        self._side_tri_kp_host = (
            np.asarray(tri_kp[:n_tris], dtype=np.int32)
            if n_tris > 0 else np.zeros((0, 3), dtype=np.int32)
        )

        # --- fill Taichi fields ---
        self.side_n_tris[None] = int(n_tris)
        self.side_n_edges[None] = int(n_edges)

        kp_buf = np.zeros((max_tris, 3), dtype=np.int32)
        panel_buf = np.full(max_tris, -1, dtype=np.int32)
        parent_buf = np.full(max_tris, -1, dtype=np.int32)
        unit_buf = np.full(max_tris, -1, dtype=np.int32)
        layer_buf = np.full(max_tris, -1, dtype=np.int32)
        lh_buf = np.zeros(max_tris, dtype=numpy_data_type)
        hlo_buf = np.zeros(max_tris, dtype=numpy_data_type)
        hhi_buf = np.zeros(max_tris, dtype=numpy_data_type)
        id_buf = np.full(max_tris, -1, dtype=np.int32)
        if n_tris > 0:
            arr = np.asarray(tri_kp[:n_tris], dtype=np.int32)
            kp_buf[:n_tris] = arr
            panel_buf[:n_tris] = np.asarray(tri_panel[:n_tris], dtype=np.int32)
            parent_buf[:n_tris] = np.asarray(tri_parent[:n_tris], dtype=np.int32)
            unit_buf[:n_tris] = np.asarray(tri_unit[:n_tris], dtype=np.int32)
            layer_buf[:n_tris] = np.asarray(tri_layer[:n_tris], dtype=np.int32)
            lh_buf[:n_tris] = np.asarray(tri_layer_h[:n_tris], dtype=numpy_data_type)
            hlo_buf[:n_tris] = np.asarray(tri_h_lo[:n_tris], dtype=numpy_data_type)
            hhi_buf[:n_tris] = np.asarray(tri_h_hi[:n_tris], dtype=numpy_data_type)
            id_buf[:n_tris] = np.asarray(tri_id[:n_tris], dtype=np.int32)
        self.side_tri_kp.from_numpy(kp_buf)
        self.side_tri_panel.from_numpy(panel_buf)
        self.side_tri_parent.from_numpy(parent_buf)
        self.side_tri_unit.from_numpy(unit_buf)
        self.side_tri_layer.from_numpy(layer_buf)
        self.side_tri_layer_h.from_numpy(lh_buf)
        self.side_tri_h_lo.from_numpy(hlo_buf)
        self.side_tri_h_hi.from_numpy(hhi_buf)
        self.side_tri_id.from_numpy(id_buf)
        # Host mirrors for Phase-2 batch postprocess (avoid per-frame field download)
        self._side_tri_kp_host = kp_buf[:n_tris].copy() if n_tris > 0 else np.zeros((0, 3), dtype=np.int32)
        self._side_tri_id_host = id_buf[:n_tris].copy() if n_tris > 0 else np.zeros(0, dtype=np.int32)
        self._side_face_from_kps = face_from_kps

        idx_buf = np.zeros(max_idx, dtype=np.int32)
        if n_idx > 0:
            idx_buf[:n_idx] = np.asarray(idx_flat[:n_idx], dtype=np.int32)
        self.side_panel_indices.from_numpy(idx_buf)

        edge_buf = np.zeros((max_edges, 2), dtype=np.int32)
        if n_edges > 0:
            edge_buf[:n_edges] = np.asarray(edge_pairs[:n_edges], dtype=np.int32)
        self.side_edge_kp.from_numpy(edge_buf)
        self.side_panel_edge_verts.fill(0)
        self.side_hit_count[None] = 0

    @ti.kernel
    def _kernel_update_side_panel_edge_lines(self):
        """Taichi: copy live self.x → side wall outline line verts (f32)."""
        n = self.side_n_edges[None]
        for i in range(n):
            a = self.side_edge_kp[i, 0]
            b = self.side_edge_kp[i, 1]
            self.side_panel_edge_verts[2 * i] = ti.cast(self.x[a], ti.f32)
            self.side_panel_edge_verts[2 * i + 1] = ti.cast(self.x[b], ti.f32)

    def _ghost_shell_live_verts(self, ghost, positions):
        """Blend live physical shell verts → ghost shell vertex array (n,3)."""
        a = float(ghost["alpha"])
        lo = np.asarray(positions[ghost["kp_lo"]], dtype=float)
        hi = np.asarray(positions[ghost["kp_hi"]], dtype=float)
        return (1.0 - a) * lo + a * hi

    # ==================================================================
    # 2. Collision logic (main/ghost: AABB+tri-tri, side: OBB+tri-tri)
    # ==================================================================

    @ti.kernel
    def _kernel_detect_side_side_contacts(
        self, hit_eps: data_type, aabb_pad: data_type, prox_eps: data_type,
    ):
        """
        Taichi narrowphase: side panel ↔ side panel (different parents).
        Sensitive: allow coplanar, expanded AABB, proximity fallback.
        Extra cull: per-tri OBB (tight under rotation) before contact_ex.
        Resets side_hit_* then writes hits.
        """
        self.side_hit_count[None] = 0
        n = self.side_n_tris[None]
        max_hits = self.side_hit_kind.shape[0]
        for ta, tb in ti.ndrange(n, n):
            if tb <= ta:
                continue
            parent_a = self.side_tri_parent[ta]
            parent_b = self.side_tri_parent[tb]
            if parent_a < 0 or parent_b < 0 or parent_a == parent_b:
                continue
            panel_a = self.side_tri_panel[ta]
            panel_b = self.side_tri_panel[tb]
            if panel_a == panel_b:
                continue
            i0 = self.side_tri_kp[ta, 0]
            i1 = self.side_tri_kp[ta, 1]
            i2 = self.side_tri_kp[ta, 2]
            j0 = self.side_tri_kp[tb, 0]
            j1 = self.side_tri_kp[tb, 1]
            j2 = self.side_tri_kp[tb, 2]
            # Skip only shared *edges* (2+ verts). A single shared corner is
            # common at creases and must still be tested.
            share = 0
            if i0 == j0 or i0 == j1 or i0 == j2:
                share += 1
            if i1 == j0 or i1 == j1 or i1 == j2:
                share += 1
            if i2 == j0 or i2 == j1 or i2 == j2:
                share += 1
            if share >= 2:
                continue
            a0 = self.x[i0]
            a1 = self.x[i1]
            a2 = self.x[i2]
            b0 = self.x[j0]
            b1 = self.x[j1]
            b2 = self.x[j2]
            # Oriented cull: thin rotated walls have fat world AABBs; OBB is
            # tight in the wall frame. Pad covers existing aabb/prox tolerance
            # so sensitivity knobs are unchanged — only false far pairs drop.
            obb_pad = ti.max(aabb_pad, prox_eps)
            if self._ti_side_obb_cull(a0, a1, a2, b0, b1, b2, obb_pad) == 0:
                continue
            # Side walls are often near-coplanar — allow coplanar + optional prox
            kind, p0, p1 = self._ti_tri_tri_contact_ex(
                a0, a1, a2, b0, b1, b2, hit_eps,
                aabb_pad, 1, prox_eps,
            )
            if kind == 0:
                continue
            idx = ti.atomic_add(self.side_hit_count[None], 1)
            if idx < max_hits:
                # stable order by panel index
                if panel_a <= panel_b:
                    self.side_hit_kind[idx] = kind
                    self.side_hit_p0[idx] = p0
                    self.side_hit_p1[idx] = p1
                    self.side_hit_tri_a[idx] = ta
                    self.side_hit_tri_b[idx] = tb
                    self.side_hit_unit_a[idx] = self.side_tri_unit[ta]
                    self.side_hit_unit_b[idx] = self.side_tri_unit[tb]
                    self.side_hit_panel_a[idx] = panel_a
                    self.side_hit_panel_b[idx] = panel_b
                    self.side_hit_parent_a[idx] = parent_a
                    self.side_hit_parent_b[idx] = parent_b
                    self.side_hit_layer_a[idx] = self.side_tri_layer[ta]
                    self.side_hit_layer_b[idx] = self.side_tri_layer[tb]
                    self.side_hit_layer_h_a[idx] = self.side_tri_layer_h[ta]
                    self.side_hit_layer_h_b[idx] = self.side_tri_layer_h[tb]
                    self.side_hit_h_lo_a[idx] = self.side_tri_h_lo[ta]
                    self.side_hit_h_hi_a[idx] = self.side_tri_h_hi[ta]
                else:
                    self.side_hit_kind[idx] = kind
                    self.side_hit_p0[idx] = p0
                    self.side_hit_p1[idx] = p1
                    self.side_hit_tri_a[idx] = tb
                    self.side_hit_tri_b[idx] = ta
                    self.side_hit_unit_a[idx] = self.side_tri_unit[tb]
                    self.side_hit_unit_b[idx] = self.side_tri_unit[ta]
                    self.side_hit_panel_a[idx] = panel_b
                    self.side_hit_panel_b[idx] = panel_a
                    self.side_hit_parent_a[idx] = parent_b
                    self.side_hit_parent_b[idx] = parent_a
                    self.side_hit_layer_a[idx] = self.side_tri_layer[tb]
                    self.side_hit_layer_b[idx] = self.side_tri_layer[ta]
                    self.side_hit_layer_h_a[idx] = self.side_tri_layer_h[tb]
                    self.side_hit_layer_h_b[idx] = self.side_tri_layer_h[ta]
                    self.side_hit_h_lo_a[idx] = self.side_tri_h_lo[tb]
                    self.side_hit_h_hi_a[idx] = self.side_tri_h_hi[tb]
                self.side_hit_partner_kind[idx] = 0  # side↔side

    @ti.kernel
    def _kernel_detect_side_phys_contacts(
        self, hit_eps: data_type, aabb_pad: data_type, prox_eps: data_type,
    ):
        """
        Taichi narrowphase: side panel ↔ physical shells (other parents).
        Sensitive: allow coplanar, expanded Z band, proximity fallback.
        Extra cull: per-tri OBB before contact_ex (same pad as side↔side).
        Appends into side_hit_* (does not reset count).
        """
        n_s = self.side_n_tris[None]
        n_p = self.coll_n_tris[None]
        max_hits = self.side_hit_kind.shape[0]
        for ta, tb in ti.ndrange(n_s, n_p):
            parent = self.side_tri_parent[ta]
            panel_p = self.coll_tri_panel[tb]
            if parent < 0 or panel_p < 0 or panel_p == parent:
                continue
            # Z-band: physical shells near the wall height span
            lh = self.coll_tri_h[tb]
            h_lo = self.side_tri_h_lo[ta]
            h_hi = self.side_tri_h_hi[ta]
            zmin = ti.min(h_lo, h_hi)
            zmax = ti.max(h_lo, h_hi)
            span = zmax - zmin
            margin = ti.max(data_type(0.15), data_type(0.05) * span)
            if lh < zmin - margin or lh > zmax + margin:
                continue
            i0 = self.side_tri_kp[ta, 0]
            i1 = self.side_tri_kp[ta, 1]
            i2 = self.side_tri_kp[ta, 2]
            j0 = self.coll_tri_kp[tb, 0]
            j1 = self.coll_tri_kp[tb, 1]
            j2 = self.coll_tri_kp[tb, 2]
            # Skip only shared edges (2+ verts); single shared corner OK
            share = 0
            if i0 == j0 or i0 == j1 or i0 == j2:
                share += 1
            if i1 == j0 or i1 == j1 or i1 == j2:
                share += 1
            if i2 == j0 or i2 == j1 or i2 == j2:
                share += 1
            if share >= 2:
                continue
            a0 = self.x[i0]
            a1 = self.x[i1]
            a2 = self.x[i2]
            b0 = self.x[j0]
            b1 = self.x[j1]
            b2 = self.x[j2]
            # Same OBB cull as side↔side (no sensitivity changes)
            obb_pad = ti.max(aabb_pad, prox_eps)
            if self._ti_side_obb_cull(a0, a1, a2, b0, b1, b2, obb_pad) == 0:
                continue
            # Allow coplanar + proximity so walls register against shells
            kind, p0, p1 = self._ti_tri_tri_contact_ex(
                a0, a1, a2, b0, b1, b2, hit_eps,
                aabb_pad, 1, prox_eps,
            )
            if kind == 0:
                continue
            idx = ti.atomic_add(self.side_hit_count[None], 1)
            if idx < max_hits:
                self.side_hit_kind[idx] = kind
                self.side_hit_p0[idx] = p0
                self.side_hit_p1[idx] = p1
                self.side_hit_tri_a[idx] = ta
                self.side_hit_tri_b[idx] = tb
                self.side_hit_unit_a[idx] = self.side_tri_unit[ta]
                self.side_hit_unit_b[idx] = self.coll_tri_unit[tb]
                self.side_hit_panel_a[idx] = self.side_tri_panel[ta]
                self.side_hit_panel_b[idx] = panel_p
                self.side_hit_parent_a[idx] = parent
                self.side_hit_parent_b[idx] = panel_p
                self.side_hit_layer_a[idx] = self.side_tri_layer[ta]
                self.side_hit_layer_b[idx] = self.coll_tri_layer[tb]
                self.side_hit_layer_h_a[idx] = self.side_tri_layer_h[ta]
                self.side_hit_layer_h_b[idx] = lh
                self.side_hit_h_lo_a[idx] = h_lo
                self.side_hit_h_hi_a[idx] = h_hi
                self.side_hit_partner_kind[idx] = 1  # side↔physical

    def _detect_side_panel_contacts(
        self,
        positions,
        flat_kps,
        contact_points,
        contact_segments,
        contact_points_flat,
        contact_segments_flat,
        max_pts,
        max_segs,
    ):
        """
        Side-panel contacts via Taichi kernels; Python only maps hits → design xy.

        Tests:
          1) side ↔ side (different parents)
          2) side ↔ physical shells of other design panels (Z-band overlap)
        """
        if not hasattr(self, "side_n_tris"):
            return
        if not bool(getattr(self, "thick_side_panels", True)):
            return
        n_side = int(self.side_n_tris[None])
        if n_side <= 0:
            return

        # Side walls need milder filters than physical shells: often near-coplanar
        # and share a crease corner with neighbors. Still no huge proximity balloon.
        ms = float(getattr(self, "max_size", 100.0) or 100.0)
        hit_eps = max(5e-4, ms * 5e-6)
        aabb_pad = max(0.02, hit_eps * 10.0)
        # Slightly tighter than prior (0.08 / 8e-4) — OBB cull handles rotation;
        # less prox balloon cuts residual false near-miss paint.
        prox_eps = max(0.05, ms * 5e-4)
        self._kernel_detect_side_side_contacts(hit_eps, aabb_pad, prox_eps)
        if hasattr(self, "coll_n_tris") and int(self.coll_n_tris[None]) > 0:
            self._kernel_detect_side_phys_contacts(hit_eps, aabb_pad, prox_eps)

        n_hits_raw = int(self.side_hit_count[None])
        max_hits = int(self.side_hit_kind.shape[0])
        n_hits = min(n_hits_raw, max_hits)
        if n_hits <= 0:
            return

        # Phase-2: single to_numpy per field
        kinds = self.side_hit_kind.to_numpy()[:n_hits]
        p0s = self.side_hit_p0.to_numpy()[:n_hits]
        p1s = self.side_hit_p1.to_numpy()[:n_hits]
        tri_as = self.side_hit_tri_a.to_numpy()[:n_hits]
        tri_bs = self.side_hit_tri_b.to_numpy()[:n_hits]
        unit_as = self.side_hit_unit_a.to_numpy()[:n_hits]
        unit_bs = self.side_hit_unit_b.to_numpy()[:n_hits]
        panel_as = self.side_hit_panel_a.to_numpy()[:n_hits]
        panel_bs = self.side_hit_panel_b.to_numpy()[:n_hits]
        parent_as = self.side_hit_parent_a.to_numpy()[:n_hits]
        parent_bs = self.side_hit_parent_b.to_numpy()[:n_hits]
        layer_as = self.side_hit_layer_a.to_numpy()[:n_hits]
        layer_bs = self.side_hit_layer_b.to_numpy()[:n_hits]
        lh_as = self.side_hit_layer_h_a.to_numpy()[:n_hits]
        lh_bs = self.side_hit_layer_h_b.to_numpy()[:n_hits]
        hlo_as = self.side_hit_h_lo_a.to_numpy()[:n_hits]
        hhi_as = self.side_hit_h_hi_a.to_numpy()[:n_hits]
        pkinds = self.side_hit_partner_kind.to_numpy()[:n_hits]

        side_kp = getattr(self, "_side_tri_kp_host", None)
        if side_kp is None:
            side_kp = np.asarray(self.side_tri_kp.to_numpy(), dtype=np.int32)
            self._side_tri_kp_host = side_kp
        phys_kp = getattr(self, "_collision_tri_kp_indices", None)
        side_id_host = getattr(self, "_side_tri_id_host", None)
        if side_id_host is None:
            side_id_host = np.asarray(self.side_tri_id.to_numpy(), dtype=np.int32)
            self._side_tri_id_host = side_id_host

        n_side_tris = int(side_kp.shape[0]) if side_kp is not None else 0
        n_phys = int(phys_kp.shape[0]) if phys_kp is not None else 0

        # Resolve triangle kp rows + synthetic ids once; batch-map all jobs
        job_p = []
        job_tri = []
        job_hi = []
        job_tag = []  # 0=p0A 1=p1A 2=p0B 3=p1B
        hit_meta = [None] * n_hits  # (ka, kb, id_a, id_b, sk_b) or None if skip

        for hi in range(n_hits):
            kind = int(kinds[hi])
            if kind <= 0:
                continue
            ta = int(tri_as[hi])
            tb = int(tri_bs[hi])
            partner = int(pkinds[hi])
            if ta < 0 or ta >= n_side_tris:
                continue
            ka = side_kp[ta]
            id_a = int(side_id_host[ta]) if ta < len(side_id_host) else ta
            if partner == 0:
                if tb < 0 or tb >= n_side_tris:
                    continue
                kb = side_kp[tb]
                id_b = int(side_id_host[tb]) if tb < len(side_id_host) else tb
                sk_b = "side"
            else:
                if phys_kp is None or tb < 0 or tb >= n_phys:
                    continue
                kb = phys_kp[tb]
                id_b = int(tb)
                sk_b = "physical"
            hit_meta[hi] = (ka, kb, id_a, id_b, sk_b, partner)

            p0 = p0s[hi]
            job_p.append(p0)
            job_tri.append(ka)
            job_hi.append(hi)
            job_tag.append(0)
            job_p.append(p0)
            job_tri.append(kb)
            job_hi.append(hi)
            job_tag.append(2)
            if kind >= 2:
                p1 = p1s[hi]
                job_p.append(p1)
                job_tri.append(ka)
                job_hi.append(hi)
                job_tag.append(1)
                job_p.append(p1)
                job_tri.append(kb)
                job_hi.append(hi)
                job_tag.append(3)

        map_A0 = [None] * n_hits
        map_A1 = [None] * n_hits
        map_B0 = [None] * n_hits
        map_B1 = [None] * n_hits
        if job_p:
            P = np.asarray(job_p, dtype=np.float64)
            T = np.asarray(job_tri, dtype=np.intp)
            xy_b, loc_b, ok_b = map_points_on_tris_batch(
                P, T, positions, flat_kps
            )
            for j in range(len(job_hi)):
                if not bool(ok_b[j]):
                    continue
                hi = int(job_hi[j])
                tag = int(job_tag[j])
                cell = (xy_b[j], loc_b[j])
                if tag == 0:
                    map_A0[hi] = cell
                elif tag == 1:
                    map_A1[hi] = cell
                elif tag == 2:
                    map_B0[hi] = cell
                else:
                    map_B1[hi] = cell

        # Side walls: kind=2 segments often land on the wall-quad mesh diagonal
        # (two edge hits). Always collapse to a midpoint contact for sides so
        # red markers + paint never draw that chord. Detection still registers.
        for hi in range(n_hits):
            meta = hit_meta[hi]
            if meta is None:
                continue
            kind = int(kinds[hi])
            if kind <= 0:
                continue
            if kind >= 2:
                p0s[hi, 0] = 0.5 * (float(p0s[hi, 0]) + float(p1s[hi, 0]))
                p0s[hi, 1] = 0.5 * (float(p0s[hi, 1]) + float(p1s[hi, 1]))
                p0s[hi, 2] = 0.5 * (float(p0s[hi, 2]) + float(p1s[hi, 2]))
                p1s[hi] = p0s[hi]
                kinds[hi] = 1
                kind = 1
                map_A1[hi] = None
                map_B1[hi] = None
                # Remap would need mid bary; keep map_A0/B0 (near mid when short)
            ka, kb, id_a, id_b, sk_b, partner = meta
            p_side = int(panel_as[hi])
            p_other = int(panel_bs[hi])
            parent = int(parent_as[hi])
            parent_b = int(parent_bs[hi])
            u_a = int(unit_as[hi])
            u_b = int(unit_bs[hi])
            li_a = int(layer_as[hi])
            li_b = int(layer_bs[hi])
            lh_a = float(lh_as[hi])
            lh_b = float(lh_bs[hi])
            h_lo = float(hlo_as[hi])
            h_hi = float(hhi_as[hi])
            p_lo = min(p_side, p_other)
            p_hi = max(p_side, p_other)
            depth = self._stack_depth_fields(parent, lh_a)

            if kind == 1:
                p0 = [float(p0s[hi, 0]), float(p0s[hi, 1]), float(p0s[hi, 2])]
                if len(contact_points) >= max_pts:
                    continue
                contact_points.append(p0)
                ent = {
                    "unit_a": u_a,
                    "unit_b": u_b,
                    "panel_of_unit_a": p_side,
                    "panel_of_unit_b": p_other,
                    "panel_a": p_lo,
                    "panel_b": p_hi,
                    "layer_idx": li_a,
                    "layer_h": lh_a,
                    "layer_idx_a": li_a,
                    "layer_h_a": lh_a,
                    "layer_idx_b": li_b,
                    "layer_h_b": lh_b,
                    "tri_a": id_a,
                    "tri_b": id_b,
                    "tri_lo": id_a,
                    "tri_hi": id_b,
                    "shell_kind": "side",
                    "shell_kind_a": "side",
                    "shell_kind_b": sk_b,
                    "parent_panel_a": parent,
                    "parent_panel_b": parent_b,
                    "p_3d": p0,
                    **depth,
                }
                # Promote point hits to degenerate segments so trail/paint/stamp
                # (which key on p0/p1) include side walls — most wall contacts
                # arrive as kind=1 points under strict filters.
                if map_A0[hi] is not None:
                    xy, loc = map_A0[hi]
                    sa = self._side_dict_from_map(
                        u_a, p_side, li_a, lh_a, "side",
                        xy=xy, loc=loc,
                        xy0=xy, loc0=loc, xy1=xy, loc1=loc,
                        parent_panel=parent, h_lo=h_lo, h_hi=h_hi,
                        depth_panel=parent,
                    )
                    if sa is not None:
                        # point form + segment form for stamp/track
                        sa["p"] = [float(xy[0]), float(xy[1])]
                        ent["side_a"] = sa
                        ent["p"] = sa["p"]
                        ent["p0"] = sa["p0"]
                        ent["p1"] = sa["p1"]
                if map_B0[hi] is not None:
                    xy, loc = map_B0[hi]
                    hlo_b = h_lo if partner == 0 else lh_b
                    hhi_b = h_hi if partner == 0 else lh_b
                    sb = self._side_dict_from_map(
                        u_b, p_other, li_b, lh_b, sk_b,
                        xy=xy, loc=loc,
                        xy0=xy, loc0=loc, xy1=xy, loc1=loc,
                        parent_panel=parent_b if sk_b == "side" else None,
                        h_lo=hlo_b if sk_b == "side" else None,
                        h_hi=hhi_b if sk_b == "side" else None,
                        depth_panel=parent_b if sk_b == "side" else p_other,
                    )
                    if sb is not None:
                        sb["p"] = [float(xy[0]), float(xy[1])]
                        ent["side_b"] = sb
                        if "p" not in ent:
                            ent["p"] = sb["p"]
                        if "p0" not in ent:
                            ent["p0"] = sb["p0"]
                            ent["p1"] = sb["p1"]
                contact_points_flat.append(ent)
                # Also register as a segment so active_segments + paint stamp run
                if (
                    len(contact_segments) < max_segs
                    and ("side_a" in ent or "side_b" in ent)
                    and "p0" in ent
                ):
                    contact_segments.append((p0, p0))
                    ent_seg = dict(ent)
                    ent_seg["p0_3d"] = p0
                    ent_seg["p1_3d"] = p0
                    contact_segments_flat.append(ent_seg)
            else:
                p0 = [float(p0s[hi, 0]), float(p0s[hi, 1]), float(p0s[hi, 2])]
                p1 = [float(p1s[hi, 0]), float(p1s[hi, 1]), float(p1s[hi, 2])]
                if len(contact_segments) >= max_segs:
                    continue
                contact_segments.append((p0, p1))
                ent = {
                    "unit_a": u_a,
                    "unit_b": u_b,
                    "panel_of_unit_a": p_side,
                    "panel_of_unit_b": p_other,
                    "panel_a": p_lo,
                    "panel_b": p_hi,
                    "layer_idx": li_a,
                    "layer_h": lh_a,
                    "layer_idx_a": li_a,
                    "layer_h_a": lh_a,
                    "layer_idx_b": li_b,
                    "layer_h_b": lh_b,
                    "tri_a": id_a,
                    "tri_b": id_b,
                    "tri_lo": id_a,
                    "tri_hi": id_b,
                    "shell_kind": "side",
                    "shell_kind_a": "side",
                    "shell_kind_b": sk_b,
                    "parent_panel_a": parent,
                    "parent_panel_b": parent_b,
                    "p0_3d": p0,
                    "p1_3d": p1,
                    **depth,
                }
                if map_A0[hi] is not None and map_A1[hi] is not None:
                    xy0, loc0 = map_A0[hi]
                    xy1, loc1 = map_A1[hi]
                    sa = self._side_dict_from_map(
                        u_a, p_side, li_a, lh_a, "side",
                        xy0=xy0, loc0=loc0, xy1=xy1, loc1=loc1,
                        parent_panel=parent, h_lo=h_lo, h_hi=h_hi,
                        depth_panel=parent,
                    )
                    if sa is not None:
                        ent["side_a"] = sa
                        ent["p0"], ent["p1"] = sa["p0"], sa["p1"]
                if map_B0[hi] is not None and map_B1[hi] is not None:
                    xy0, loc0 = map_B0[hi]
                    xy1, loc1 = map_B1[hi]
                    hlo_b = h_lo if partner == 0 else lh_b
                    hhi_b = h_hi if partner == 0 else lh_b
                    sb = self._side_dict_from_map(
                        u_b, p_other, li_b, lh_b, sk_b,
                        xy0=xy0, loc0=loc0, xy1=xy1, loc1=loc1,
                        parent_panel=parent_b if sk_b == "side" else None,
                        h_lo=hlo_b if sk_b == "side" else None,
                        h_hi=hhi_b if sk_b == "side" else None,
                        depth_panel=parent_b if sk_b == "side" else p_other,
                    )
                    if sb is not None:
                        ent["side_b"] = sb
                        if "p0" not in ent:
                            ent["p0"], ent["p1"] = sb["p0"], sb["p1"]
                contact_segments_flat.append(ent)
                if len(contact_points) < max_pts:
                    contact_points.append(p0)
                if len(contact_points) < max_pts:
                    contact_points.append(p1)

    def _flat_from_ghost_bary(self, p3d, ghost, verts, i0, i1, i2, flat_kps):
        """Map a 3D ghost contact onto design xy via barycentric on blended tri."""
        a, b, c = verts[i0], verts[i1], verts[i2]
        _q, u, v, w = closest_point_on_triangle_3d(p3d, a, b, c)
        k0 = int(ghost["kp_lo"][i0])
        k1 = int(ghost["kp_lo"][i1])
        k2 = int(ghost["kp_lo"][i2])
        f0 = np.asarray(flat_kps[k0, :2], dtype=float)
        f1 = np.asarray(flat_kps[k1, :2], dtype=float)
        f2 = np.asarray(flat_kps[k2, :2], dtype=float)
        xy = u * f0 + v * f1 + w * f2
        return {
            "p": [float(xy[0]), float(xy[1])],
            "loc": (k0, k1, k2, float(u), float(v), float(w)),
        }

    def _annotate_side_depth(self, side, panel_idx, layer_h, shell_kind):
        """Attach shell_kind + stack depth fields to a contact side dict."""
        if side is None:
            return
        side["shell_kind"] = shell_kind
        side.update(self._stack_depth_fields(panel_idx, layer_h))

    def _detect_ghost_shell_contacts(
        self,
        positions,
        flat_kps,
        contact_points,
        contact_segments,
        contact_points_flat,
        contact_segments_flat,
        max_pts,
        max_segs,
    ):
        """
        Host-side narrowphase for collision-only ghost shells.

        Ghosts at the same absolute layer_h on different panels are tested with
        triangle_intersection_contacts_3d. Grouping by height (not per-panel
        stack rank) so support-gap ghosts pair with other panels' samples.
        No physics involvement.
        """
        ghosts = getattr(self, "_collision_ghost_shells", None) or []
        if not ghosts:
            return

        spacing = float(getattr(self, "thick_ghost_spacing_mm", 0.0) or 0.0)
        # Bucket width for height matching (support-gap ghosts use absolute h)
        h_tol = max(0.5 * spacing, 1e-4) if spacing > 1e-12 else 1e-3

        # Prepare live geometry once per ghost
        prepared = []
        for g in ghosts:
            verts = self._ghost_shell_live_verts(g, positions)
            tris = []
            for ti, (i0, i1, i2) in enumerate(g["local_tris"]):
                tris.append({
                    "local": (i0, i1, i2),
                    "tri_idx": int(g["ghost_id"]) * 1024 + ti,  # synthetic id
                    "corners": [
                        verts[i0].tolist(),
                        verts[i1].tolist(),
                        verts[i2].tolist(),
                    ],
                })
            prepared.append({"g": g, "verts": verts, "tris": tris})

        # Group by absolute height (rounded). Fall back to layer_idx only when
        # height is missing.
        by_h = {}
        for prep in prepared:
            lh = prep["g"].get("layer_h")
            if lh is not None:
                # Quantize to spacing grid when available
                if spacing > 1e-12:
                    key = round(float(lh) / spacing) * spacing
                    key = round(key, 6)
                else:
                    key = round(float(lh), 6)
            else:
                key = ("idx", int(prep["g"].get("layer_idx", -1)))
            by_h.setdefault(key, []).append(prep)

        for h_key, group in by_h.items():
            n = len(group)
            for ia in range(n):
                for ib in range(ia + 1, n):
                    pa = group[ia]
                    pb = group[ib]
                    ga = pa["g"]
                    gb = pb["g"]
                    if int(ga["panel_idx"]) == int(gb["panel_idx"]):
                        continue
                    # Extra height check when bucket is coarse
                    if abs(float(ga["layer_h"]) - float(gb["layer_h"])) > h_tol * 2:
                        continue
                    for ta in pa["tris"]:
                        for tb in pb["tris"]:
                            pts = triangle_intersection_contacts_3d(
                                ta["corners"], tb["corners"], eps=1e-4
                            )
                            if not pts:
                                continue
                            # Dedup → point or segment
                            if len(pts) == 1:
                                kind = 1
                                p0 = [float(pts[0][0]), float(pts[0][1]), float(pts[0][2])]
                                p1 = None
                            else:
                                kind = 2
                                p0 = [float(pts[0][0]), float(pts[0][1]), float(pts[0][2])]
                                p1 = [float(pts[-1][0]), float(pts[-1][1]), float(pts[-1][2])]

                            # Stable unit order for keys (swap both geometry handles)
                            left, right = pa, pb
                            t_left, t_right = ta, tb
                            if int(left["g"]["unit_lo"]) > int(right["g"]["unit_lo"]):
                                left, right = right, left
                                t_left, t_right = t_right, t_left

                            gL, gR = left["g"], right["g"]
                            u_lo = int(gL["unit_lo"])
                            u_hi = int(gR["unit_lo"])
                            p_of_ulo = int(gL["panel_idx"])
                            p_of_uhi = int(gR["panel_idx"])
                            layer_h_a = float(gL["layer_h"])
                            layer_h_b = float(gR["layer_h"])
                            layer_idx_a = int(gL.get("layer_idx", -1))
                            layer_idx_b = int(gR.get("layer_idx", -1))
                            p_lo = min(p_of_ulo, p_of_uhi)
                            p_hi = max(p_of_ulo, p_of_uhi)
                            i0a, i1a, i2a = t_left["local"]
                            i0b, i1b, i2b = t_right["local"]

                            pair_meta = {
                                "unit_a": u_lo,
                                "unit_b": u_hi,
                                "panel_of_unit_a": p_of_ulo,
                                "panel_of_unit_b": p_of_uhi,
                                "panel_a": p_lo,
                                "panel_b": p_hi,
                                "layer_idx": layer_idx_a,
                                "layer_h": layer_h_a,
                                "layer_idx_a": layer_idx_a,
                                "layer_h_a": layer_h_a,
                                "layer_idx_b": layer_idx_b,
                                "layer_h_b": layer_h_b,
                                "tri_a": int(t_left["tri_idx"]),
                                "tri_b": int(t_right["tri_idx"]),
                                "tri_lo": int(t_left["tri_idx"]),
                                "tri_hi": int(t_right["tri_idx"]),
                                "shell_kind": "ghost",
                                "shell_kind_a": "ghost",
                                "shell_kind_b": "ghost",
                            }

                            def _side_for(g, verts, loc_i, p3d, panel, lh, sk):
                                flat = self._flat_from_ghost_bary(
                                    p3d, g, verts, loc_i[0], loc_i[1], loc_i[2], flat_kps
                                )
                                side = {
                                    "unit": int(g["unit_lo"]),
                                    "panel": int(panel),
                                    "layer_idx": int(g["layer_idx"]),
                                    "layer_h": float(lh),
                                    "shell_kind": sk,
                                }
                                return flat, side

                            if kind == 1:
                                if len(contact_points) >= max_pts:
                                    continue
                                contact_points.append(p0)
                                ent = dict(pair_meta)
                                fa, sa = _side_for(
                                    gL, left["verts"], (i0a, i1a, i2a), p0,
                                    p_of_ulo, layer_h_a, "ghost",
                                )
                                fb, sb = _side_for(
                                    gR, right["verts"], (i0b, i1b, i2b), p0,
                                    p_of_uhi, layer_h_b, "ghost",
                                )
                                sa["p"] = fa["p"]
                                sa["loc"] = fa["loc"]
                                sb["p"] = fb["p"]
                                sb["loc"] = fb["loc"]
                                self._annotate_side_depth(sa, p_of_ulo, layer_h_a, "ghost")
                                self._annotate_side_depth(sb, p_of_uhi, layer_h_b, "ghost")
                                ent["side_a"] = sa
                                ent["side_b"] = sb
                                ent["p"] = fa["p"]
                                ent["p_3d"] = p0
                                ent.update(self._stack_depth_fields(p_of_ulo, layer_h_a))
                                contact_points_flat.append(ent)
                            else:
                                if len(contact_segments) >= max_segs:
                                    continue
                                contact_segments.append((p0, p1))
                                ent = dict(pair_meta)
                                fa0, sa = _side_for(
                                    gL, left["verts"], (i0a, i1a, i2a), p0,
                                    p_of_ulo, layer_h_a, "ghost",
                                )
                                fa1, _ = _side_for(
                                    gL, left["verts"], (i0a, i1a, i2a), p1,
                                    p_of_ulo, layer_h_a, "ghost",
                                )
                                fb0, sb = _side_for(
                                    gR, right["verts"], (i0b, i1b, i2b), p0,
                                    p_of_uhi, layer_h_b, "ghost",
                                )
                                fb1, _ = _side_for(
                                    gR, right["verts"], (i0b, i1b, i2b), p1,
                                    p_of_uhi, layer_h_b, "ghost",
                                )
                                sa["p0"], sa["p1"] = fa0["p"], fa1["p"]
                                sa["loc0"], sa["loc1"] = fa0["loc"], fa1["loc"]
                                sb["p0"], sb["p1"] = fb0["p"], fb1["p"]
                                sb["loc0"], sb["loc1"] = fb0["loc"], fb1["loc"]
                                self._annotate_side_depth(sa, p_of_ulo, layer_h_a, "ghost")
                                self._annotate_side_depth(sb, p_of_uhi, layer_h_b, "ghost")
                                ent["side_a"] = sa
                                ent["side_b"] = sb
                                ent["p0"], ent["p1"] = fa0["p"], fa1["p"]
                                ent["p0_3d"] = p0
                                ent["p1_3d"] = p1
                                ent.update(self._stack_depth_fields(p_of_ulo, layer_h_a))
                                contact_segments_flat.append(ent)
                                if len(contact_points) < max_pts:
                                    contact_points.append(p0)
                                if len(contact_points) < max_pts:
                                    contact_points.append(p1)

    def _detect_support_shell_contacts(
        self,
        positions,
        flat_kps,
        contact_points,
        contact_segments,
        contact_points_flat,
        contact_segments_flat,
        max_pts,
        max_segs,
    ):
        """
        Host-side narrowphase for collision-only support shells.

        Support shells pad missing design panels at each global physical height
        so every layer has the same panel count. Tests:
          - support ↔ support (same layer_h, different panels)
          - support ↔ physical (same layer_h, different panels)
          - support ↔ ghost (same layer_h — support-gap ghosts included)
        No physics involvement.
        """
        supports = getattr(self, "_collision_support_shells", None) or []
        if not supports:
            return

        def _prep_shell(shell, verts, tris_local, shell_kind, synthetic_base):
            tris = []
            for ti_, (i0, i1, i2) in enumerate(tris_local):
                tris.append({
                    "local": (i0, i1, i2),
                    "tri_idx": int(synthetic_base) * 1024 + ti_,
                    "corners": [
                        verts[i0].tolist(),
                        verts[i1].tolist(),
                        verts[i2].tolist(),
                    ],
                })
            return {
                "shell": shell,
                "verts": verts,
                "tris": tris,
                "shell_kind": shell_kind,
            }

        # Support geometry
        by_h = {}
        for s in supports:
            verts = self._support_shell_live_verts(s, positions)
            prep = _prep_shell(
                s, verts, s["local_tris"], "support",
                100000 + int(s["support_id"]),
            )
            hk = round(float(s["layer_h"]), 6)
            by_h.setdefault(hk, []).append(prep)

        # Physical shells at the same global heights (for support↔physical)
        mapping = self._panel_layer_mapping()
        coll_map = getattr(self, "_unit_coll_layer_idx", None) or {}
        meta = getattr(self, "_collision_unit_layer_meta", None) or {}
        for panel_idx, unit_ids in enumerate(mapping):
            for uid in unit_ids or []:
                uid = int(uid)
                h = float(self._unit_layer_height_z(uid))
                hk = round(h, 6)
                if hk not in by_h:
                    # No support shells at this height — still may be useful but
                    # support↔physical only matters where supports exist.
                    continue
                kps = list(self._unit_kp_indices(uid))
                if len(kps) < 3:
                    continue
                local_tris = self._local_tris_for_unit(uid, kps)
                if not local_tris:
                    continue
                verts = np.asarray(positions[kps], dtype=float)
                layer_idx = int(coll_map.get(uid, meta.get(uid, {}).get("layer_idx", 0)))
                phys_shell = {
                    "panel_idx": int(panel_idx),
                    "layer_h": float(h),
                    "layer_idx": layer_idx,
                    "unit_lo": uid,
                    "unit_hi": uid,
                    "kp_lo": kps,
                    "kp_hi": kps,
                    "alpha": 0.0,
                    "shell_kind": "physical",
                }
                prep = _prep_shell(
                    phys_shell, verts, local_tris, "physical",
                    200000 + uid,
                )
                by_h[hk].append(prep)

        # Ghost shells near support heights (support-gap + regular ghosts)
        # so support↔ghost intermediate contacts are detected.
        spacing = float(getattr(self, "thick_ghost_spacing_mm", 0.0) or 0.0)
        h_tol = max(0.5 * spacing, 1e-3) if spacing > 1e-12 else 1e-3
        support_heights = list(by_h.keys())
        for g in getattr(self, "_collision_ghost_shells", None) or []:
            lh = float(g.get("layer_h", 0.0))
            best_hk = None
            best_d = 1e300
            for hk in support_heights:
                d = abs(float(hk) - lh)
                if d < best_d:
                    best_d = d
                    best_hk = hk
            if best_hk is None or best_d > h_tol * 2:
                continue
            verts = self._ghost_shell_live_verts(g, positions)
            prep = _prep_shell(
                g, verts, g["local_tris"], "ghost",
                300000 + int(g.get("ghost_id", 0)),
            )
            by_h[best_hk].append(prep)

        def _side_for(shell, verts, loc_i, p3d, panel, lh, sk):
            # Reuse ghost bary mapper (same kp_lo / verts layout)
            flat = self._flat_from_ghost_bary(
                p3d, shell, verts, loc_i[0], loc_i[1], loc_i[2], flat_kps
            )
            side = {
                "unit": int(shell.get("unit_lo", -1)),
                "panel": int(panel),
                "layer_idx": int(shell.get("layer_idx", -1)),
                "layer_h": float(lh),
                "shell_kind": sk,
            }
            return flat, side

        def _emit_pair(left, right, t_left, t_right, kind, p0, p1, layer_h):
            gL, gR = left["shell"], right["shell"]
            sk_a = left["shell_kind"]
            sk_b = right["shell_kind"]
            # Prefer support as the primary shell_kind when either side is support
            if sk_a == "support" or sk_b == "support":
                pair_sk = "support"
            else:
                pair_sk = sk_a
            u_lo = int(gL["unit_lo"])
            u_hi = int(gR["unit_lo"])
            p_of_ulo = int(gL["panel_idx"])
            p_of_uhi = int(gR["panel_idx"])
            layer_h_a = float(gL["layer_h"])
            layer_h_b = float(gR["layer_h"])
            layer_idx_a = int(gL.get("layer_idx", -1))
            layer_idx_b = int(gR.get("layer_idx", -1))
            p_lo = min(p_of_ulo, p_of_uhi)
            p_hi = max(p_of_ulo, p_of_uhi)
            i0a, i1a, i2a = t_left["local"]
            i0b, i1b, i2b = t_right["local"]
            pair_meta = {
                "unit_a": u_lo,
                "unit_b": u_hi,
                "panel_of_unit_a": p_of_ulo,
                "panel_of_unit_b": p_of_uhi,
                "panel_a": p_lo,
                "panel_b": p_hi,
                "layer_idx": int(gL.get("layer_idx", -1)),
                "layer_h": float(layer_h),
                "layer_idx_a": layer_idx_a,
                "layer_h_a": layer_h_a,
                "layer_idx_b": layer_idx_b,
                "layer_h_b": layer_h_b,
                "tri_a": int(t_left["tri_idx"]),
                "tri_b": int(t_right["tri_idx"]),
                "tri_lo": int(t_left["tri_idx"]),
                "tri_hi": int(t_right["tri_idx"]),
                "shell_kind": pair_sk,
                "shell_kind_a": sk_a,
                "shell_kind_b": sk_b,
            }
            if kind == 1:
                if len(contact_points) >= max_pts:
                    return
                contact_points.append(p0)
                ent = dict(pair_meta)
                fa, sa = _side_for(
                    gL, left["verts"], (i0a, i1a, i2a), p0,
                    p_of_ulo, layer_h_a, sk_a,
                )
                fb, sb = _side_for(
                    gR, right["verts"], (i0b, i1b, i2b), p0,
                    p_of_uhi, layer_h_b, sk_b,
                )
                sa["p"] = fa["p"]
                sa["loc"] = fa["loc"]
                sb["p"] = fb["p"]
                sb["loc"] = fb["loc"]
                self._annotate_side_depth(sa, p_of_ulo, layer_h_a, sk_a)
                self._annotate_side_depth(sb, p_of_uhi, layer_h_b, sk_b)
                ent["side_a"] = sa
                ent["side_b"] = sb
                ent["p"] = fa["p"]
                ent["p_3d"] = p0
                ent.update(self._stack_depth_fields(p_of_ulo, layer_h_a))
                contact_points_flat.append(ent)
            else:
                if len(contact_segments) >= max_segs:
                    return
                contact_segments.append((p0, p1))
                ent = dict(pair_meta)
                fa0, sa = _side_for(
                    gL, left["verts"], (i0a, i1a, i2a), p0,
                    p_of_ulo, layer_h_a, sk_a,
                )
                fa1, _ = _side_for(
                    gL, left["verts"], (i0a, i1a, i2a), p1,
                    p_of_ulo, layer_h_a, sk_a,
                )
                fb0, sb = _side_for(
                    gR, right["verts"], (i0b, i1b, i2b), p0,
                    p_of_uhi, layer_h_b, sk_b,
                )
                fb1, _ = _side_for(
                    gR, right["verts"], (i0b, i1b, i2b), p1,
                    p_of_uhi, layer_h_b, sk_b,
                )
                sa["p0"], sa["p1"] = fa0["p"], fa1["p"]
                sa["loc0"], sa["loc1"] = fa0["loc"], fa1["loc"]
                sb["p0"], sb["p1"] = fb0["p"], fb1["p"]
                sb["loc0"], sb["loc1"] = fb0["loc"], fb1["loc"]
                self._annotate_side_depth(sa, p_of_ulo, layer_h_a, sk_a)
                self._annotate_side_depth(sb, p_of_uhi, layer_h_b, sk_b)
                ent["side_a"] = sa
                ent["side_b"] = sb
                ent["p0"], ent["p1"] = fa0["p"], fa1["p"]
                ent["p0_3d"] = p0
                ent["p1_3d"] = p1
                ent.update(self._stack_depth_fields(p_of_ulo, layer_h_a))
                contact_segments_flat.append(ent)
                if len(contact_points) < max_pts:
                    contact_points.append(p0)
                if len(contact_points) < max_pts:
                    contact_points.append(p1)

        for hk, group in by_h.items():
            n = len(group)
            # Only run pairs that involve at least one support shell
            for ia in range(n):
                for ib in range(ia + 1, n):
                    pa = group[ia]
                    pb = group[ib]
                    # Skip pure physical↔physical (already handled by Taichi)
                    if pa["shell_kind"] == "physical" and pb["shell_kind"] == "physical":
                        continue
                    # Skip pure ghost↔ghost (handled in _detect_ghost_shell_contacts)
                    if pa["shell_kind"] == "ghost" and pb["shell_kind"] == "ghost":
                        continue
                    # Require support on at least one side
                    if (
                        pa["shell_kind"] != "support"
                        and pb["shell_kind"] != "support"
                    ):
                        continue
                    if int(pa["shell"]["panel_idx"]) == int(pb["shell"]["panel_idx"]):
                        continue
                    for ta in pa["tris"]:
                        for tb in pb["tris"]:
                            pts = triangle_intersection_contacts_3d(
                                ta["corners"], tb["corners"], eps=1e-4
                            )
                            if not pts:
                                continue
                            if len(pts) == 1:
                                kind = 1
                                p0 = [float(pts[0][0]), float(pts[0][1]), float(pts[0][2])]
                                p1 = None
                            else:
                                kind = 2
                                p0 = [float(pts[0][0]), float(pts[0][1]), float(pts[0][2])]
                                p1 = [float(pts[-1][0]), float(pts[-1][1]), float(pts[-1][2])]
                            left, right = pa, pb
                            t_left, t_right = ta, tb
                            if int(left["shell"]["unit_lo"]) > int(right["shell"]["unit_lo"]):
                                left, right = right, left
                                t_left, t_right = t_right, t_left
                            _emit_pair(
                                left, right, t_left, t_right,
                                kind, p0, p1, float(hk),
                            )

    @ti.func
    def _ti_v3(self, x, y, z):
        return ti.Vector([x, y, z], dt=data_type)

    @ti.func
    def _ti_point_in_triangle(
        self, p: ti.template(), a: ti.template(), b: ti.template(), c: ti.template(),
        eps: data_type,
    ) -> ti.i32:
        n = (b - a).cross(c - a)
        nn = n.norm()
        ok = 0
        if nn >= 1e-12:
            n = n / nn
            if ti.abs((p - a).dot(n)) <= eps:
                v0 = b - a
                v1 = c - a
                v2 = p - a
                d00 = v0.dot(v0)
                d01 = v0.dot(v1)
                d11 = v1.dot(v1)
                d20 = v2.dot(v0)
                d21 = v2.dot(v1)
                denom = d00 * d11 - d01 * d01
                if ti.abs(denom) >= 1e-18:
                    v = (d11 * d20 - d01 * d21) / denom
                    w = (d00 * d21 - d01 * d20) / denom
                    u = data_type(1.0) - v - w
                    tol = eps * data_type(10.0)
                    if u >= -tol and v >= -tol and w >= -tol:
                        ok = 1
        return ok

    @ti.func
    def _ti_seg_tri_intersect(
        self, s0: ti.template(), s1: ti.template(),
        a: ti.template(), b: ti.template(), c: ti.template(),
        eps: data_type,
    ):
        n = (b - a).cross(c - a)
        nn = n.norm()
        hit = 0
        p = self._ti_v3(0.0, 0.0, 0.0)
        if nn >= 1e-12:
            n = n / nn
            d = s1 - s0
            denom = n.dot(d)
            if ti.abs(denom) >= eps:
                t = n.dot(a - s0) / denom
                if t >= -eps and t <= data_type(1.0) + eps:
                    cand = s0 + t * d
                    if self._ti_point_in_triangle(cand, a, b, c, eps * data_type(10.0)) == 1:
                        hit = 1
                        p = cand
        return hit, p

    @ti.func
    def _ti_tris_coplanar(
        self, a0: ti.template(), a1: ti.template(), a2: ti.template(),
        b0: ti.template(), b1: ti.template(), b2: ti.template(),
        eps: data_type,
    ) -> ti.i32:
        n = (a1 - a0).cross(a2 - a0)
        nn = n.norm()
        coplanar = 0
        if nn < 1e-12:
            coplanar = 1
        else:
            n = n / nn
            coplanar = 1
            if ti.abs((b0 - a0).dot(n)) > eps:
                coplanar = 0
            if ti.abs((b1 - a0).dot(n)) > eps:
                coplanar = 0
            if ti.abs((b2 - a0).dot(n)) > eps:
                coplanar = 0
        return coplanar

    @ti.func
    def _ti_aabb_overlap_tris(
        self, a0: ti.template(), a1: ti.template(), a2: ti.template(),
        b0: ti.template(), b1: ti.template(), b2: ti.template(),
        eps: data_type,
    ) -> ti.i32:
        ok = 1
        for k in ti.static(range(3)):
            amin = ti.min(a0[k], ti.min(a1[k], a2[k]))
            amax = ti.max(a0[k], ti.max(a1[k], a2[k]))
            bmin = ti.min(b0[k], ti.min(b1[k], b2[k]))
            bmax = ti.max(b0[k], ti.max(b1[k], b2[k]))
            if amax < bmin - eps or bmax < amin - eps:
                ok = 0
        return ok

    @ti.func
    def _ti_tri_obb(
        self, a0: ti.template(), a1: ti.template(), a2: ti.template(),
        pad: data_type,
    ):
        """
        Tight OBB for one triangle in its own frame (edge, in-plane ortho, normal).

        Returns (valid, center, ax0, ax1, ax2, half_extents).
        valid=0 → skip OBB cull (degenerate tri); caller falls through to AABB path.
        half_extents are expanded by ``pad`` so existing prox/pad contacts still pass.
        """
        valid = 1
        e0 = a1 - a0
        len0 = e0.norm()
        n = e0.cross(a2 - a0)
        nn = n.norm()
        c = self._ti_v3(0.0, 0.0, 0.0)
        ax0 = self._ti_v3(1.0, 0.0, 0.0)
        ax1 = self._ti_v3(0.0, 1.0, 0.0)
        ax2 = self._ti_v3(0.0, 0.0, 1.0)
        h = self._ti_v3(0.0, 0.0, 0.0)
        if len0 < data_type(1e-14) or nn < data_type(1e-14):
            valid = 0
        else:
            ax0 = e0 / len0
            ax2 = n / nn
            ax1 = ax2.cross(ax0)
            # Project the three verts onto the local frame (origin at a0)
            p0_0 = data_type(0.0)
            p0_1 = data_type(0.0)
            p0_2 = data_type(0.0)
            d1 = a1 - a0
            p1_0 = d1.dot(ax0)
            p1_1 = d1.dot(ax1)
            p1_2 = d1.dot(ax2)
            d2 = a2 - a0
            p2_0 = d2.dot(ax0)
            p2_1 = d2.dot(ax1)
            p2_2 = d2.dot(ax2)
            mn0 = ti.min(p0_0, ti.min(p1_0, p2_0))
            mx0 = ti.max(p0_0, ti.max(p1_0, p2_0))
            mn1 = ti.min(p0_1, ti.min(p1_1, p2_1))
            mx1 = ti.max(p0_1, ti.max(p1_1, p2_1))
            mn2 = ti.min(p0_2, ti.min(p1_2, p2_2))
            mx2 = ti.max(p0_2, ti.max(p1_2, p2_2))
            mid0 = data_type(0.5) * (mn0 + mx0)
            mid1 = data_type(0.5) * (mn1 + mx1)
            mid2 = data_type(0.5) * (mn2 + mx2)
            c = a0 + ax0 * mid0 + ax1 * mid1 + ax2 * mid2
            # Flat tris need a small thickness; pad keeps prox-near pairs alive
            thick = ti.max(pad, data_type(1e-6))
            h = self._ti_v3(
                data_type(0.5) * (mx0 - mn0) + pad,
                data_type(0.5) * (mx1 - mn1) + pad,
                data_type(0.5) * (mx2 - mn2) + thick,
            )
        return valid, c, ax0, ax1, ax2, h

    @ti.func
    def _ti_obb_overlap(
        self,
        cA: ti.template(), axA0: ti.template(), axA1: ti.template(),
        axA2: ti.template(), hA: ti.template(),
        cB: ti.template(), axB0: ti.template(), axB1: ti.template(),
        axB2: ti.template(), hB: ti.template(),
    ) -> ti.i32:
        """
        Separating-axis OBB–OBB test (Ericson). Returns 1 if overlap, 0 if separate.
        Used only as a cull for side-panel pairs (does not change contact math).
        """
        ok = 1
        eps = data_type(1e-6)
        # Rotation of B in A's frame: R[i][j] = axA_i · axB_j
        r00 = axA0.dot(axB0)
        r01 = axA0.dot(axB1)
        r02 = axA0.dot(axB2)
        r10 = axA1.dot(axB0)
        r11 = axA1.dot(axB1)
        r12 = axA1.dot(axB2)
        r20 = axA2.dot(axB0)
        r21 = axA2.dot(axB1)
        r22 = axA2.dot(axB2)
        ar00 = ti.abs(r00) + eps
        ar01 = ti.abs(r01) + eps
        ar02 = ti.abs(r02) + eps
        ar10 = ti.abs(r10) + eps
        ar11 = ti.abs(r11) + eps
        ar12 = ti.abs(r12) + eps
        ar20 = ti.abs(r20) + eps
        ar21 = ti.abs(r21) + eps
        ar22 = ti.abs(r22) + eps
        t = cB - cA
        t0 = t.dot(axA0)
        t1 = t.dot(axA1)
        t2 = t.dot(axA2)
        # A's face axes
        if ti.abs(t0) > hA[0] + (hB[0] * ar00 + hB[1] * ar01 + hB[2] * ar02):
            ok = 0
        if ok == 1 and ti.abs(t1) > hA[1] + (hB[0] * ar10 + hB[1] * ar11 + hB[2] * ar12):
            ok = 0
        if ok == 1 and ti.abs(t2) > hA[2] + (hB[0] * ar20 + hB[1] * ar21 + hB[2] * ar22):
            ok = 0
        # B's face axes
        if ok == 1:
            tb0 = t0 * r00 + t1 * r10 + t2 * r20
            if ti.abs(tb0) > hB[0] + (hA[0] * ar00 + hA[1] * ar10 + hA[2] * ar20):
                ok = 0
        if ok == 1:
            tb1 = t0 * r01 + t1 * r11 + t2 * r21
            if ti.abs(tb1) > hB[1] + (hA[0] * ar01 + hA[1] * ar11 + hA[2] * ar21):
                ok = 0
        if ok == 1:
            tb2 = t0 * r02 + t1 * r12 + t2 * r22
            if ti.abs(tb2) > hB[2] + (hA[0] * ar02 + hA[1] * ar12 + hA[2] * ar22):
                ok = 0
        # A_i × B_j edge axes
        if ok == 1:
            # A0 × B0
            ra = hA[1] * ar20 + hA[2] * ar10
            rb = hB[1] * ar02 + hB[2] * ar01
            if ti.abs(t2 * r10 - t1 * r20) > ra + rb:
                ok = 0
        if ok == 1:
            # A0 × B1
            ra = hA[1] * ar21 + hA[2] * ar11
            rb = hB[0] * ar02 + hB[2] * ar00
            if ti.abs(t2 * r11 - t1 * r21) > ra + rb:
                ok = 0
        if ok == 1:
            # A0 × B2
            ra = hA[1] * ar22 + hA[2] * ar12
            rb = hB[0] * ar01 + hB[1] * ar00
            if ti.abs(t2 * r12 - t1 * r22) > ra + rb:
                ok = 0
        if ok == 1:
            # A1 × B0
            ra = hA[0] * ar20 + hA[2] * ar00
            rb = hB[1] * ar12 + hB[2] * ar11
            if ti.abs(t0 * r20 - t2 * r00) > ra + rb:
                ok = 0
        if ok == 1:
            # A1 × B1
            ra = hA[0] * ar21 + hA[2] * ar01
            rb = hB[0] * ar12 + hB[2] * ar10
            if ti.abs(t0 * r21 - t2 * r01) > ra + rb:
                ok = 0
        if ok == 1:
            # A1 × B2
            ra = hA[0] * ar22 + hA[2] * ar02
            rb = hB[0] * ar11 + hB[1] * ar10
            if ti.abs(t0 * r22 - t2 * r02) > ra + rb:
                ok = 0
        if ok == 1:
            # A2 × B0
            ra = hA[0] * ar10 + hA[1] * ar00
            rb = hB[1] * ar22 + hB[2] * ar21
            if ti.abs(t1 * r00 - t0 * r10) > ra + rb:
                ok = 0
        if ok == 1:
            # A2 × B1
            ra = hA[0] * ar11 + hA[1] * ar01
            rb = hB[0] * ar22 + hB[2] * ar20
            if ti.abs(t1 * r01 - t0 * r11) > ra + rb:
                ok = 0
        if ok == 1:
            # A2 × B2
            ra = hA[0] * ar12 + hA[1] * ar02
            rb = hB[0] * ar21 + hB[1] * ar20
            if ti.abs(t1 * r02 - t0 * r12) > ra + rb:
                ok = 0
        return ok

    @ti.func
    def _ti_side_obb_cull(
        self, a0: ti.template(), a1: ti.template(), a2: ti.template(),
        b0: ti.template(), b1: ti.template(), b2: ti.template(),
        pad: data_type,
    ) -> ti.i32:
        """
        1 = pair may contact (run existing narrowphase), 0 = reject.
        Degenerate tris → 1 (fall back to AABB inside contact_ex).
        """
        allow = 1
        va, cA, axA0, axA1, axA2, hA = self._ti_tri_obb(a0, a1, a2, pad)
        vb, cB, axB0, axB1, axB2, hB = self._ti_tri_obb(b0, b1, b2, pad)
        if va == 1 and vb == 1:
            allow = self._ti_obb_overlap(
                cA, axA0, axA1, axA2, hA,
                cB, axB0, axB1, axB2, hB,
            )
        return allow

    @ti.func
    def _ti_share_vert(
        self, i0: ti.i32, i1: ti.i32, i2: ti.i32,
        j0: ti.i32, j1: ti.i32, j2: ti.i32,
    ) -> ti.i32:
        shared = 0
        if (
            i0 == j0 or i0 == j1 or i0 == j2
            or i1 == j0 or i1 == j1 or i1 == j2
            or i2 == j0 or i2 == j1 or i2 == j2
        ):
            shared = 1
        return shared

    @ti.func
    def _ti_push_contact_pt(
        self, n_hits: ti.i32,
        q0: ti.template(), q1: ti.template(),
        p: ti.template(), dedupe: data_type,
    ):
        """
        Maintain contact set as 0/1/2 points (segment endpoints).
        Returns updated (n_hits, q0, q1).
        """
        out_n = n_hits
        out0 = q0
        out1 = q1
        if n_hits == 0:
            out0 = p
            out_n = 1
        elif n_hits == 1:
            if (p - q0).norm() > dedupe:
                out1 = p
                out_n = 2
        else:
            d = q1 - q0
            dn = d.norm()
            if dn < 1e-14:
                if (p - q0).norm() > dedupe:
                    out1 = p
            else:
                dirv = d / dn
                t0 = q0.dot(dirv)
                t1 = q1.dot(dirv)
                tp = p.dot(dirv)
                # Keep extreme projections (contact line endpoints)
                if tp < t0:
                    out0 = p
                elif tp > t1:
                    out1 = p
                # else interior of current segment — ignore
        return out_n, out0, out1

    @ti.func
    def _ti_closest_on_triangle(
        self, p: ti.template(), a: ti.template(), b: ti.template(), c: ti.template(),
    ):
        """Closest point on triangle abc to p (Ericson). Returns (q, dist)."""
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = ab.dot(ap)
        d2 = ac.dot(ap)
        q = a
        if d1 <= 0.0 and d2 <= 0.0:
            q = a
        else:
            bp = p - b
            d3 = ab.dot(bp)
            d4 = ac.dot(bp)
            if d3 >= 0.0 and d4 <= d3:
                q = b
            else:
                vc = d1 * d4 - d3 * d2
                if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                    v = d1 / (d1 - d3) if ti.abs(d1 - d3) > 1e-18 else data_type(0.0)
                    q = a + v * ab
                else:
                    cp = p - c
                    d5 = ab.dot(cp)
                    d6 = ac.dot(cp)
                    if d6 >= 0.0 and d5 <= d6:
                        q = c
                    else:
                        vb = d5 * d2 - d1 * d6
                        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                            w = d2 / (d2 - d6) if ti.abs(d2 - d6) > 1e-18 else data_type(0.0)
                            q = a + w * ac
                        else:
                            va = d3 * d6 - d5 * d4
                            if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                                denom_bc = (d4 - d3) + (d5 - d6)
                                w = (
                                    (d4 - d3) / denom_bc
                                    if ti.abs(denom_bc) > 1e-18
                                    else data_type(0.0)
                                )
                                q = b + w * (c - b)
                            else:
                                denom = va + vb + vc
                                if ti.abs(denom) < 1e-18:
                                    q = a
                                else:
                                    v = vb / denom
                                    w = vc / denom
                                    q = a + ab * v + ac * w
        return q, (p - q).norm()

    @ti.func
    def _ti_tri_tri_contact(
        self, a0: ti.template(), a1: ti.template(), a2: ti.template(),
        b0: ti.template(), b1: ti.template(), b2: ti.template(),
        eps: data_type,
    ):
        """
        Returns (kind, p0, p1): kind 0=none, 1=point, 2=segment.
        Strict mode: rejects coplanar pairs (used by physical shells).
        """
        return self._ti_tri_tri_contact_ex(
            a0, a1, a2, b0, b1, b2, eps,
            data_type(1e-9),  # aabb pad
            0,               # allow_coplanar
            data_type(0.0),  # prox_eps (off)
        )

    @ti.func
    def _ti_tri_tri_contact_ex(
        self, a0: ti.template(), a1: ti.template(), a2: ti.template(),
        b0: ti.template(), b1: ti.template(), b2: ti.template(),
        eps: data_type,
        aabb_pad: data_type,
        allow_coplanar: ti.i32,
        prox_eps: data_type,
    ):
        """
        Returns (kind, p0, p1): kind 0=none, 1=point, 2=segment.

        Side-panel sensitive mode: larger AABB pad, allow coplanar, optional
        proximity fallback when faces approach within prox_eps without crossing.
        """
        kind = 0
        p0 = self._ti_v3(0.0, 0.0, 0.0)
        p1 = self._ti_v3(0.0, 0.0, 0.0)
        do_test = 1
        pad = ti.max(aabb_pad, eps)
        if self._ti_aabb_overlap_tris(a0, a1, a2, b0, b1, b2, pad) == 0:
            do_test = 0
        if (
            do_test == 1
            and allow_coplanar == 0
            and self._ti_tris_coplanar(
                a0, a1, a2, b0, b1, b2, data_type(1e-2)
            ) == 1
        ):
            do_test = 0

        if do_test == 1:
            n_hits = 0
            q0 = self._ti_v3(0.0, 0.0, 0.0)
            q1 = self._ti_v3(0.0, 0.0, 0.0)
            dedupe = ti.max(eps * data_type(10.0), data_type(1e-6))

            # Edges of A vs triangle B
            h, p = self._ti_seg_tri_intersect(a0, a1, b0, b1, b2, eps)
            if h == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, p, dedupe)
            h, p = self._ti_seg_tri_intersect(a1, a2, b0, b1, b2, eps)
            if h == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, p, dedupe)
            h, p = self._ti_seg_tri_intersect(a2, a0, b0, b1, b2, eps)
            if h == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, p, dedupe)
            # Edges of B vs triangle A
            h, p = self._ti_seg_tri_intersect(b0, b1, a0, a1, a2, eps)
            if h == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, p, dedupe)
            h, p = self._ti_seg_tri_intersect(b1, b2, a0, a1, a2, eps)
            if h == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, p, dedupe)
            h, p = self._ti_seg_tri_intersect(b2, b0, a0, a1, a2, eps)
            if h == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, p, dedupe)

            # Vertex-in-triangle (touching cases) — looser in-plane eps
            pin_eps = ti.max(eps * data_type(10.0), data_type(1e-5))
            if self._ti_point_in_triangle(a0, b0, b1, b2, pin_eps) == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, a0, dedupe)
            if self._ti_point_in_triangle(a1, b0, b1, b2, pin_eps) == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, a1, dedupe)
            if self._ti_point_in_triangle(a2, b0, b1, b2, pin_eps) == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, a2, dedupe)
            if self._ti_point_in_triangle(b0, a0, a1, a2, pin_eps) == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, b0, dedupe)
            if self._ti_point_in_triangle(b1, a0, a1, a2, pin_eps) == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, b1, dedupe)
            if self._ti_point_in_triangle(b2, a0, a1, a2, pin_eps) == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, b2, dedupe)

            if n_hits == 1:
                kind = 1
                p0 = q0
                p1 = q0
            elif n_hits >= 2:
                kind = 2
                p0 = q0
                p1 = q1
            elif prox_eps > data_type(1e-12):
                # Proximity fallback: nearest vertex→triangle within prox_eps
                best_d = prox_eps + data_type(1.0)
                best_q = self._ti_v3(0.0, 0.0, 0.0)
                q, d = self._ti_closest_on_triangle(a0, b0, b1, b2)
                if d < best_d:
                    best_d, best_q = d, q
                q, d = self._ti_closest_on_triangle(a1, b0, b1, b2)
                if d < best_d:
                    best_d, best_q = d, q
                q, d = self._ti_closest_on_triangle(a2, b0, b1, b2)
                if d < best_d:
                    best_d, best_q = d, q
                q, d = self._ti_closest_on_triangle(b0, a0, a1, a2)
                if d < best_d:
                    best_d, best_q = d, q
                q, d = self._ti_closest_on_triangle(b1, a0, a1, a2)
                if d < best_d:
                    best_d, best_q = d, q
                q, d = self._ti_closest_on_triangle(b2, a0, a1, a2)
                if d < best_d:
                    best_d, best_q = d, q
                if best_d <= prox_eps:
                    kind = 1
                    p0 = best_q
                    p1 = best_q
        return kind, p0, p1

    @ti.kernel
    def _kernel_detect_panel_contacts(self, hit_eps: data_type):
        """
        All triangle pairs: different panels, same shell layer_idx, no shared verts.

        coll_tri_layer = per-panel layer ordinal (0 = bottom shell of that panel).
        Writes contact hits into coll_hit_* (float64).
        """
        self.coll_hit_count[None] = 0
        n = self.coll_n_tris[None]
        max_hits = self.coll_hit_kind.shape[0]
        for ta, tb in ti.ndrange(n, n):
            if tb <= ta:
                continue
            panel_a = self.coll_tri_panel[ta]
            layer_a = self.coll_tri_layer[ta]
            panel_b = self.coll_tri_panel[tb]
            layer_b = self.coll_tri_layer[tb]
            if panel_a < 0 or panel_b < 0 or panel_a == panel_b or layer_a != layer_b:
                continue
            unit_a = self.coll_tri_unit[ta]
            unit_b = self.coll_tri_unit[tb]
            if unit_a == unit_b:
                continue
            i0 = self.coll_tri_kp[ta, 0]
            i1 = self.coll_tri_kp[ta, 1]
            i2 = self.coll_tri_kp[ta, 2]
            j0 = self.coll_tri_kp[tb, 0]
            j1 = self.coll_tri_kp[tb, 1]
            j2 = self.coll_tri_kp[tb, 2]
            if self._ti_share_vert(i0, i1, i2, j0, j1, j2) == 1:
                continue
            a0 = self.x[i0]
            a1 = self.x[i1]
            a2 = self.x[i2]
            b0 = self.x[j0]
            b1 = self.x[j1]
            b2 = self.x[j2]
            kind, p0, p1 = self._ti_tri_tri_contact(
                a0, a1, a2, b0, b1, b2, hit_eps
            )
            if kind == 0:
                continue
            idx = ti.atomic_add(self.coll_hit_count[None], 1)
            if idx < max_hits:
                self.coll_hit_kind[idx] = kind
                self.coll_hit_p0[idx] = p0
                self.coll_hit_p1[idx] = p1
                # ordered tri / unit for stable keys
                if unit_a < unit_b:
                    self.coll_hit_tri_a[idx] = ta
                    self.coll_hit_tri_b[idx] = tb
                    self.coll_hit_unit_a[idx] = unit_a
                    self.coll_hit_unit_b[idx] = unit_b
                    self.coll_hit_panel_a[idx] = panel_a
                    self.coll_hit_panel_b[idx] = panel_b
                else:
                    self.coll_hit_tri_a[idx] = tb
                    self.coll_hit_tri_b[idx] = ta
                    self.coll_hit_unit_a[idx] = unit_b
                    self.coll_hit_unit_b[idx] = unit_a
                    self.coll_hit_panel_a[idx] = panel_b
                    self.coll_hit_panel_b[idx] = panel_a
                self.coll_hit_layer[idx] = layer_a

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
        """Find inter-panel contact geometry.

        Narrowphase triangle–triangle tests run in a Taichi kernel (float64)
        reading live ``self.x``. Python only maps hits to design-xy + tracking.
        """
        if not hasattr(self, "_collision_tri_kp_indices"):
            return
        if not hasattr(self, "collision_contact_points"):
            return
        if not hasattr(self, "coll_tri_kp"):
            # Kernel buffers not allocated (collision_shading off at field create)
            return

        positions = self._cache_frame_positions()
        tri_kp_indices = self._collision_tri_kp_indices

        # Flat design keypoints (JSON unfolded layout) for 2D coordinate readout
        flat_kps = getattr(self, "_flat_kps_np", None)
        if flat_kps is None:
            flat_kps = np.asarray(self.kps, dtype=numpy_data_type)
            self._flat_kps_np = flat_kps
        if flat_kps.ndim != 2 or flat_kps.shape[0] < self.kp_num:
            flat_kps = positions  # fallback (should not happen after init)

        # --- Taichi kernel: all qualifying triangle pairs → contact hits ---
        self._kernel_detect_panel_contacts(1e-4)
        n_hits_raw = int(self.coll_hit_count[None])
        max_hits = int(self._collision_contact_max)
        n_hits = min(n_hits_raw, max_hits)

        # Phase-2: pull each hit field once (slice views, then use)
        if n_hits > 0:
            kinds = self.coll_hit_kind.to_numpy()[:n_hits]
            p0s = self.coll_hit_p0.to_numpy()[:n_hits]
            p1s = self.coll_hit_p1.to_numpy()[:n_hits]
            tri_as = self.coll_hit_tri_a.to_numpy()[:n_hits]
            tri_bs = self.coll_hit_tri_b.to_numpy()[:n_hits]
            unit_as = self.coll_hit_unit_a.to_numpy()[:n_hits]
            unit_bs = self.coll_hit_unit_b.to_numpy()[:n_hits]
            panel_as = self.coll_hit_panel_a.to_numpy()[:n_hits]
            panel_bs = self.coll_hit_panel_b.to_numpy()[:n_hits]
            layers = self.coll_hit_layer.to_numpy()[:n_hits]
        else:
            kinds = np.zeros(0, np.int32)
            p0s = np.zeros((0, 3), numpy_data_type)
            p1s = np.zeros((0, 3), numpy_data_type)
            tri_as = tri_bs = unit_as = unit_bs = panel_as = panel_bs = layers = (
                np.zeros(0, np.int32)
            )

        contact_points = []
        contact_segments = []
        contact_points_flat = []
        contact_segments_flat = []
        max_pts = self._collision_contact_max
        max_segs = self._collision_contact_max

        # Dense layer tables (Phase-2); fallback to legacy meta if missing
        li_arr = getattr(self, "_unit_layer_idx_arr", None)
        lh_arr = getattr(self, "_unit_layer_h_arr", None)
        n_units_tab = int(li_arr.shape[0]) if li_arr is not None else 0

        if n_hits > 0:
            # --- Batch design-xy maps for all physical hits ---
            # Jobs: for each hit endpoint on each triangle side
            # kind1: p0@A, p0@B ; kind2: p0@A, p1@A, p0@B, p1@B
            job_p = []
            job_tri = []
            job_hi = []
            job_tag = []  # 0=p0A 1=p1A 2=p0B 3=p1B

            for hi in range(n_hits):
                kind = int(kinds[hi])
                if kind <= 0:
                    continue
                ta = int(tri_as[hi])
                tb = int(tri_bs[hi])
                if ta < 0 or tb < 0 or ta >= len(tri_kp_indices) or tb >= len(tri_kp_indices):
                    continue
                p0 = p0s[hi]
                job_p.append(p0)
                job_tri.append(tri_kp_indices[ta])
                job_hi.append(hi)
                job_tag.append(0)  # p0 on A
                job_p.append(p0)
                job_tri.append(tri_kp_indices[tb])
                job_hi.append(hi)
                job_tag.append(2)  # p0 on B
                if kind >= 2:
                    p1 = p1s[hi]
                    job_p.append(p1)
                    job_tri.append(tri_kp_indices[ta])
                    job_hi.append(hi)
                    job_tag.append(1)  # p1 on A
                    job_p.append(p1)
                    job_tri.append(tri_kp_indices[tb])
                    job_hi.append(hi)
                    job_tag.append(3)  # p1 on B

            # Per-hit slots: (xy_A0, loc_A0, xy_A1, loc_A1, xy_B0, loc_B0, xy_B1, loc_B1, ok flags)
            map_A0 = [None] * n_hits
            map_A1 = [None] * n_hits
            map_B0 = [None] * n_hits
            map_B1 = [None] * n_hits

            if job_p:
                P = np.asarray(job_p, dtype=np.float64)
                T = np.asarray(job_tri, dtype=np.intp)
                xy_b, loc_b, ok_b = map_points_on_tris_batch(
                    P, T, positions, flat_kps
                )
                for j in range(len(job_hi)):
                    if not bool(ok_b[j]):
                        continue
                    hi = int(job_hi[j])
                    tag = int(job_tag[j])
                    cell = (xy_b[j], loc_b[j])
                    if tag == 0:
                        map_A0[hi] = cell
                    elif tag == 1:
                        map_A1[hi] = cell
                    elif tag == 2:
                        map_B0[hi] = cell
                    else:
                        map_B1[hi] = cell

            for hi in range(n_hits):
                kind = int(kinds[hi])
                if kind <= 0:
                    continue
                u_lo = int(unit_as[hi])
                u_hi = int(unit_bs[hi])
                p_of_ulo = int(panel_as[hi])
                p_of_uhi = int(panel_bs[hi])
                tri_lo = int(tri_as[hi])
                tri_hi = int(tri_bs[hi])

                if 0 <= u_lo < n_units_tab:
                    layer_idx_a = int(li_arr[u_lo])
                    layer_h_a = float(lh_arr[u_lo])
                else:
                    layer_idx_a, layer_h_a = 0, 0.0
                if 0 <= u_hi < n_units_tab:
                    layer_idx_b = int(li_arr[u_hi])
                    layer_h_b = float(lh_arr[u_hi])
                else:
                    layer_idx_b, layer_h_b = 0, 0.0
                layer_idx_hit = int(layers[hi]) if int(layers[hi]) >= 0 else layer_idx_a

                p_lo = p_of_ulo if p_of_ulo <= p_of_uhi else p_of_uhi
                p_hi = p_of_uhi if p_of_ulo <= p_of_uhi else p_of_ulo
                depth = self._stack_depth_fields(p_of_ulo, layer_h_a)

                if kind == 1:
                    p0 = [float(p0s[hi, 0]), float(p0s[hi, 1]), float(p0s[hi, 2])]
                    if len(contact_points) >= max_pts:
                        continue
                    contact_points.append(p0)
                    ent = {
                        "unit_a": u_lo,
                        "unit_b": u_hi,
                        "panel_of_unit_a": p_of_ulo,
                        "panel_of_unit_b": p_of_uhi,
                        "panel_a": p_lo,
                        "panel_b": p_hi,
                        "layer_idx": layer_idx_hit,
                        "layer_h": layer_h_a,
                        "layer_idx_a": layer_idx_a,
                        "layer_h_a": layer_h_a,
                        "layer_idx_b": layer_idx_b,
                        "layer_h_b": layer_h_b,
                        "tri_a": tri_lo,
                        "tri_b": tri_hi,
                        "tri_lo": tri_lo,
                        "tri_hi": tri_hi,
                        "shell_kind": "physical",
                        "shell_kind_a": "physical",
                        "shell_kind_b": "physical",
                        "p_3d": p0,
                        **depth,
                    }
                    if map_A0[hi] is not None:
                        xy, loc = map_A0[hi]
                        sa = self._side_dict_from_map(
                            u_lo, p_of_ulo, layer_idx_a, layer_h_a, "physical",
                            xy=xy, loc=loc, depth_panel=p_of_ulo,
                        )
                        if sa is not None:
                            ent["side_a"] = sa
                            ent["p"] = sa["p"]
                    if map_B0[hi] is not None:
                        xy, loc = map_B0[hi]
                        sb = self._side_dict_from_map(
                            u_hi, p_of_uhi, layer_idx_b, layer_h_b, "physical",
                            xy=xy, loc=loc, depth_panel=p_of_uhi,
                        )
                        if sb is not None:
                            ent["side_b"] = sb
                            if "p" not in ent:
                                ent["p"] = sb["p"]
                    contact_points_flat.append(ent)
                else:
                    p0 = [float(p0s[hi, 0]), float(p0s[hi, 1]), float(p0s[hi, 2])]
                    p1 = [float(p1s[hi, 0]), float(p1s[hi, 1]), float(p1s[hi, 2])]
                    if len(contact_segments) < max_segs:
                        contact_segments.append((p0, p1))
                        ent = {
                            "unit_a": u_lo,
                            "unit_b": u_hi,
                            "panel_of_unit_a": p_of_ulo,
                            "panel_of_unit_b": p_of_uhi,
                            "panel_a": p_lo,
                            "panel_b": p_hi,
                            "layer_idx": layer_idx_hit,
                            "layer_h": layer_h_a,
                            "layer_idx_a": layer_idx_a,
                            "layer_h_a": layer_h_a,
                            "layer_idx_b": layer_idx_b,
                            "layer_h_b": layer_h_b,
                            "tri_a": tri_lo,
                            "tri_b": tri_hi,
                            "tri_lo": tri_lo,
                            "tri_hi": tri_hi,
                            "shell_kind": "physical",
                            "shell_kind_a": "physical",
                            "shell_kind_b": "physical",
                            "p0_3d": p0,
                            "p1_3d": p1,
                            **depth,
                        }
                        if map_A0[hi] is not None and map_A1[hi] is not None:
                            xy0, loc0 = map_A0[hi]
                            xy1, loc1 = map_A1[hi]
                            sa = self._side_dict_from_map(
                                u_lo, p_of_ulo, layer_idx_a, layer_h_a, "physical",
                                xy0=xy0, loc0=loc0, xy1=xy1, loc1=loc1,
                                depth_panel=p_of_ulo,
                            )
                            if sa is not None:
                                ent["side_a"] = sa
                                ent["p0"], ent["p1"] = sa["p0"], sa["p1"]
                        if map_B0[hi] is not None and map_B1[hi] is not None:
                            xy0, loc0 = map_B0[hi]
                            xy1, loc1 = map_B1[hi]
                            sb = self._side_dict_from_map(
                                u_hi, p_of_uhi, layer_idx_b, layer_h_b, "physical",
                                xy0=xy0, loc0=loc0, xy1=xy1, loc1=loc1,
                                depth_panel=p_of_uhi,
                            )
                            if sb is not None:
                                ent["side_b"] = sb
                                if "p0" not in ent:
                                    ent["p0"], ent["p1"] = sb["p0"], sb["p1"]
                        contact_segments_flat.append(ent)
                    if len(contact_points) < max_pts:
                        contact_points.append(p0)
                    if len(contact_points) < max_pts:
                        contact_points.append(p1)

        # Collision-only ghost intermediate shells (no physics)
        try:
            self._detect_ghost_shell_contacts(
                positions,
                flat_kps,
                contact_points,
                contact_segments,
                contact_points_flat,
                contact_segments_flat,
                max_pts,
                max_segs,
            )
        except Exception as exc:
            if not getattr(self, "_ghost_contact_error_logged", False):
                self._ghost_contact_error_logged = True
                print(f"[Contact] ghost shell detect failed: {exc}")

        # Collision-only support shells (pad layer panel counts; no physics)
        try:
            self._detect_support_shell_contacts(
                positions,
                flat_kps,
                contact_points,
                contact_segments,
                contact_points_flat,
                contact_segments_flat,
                max_pts,
                max_segs,
            )
        except Exception as exc:
            if not getattr(self, "_support_contact_error_logged", False):
                self._support_contact_error_logged = True
                print(f"[Contact] support shell detect failed: {exc}")

        # Collision-only vertical side panels (unique indices, no physics)
        try:
            self._detect_side_panel_contacts(
                positions,
                flat_kps,
                contact_points,
                contact_segments,
                contact_points_flat,
                contact_segments_flat,
                max_pts,
                max_segs,
            )
        except Exception as exc:
            if not getattr(self, "_side_contact_error_logged", False):
                self._side_contact_error_logged = True
                print(f"[Contact] side panel detect failed: {exc}")

        self._collision_contact_count = len(contact_points)
        self._collision_segment_count = len(contact_segments)
        self._collision_contact_points_list = list(contact_points)
        self._collision_contact_segments_list = list(contact_segments)
        self._collision_contact_points_flat = list(contact_points_flat)
        self._collision_contact_segments_flat = list(contact_segments_flat)

        self._track_per_pair_flat_contacts()
        if not getattr(self, "_collision_coords_exported", False):
            try:
                self._stamp_paint_canvas_from_contacts()
                self._stamp_paint_needed = False
            except Exception as exc:
                if not getattr(self, "_stamp_paint_error_logged", False):
                    self._stamp_paint_error_logged = True
                    print(f"[Paint] stamp failed: {exc}")

        # Upload red markers for GUI (f32 renderer buffers)
        pt_buf = np.zeros((max_pts, 3), dtype=np.float32)
        if contact_points:
            pt_buf[: len(contact_points)] = np.asarray(contact_points, dtype=np.float32)
        self.collision_contact_points.from_numpy(pt_buf)

        ln_buf = np.zeros((max_segs * 2, 3), dtype=np.float32)
        if contact_segments:
            seg_arr = np.asarray(contact_segments, dtype=np.float32).reshape(-1, 3)
            ln_buf[: seg_arr.shape[0]] = seg_arr
        self.collision_contact_lines.from_numpy(ln_buf)

        self._collision_ran_once = True

    # ==================================================================
    # 3. Sweeping visualization (main, ghost, side) — host, no kernels
    # ==================================================================

    def _track_per_pair_flat_contacts(self):
        """
        Track every red contact (one slot per triangle–triangle pair).

        - **live p0/p1**: current flat-mapped contact line
        - **sweep**: successive p0–p1 samples while the contact is live
          (reprojected onto panels as locus strokes)
        - **lock**: when the contact disappears, or fold hits π

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
        # Min mid-point travel (design units) before recording another sweep sample.
        # Keep in sync with on-panel paint density (_paint_min_move_dist).
        sweep_min_move = self._paint_min_move_dist()

        # One entry per triangle–triangle contact (same as each red GUI segment/point).
        # Adopt collision hits by reference — rebuilt fresh each detection frame.
        segs_now = {}
        for s in self._collision_contact_segments_flat:
            s["folding_angle"] = angle
            segs_now[contact_pair_key(s)] = s

        pts_now = {}
        for e in self._collision_contact_points_flat:
            key = contact_pair_key(e)
            # Same tri-pair already produced a segment this frame → skip lone point
            if key in segs_now:
                continue
            e["folding_angle"] = angle
            pts_now[key] = e

        def _carry_sweep(ent, prev):
            """Move prior 2D sweep trail onto this frame's sides (by reference)."""
            if prev is None:
                return
            for sk in ("side_a", "side_b"):
                side = ent.get(sk)
                pside = prev.get(sk)
                if side is None or pside is None:
                    continue
                if side.get("sweep_samples"):
                    continue
                samples = pside.get("sweep_samples")
                if samples:
                    side["sweep_samples"] = samples

        def _record_sweep(ent, force_last=False):
            """Append current 2D contact line (min_move density; force locks tip)."""
            min_move = 0.0 if force_last else sweep_min_move
            for sk in ("side_a", "side_b"):
                side = ent.get(sk)
                if side is None:
                    continue
                append_sweep_sample_2d(side, angle, min_move=min_move)

        def _is_locked(key):
            fin = self._collision_fixed_segments.get(key)
            return bool(fin and fin.get("coords_locked"))

        for key, ent in segs_now.items():
            already = _is_locked(key)
            prev = self._collision_active_segments.get(key)

            if already:
                # Locked trail is frozen — keep the fixed entry for display
                self._collision_active_segments[key] = self._collision_fixed_segments[key]
                continue

            if ent.get("side_a") and "p0" in ent["side_a"]:
                sa = ent["side_a"]
                ent["p0"], ent["p1"] = sa["p0"], sa["p1"]
            elif ent.get("side_b") and "p0" in ent["side_b"]:
                sb = ent["side_b"]
                ent["p0"], ent["p1"] = sb["p0"], sb["p1"]
            if prev is not None:
                if ent.get("p0_3d") is None and prev.get("p0_3d") is not None:
                    ent["p0_3d"] = prev["p0_3d"]
                    ent["p1_3d"] = prev["p1_3d"]
            _carry_sweep(ent, prev)

            # 2D paint: record the mapped two-node line on each panel's design xy
            _record_sweep(ent, force_last=False)
            self._collision_active_segments[key] = ent

        for key, ent in pts_now.items():
            if key in self._collision_fixed_segments and self._collision_fixed_segments[key].get("coords_locked"):
                self._collision_active_points.pop(key, None)
                continue
            if ent.get("side_a"):
                ent["p"] = ent["side_a"]["p"]
            self._collision_active_points[key] = ent
            self._collision_fixed_points.pop(key, None)

        # Contact ended → freeze trail
        for key in list(self._collision_active_segments.keys()):
            if key not in segs_now:
                fin = self._collision_active_segments.pop(key)
                if not (self._collision_fixed_segments.get(key) or {}).get("coords_locked"):
                    for sk in ("side_a", "side_b"):
                        side = fin.get(sk)
                        if side is not None and "p0" in side:
                            append_sweep_sample_2d(
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

    def _stop_sweep_drawing_at_pi(self):
        """
        Immediately freeze locus paint when θ hits π.

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
        if not getattr(self, "_collision_coords_exported", False):
            angle = float(getattr(self, "folding_angle", 3.1415))
            for ent in (getattr(self, "_collision_active_segments", None) or {}).values():
                for sk in ("side_a", "side_b"):
                    side = ent.get(sk)
                    if side is not None and "p0" in side:
                        try:
                            append_sweep_sample_2d(side, angle, min_move=0.0)
                        except Exception:
                            pass
            try:
                self._seal_fixed_flat_contacts(reason="fold_pi")
            except Exception as exc:
                if getattr(self, "verbose", False):
                    print(f"[Contact] seal@π failed: {exc}")
                self._collision_coords_exported = True
        # Always write panel_trimming/trimmedData/<name>-trimmed.json once θ hits π
        try:
            self.export_trimmed_json(reason="fold_pi")
        except Exception as exc:
            print(f"[Contact] trimmed JSON export @π failed: {exc}")

    def _seal_fixed_flat_contacts(self, reason="fold_complete"):
        """Move remaining actives into fixed and stop further trail growth.

        reason: fixed_reason tag for still-live contacts (e.g. fold_pi).
        No export / deep-copy — entries kept by reference.
        """
        if self._collision_coords_exported:
            if self._collision_fixed_segments or self._collision_fixed_points:
                return
            self._collision_coords_exported = False
        seal_reason = reason or "fold_complete"
        for key, ent in list(self._collision_active_segments.items()):
            existing = self._collision_fixed_segments.get(key)
            if existing and existing.get("coords_locked"):
                for sk in ("side_a", "side_b"):
                    es = (ent.get(sk) or {}).get("sweep_samples") or []
                    xs = (existing.get(sk) or {}).get("sweep_samples") or []
                    if len(es) > len(xs) and existing.get(sk) is not None:
                        existing[sk]["sweep_samples"] = es
                continue
            ent["coords_locked"] = True
            ent["fixed_reason"] = ent.get("fixed_reason") or seal_reason
            self._collision_fixed_segments[key] = ent
        self._collision_active_segments.clear()
        for key, ent in list(self._collision_active_points.items()):
            if key in self._collision_fixed_segments:
                continue
            ent["coords_locked"] = True
            ent["fixed_reason"] = ent.get("fixed_reason") or seal_reason
            self._collision_fixed_points.setdefault(key, ent)
        self._collision_active_points.clear()
        for key in list(self._collision_fixed_points.keys()):
            if key in self._collision_fixed_segments:
                del self._collision_fixed_points[key]

        if (
            seal_reason == "fold_pi"
            and not self._collision_fixed_segments
            and getattr(self, "_collision_contact_segments_flat", None)
        ):
            for s in self._collision_contact_segments_flat:
                try:
                    key = contact_pair_key(s)
                except Exception:
                    continue
                if key in self._collision_fixed_segments:
                    continue
                s["coords_locked"] = True
                s["fixed_reason"] = "fold_pi"
                self._collision_fixed_segments[key] = s

        # Mark sealed so paint / tracking stop; pack lean shaded payload for JSON
        if (
            self._collision_fixed_segments
            or self._collision_fixed_points
            or seal_reason == "fold_pi"
        ):
            self._collision_coords_exported = True
            try:
                self._append_shaded_export_to_json()
            except Exception as exc:
                if getattr(self, "verbose", False):
                    print(f"[Contact] shaded pack@seal failed: {exc}")

    def _paint_min_move_dist(self):
        """Min mid-point travel between successive stroke samples."""
        if getattr(self, "_paint_min_move", None) is None:
            self._paint_min_move = max(
                float(getattr(self, "max_size", 100.0)) * 5e-5, 0.002
            )
        return float(self._paint_min_move)

    def _paint_trail_key(self, side, ent=None):
        """
        Stable paint canvas key for a contact side.

        - physical / ghost: ("phys", unit_id)
        - side: one trail per wall *quad face* (both mesh tris share a key via
          ``_side_face_from_kps``). Whole-panel keys mixed outline edges into
          diagonal chords; per-tri keys split across the mesh diagonal.
        """
        sk = str(
            (side or {}).get("shell_kind")
            or (ent or {}).get("shell_kind")
            or "physical"
        ).strip().lower()
        if sk == "side":
            panel = side.get("panel")
            if panel is None:
                return None
            try:
                pid = int(panel)
            except (TypeError, ValueError):
                return None
            loc = side.get("loc0") or side.get("loc")
            face_map = getattr(self, "_side_face_from_kps", None) or {}
            if loc is not None and face_map:
                try:
                    fs = frozenset(
                        (int(loc[0]), int(loc[1]), int(loc[2]))
                    )
                    fk = face_map.get(fs)
                    if fk is not None:
                        # (panel, side_id, edge_j)
                        return ("side", int(fk[0]), int(fk[1]), int(fk[2]))
                except (TypeError, ValueError, IndexError):
                    pass
            if loc is not None:
                try:
                    a, b, c = sorted(
                        (int(loc[0]), int(loc[1]), int(loc[2]))
                    )
                    return ("side", pid, a, b, c)
                except (TypeError, ValueError, IndexError):
                    pass
            return ("side", pid)
        uid = side.get("unit")
        if uid is None:
            return None
        try:
            return ("phys", int(uid))
        except (TypeError, ValueError):
            return None

    def _stamp_paint_canvas_from_contacts(self):
        """
        Record locus using barycentric locs from the collision hit triangle.

        Physical: trails keyed by unit; min_move in design-xy.
        Side panels: trails keyed by face; min_move in 3D (design-xy collapses).
        """
        if not getattr(self, "collision_shading", False):
            return
        if getattr(self, "_sweep_draw_stopped", False) or getattr(
            self, "_collision_coords_exported", False
        ):
            return
        if not hasattr(self, "_paint_by_unit") or self._paint_by_unit is None:
            self._paint_by_unit = {}

        active_segs = getattr(self, "_collision_active_segments", None) or {}
        active_pts = getattr(self, "_collision_active_points", None) or {}
        if not active_segs and not active_pts:
            return

        min_move = self._paint_min_move_dist()
        min_move_3d = max(min_move, float(getattr(self, "max_size", 100.0)) * 1e-4)
        dirty = False
        by_u = self._paint_by_unit

        def _iter_stamp_entries():
            for ent in active_segs.values():
                yield ent
            # Point contacts (many side hits) — promote p/loc → p0/p1 for stamp
            for ent in active_pts.values():
                yield ent

        for ent in _iter_stamp_entries():
            for sk in ("side_a", "side_b"):
                side = ent.get(sk)
                if not side:
                    continue
                # Segment form preferred; point form (p/loc) used for side kind=1
                loc0 = copy_loc(side.get("loc0") or side.get("loc"))
                loc1 = copy_loc(side.get("loc1") or side.get("loc"))
                if loc0 is None or loc1 is None:
                    continue
                if "p0" in side and "p1" in side:
                    p0 = [float(side["p0"][0]), float(side["p0"][1])]
                    p1 = [float(side["p1"][0]), float(side["p1"][1])]
                elif "p" in side:
                    p0 = [float(side["p"][0]), float(side["p"][1])]
                    p1 = [float(side["p"][0]), float(side["p"][1])]
                else:
                    continue
                trail_key = self._paint_trail_key(side, ent)
                if trail_key is None:
                    continue
                is_side = trail_key[0] == "side"
                mid3 = None
                if is_side:
                    mid3 = seg_mid3(
                        ent.get("p0_3d") or ent.get("p_3d"),
                        ent.get("p1_3d") or ent.get("p_3d"),
                    )
                    # Side walls: design-xy collapses — paint midpoint only.
                    if (
                        loc0[0] == loc1[0]
                        and loc0[1] == loc1[1]
                        and loc0[2] == loc1[2]
                    ):
                        loc_m = (
                            loc0[0], loc0[1], loc0[2],
                            0.5 * (float(loc0[3]) + float(loc1[3])),
                            0.5 * (float(loc0[4]) + float(loc1[4])),
                            0.5 * (float(loc0[5]) + float(loc1[5])),
                        )
                    else:
                        loc_m = loc0
                    loc0 = loc1 = loc_m
                    p_m = [
                        0.5 * (float(p0[0]) + float(p1[0])),
                        0.5 * (float(p0[1]) + float(p1[1])),
                    ]
                    p0 = p1 = p_m

                buf = by_u.get(trail_key)
                if buf is None:
                    buf = paint_buf_new(64, is_side=is_side)
                    by_u[trail_key] = buf
                n = int(buf["n"])

                if n > 0:
                    last_i = n - 1
                    last_p0 = buf["p0"][last_i]
                    last_p1 = buf["p1"][last_i]
                    if not is_side:
                        p0o, p1o = order_segment_endpoints_2d(
                            p0, p1,
                            [float(last_p0[0]), float(last_p0[1])],
                            [float(last_p1[0]), float(last_p1[1])],
                        )
                        if p0o[0] != p0[0] or p0o[1] != p0[1]:
                            loc0, loc1 = loc1, loc0
                        p0, p1 = p0o, p1o

                    if is_side and mid3 is not None and "mid3" in buf:
                        lm = buf["mid3"][last_i]
                        dx = float(mid3[0]) - float(lm[0])
                        dy = float(mid3[1]) - float(lm[1])
                        dz = float(mid3[2]) - float(lm[2])
                        moved = (dx * dx + dy * dy + dz * dz) ** 0.5
                        move_lim = min_move_3d
                    else:
                        moved = segment_mid_dist_2d(
                            [float(last_p0[0]), float(last_p0[1])],
                            [float(last_p1[0]), float(last_p1[1])],
                            p0, p1,
                        )
                        move_lim = min_move

                    if moved < move_lim:
                        # Refresh tip in place
                        buf["p0"][last_i, 0] = p0[0]
                        buf["p0"][last_i, 1] = p0[1]
                        buf["p1"][last_i, 0] = p1[0]
                        buf["p1"][last_i, 1] = p1[1]
                        buf["loc0"][last_i, 0] = loc0[0]
                        buf["loc0"][last_i, 1] = loc0[1]
                        buf["loc0"][last_i, 2] = loc0[2]
                        buf["loc0"][last_i, 3] = loc0[3]
                        buf["loc0"][last_i, 4] = loc0[4]
                        buf["loc0"][last_i, 5] = loc0[5]
                        buf["loc1"][last_i, 0] = loc1[0]
                        buf["loc1"][last_i, 1] = loc1[1]
                        buf["loc1"][last_i, 2] = loc1[2]
                        buf["loc1"][last_i, 3] = loc1[3]
                        buf["loc1"][last_i, 4] = loc1[4]
                        buf["loc1"][last_i, 5] = loc1[5]
                        if mid3 is not None and "mid3" in buf:
                            buf["mid3"][last_i, 0] = mid3[0]
                            buf["mid3"][last_i, 1] = mid3[1]
                            buf["mid3"][last_i, 2] = mid3[2]
                        dirty = True
                        continue

                # Append new stroke
                need = n + 1
                if need > int(buf["cap"]):
                    buf = paint_buf_grow(buf, need)
                    by_u[trail_key] = buf
                i = n
                buf["p0"][i, 0] = p0[0]
                buf["p0"][i, 1] = p0[1]
                buf["p1"][i, 0] = p1[0]
                buf["p1"][i, 1] = p1[1]
                buf["loc0"][i, 0] = loc0[0]
                buf["loc0"][i, 1] = loc0[1]
                buf["loc0"][i, 2] = loc0[2]
                buf["loc0"][i, 3] = loc0[3]
                buf["loc0"][i, 4] = loc0[4]
                buf["loc0"][i, 5] = loc0[5]
                buf["loc1"][i, 0] = loc1[0]
                buf["loc1"][i, 1] = loc1[1]
                buf["loc1"][i, 2] = loc1[2]
                buf["loc1"][i, 3] = loc1[3]
                buf["loc1"][i, 4] = loc1[4]
                buf["loc1"][i, 5] = loc1[5]
                if mid3 is not None and "mid3" in buf:
                    buf["mid3"][i, 0] = mid3[0]
                    buf["mid3"][i, 1] = mid3[1]
                    buf["mid3"][i, 2] = mid3[2]
                buf["n"] = need
                dirty = True

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
        """Reproject stored barycentric paint trails onto current mesh pose."""
        if not hasattr(self, "paint_line_vertices") and not getattr(
            self, "collision_shading", False
        ):
            self._paint_line_vert_count = 0
            self._paint_n_strokes = 0
            return

        if getattr(self, "_sweep_draw_stopped", False):
            return

        by_u = getattr(self, "_paint_by_unit", None) or {}
        n_strokes = paint_total_strokes(by_u)
        if n_strokes <= 0:
            if self._paint_line_vert_count != 0:
                self._paint_line_vert_count = 0
                self._paint_n_strokes = 0
            return

        # Skip when mesh pose and canvas are both unchanged (pause / no advance).
        mesh_adv = bool(getattr(self, "_mesh_advanced", True))
        dirty = bool(getattr(self, "_paint_dirty", False))
        if (
            not mesh_adv
            and not dirty
            and int(getattr(self, "_paint_line_vert_count", 0) or 0) >= 2
        ):
            return

        lift = max(float(getattr(self, "max_size", 100.0)) * 5e-5, 0.005)
        positions = self._cache_frame_positions()

        loc0_parts = []
        loc1_parts = []
        unit_ranges = []  # (start, n, is_side)
        cursor = 0
        for _uid, buf in by_u.items():
            n = int(buf.get("n", 0) or 0)
            if n <= 0:
                continue
            loc0_parts.append(buf["loc0"][:n])
            loc1_parts.append(buf["loc1"][:n])
            unit_ranges.append((cursor, n, bool(buf.get("is_side"))))
            cursor += n

        if cursor <= 0:
            self._paint_line_vert_count = 0
            self._paint_n_strokes = 0
            self._paint_dirty = False
            return

        all_loc0 = np.vstack(loc0_parts)
        all_loc1 = np.vstack(loc1_parts)
        pts0_all = eval_bary_batch(all_loc0, positions, lift=lift)
        pts1_all = eval_bary_batch(all_loc1, positions, lift=lift)

        ms = float(getattr(self, "max_size", 100.0) or 100.0)
        side_max_link = max(ms * 0.02, 0.25)

        packed = []
        for start, n, is_side in unit_ranges:
            chunk = pack_unit_locus_lines(
                pts0_all[start : start + n],
                pts1_all[start : start + n],
                max_link=(side_max_link if is_side else None),
                side_mode=bool(is_side),
            )
            if chunk.shape[0] > 0:
                packed.append(chunk)

        if not packed:
            self._paint_line_vert_count = 0
            self._paint_n_strokes = 0
            self._paint_dirty = False
            return

        ln = np.vstack(packed)
        li = int(ln.shape[0])
        self._ensure_paint_line_capacity(li)
        max_lv = int(self._paint_max_line_verts)
        out_buf = np.zeros((max_lv, 3), dtype=np.float32)
        n_write = min(li, max_lv)
        out_buf[:n_write] = ln[:n_write]
        self.paint_line_vertices.from_numpy(out_buf)
        self._paint_line_vert_count = n_write
        self._paint_n_strokes = n_write // 2
        self._paint_dirty = False

    def _render_panel_meshes(self, scene):
        """
        Draw base mesh + side walls + green support shells (not ghosts)
        + surface sweep paint + red contact markers.
        """
        scene.mesh(
            self.vertices,
            indices=self.indices,
            color=(0.80, 0.82, 0.93),
            two_sided=True,
        )

        # Collision-only vertical side panels (pale mint, two-sided).
        # GGUI mesh has no true alpha; light mint fill + green outline.
        # Topology/contact/edge-copy are all Taichi; draw uses live vertices.
        n_side_idx = int(getattr(self, "_side_panel_index_count", 0) or 0)
        if (
            self.collision_shading
            and n_side_idx >= 3
            and hasattr(self, "side_panel_indices")
        ):
            try:
                scene.mesh(
                    self.vertices,
                    indices=self.side_panel_indices,
                    color=(0.55, 0.88, 0.58),
                    two_sided=True,
                    index_count=n_side_idx,
                )
            except Exception as exc:
                if not getattr(self, "_side_mesh_error_logged", False):
                    self._side_mesh_error_logged = True
                    print(f"[Contact] side panel mesh draw failed: {exc}")
            n_edge_v = int(getattr(self, "_side_panel_edge_vert_count", 0) or 0)
            if (
                n_edge_v >= 2
                and hasattr(self, "side_panel_edge_verts")
                and hasattr(self, "_kernel_update_side_panel_edge_lines")
            ):
                try:
                    self._kernel_update_side_panel_edge_lines()
                    scene.lines(
                        self.side_panel_edge_verts,
                        width=1.8,
                        color=(0.18, 0.52, 0.28),
                        vertex_count=n_edge_v,
                    )
                except Exception as exc:
                    if not getattr(self, "_side_edge_error_logged", False):
                        self._side_edge_error_logged = True
                        print(f"[Contact] side panel edge draw failed: {exc}")

        # Support panels (pad missing layer panels) — green fill, not ghosts.
        # Live verts blended from physical shells each frame.
        n_sup_idx = int(getattr(self, "_support_panel_index_count", 0) or 0)
        if (
            self.collision_shading
            and bool(getattr(self, "thick_support_panels", True))
            and n_sup_idx >= 3
            and hasattr(self, "support_panel_verts")
            and hasattr(self, "support_panel_indices")
        ):
            try:
                self._update_support_panel_draw_verts()
                scene.mesh(
                    self.support_panel_verts,
                    indices=self.support_panel_indices,
                    # Bright green so missing-layer pads (e.g. miura @3 / @-9) stand out
                    color=(0.12, 0.78, 0.18),
                    two_sided=True,
                    index_count=n_sup_idx,
                )
            except Exception as exc:
                if not getattr(self, "_support_mesh_error_logged", False):
                    self._support_mesh_error_logged = True
                    print(f"[Contact] support panel mesh draw failed: {exc}")
            n_sup_edge = int(getattr(self, "_support_panel_edge_vert_count", 0) or 0)
            if n_sup_edge >= 2 and hasattr(self, "support_panel_edge_verts"):
                try:
                    scene.lines(
                        self.support_panel_edge_verts,
                        width=1.6,
                        color=(0.08, 0.40, 0.10),
                        vertex_count=n_sup_edge,
                    )
                except Exception as exc:
                    if not getattr(self, "_support_edge_error_logged", False):
                        self._support_edge_error_logged = True
                        print(f"[Contact] support panel edge draw failed: {exc}")

        n_pts = self._collision_contact_count
        n_segs = self._collision_segment_count

        if self.collision_shading and hasattr(self, "paint_line_vertices"):
            try:
                self._update_surface_sweep_paint()
            except Exception as exc:
                if not getattr(self, "_paint_update_error_logged", False):
                    self._paint_update_error_logged = True
                    print(f"[Paint] surface sweep update failed: {exc}")
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

    # ==================================================================
    # 4. JSON export helper — host, no kernels
    # ==================================================================

    def _default_trimmed_json_path(self):
        """panel_trimming/trimmedData/<name>-trimmed.json (never clobber source descriptionData)."""
        name = str(getattr(self, "origami_name", "export") or "export")
        # avoid name-trimmed-trimmed if already a trimmed stem
        if name.endswith("-trimmed"):
            stem = name
        else:
            stem = f"{name}-trimmed"
        # Prefer absolute under project root so cwd does not matter
        root = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(root, "panel_trimming", "trimmedData", f"{stem}.json")

    def export_trimmed_json(self, path=None, reason="fold_pi", force=False):
        """
        Write design JSON + dual-curve shaded_regions to
        panel_trimming/trimmedData/*-trimmed.json.

        Called automatically when θ hits π (collision_shading). Source design
        ``descriptionData/<name>.json`` is left unchanged.
        """
        if not getattr(self, "collision_shading", False):
            return None
        if getattr(self, "_trimmed_json_exported", False) and not force:
            return getattr(self, "_trimmed_json_path", None)
        if not hasattr(self, "input_json") or self.input_json is None:
            return None

        # Pack shaded dual-curves into input_json (in-place attributes only)
        try:
            self._append_shaded_export_to_json()
        except Exception as exc:
            print(f"[Contact] shaded pack before trimmed write failed: {exc}")

        # Latest crease targets if PD fields exist
        try:
            self.input_json["crease_angle"] = [
                max(min(self.crease_angle[i], 1.), -1.)
                for i in range(self.crease_pairs_num)
            ]
            self.input_json["crease_info"] = [
                [self.kps[self.crease_pairs[i, 0]], self.kps[self.crease_pairs[i, 1]]]
                for i in range(self.crease_pairs_num)
            ]
        except Exception:
            pass

        self.input_json["export_meta"] = {
            "source": str(getattr(self, "origami_name", "")),
            "reason": str(reason or "fold_pi"),
            "folding_angle": float(getattr(self, "folding_angle", 0.0)),
            "schema": "dual_curve_v1",
            "n_shaded_regions": len(self.input_json.get("shaded_regions") or []),
        }

        out_path = path or self._default_trimmed_json_path()
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(self.input_json, fw, indent=4)

        self._trimmed_json_exported = True
        self._trimmed_json_path = out_path
        n_sh = len(self.input_json.get("shaded_regions") or [])
        print(
            f"[Contact] auto-exported trimmed JSON @π "
            f"(shaded_regions={n_sh}) → {out_path}"
        )
        return out_path

    ribbon_ring_from_curves = staticmethod(ribbon_ring_from_curves)

    sweep_paint_polygon_2d = staticmethod(sweep_paint_polygon_2d)

    def build_shaded_regions(self):
        """
        Compact dual-curve shaded regions from locked (or active) contacts.

        Walks tracker entries **by reference** — no deep-copy of side/loc/3d
        trees. Builds only the thin parallel arrays needed for JSON.
        """
        regions = []
        # Prefer sealed trails; fall back to still-active if seal was empty
        ents = list((getattr(self, "_collision_fixed_segments", None) or {}).values())
        if not ents:
            ents = list(
                (getattr(self, "_collision_active_segments", None) or {}).values()
            )
        # Stable order for diffs / visualizer
        ents.sort(key=lambda s: (
            int(s.get("panel_a", -1)), int(s.get("panel_b", -1)),
            int(s.get("layer_idx", -1)),
            int(s.get("unit_a", -1)), int(s.get("unit_b", -1)),
            int(s.get("tri_a", -1)), int(s.get("tri_b", -1)),
        ))

        for seg in ents:
            for side_key in ("side_a", "side_b"):
                side = seg.get(side_key)
                if not side or "p0" not in side:
                    continue
                samples = side.get("sweep_samples") or []
                # Seed with crease-snapped first if present but not in trail
                if not samples and "first_p0" in side and "first_p1" in side:
                    samples = [{
                        "angle": float(side.get("first_folding_angle", 0.0)),
                        "p0": side["first_p0"],
                        "p1": side["first_p1"],
                    }, {
                        "angle": float(seg.get("folding_angle", 0.0)),
                        "p0": side["p0"],
                        "p1": side["p1"],
                    }]
                theta, c0, c1 = pack_dual_curves(samples)
                n = len(c0)
                if n < 1:
                    # Degenerate: single live line only
                    if "p0" in side and "p1" in side:
                        c0 = [xy2(side["p0"])]
                        c1 = [xy2(side["p1"])]
                        theta = [float(seg.get("folding_angle", 0.0))]
                        n = 1
                    else:
                        continue
                ring = ribbon_ring_from_curves(c0, c1)
                kind = "sweep" if ring is not None else ("line" if n >= 1 else "empty")
                area = polygon_area_2d(ring) if ring is not None else 0.0
                # Per-side thickness offset: each sim unit has its own height_z.
                # Legacy exports copied seg.layer_h (from unit_a) onto both sides —
                # wrong when panels have different stacks (miura P0@-3 vs P2@-9).
                if side.get("layer_h") is not None:
                    side_layer_h = float(side["layer_h"])
                elif side_key == "side_a" and seg.get("layer_h_a") is not None:
                    side_layer_h = float(seg["layer_h_a"])
                elif side_key == "side_b" and seg.get("layer_h_b") is not None:
                    side_layer_h = float(seg["layer_h_b"])
                else:
                    side_layer_h = float(seg.get("layer_h", 0.0))
                if side.get("layer_idx") is not None:
                    side_layer_idx = int(side["layer_idx"])
                elif side_key == "side_a" and seg.get("layer_idx_a") is not None:
                    side_layer_idx = int(seg["layer_idx_a"])
                elif side_key == "side_b" and seg.get("layer_idx_b") is not None:
                    side_layer_idx = int(seg["layer_idx_b"])
                else:
                    side_layer_idx = int(seg.get("layer_idx", -1))
                panel_id = int(side.get("panel", seg.get("panel_a", -1)))
                shell_kind = side.get("shell_kind") or seg.get("shell_kind") or "physical"
                shell_kind_s = str(shell_kind).strip().lower()
                parent_panel = side.get("parent_panel")
                if parent_panel is None:
                    parent_panel = seg.get("parent_panel_a") if side_key == "side_a" else seg.get("parent_panel_b")
                # Authoritative physical height from registry — but never for ghost,
                # support, or side shells: those keep their stamped layer_h.
                if shell_kind_s not in ("ghost", "side", "support"):
                    su = side.get("unit")
                    if su is not None:
                        try:
                            um = self.unit_to_panel_layer(int(su))
                            side_layer_h = float(um.get("height_z", side_layer_h))
                            # Prefer collision-stack ordinal when available
                            coll_map = getattr(self, "_unit_coll_layer_idx", {}) or {}
                            if int(su) in coll_map:
                                side_layer_idx = int(coll_map[int(su)])
                            else:
                                side_layer_idx = int(
                                    um.get("layer_idx", side_layer_idx)
                                )
                        except Exception:
                            pass

                # Depth uses parent design panel for side walls (unique panel id
                # is not in the stock-span registry).
                depth_panel = int(parent_panel) if parent_panel is not None else panel_id
                if shell_kind_s == "side" and parent_panel is not None:
                    depth_panel = int(parent_panel)
                # Depth: prefer values stamped on the side; recompute if missing
                if side.get("stock_span_mm") is not None and side.get("depth_from_top_mm") is not None:
                    depth_fields = {
                        "stock_span_mm": float(side["stock_span_mm"]),
                        "depth_from_top_mm": float(side["depth_from_top_mm"]),
                        "depth_from_bottom_mm": float(
                            side.get("depth_from_bottom_mm", 0.0)
                        ),
                    }
                else:
                    depth_fields = self._stack_depth_fields(depth_panel, side_layer_h)
                # Keep layer_h consistent with depth if side carried stamped depths
                # (ghost/support/side: side_layer_h already correct; recompute depths).
                if shell_kind_s in ("ghost", "side", "support"):
                    depth_fields = self._stack_depth_fields(depth_panel, side_layer_h)

                _sk_export = (
                    shell_kind_s
                    if shell_kind_s in ("ghost", "side", "support")
                    else "physical"
                )
                region = {
                    # identity
                    "panel": panel_id,
                    "unit": int(side.get("unit", -1)),
                    "side": side_key,
                    "layer_idx": side_layer_idx,
                    "layer_h": side_layer_h,
                    "shell_kind": _sk_export,
                    "unit_a": int(seg.get("unit_a", -1)),
                    "unit_b": int(seg.get("unit_b", -1)),
                    "tri_a": int(seg.get("tri_a", -1)),
                    "tri_b": int(seg.get("tri_b", -1)),
                    # dual curves (primary geometry — one source of truth)
                    "theta": theta,
                    "c0": c0,
                    "c1": c1,
                    "n": n,
                    # derived once (consumers may also rebuild from c0/c1)
                    "kind": kind,
                    "area": float(area),
                    "fixed_reason": seg.get("fixed_reason"),
                    "folding_angle": float(
                        seg.get("folding_angle", getattr(self, "folding_angle", 0.0))
                    ),
                    # stack depth (physical + ghost + side shells share fields)
                    **depth_fields,
                }
                if shell_kind_s == "side":
                    if parent_panel is not None:
                        region["parent_panel"] = int(parent_panel)
                    if side.get("h_lo") is not None:
                        region["h_lo"] = float(side["h_lo"])
                    if side.get("h_hi") is not None:
                        region["h_hi"] = float(side["h_hi"])
                regions.append(region)
        return regions

    def _append_shaded_export_to_json(self):
        """
        Mutate ``self.input_json`` in place with compact shaded attributes.

        No deep-copy of the contact tracker: only newly packed dual-curve
        arrays are attached. Existing crease / unit / line data is left alone.
        """
        if not hasattr(self, "input_json") or self.input_json is None:
            return
        regions = self.build_shaded_regions()
        self._shaded_regions_export = regions
        # Primary: dual-curve list (curves + rebuildable ribbon)
        self.input_json["shaded_regions"] = regions
        # Lean stats block for tooling / visualizer discovery (no nested
        # segment trees, no duplicated polygon aliases)
        n_fixed = len(getattr(self, "_collision_fixed_segments", {}) or {})
        n_pts = len(getattr(self, "_collision_fixed_points", {}) or {})
        n_sweep = sum(1 for r in regions if r.get("kind") == "sweep")
        n_ghost = sum(1 for r in regions if r.get("shell_kind") == "ghost")
        n_side = sum(1 for r in regions if r.get("shell_kind") == "side")
        n_support = sum(1 for r in regions if r.get("shell_kind") == "support")
        n_phys = sum(
            1 for r in regions
            if r.get("shell_kind") not in ("ghost", "side", "support")
        )
        depth_tops = [
            float(r["depth_from_top_mm"])
            for r in regions
            if r.get("depth_from_top_mm") is not None
        ]
        depth_bots = [
            float(r["depth_from_bottom_mm"])
            for r in regions
            if r.get("depth_from_bottom_mm") is not None
        ]
        # Compact side-panel registry (unique indices, geometry for viz)
        side_reg = []
        for s in getattr(self, "_collision_side_panels", None) or []:
            side_reg.append({
                "panel": int(s["panel_idx"]),
                "parent_panel": int(s["parent_panel_idx"]),
                "h_lo": float(s["h_lo"]),
                "h_hi": float(s["h_hi"]),
                "layer_h": float(s["layer_h"]),
                "layer_idx": int(s.get("layer_idx", -1)),
                "unit_lo": int(s.get("unit_lo", -1)),
                "unit_hi": int(s.get("unit_hi", -1)),
                "outline_xy": s.get("outline_xy") or [],
                "n_edges": len(s.get("edges") or []),
                "shell_kind": "side",
            })
        if side_reg:
            self.input_json["side_panels"] = side_reg
        self.input_json["collision_stats"] = {
            "schema": "dual_curve_v1",
            "depth_schema": "stack_shell_v1",
            "n_shaded_regions": len(regions),
            "n_shaded_areas": len(regions),  # alias for older readers
            "n_sweep_paints": n_sweep,
            "n_segments": n_fixed,
            "n_points": n_pts,
            "n_closed_polygons": n_sweep,
            "n_physical_regions": n_phys,
            "n_ghost_regions": n_ghost,
            "n_side_regions": n_side,
            "n_support_regions": n_support,
            "thick_ghost_spacing_mm": float(
                getattr(self, "thick_ghost_spacing_mm", 0.0) or 0.0
            ),
            "thick_support_panels": bool(
                getattr(self, "thick_support_panels", True)
            ),
            "n_ghost_shells": len(getattr(self, "_collision_ghost_shells", None) or []),
            "n_support_shells": len(
                getattr(self, "_collision_support_shells", None) or []
            ),
            "n_side_panels": len(side_reg),
            "depth_global_max_from_top": float(max(depth_tops) if depth_tops else 0.0),
            "depth_global_max_from_bottom": float(max(depth_bots) if depth_bots else 0.0),
            "folding_angle": float(getattr(self, "folding_angle", 0.0)),
            # Pointer: full geometry lives in top-level shaded_regions
            "shaded_regions_key": "shaded_regions",
        }
        if getattr(self, "verbose", False) or len(regions) > 0:
            print(
                f"[Contact] shaded export: {len(regions)} region(s) "
                f"(sweep={n_sweep}, sealed_lines={n_fixed}, "
                f"physical={n_phys}, ghost={n_ghost}, support={n_support}, "
                f"side={n_side}) "
                f"schema=dual_curve_v1 depth=stack_shell_v1 → input_json['shaded_regions']"
            )

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

    def get_collision_groups(self, include_live=True):
        """Group collision entries by the two JSON panels that collide."""
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
                }
            return buckets[key]

        for seg in self.get_fixed_flat_segments():
            pa, pb = panel_group_key(seg)
            g = _ensure(pa, pb)
            g["segments"].append(seg)
            g["_has_done"] = True
        for ent in self.get_fixed_flat_points():
            pa, pb = panel_group_key(ent)
            g = _ensure(pa, pb)
            g["points"].append(ent)
            g["_has_done"] = True

        if include_live:
            for seg in self._collision_active_segments.values():
                pa, pb = panel_group_key(seg)
                g = _ensure(pa, pb)
                g["segments"].append(seg)
                g["_has_live"] = True
            for ent in self._collision_active_points.values():
                pa, pb = panel_group_key(ent)
                g = _ensure(pa, pb)
                g["points"].append(ent)
                g["_has_live"] = True

        groups_out = []
        for key in sorted(buckets.keys()):
            g = buckets[key]
            if g["_has_done"] and g["_has_live"]:
                g["status"] = "mixed"
            elif g["_has_live"]:
                g["status"] = "live"
            else:
                g["status"] = "done"
            g["segments"].sort(key=lambda s: (
                s["layer_idx"], s["unit_a"], s["unit_b"],
                int(s.get("tri_a", -1)), int(s.get("tri_b", -1)),
            ))
            g["points"].sort(key=lambda e: (
                e["layer_idx"], e["unit_a"], e["unit_b"],
                int(e.get("tri_a", -1)), int(e.get("tri_b", -1)),
            ))
            del g["_has_done"]
            del g["_has_live"]
            groups_out.append(g)
        return groups_out
