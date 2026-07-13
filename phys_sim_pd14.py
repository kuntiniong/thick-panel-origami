import taichi as ti
import taichi.math as tm
import json
import time, os
import yaml
import gc
from spatialhash import SpatialHash
from ori_sim_sys import *
# Triangle contact narrowphase is Taichi (_kernel_detect_panel_contacts);
# utils.triangle_intersection_contacts_3d kept only for non-sim tooling.

data_type = ti.f64
numpy_data_type = np.float64
use_gpu = 0

if use_gpu:
    ti.init(arch=ti.gpu, default_fp=data_type, fast_math=False, advanced_optimization=False, kernel_profiler=True)
else:
    ti.init(arch=ti.cpu, default_fp=data_type, fast_math=False, advanced_optimization=False, cpu_max_num_threads=1) #, kernel_profiler=False, verbose=True, debug=True, gdb_trigger=True)


# ---------------------------------------------------------------------------
# 3D contact → design-plane map (needed by collision flat coords / locus paint)
# ---------------------------------------------------------------------------

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


def _map_point_3d_to_flat_2d(p3d, verts_3d, verts_flat_xy, return_bary=False):
    """
    Map a 3D point on (or near) a triangle to flat design 2D coords.

    Exact for any point on the triangle under the standard piecewise-linear
    mesh assumption: the unique affine map sending the deformed triangle
    (verts_3d) to the rest/design triangle (verts_flat_xy).

    Returns [x, y], or None if the triangle is degenerate.
    If return_bary=True, returns ([x,y], (u,v,w)) or (None, None).
    """
    a = np.asarray(verts_3d[0], dtype=float).reshape(3)[:3]
    b = np.asarray(verts_3d[1], dtype=float).reshape(3)[:3]
    c = np.asarray(verts_3d[2], dtype=float).reshape(3)[:3]
    p = np.asarray(p3d, dtype=float).reshape(-1)[:3]
    if a.shape[0] < 3 or p.shape[0] < 3:
        def _pad3(v):
            v = np.asarray(v, dtype=float).reshape(-1)
            if v.size >= 3:
                return v[:3]
            out = np.zeros(3, dtype=float)
            out[: v.size] = v
            return out
        a, b, c, p = _pad3(a), _pad3(b), _pad3(c), _pad3(p)

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
        self._collision_coords_exported = False  # sealed at π (no more trail growth)
        self._sweep_draw_stopped = False  # True once θ hits π — no more paint growth
        # Locus paint: path of two contact nodes + the line (design xy → panel).
        # Host canvas stores bary locs; reprojected onto live x (af0a695 min_move).
        self._paint_canvas = {}          # unit_id -> list[{p0,p1,loc0,loc1}]
        self._paint_line_vert_count = 0
        self._paint_n_strokes = 0
        # Starting buffer size only — grows as needed (no hard line cap)
        self._paint_max_line_verts = 8192
        self._paint_min_move = None      # lazy from max_size
        # Collision postprocess is still host-side — throttle when not folding
        self._collision_frame_i = 0
        self._collision_every_n = 4      # run broadphase every N render frames
        self._paint_dirty = False
        # Per-frame host cache of x (shared by collision / paint)
        self._frame_positions = None
        self._frame_positions_gen = -1
        self._render_gen = 0
        self._stamp_paint_needed = False
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
            # Paint strokes reprojected to panel surfaces (lines only — canvas model)
            if self.collision_shading:
                n_pl = int(self._paint_max_line_verts)
                self.paint_line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_pl)
                # Taichi kernel buffers for inter-panel triangle contact detection
                max_tris = max(int(self.maximum_indice_num) // 3, 1)
                max_hits = int(self._collision_contact_max)
                self._coll_max_tris = max_tris
                self.coll_n_tris = ti.field(dtype=ti.i32, shape=())
                self.coll_tri_kp = ti.field(dtype=ti.i32, shape=(max_tris, 3))
                self.coll_tri_unit = ti.field(dtype=ti.i32, shape=max_tris)
                self.coll_tri_panel = ti.field(dtype=ti.i32, shape=max_tris)
                self.coll_tri_layer = ti.field(dtype=ti.i32, shape=max_tris)
                self.coll_hit_count = ti.field(dtype=ti.i32, shape=())
                self.coll_hit_kind = ti.field(dtype=ti.i32, shape=max_hits)  # 1=point, 2=segment
                self.coll_hit_p0 = ti.Vector.field(3, dtype=data_type, shape=max_hits)
                self.coll_hit_p1 = ti.Vector.field(3, dtype=data_type, shape=max_hits)
                self.coll_hit_tri_a = ti.field(dtype=ti.i32, shape=max_hits)
                self.coll_hit_tri_b = ti.field(dtype=ti.i32, shape=max_hits)
                self.coll_hit_unit_a = ti.field(dtype=ti.i32, shape=max_hits)
                self.coll_hit_unit_b = ti.field(dtype=ti.i32, shape=max_hits)
                self.coll_hit_panel_a = ti.field(dtype=ti.i32, shape=max_hits)
                self.coll_hit_panel_b = ti.field(dtype=ti.i32, shape=max_hits)
                self.coll_hit_layer = ti.field(dtype=ti.i32, shape=max_hits)

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
        self._collision_ran_once = False
        self._collision_frame_i = 0
        # Surface sweep paint (red locus on panels)
        self._paint_canvas = {}
        self._paint_line_vert_count = 0
        self._paint_n_strokes = 0
        self._paint_dirty = True
        self._stamp_paint_needed = False
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
            # Hit π this step → freeze paint immediately (not next render)
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
        # Keep float64 host cache (sim fields are ti.f64 / numpy_data_type)
        pos = np.asarray(self.x.to_numpy()[: self.kp_num], dtype=numpy_data_type)
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
            # Seal every pair's final coords (including still-active)
            if self.collision_shading and not self._collision_coords_exported:
                self._seal_fixed_flat_contacts()
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
    4. sweeping visualization
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
        # Per-unit layer meta for collision paint
        self._collision_unit_layer_meta = self._build_unit_layer_meta()

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
            tri_layer = np.full(num_tris, -1, dtype=np.int32)
            for t in range(num_tris):
                u = int(tri_unit_ids[t])
                if 0 <= u < len(unit_panel_idx):
                    tri_panel[t] = int(unit_panel_idx[u])
                    tri_layer[t] = int(unit_layer_idx[u])
            kp_buf = np.zeros((max_tris, 3), dtype=np.int32)
            unit_buf = np.zeros(max_tris, dtype=np.int32)
            panel_buf = np.full(max_tris, -1, dtype=np.int32)
            layer_buf = np.full(max_tris, -1, dtype=np.int32)
            kp_buf[:num_tris] = tri_kp_indices[:num_tris]
            unit_buf[:num_tris] = tri_unit_ids[:num_tris]
            panel_buf[:num_tris] = tri_panel
            layer_buf[:num_tris] = tri_layer
            self.coll_n_tris[None] = int(num_tris)
            self.coll_tri_kp.from_numpy(kp_buf)
            self.coll_tri_unit.from_numpy(unit_buf)
            self.coll_tri_panel.from_numpy(panel_buf)
            self.coll_tri_layer.from_numpy(layer_buf)

    # ------------------------------------------------------------------
    # Taichi triangle–triangle contact detection (float64)
    # ------------------------------------------------------------------

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
    def _ti_tri_tri_contact(
        self, a0: ti.template(), a1: ti.template(), a2: ti.template(),
        b0: ti.template(), b1: ti.template(), b2: ti.template(),
        eps: data_type,
    ):
        """
        Returns (kind, p0, p1): kind 0=none, 1=point, 2=segment.
        """
        kind = 0
        p0 = self._ti_v3(0.0, 0.0, 0.0)
        p1 = self._ti_v3(0.0, 0.0, 0.0)
        do_test = 1
        if self._ti_aabb_overlap_tris(a0, a1, a2, b0, b1, b2, data_type(1e-9)) == 0:
            do_test = 0
        if do_test == 1 and self._ti_tris_coplanar(
            a0, a1, a2, b0, b1, b2, data_type(1e-2)
        ) == 1:
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

            # Vertex-in-triangle (touching cases)
            if self._ti_point_in_triangle(a0, b0, b1, b2, eps) == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, a0, dedupe)
            if self._ti_point_in_triangle(a1, b0, b1, b2, eps) == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, a1, dedupe)
            if self._ti_point_in_triangle(a2, b0, b1, b2, eps) == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, a2, dedupe)
            if self._ti_point_in_triangle(b0, a0, a1, a2, eps) == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, b0, dedupe)
            if self._ti_point_in_triangle(b1, a0, a1, a2, eps) == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, b1, dedupe)
            if self._ti_point_in_triangle(b2, a0, a1, a2, eps) == 1:
                n_hits, q0, q1 = self._ti_push_contact_pt(n_hits, q0, q1, b2, dedupe)

            if n_hits == 1:
                kind = 1
                p0 = q0
                p1 = q0
            elif n_hits >= 2:
                kind = 2
                p0 = q0
                p1 = q1
        return kind, p0, p1

    @ti.kernel
    def _kernel_detect_panel_contacts(self, hit_eps: data_type):
        """
        All triangle pairs: different panels, same layer, no shared verts.
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
        unit_layer_idx = self._collision_unit_layer_idx

        # Flat design keypoints (JSON unfolded layout) for 2D coordinate readout
        flat_kps = getattr(self, "_flat_kps_np", None)
        if flat_kps is None:
            flat_kps = np.asarray(self.kps, dtype=numpy_data_type)
            self._flat_kps_np = flat_kps
        if flat_kps.ndim != 2 or flat_kps.shape[0] < self.kp_num:
            flat_kps = positions  # fallback (should not happen after init)

        unit_layer_meta = getattr(self, "_collision_unit_layer_meta", None) or {}

        # --- Taichi kernel: all qualifying triangle pairs → contact hits ---
        self._kernel_detect_panel_contacts(1e-4)
        n_hits_raw = int(self.coll_hit_count[None])
        max_hits = int(self._collision_contact_max)
        n_hits = min(n_hits_raw, max_hits)

        if n_hits > 0:
            kinds = np.asarray(self.coll_hit_kind.to_numpy()[:n_hits], dtype=np.int32)
            p0s = np.asarray(self.coll_hit_p0.to_numpy()[:n_hits], dtype=numpy_data_type)
            p1s = np.asarray(self.coll_hit_p1.to_numpy()[:n_hits], dtype=numpy_data_type)
            tri_as = np.asarray(self.coll_hit_tri_a.to_numpy()[:n_hits], dtype=np.int32)
            tri_bs = np.asarray(self.coll_hit_tri_b.to_numpy()[:n_hits], dtype=np.int32)
            unit_as = np.asarray(self.coll_hit_unit_a.to_numpy()[:n_hits], dtype=np.int32)
            unit_bs = np.asarray(self.coll_hit_unit_b.to_numpy()[:n_hits], dtype=np.int32)
            panel_as = np.asarray(self.coll_hit_panel_a.to_numpy()[:n_hits], dtype=np.int32)
            panel_bs = np.asarray(self.coll_hit_panel_b.to_numpy()[:n_hits], dtype=np.int32)
            layers = np.asarray(self.coll_hit_layer.to_numpy()[:n_hits], dtype=np.int32)
        else:
            kinds = np.zeros(0, np.int32)
            p0s = np.zeros((0, 3), numpy_data_type)
            p1s = np.zeros((0, 3), numpy_data_type)
            tri_as = np.zeros(0, np.int32)
            tri_bs = np.zeros(0, np.int32)
            unit_as = np.zeros(0, np.int32)
            unit_bs = np.zeros(0, np.int32)
            panel_as = np.zeros(0, np.int32)
            panel_bs = np.zeros(0, np.int32)
            layers = np.zeros(0, np.int32)

        contact_points = []
        contact_segments = []
        contact_points_flat = []
        contact_segments_flat = []
        colliding_unit_pairs = set()
        max_pts = self._collision_contact_max
        max_segs = self._collision_contact_max

        def _flat_of(p3d, tri_idx):
            """Map 3D contact → design xy + material barycentric on that triangle."""
            kp = tri_kp_indices[int(tri_idx)]
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
            meta = unit_layer_meta.get(int(unit_id), {})
            layer_idx = meta.get("layer_idx")
            if layer_idx is None:
                layer_idx = int(unit_layer_idx[unit_id]) if unit_id < len(unit_layer_idx) else -1
            layer_h = meta.get("layer_h", meta.get("height_z"))
            if layer_h is None:
                kp = tri_kp_indices[int(tri_idx)]
                if flat_kps.shape[1] > 2:
                    layer_h = float(flat_kps[kp[0]][2])
                else:
                    layer_h = 0.0
            return int(layer_idx), float(layer_h)

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
            layer_idx_hit = int(layers[hi])
            _, layer_h = _layer_of(u_lo, tri_lo)
            # Prefer meta layer_idx when present
            meta_li, meta_h = _layer_of(u_lo, tri_lo)
            if meta_li >= 0:
                layer_idx_hit = meta_li
            layer_h = meta_h

            p_lo = p_of_ulo if p_of_ulo <= p_of_uhi else p_of_uhi
            p_hi = p_of_uhi if p_of_ulo <= p_of_uhi else p_of_ulo
            pair_meta = {
                "unit_a": u_lo,
                "unit_b": u_hi,
                "panel_of_unit_a": p_of_ulo,
                "panel_of_unit_b": p_of_uhi,
                "panel_a": p_lo,
                "panel_b": p_hi,
                "layer_idx": layer_idx_hit,
                "layer_h": layer_h,
                "tri_a": tri_lo,
                "tri_b": tri_hi,
                "tri_lo": tri_lo,
                "tri_hi": tri_hi,
            }

            def _side_flat(p3d0, p3d1=None, _tri_lo=tri_lo, _tri_hi=tri_hi,
                           _u_lo=u_lo, _u_hi=u_hi, _pulo=p_of_ulo, _puhi=p_of_uhi):
                a0 = _flat_of(p3d0, _tri_lo)
                b0 = _flat_of(p3d0, _tri_hi)
                if p3d1 is None:
                    if a0 is None and b0 is None:
                        return None
                    out = {}
                    if a0 is not None:
                        out["side_a"] = {
                            "unit": _u_lo, "panel": _pulo,
                            "p": a0["p"], "loc": a0["loc"],
                        }
                        out["p"] = a0["p"]
                    if b0 is not None:
                        out["side_b"] = {
                            "unit": _u_hi, "panel": _puhi,
                            "p": b0["p"], "loc": b0["loc"],
                        }
                        if "p" not in out:
                            out["p"] = b0["p"]
                    return out
                a1 = _flat_of(p3d1, _tri_lo)
                b1 = _flat_of(p3d1, _tri_hi)
                out = {}
                if a0 is not None and a1 is not None:
                    out["side_a"] = {
                        "unit": _u_lo, "panel": _pulo,
                        "p0": a0["p"], "p1": a1["p"],
                        "loc0": a0["loc"], "loc1": a1["loc"],
                    }
                    out["p0"], out["p1"] = a0["p"], a1["p"]
                if b0 is not None and b1 is not None:
                    out["side_b"] = {
                        "unit": _u_hi, "panel": _puhi,
                        "p0": b0["p"], "p1": b1["p"],
                        "loc0": b0["loc"], "loc1": b1["loc"],
                    }
                    if "p0" not in out:
                        out["p0"], out["p1"] = b0["p"], b1["p"]
                return out if out else None

            recorded = False
            if kind == 1:
                p0 = [float(p0s[hi, 0]), float(p0s[hi, 1]), float(p0s[hi, 2])]
                if len(contact_points) < max_pts:
                    contact_points.append(p0)
                    ent = dict(pair_meta)
                    both = _side_flat(p0)
                    if both is not None:
                        ent.update(both)
                    ent["p_3d"] = p0
                    contact_points_flat.append(ent)
                    recorded = True
            else:
                p0 = [float(p0s[hi, 0]), float(p0s[hi, 1]), float(p0s[hi, 2])]
                p1 = [float(p1s[hi, 0]), float(p1s[hi, 1]), float(p1s[hi, 2])]
                if len(contact_segments) < max_segs:
                    contact_segments.append((p0, p1))
                    ent = dict(pair_meta)
                    both = _side_flat(p0, p1)
                    if both is not None:
                        ent.update(both)
                    ent["p0_3d"] = p0
                    ent["p1_3d"] = p1
                    contact_segments_flat.append(ent)
                    recorded = True
                if len(contact_points) < max_pts:
                    contact_points.append(p0)
                    recorded = True
                if len(contact_points) < max_pts:
                    contact_points.append(p1)
                    recorded = True

            if recorded:
                colliding_unit_pairs.add((min(u_lo, u_hi), max(u_lo, u_hi)))

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
    # Contact dicts from collision are adopted by reference (no deep-copy /
    # export snapshots). Trail lists are carried across frames by reference.

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
        Append current side p0–p1 to a 2D sweep trail (af0a695 non-Taichi).

        - Seeds trail with crease-snapped first_* when present
        - Skips samples that barely moved (min_move)
        - Endpoint order stabilized against previous sample
        - No sample count cap
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
            mm = float(min_move)
            if moved < mm and end_move < mm:
                if moved < 1e-9:
                    return
                samples[-1] = {"angle": float(angle), "p0": p0, "p1": p1}
                return

        samples.append({"angle": float(angle), "p0": p0, "p1": p1})

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
        # Min mid-point travel (design units) before recording another sweep sample
        sweep_min_move = max(float(getattr(self, "max_size", 100.0)) * 3e-4, 0.012)

        # One entry per triangle–triangle contact (same as each red GUI segment/point).
        # Adopt collision hits by reference — rebuilt fresh each detection frame.
        segs_now = {}
        for s in self._collision_contact_segments_flat:
            s["folding_angle"] = angle
            segs_now[self._contact_pair_key(s)] = s

        pts_now = {}
        for e in self._collision_contact_points_flat:
            key = self._contact_pair_key(e)
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
                self._append_sweep_sample_2d(side, angle, min_move=min_move)

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
            self._seal_fixed_flat_contacts(reason="fold_pi")
        except Exception as exc:
            if getattr(self, "verbose", False):
                print(f"[Contact] seal@π failed: {exc}")
            self._collision_coords_exported = True

    # 5. seal trails at π

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
                    key = self._contact_pair_key(s)
                except Exception:
                    continue
                if key in self._collision_fixed_segments:
                    continue
                s["coords_locked"] = True
                s["fixed_reason"] = "fold_pi"
                self._collision_fixed_segments[key] = s

        # Mark sealed so paint / tracking stop (no JSON write)
        if (
            self._collision_fixed_segments
            or self._collision_fixed_points
            or seal_reason == "fold_pi"
        ):
            self._collision_coords_exported = True

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
            pa, pb = self._panel_group_key(seg)
            g = _ensure(pa, pb)
            g["segments"].append(seg)
            g["_has_done"] = True
        for ent in self.get_fixed_flat_points():
            pa, pb = self._panel_group_key(ent)
            g = _ensure(pa, pb)
            g["points"].append(ent)
            g["_has_done"] = True

        if include_live:
            for seg in self._collision_active_segments.values():
                pa, pb = self._panel_group_key(seg)
                g = _ensure(pa, pb)
                g["segments"].append(seg)
                g["_has_live"] = True
            for ent in self._collision_active_points.values():
                pa, pb = self._panel_group_key(ent)
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

    def _iter_live_sweep_sides(self):
        """Yield (side_dict, ent) for active + fixed contacts."""
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
            for sk in ("side_a", "side_b"):
                side = ent.get(sk)
                if not side or "p0" not in side:
                    continue
                yield side, ent

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

    def _paint_min_move_dist(self):
        # Match export sweep density so the on-panel locus is not sparse
        if getattr(self, "_paint_min_move", None) is None:
            self._paint_min_move = max(
                float(getattr(self, "max_size", 100.0)) * 3e-4, 0.012
            )
        return float(self._paint_min_move)

    def _stamp_paint_canvas_from_contacts(self):
        """
        Record locus using barycentric locs from the collision hit triangle.
        af0a695 min_move sampling: append when mid-point moved enough; else
        refresh tip locs so the last sample tracks the live contact.
        """
        if not getattr(self, "collision_shading", False):
            return
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
                    if p0o[0] != p0[0] or p0o[1] != p0[1]:
                        loc0, loc1 = loc1, loc0
                    p0, p1 = p0o, p1o
                    moved = mid_dist(last["p0"], last["p1"], p0, p1)
                    if moved < min_move:
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

        Restored from af0a695 (non-Taichi numpy reproject + from_numpy upload).
        """
        if not hasattr(self, "paint_line_vertices") and not getattr(
            self, "collision_shading", False
        ):
            self._paint_line_vert_count = 0
            self._paint_n_strokes = 0
            return

        if getattr(self, "_sweep_draw_stopped", False):
            return

        canvas = getattr(self, "_paint_canvas", None) or {}
        if not canvas:
            if self._paint_line_vert_count != 0:
                self._paint_line_vert_count = 0
                self._paint_n_strokes = 0
            return

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
        lift = max(float(getattr(self, "max_size", 100.0)) * 5e-5, 0.005)

        all_loc0 = []
        all_loc1 = []
        unit_ranges = []
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

        n_total = len(all_loc0)
        est_verts = max(4 * n_total * 2, 64)
        self._ensure_paint_line_capacity(est_verts)
        max_lv = int(self._paint_max_line_verts)
        ln_buf = np.zeros((max_lv, 3), dtype=np.float32)
        li = 0

        for start, n in unit_ranges:
            pts0 = pts0_all[start : start + n]
            pts1 = pts1_all[start : start + n]
            good0 = np.isfinite(pts0).all(axis=1)
            good1 = np.isfinite(pts1).all(axis=1)
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
            max_lv = int(self._paint_max_line_verts)
            if ln_buf.shape[0] != max_lv:
                full = np.zeros((max_lv, 3), dtype=np.float32)
                full[:li] = ln_buf[:li]
                ln_buf = full
            self.paint_line_vertices.from_numpy(ln_buf)

    def _render_panel_meshes(self, scene):
        """
        Draw base mesh + surface sweep paint (design-xy mapped onto panels)
        + red contact markers. (af0a695 non-Taichi draw path)
        """
        scene.mesh(
            self.vertices,
            indices=self.indices,
            color=(0.80, 0.82, 0.93),
            two_sided=True,
        )

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
            # Collision shading is off by default; use panel-trimming/ runner to enable it.
            collision_shading=sim.get("collision_shading", False),
        )

        ori.start(
            filepath=name,
            unit_edge_max=sim.get("unit_edge_max", 4),
            thick_mode=sim.get("thick_mode", False),
        )

        ori.run()