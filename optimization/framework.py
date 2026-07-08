import copy
import gc
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phys_sim_pd14 import PD_Origami_Simulator as OrigamiSimulator
from phys_sim_pd14 import data_type, ensure_taichi_init, mark_taichi_reset, ti, use_gpu
# from symmetry_grouping.symmetry import detect_crease_symmetry_groups


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    aliases: Tuple[str, ...]
    result_prefix: str
    config_key: str
    framework_loader: Callable[[], Type["ThickPanelDesignFramework"]]
    optimize_kwargs_builder: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


_ALGORITHM_SPECS: Dict[str, AlgorithmSpec] = {}
_ALGORITHM_ALIASES: Dict[str, str] = {}


def register_algorithm(spec: AlgorithmSpec) -> None:
    """Register an optimization algorithm. Call from framework or algo modules."""
    _ALGORITHM_SPECS[spec.key] = spec
    for alias in spec.aliases:
        _ALGORITHM_ALIASES[alias.lower()] = spec.key


def resolve_algorithm(name: str) -> str:
    key = _ALGORITHM_ALIASES.get(name.lower())
    if key is None:
        supported = ", ".join(sorted(_ALGORITHM_SPECS))
        raise ValueError(f"Unknown algorithm '{name}'. Supported values: {supported}")
    return key


def default_n_processes() -> int:
    """Reasonable fallback when n_processes is omitted from config (cap at 4)."""
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, 4))


def resolve_n_processes(value: Optional[Any]) -> int:
    """Normalize n_processes from config; omitted/null uses default_n_processes()."""
    if value is None:
        return default_n_processes()
    return max(1, int(value))


class ThickPanelDesignFramework:
    """
    厚板折纸高度偏移量计算设计框架

    该框架通过优化算法来设计折痕的高度偏移量，使得厚板折纸能够尽可能完全地折叠。

    约束条件/Constraints:
    1. 优化器在倍率空间搜索 [1, max_offset/min_thickness]，物理高度 = 倍率 × min_thickness
    2. 高度偏移量绝对值至少为min_thickness（默认2mm），保证板具有厚度
    3. 高度偏移量绝对值按discrete_step（默认0.4mm）离散化
    4. 对于同时包含山折痕和谷折痕的板，谷折痕的有符号高度偏移量必须大于山折痕的有符号高度偏移量，且至少相差4mm
    5. 使用OrigamiSimulator评估折叠程度

    :param json_path: 输入JSON文件路径 / Input JSON file path
    :param batch_size: 批量评估大小 / Batch size for parallel evaluation
    :param n_processes: 并行仿真进程数 / Number of parallel simulator worker processes
    :param min_thickness: 最小厚度（高度偏移量绝对值的最小值）/ Minimum thickness
    :param discrete_step: 高度偏移量离散化步长 / Discretization step for height offset
    :param max_offset: 最大高度偏移量 / Maximum height offset
    :param use_gui: 是否使用GUI / Whether to use GUI
    :param symm_mode: 是否启用对称检测与降维优化 / Enable symmetry detection and reduced-dimension optimization
    :param algorithm_key: Canonical algorithm id (e.g. cma_es, bo)
    :param result_prefix: Prefix for physResult export folder names
    """

    algorithm_key: str = ""
    result_prefix: str = ""

    def __init__(
        self,
        json_path: str,
        batch_size: int = 64,
        population_size: int = 256,
        n_processes: int = 1,
        min_thickness: float = 2.0,
        discrete_step: float = 0.4,
        max_offset: float = 50.0,
        use_gui: bool = False,
        symm_mode: bool = True,
        algorithm_key: Optional[str] = None,
        result_prefix: Optional[str] = None,
        _batch_json_dir: Optional[str] = None,
        _simulator_name_override: Optional[str] = None,
        _quiet: bool = False,
        _force_cpu: bool = False,
        _reuse_ti_runtime: bool = False,
        max_steps: int = 60,
        fold_angle_step: float = 0.105,
        ref_target: bool = True,
        reuse_simulator: bool = True,
        initial_offsets: Optional[np.ndarray] = None,
    ):
        self.json_path = json_path
        self.batch_size = batch_size
        self.population_size = population_size
        self.n_processes = max(1, int(n_processes))
        self._batch_json_dir = _batch_json_dir
        self._simulator_name_override = _simulator_name_override
        self._quiet = _quiet
        self._force_cpu = _force_cpu
        self._reuse_ti_runtime = _reuse_ti_runtime
        self.max_steps = max(1, int(max_steps))
        self.fold_angle_step = float(fold_angle_step)
        self.ref_target = bool(ref_target)
        self.reuse_simulator = bool(reuse_simulator)
        self.min_thickness = min_thickness
        self.discrete_step = discrete_step
        self.max_offset = max_offset
        self.use_gui = use_gui
        self.symm_mode = symm_mode

        self.algorithm_key = algorithm_key or self.algorithm_key
        if result_prefix is None:
            self.result_prefix = self.result_prefix or self.algorithm_key
        else:
            self.result_prefix = result_prefix

        self.original_data = self._load_json(json_path)

        self.crease_info = self._parse_crease_info()
        self.num_creases = len(self.crease_info)

        self._initial_offsets_full: Optional[np.ndarray] = None
        if initial_offsets is not None:
            offsets = np.asarray(initial_offsets, dtype=float).reshape(-1)
            if offsets.size != self.num_creases:
                raise ValueError(
                    "initial_offsets length must match the number of creases "
                    f"({self.num_creases}); got {offsets.size}"
                )
            self._initial_offsets_full = offsets

        # --- symmetry state (disabled) ---
        # self._init_symmetry_state()
        self.line_symmetry_groups = []
        self.symmetry_groups = []
        self.optimization_groups = [[i] for i in range(self.num_creases)]
        self.independent_indices = np.arange(self.num_creases, dtype=int)
        self.num_independent = self.num_creases

        self.batch_json_path = self._create_batch_json()

        self.simulator = None
        self._mp_pool: Optional[Any] = None
        self._mp_pool_worker_count = 0

        self.data = []
        self.extract_data = {
            "gen": [],
            "avg": [],
            "std": [],
            "min": [],
            "min_without_var": [],
            "num": self.population_size,
            "best_offset": [],
            "best_offset_normalized": [],
            "pop_rewards": [],
            "pop_variances": [],
            "pop_fitness": [],
        }

        if not self._quiet:
            print("[初始化] 厚板折纸设计框架初始化完成")
            print("[Init] Thick panel design framework initialized")
            print(f"  - 对称模式/Symmetry mode: off (disabled)")
            print(f"  - 折痕数量/Number of creases: {self.num_creases}")
        # --- symmetry print block (disabled) ---
        # if self.symm_mode and not self._quiet:
        #     print(f"  - 折痕线对称分组/Line symmetry groups: ...")
        #     ...
        if not self._quiet:
            print(f"  - 批量大小/Batch size: {batch_size}")
            print(f"  - 并行进程数/Parallel processes (n_processes): {self.n_processes}")
            print(f"  - 最小厚度/Min thickness: {min_thickness}mm")
            print(f"  - 离散步长/Discrete step: {discrete_step}mm")
            print(
                f"  - 优化器倍率范围/Optimizer multiplicand range: "
                f"[{self._optimizer_bound_lo():.4g}, {self._optimizer_bound_hi():.4g}] "
                f"(× {min_thickness}mm)"
            )
            if self.result_prefix:
                print(f"  - 结果前缀/Result prefix: {self.result_prefix}")
            print(f"  - 仿真步数/Simulation max steps: {self.max_steps}")
            print(f"  - 折角步长/Fold angle step: {self.fold_angle_step}")
            print(f"  - 复用仿真器/Reuse simulator: {self.reuse_simulator}")
            if self._initial_offsets_full is not None:
                print("  - 初始偏移量/Initial offsets: from framework.initial_offsets")

    def has_initial_offsets(self) -> bool:
        return self._initial_offsets_full is not None

    def _batch_json_stem(self) -> str:
        return os.path.basename(self.batch_json_path).replace(".json", "")

    def simulator_origami_name(self) -> str:
        """Export folder key: optional algo/config prefix + batch stem (never _mp)."""
        if self._simulator_name_override:
            return self._simulator_name_override
        stem = self._batch_json_stem()
        if self.result_prefix:
            return f"{self.result_prefix}-{stem}"
        return stem

    def result_dir(self) -> str:
        """Same folder as PD_Origami_Simulator.outputFigure (physResult/{export name})."""
        return os.path.join("./physResult", self.simulator_origami_name())

    def _export_figure_stride(self) -> int:
        """Export PNGs every N evaluations (CDF-style throttling)."""
        return max(1, int(self.population_size / self.batch_size * self.num_creases))

    def _should_export_figure(self, algo_step: int) -> bool:
        return algo_step % self._export_figure_stride() == 0

    def _create_simulator(self, simulator_name: str, algo_step: int) -> OrigamiSimulator:
        simulator = OrigamiSimulator(
            origami_name=simulator_name,
            use_gui=self.use_gui,
            fast=True,
            ref_target=self.ref_target,
            verbose=0,
        )
        simulator.ID = algo_step
        return simulator

    def _teardown_simulator(self) -> None:
        if self.simulator is None:
            return
        self.simulator.window.destroy()
        self.simulator = None
        gc.collect()
        if not self._reuse_ti_runtime:
            ti.reset()
            mark_taichi_reset()

    def _recover_simulator_after_failed_start(
        self,
        simulator_name: str,
        batch_json_path: str,
        algo_step: int,
    ) -> bool:
        self._teardown_simulator()
        self._init_ti()
        self.simulator = self._create_simulator(simulator_name, algo_step)
        return self.simulator.start(batch_json_path, 4, thick_mode=1)

    def _record_best_offset(self, best_solution: Optional[np.ndarray]) -> None:
        """
        Append the current best offset (denormalized and normalized by crease
        length) to extract_data.  Call once per generation after stats are
        recorded.

        :param best_solution: Full crease offset vector (num_creases,), or None
            if no best solution has been found yet (first generation with no
            improvement).
        """
        if best_solution is None:
            self.extract_data["best_offset"].append(None)
            self.extract_data["best_offset_normalized"].append(None)
        else:
            offsets = np.asarray(best_solution, dtype=float)
            normalized = self._normalize_offsets_by_crease_length(offsets)
            self.extract_data["best_offset"].append(offsets.tolist())
            self.extract_data["best_offset_normalized"].append(normalized.tolist())

    @staticmethod
    def _compact_json_arrays(text: str) -> str:
        """Collapse inner numeric arrays of best_offset / best_offset_normalized onto one line each."""
        import re

        _num = r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
        _item = rf"(?:{_num}|null)"
        inner_pat = re.compile(
            rf"\[\s*({_item}(?:\s*,\s*{_item})*)\s*\]",
            re.DOTALL,
        )

        def _collapse_inner(section: str) -> str:
            return inner_pat.sub(
                lambda m: "[" + re.sub(r"\s+", " ", m.group(1)).strip() + "]",
                section,
            )

        # Only target the value blocks of the listed keys.
        # json.dumps(indent=4) closes a top-level array value with "\n    ]".
        for key in (
            "best_offset",
            "best_offset_normalized",
            "pop_rewards",
            "pop_variances",
            "pop_fitness",
        ):
            key_re = re.compile(
                rf'("{re.escape(key)}":\s*)(\[.*?\n    \])',
                re.DOTALL,
            )
            text = key_re.sub(lambda m: m.group(1) + _collapse_inner(m.group(2)), text)

        return text

    def save_extract_data(self) -> None:
        result_dir = self.result_dir()
        os.makedirs(result_dir, exist_ok=True)
        raw = json.dumps(self.extract_data, indent=4)
        compact = self._compact_json_arrays(raw)
        with open(os.path.join(result_dir, "data.json"), "w", encoding="utf-8") as f:
            f.write(compact)

    def _load_json(self, path: str) -> Dict:
        """加载JSON文件 / Load JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _parse_crease_info(self) -> List[Dict]:
        """
        解析折痕信息

        从JSON中提取折痕的几何信息和类型信息。
        注意：只关注type为0（valley谷折）和1（mountain山折）的折痕，type为2的是边界。

        :return: 折痕信息列表，每个元素包含：
            - index: 折痕在line_features中的索引
            - type: 折痕类型 (0=valley, 1=mountain)
            - line_index: 对应的lines索引
        """
        crease_info = []
        line_features = self.original_data.get("line_features", [])

        for i, feature in enumerate(line_features):
            crease_type = feature.get("type", 2)
            if crease_type in [0, 1]:
                crease_info.append(
                    {
                        "index": i,
                        "type": crease_type,
                        "line_index": i,
                        "original_height": feature.get("thick_panel_height", 0.0),
                    }
                )

        return crease_info

    def _init_symmetry_state(self) -> None:
        """Initialize symmetry groups and independent optimization variables."""
        # --- symmetry detection disabled; always use flat (no-symm) init ---
        # if self.symm_mode:
        #     self.line_symmetry_groups, self.symmetry_groups, _ = self._detect_symmetry_groups()
        #     self.optimization_groups, self.independent_indices = self._build_optimization_groups()
        # else:
        self.line_symmetry_groups = []
        self.symmetry_groups = []
        self.optimization_groups = [[i] for i in range(self.num_creases)]
        self.independent_indices = np.arange(self.num_creases, dtype=int)
        self.num_independent = len(self.optimization_groups)

    def _detect_symmetry_groups(self) -> Tuple[List[List[int]], List[List[int]], Optional[dict]]:
        """Detect mirror-symmetric line pairs via PCA and map to crease indices."""
        # --- disabled ---
        # crease_line_indices = [info["index"] for info in self.crease_info]
        # return detect_crease_symmetry_groups(self.original_data, crease_line_indices)
        return [], [], None

    def _build_optimization_groups(self) -> Tuple[List[List[int]], np.ndarray]:
        """
        Partition creases into independent DOF groups once at init.

        Each symmetry-linked crease set shares one variable; unpaired creases
        remain singleton groups. Returns (groups, representative crease indices).
        """
        # --- disabled; always return flat singleton groups ---
        # assigned = set()
        # groups: List[List[int]] = []
        # for group in self.symmetry_groups:
        #     if len(group) < 2:
        #         continue
        #     sorted_group = sorted(group)
        #     groups.append(sorted_group)
        #     assigned.update(sorted_group)
        # for crease_idx in range(self.num_creases):
        #     if crease_idx not in assigned:
        #         groups.append([crease_idx])
        # groups.sort(key=lambda group: group[0])
        # independent_indices = np.array([group[0] for group in groups], dtype=int)
        # return groups, independent_indices
        groups = [[i] for i in range(self.num_creases)]
        independent_indices = np.arange(self.num_creases, dtype=int)
        return groups, independent_indices

    def _expand_offsets(self, reduced: np.ndarray) -> np.ndarray:
        """Map reduced independent variables to the full crease offset vector."""
        full = np.zeros(self.num_creases, dtype=float)
        for group_idx, group in enumerate(self.optimization_groups):
            value = reduced[group_idx]
            for crease_idx in group:
                full[crease_idx] = value
        return full

    def _reduce_offsets(self, full: np.ndarray) -> np.ndarray:
        """Extract independent variables from a full crease offset vector."""
        return np.copy(full[self.independent_indices])

    def _create_batch_json(self) -> str:
        """
        创建批量仿真用的JSON文件

        将原始折纸结构复制batch_size份，每份在x/y方向上有偏移，避免相互干涉。
        这样可以一次性评估多个候选解。

        :return: 生成的JSON文件路径
        """
        batch_data = self._construct_batch_data(self.original_data, self.batch_size)

        base_name = os.path.basename(self.json_path).replace(".json", "")
        batch_json_name = f"{base_name}_batch_{self.batch_size}.json"
        if self._batch_json_dir:
            os.makedirs(self._batch_json_dir, exist_ok=True)
            batch_json_path = os.path.join(self._batch_json_dir, batch_json_name)
        else:
            batch_json_path = os.path.join(os.path.dirname(self.json_path), batch_json_name)

        with open(batch_json_path, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, indent=2)

        if not self._quiet:
            print(f"[批量JSON] 已创建批量仿真文件: {batch_json_path}")
            print("[Batch JSON] Batch simulation file created")

        return batch_json_path

    def _construct_batch_data(self, original_data: Dict, batch_size: int) -> Dict:
        """
        构造批量数据

        将原始折纸数据复制batch_size份，在横向(x)和纵向(y)形成网格布局。
        """
        batch_data = {
            "kps": [],
            "lines": [],
            "units": [],
            "line_features": [],
            "strings": original_data.get("strings", {"type": [], "id": [], "reverse": []}),
            "P_candidators": original_data.get("P_candidators", {"points": [], "connections": []}),
            "contributions": [],
            "split_num": batch_size,
        }

        if "crease_angle" in original_data:
            batch_data["crease_angle"] = []

        kps = original_data.get("kps", [])
        if len(kps) > 0:
            xs = [kp[0] for kp in kps]
            ys = [kp[1] for kp in kps]
            width = max(xs) - min(xs) if len(xs) > 1 else 100
            height = max(ys) - min(ys) if len(ys) > 1 else 100
        else:
            width = 100
            height = 100

        grid_cols = int(np.ceil(np.sqrt(batch_size)))
        grid_rows = int(np.ceil(batch_size / grid_cols))

        spacing_x = width * 1.2
        spacing_y = height * 1.2

        if not getattr(self, "_quiet", False):
            print(f"[网格布局] Grid layout: {grid_rows} rows x {grid_cols} cols")
            print(f"[间距] Spacing: x={spacing_x:.1f}mm, y={spacing_y:.1f}mm")

        for b in range(batch_size):
            row = b // grid_cols
            col = b % grid_cols

            offset_x = col * spacing_x
            offset_y = row * spacing_y

            for kp in original_data.get("kps", []):
                batch_data["kps"].append([kp[0] + offset_x, kp[1] + offset_y])

            for line in original_data.get("lines", []):
                batch_data["lines"].append(
                    [
                        [line[0][0] + offset_x, line[0][1] + offset_y],
                        [line[1][0] + offset_x, line[1][1] + offset_y],
                    ]
                )

            for unit in original_data.get("units", []):
                new_unit = []
                for point in unit:
                    new_unit.append([point[0] + offset_x, point[1] + offset_y, point[2]])
                batch_data["units"].append(new_unit)

            for feature in original_data.get("line_features", []):
                batch_data["line_features"].append(copy.deepcopy(feature))

            for contrib in original_data.get("contributions", []):
                batch_data["contributions"].append(copy.deepcopy(contrib))

            if "crease_angle" in original_data:
                for target in original_data["crease_angle"]:
                    batch_data["crease_angle"].append(copy.deepcopy(target))

        return batch_data

    def _compute_crease_lengths(self) -> np.ndarray:
        """
        计算每条折痕的欧氏长度 / Compute the Euclidean length of each crease.

        :return: 长度数组，索引对应 crease_info / Array of lengths indexed by crease_info position.
        """
        lines = self.original_data.get("lines", [])
        lengths = np.zeros(self.num_creases, dtype=float)
        for i, info in enumerate(self.crease_info):
            line_idx = info["line_index"]
            if line_idx < len(lines):
                p0 = lines[line_idx][0]
                p1 = lines[line_idx][1]
                lengths[i] = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
            else:
                lengths[i] = 1.0  # fallback to avoid division by zero
        return lengths

    def _normalize_offsets_by_crease_length(self, offsets: np.ndarray) -> np.ndarray:
        """
        按折痕长度归一化高度偏移量 / Normalize height offsets by crease length.

        Each offset is divided by the length of its corresponding crease so
        that the resulting values are dimensionless fractions of crease length.
        This puts creases of different sizes on a comparable scale and is
        available to all optimization algorithms.

        :param offsets: 绝对高度偏移量 (num_creases,) / Absolute height offsets.
        :return: 归一化偏移量 / Normalized offsets (offset / crease_length).
        """
        # lengths = self._compute_crease_lengths()
        # return offsets / np.where(lengths > 0, lengths, 1.0)

        return offsets  # disabled; return raw offsets without normalization

    def _denormalize_offsets_by_crease_length(self, normalized_offsets: np.ndarray) -> np.ndarray:
        """
        将归一化偏移量还原为绝对值 / Denormalize offsets back to absolute values.

        :param normalized_offsets: 归一化偏移量 / Normalized offsets (offset / crease_length).
        :return: 绝对高度偏移量 / Absolute height offsets (normalized * crease_length).
        """
        # lengths = self._compute_crease_lengths()
        # return normalized_offsets * lengths
        
        return normalized_offsets  # disabled; return raw offsets without denormalization

    def _horizontal_crease_mask(self, angle_threshold_deg: float = 10.0) -> np.ndarray:
        """
        Return a boolean array (num_creases,) that is True for creases within
        angle_threshold_deg degrees of horizontal (|dy|/length < sin(threshold)).
        """
        lines = self.original_data.get("lines", [])
        sin_threshold = np.sin(np.deg2rad(angle_threshold_deg))
        mask = np.zeros(self.num_creases, dtype=bool)
        for i, info in enumerate(self.crease_info):
            idx = info["line_index"]
            if idx < len(lines):
                p0, p1 = lines[idx][0], lines[idx][1]
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                length = np.hypot(dx, dy)
                if length > 0 and abs(dy) / length < sin_threshold:
                    mask[i] = True
        return mask

    def _optimizer_bound_lo(self) -> float:
        """Lower bound for each optimizer multiplicand (dimensionless, ≥ 1)."""
        return 1.0

    def _optimizer_bound_hi(self) -> float:
        """Upper bound for each optimizer multiplicand: max_offset / min_thickness."""
        return self.max_offset / self.min_thickness

    def _build_optimizer_bounds(self) -> np.ndarray:
        """
        Per-independent-variable bounds in multiplicand space.

        Optimizers search unsigned multiplicands in [1, max_offset/min_thickness];
        physical magnitude is multiplicand × min_thickness. Valley/mountain sign is
        applied later in _apply_constraints.
        """
        lo = self._optimizer_bound_lo()
        hi = self._optimizer_bound_hi()
        return np.array([[lo, hi] for _ in range(self.num_independent)])

    def _build_discrete_magnitude_values(self) -> np.ndarray:
        """Quantised multiplicand levels for margin CMA-ES variants."""
        eps = self.discrete_step * 0.5
        heights = np.arange(self.min_thickness, self.max_offset + eps, self.discrete_step)
        return heights / self.min_thickness

    def _optimizer_vars_to_magnitudes(self, optimizer_vars: np.ndarray) -> np.ndarray:
        """
        Convert optimizer multiplicands to unsigned height magnitudes (mm).

        :param optimizer_vars: Independent multiplicands (× min_thickness → mm).
        :return: Unsigned magnitudes in mm, same shape as optimizer_vars.
        """
        return np.asarray(optimizer_vars, dtype=float) * self.min_thickness

    def _magnitudes_to_optimizer_vars(self, magnitudes: np.ndarray) -> np.ndarray:
        """Convert unsigned height magnitudes (mm) to optimizer multiplicands."""
        return np.asarray(magnitudes, dtype=float) / self.min_thickness

    def _build_initial_mean(self) -> np.ndarray:
        """
        Build the initial mean vector for optimization algorithms.

        Values are unsigned multiplicands in [1, max_offset/min_thickness]. When
        initial_offsets were loaded, their constrained absolute values are converted
        to multiplicands. Otherwise every crease starts at multiplicand 1
        (height = min_thickness).

        :return: Mean vector of shape (num_independent,).
        """
        if self._initial_offsets_full is not None:
            constrained = self._apply_constraints(self._initial_offsets_full)
            multiplicands = self._magnitudes_to_optimizer_vars(np.abs(constrained))
            return self._reduce_offsets(multiplicands)

        mean = np.ones(self.num_creases, dtype=float)
        return self._reduce_offsets(mean)

    def _discretize_offset(self, offset: float) -> float:
        """离散化高度偏移量 / Discretize height offset."""
        sign = 1 if offset >= 0 else -1
        abs_offset = abs(offset)

        abs_offset = max(abs_offset, self.min_thickness)

        n_steps = round((abs_offset - self.min_thickness) / self.discrete_step)
        discrete_abs = self.min_thickness + n_steps * self.discrete_step

        discrete_abs = min(discrete_abs, self.max_offset)

        return sign * discrete_abs

    def _enforce_symmetry(self, offsets: np.ndarray) -> np.ndarray:
        """
        Enforce symmetry on height offsets using PCA-detected crease line pairs.
        Corresponding creases share the same signed height offset (averaged
        within each group).
        """
        # --- disabled ---
        # if not self.symm_mode or not self.symmetry_groups:
        #     return np.copy(offsets)
        # constrained = np.copy(offsets)
        # for group in self.symmetry_groups:
        #     if len(group) < 2:
        #         continue
        #     avg = float(np.mean([constrained[i] for i in group]))
        #     for i in group:
        #         constrained[i] = avg
        # return constrained
        return np.copy(offsets)

    def _apply_constraints(self, offsets: np.ndarray) -> np.ndarray:
        """
        应用约束条件

        Step 1 – Sign enforcement:  valley (type 0) → positive, mountain (type 1) → negative.
        Step 2 – Discretisation:    snap to the grid  [min_thickness, min_thickness+step, …].
        Step 3 – Global M/V gap:    ensure every valley height ≥ every mountain height + 4 mm.
        """
        constrained = np.copy(offsets)

        # ── Step 1: sign enforcement ──────────────────────────────────────────
        for i, info in enumerate(self.crease_info):
            if info["type"] == 0:
                constrained[i] = abs(constrained[i])
            else:
                constrained[i] = -abs(constrained[i])

        # constrained = self._enforce_symmetry(constrained)  # disabled

        # ── Step 2: discretisation ────────────────────────────────────────────
        for i in range(len(constrained)):
            constrained[i] = self._discretize_offset(constrained[i])

        # ── Step 3: global valley-mountain gap ≥ 4 mm ────────────────────────
        valley_indices = [i for i, info in enumerate(self.crease_info) if info["type"] == 0]
        mountain_indices = [i for i, info in enumerate(self.crease_info) if info["type"] == 1]

        if len(valley_indices) > 0 and len(mountain_indices) > 0:
            min_valley = min(constrained[i] for i in valley_indices)
            max_mountain = max(constrained[j] for j in mountain_indices)

            if min_valley < max_mountain + 4.0:
                gap = (max_mountain + 4.0) - min_valley

                for i in valley_indices:
                    constrained[i] += gap / 2 + 0.2
                for j in mountain_indices:
                    constrained[j] -= gap / 2 - 0.2

                for i in range(len(constrained)):
                    constrained[i] = self._discretize_offset(constrained[i])

        # constrained = self._enforce_symmetry(constrained)  # disabled

        return constrained

    def _init_ti(self):
        """安全初始化 Taichi，避免重复初始化。"""
        try:
            ensure_taichi_init(force_cpu=self._force_cpu)
        except Exception:
            pass

    def _set_heights_in_batch_json(self, height_matrix: np.ndarray):
        """在批量JSON中设置高度偏移量 / Set height offsets in batch JSON."""
        with open(self.batch_json_path, "r", encoding="utf-8") as f:
            batch_data = json.load(f)

        line_features = batch_data["line_features"]

        for b in range(self.batch_size):
            offsets = height_matrix[b]

            for c, crease in enumerate(self.crease_info):
                global_index = b * len(self.original_data.get("line_features", [])) + crease["index"]

                if global_index < len(line_features):
                    line_features[global_index]["thick_panel_height"] = float(offsets[c])

        with open(self.batch_json_path, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, indent=2)

    def evaluate_batch(self, height_matrix: np.ndarray, algo_step: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量评估候选解 / Batch-evaluate candidate height offset configurations.

        :param height_matrix: 形状为(batch_size, num_creases)的矩阵
        :return: (每个候选解的折叠百分比, 约束后的高度偏移量矩阵)
        """
        constrained_matrix = np.zeros_like(height_matrix)
        for i in range(self.batch_size):
            constrained_matrix[i] = self._apply_constraints(height_matrix[i])

        self._set_heights_in_batch_json(constrained_matrix)

        if os.environ.get("FRAMEWORK_MP_DRY_RUN") == "1":
            print(
                "[DRY RUN] FRAMEWORK_MP_DRY_RUN=1, skipping simulator and "
                "returning synthetic fitness values"
            )
            fitnesses = np.arange(self.batch_size, dtype=float) + float(algo_step)
            return fitnesses, constrained_matrix

        simulator_name = self.simulator_origami_name()
        batch_json_path = os.path.abspath(self.batch_json_path)

        if self.simulator is None:
            self._init_ti()
            self.simulator = self._create_simulator(simulator_name, algo_step)
            ok = self.simulator.start(batch_json_path, 4, thick_mode=1)
            if not ok:
                ok = self._recover_simulator_after_failed_start(
                    simulator_name,
                    batch_json_path,
                    algo_step,
                )
        else:
            self.simulator.ID = algo_step
            ok = self.simulator.start(batch_json_path, 4, thick_mode=1)
            if not ok:
                ok = self._recover_simulator_after_failed_start(
                    simulator_name,
                    batch_json_path,
                    algo_step,
                )

        step_count = 0
        self.simulator.initializeRunning()
        self.simulator.enable_add_folding_angle = self.fold_angle_step

        while step_count < self.max_steps and self.simulator.window.running:
            self.simulator.step()
            if self.use_gui:
                self.simulator.render()
            if self.simulator.stop():
                if self._should_export_figure(algo_step):
                    self.simulator.outputFigure()
                self.simulator.backupSimulationSetting()
                break
            step_count += 1

        if step_count == self.max_steps:
            if self._should_export_figure(algo_step):
                self.simulator.outputFigure()
            self.simulator.backupSimulationSetting()

        folding_percentages = self._extract_folding_percentages()

        if not self.reuse_simulator:
            self._teardown_simulator()

        return folding_percentages, constrained_matrix

    def _warmup_taichi_jit(
        self,
        height_matrix: np.ndarray,
        algo_step: int,
    ) -> None:
        """Compile Taichi kernels once; caller must hold the cross-process lock."""
        constrained_matrix = np.zeros_like(height_matrix)
        for i in range(self.batch_size):
            constrained_matrix[i] = self._apply_constraints(height_matrix[i])

        self._set_heights_in_batch_json(constrained_matrix)

        self._init_ti()
        self.simulator = self._create_simulator(
            self.simulator_origami_name(),
            algo_step,
        )
        self.simulator.start(os.path.abspath(self.batch_json_path), 4, thick_mode=1)
        self.simulator.initializeRunning()
        self.simulator.enable_add_folding_angle = self.fold_angle_step
        self.simulator.step()

    def _extract_folding_percentages(self) -> np.ndarray:
        """从仿真器中提取每个折纸的折叠百分比 / Extract folding percentages."""
        return self.simulator.reward()

    def _effective_n_processes(self) -> int:
        return self.n_processes

    def shutdown_workers(self) -> None:
        """Close the persistent multiprocessing pool (no-op if not started)."""
        if self._mp_pool is None:
            return
        self._mp_pool.close()
        self._mp_pool.join()
        self._mp_pool = None
        self._mp_pool_worker_count = 0

    def _acquire_mp_pool(self, worker_count: int) -> Any:
        """Return a process pool reused across generations (spawn + Taichi-safe init)."""
        if self._mp_pool is not None and self._mp_pool_worker_count == worker_count:
            return self._mp_pool

        self.shutdown_workers()
        ctx = mp.get_context("spawn")
        ti_lock = ctx.Lock()
        self._mp_pool = ctx.Pool(
            processes=worker_count,
            initializer=_init_mp_worker,
            initargs=(ti_lock,),
        )
        self._mp_pool_worker_count = worker_count
        if not self._quiet:
            print(
                f"[多进程/Multiprocessing] Persistent worker pool started "
                f"(n_processes={worker_count})"
            )
        return self._mp_pool

    def _shared_simulator_name(self) -> str:
        """Export name shared by all MP workers (prefix + batch stem, no _mp)."""
        base = os.path.basename(self.json_path).replace(".json", "")
        stem = f"{base}_batch_{self.batch_size}"
        if self.result_prefix:
            return f"{self.result_prefix}-{stem}"
        return stem

    def _build_worker_payload(self, task: Dict[str, Any]) -> Dict[str, Any]:
        worker_tag = f"b{task['batch_idx']}_{task['algo_step']}"
        return {
            "json_path": self.json_path,
            "batch_size": self.batch_size,
            "population_size": self.population_size,
            "min_thickness": self.min_thickness,
            "discrete_step": self.discrete_step,
            "max_offset": self.max_offset,
            "use_gui": self.use_gui,
            "symm_mode": self.symm_mode,
            "algorithm_key": self.algorithm_key,
            "result_prefix": self.result_prefix,
            "max_steps": self.max_steps,
            "fold_angle_step": self.fold_angle_step,
            "ref_target": self.ref_target,
            "reuse_simulator": self.reuse_simulator,
            "simulator_name": self._shared_simulator_name(),
            "worker_tag": worker_tag,
            "height_matrix": task["height_matrix"],
            "algo_step": task["algo_step"],
            "batch_size_actual": task["batch_size_actual"],
            "batch_idx": task["batch_idx"],
        }

    def _composite_fitness(
        self,
        rewards: List[float],
        constrained: List[np.ndarray],
        diversity_weight: float = 0,
    ) -> List[float]:
        """
        Combined fitness for minimization with optimal value of 0.

        score_i = reward_i / (1 + diversity_weight * std(constrained_offsets_i))

        - reward_i = 0 (best folding, mode 1 or 2) → score = 0 always. ✓
        - Higher reward (worse folding) → larger numerator → higher score.
        - Higher offset std → larger denominator → lower score (exploration bonus).
        - Always in [0, 1] for reward values in [0, 1].

        :param rewards: Raw values from simulator.reward() — lower is better.
        :param constrained: Constrained offset vectors for each candidate.
        :param diversity_weight: Weight alpha on the std term; tune to balance
            exploitation vs. exploration (default 0.3).
            Offsets are signed (valley +, mountain -) so std across creases is
            non-trivial even at minimum thickness.  Practical scale:
              0.01–0.05  mild     (5–20% max score reduction)
              0.05–0.15  moderate (up to ~50% reduction)
              0.15–0.40  strong   (routinely cuts score to <50% of raw reward)
              >0.40      diversity dominates over folding quality
            At the default 0.3 the score is typically reduced by 55–80% for
            mixed mountain/valley patterns — this is on the strong end.
        :return: Composite score list; intended to be minimized by the caller.
        """
        return [
            r / (1.0 + diversity_weight * float(np.std(c)))
            for r, c in zip(rewards, constrained)
        ]

    def evaluate_population(
        self,
        candidate_heights: List[np.ndarray],
        algo_step: int,
    ) -> Tuple[List[float], List[np.ndarray]]:
        """
        Evaluate a population of reduced-offset candidate vectors.

        When n_processes > 1 and use_gui is False, batch chunks run in parallel
        via multiprocessing.Pool (spawn context for Windows safety). The call
        blocks until every batch in this population finishes before returning.
        """
        population_size = len(candidate_heights)
        if population_size == 0:
            return [], []

        num_batches = (population_size + self.batch_size - 1) // self.batch_size
        tasks: List[Dict[str, Any]] = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, population_size)
            batch_size_actual = end_idx - start_idx

            height_matrix = np.zeros((self.batch_size, self.num_creases))
            for i in range(batch_size_actual):
                magnitudes = self._optimizer_vars_to_magnitudes(
                    candidate_heights[start_idx + i]
                )
                height_matrix[i] = self._expand_offsets(magnitudes)

            tasks.append(
                {
                    "batch_idx": batch_idx,
                    "start_idx": start_idx,
                    "batch_size_actual": batch_size_actual,
                    "height_matrix": height_matrix,
                    "algo_step": algo_step + start_idx,
                }
            )

        effective_n_processes = self._effective_n_processes()
        use_parallel = (
            effective_n_processes > 1
            and not self.use_gui
            and num_batches > 1
        )

        if use_parallel:
            if use_gpu:
                print(
                    "[信息/Info] GPU mode: parallel workers run Taichi on CPU "
                    "to avoid device contention"
                )
            print(
                f"[多进程/Multiprocessing] Evaluating {num_batches} batches "
                f"with n_processes={effective_n_processes} "
                f"(shared result folder: {self._shared_simulator_name()})"
            )
            payloads = [self._build_worker_payload(task) for task in tasks]

            pool = self._acquire_mp_pool(min(effective_n_processes, num_batches))
            raw_results = pool.map(_run_mp_worker, payloads)
            raw_results.sort(key=lambda item: item[2])
            print(
                f"[多进程/Multiprocessing] All {num_batches} batches finished; "
                "proceeding to next optimizer step"
            )

            fitness_list: List[float] = []
            constrained_list: List[np.ndarray] = []
            for fitnesses, constrained, _batch_idx, batch_size_actual in raw_results:
                for i in range(batch_size_actual):
                    fitness_list.append(float(fitnesses[i]))
                    constrained_list.append(constrained[i])
        else:
            fitness_list = []
            constrained_list = []
            for task in tasks:
                fitnesses, constrained = self.evaluate_batch(
                    task["height_matrix"],
                    task["algo_step"],
                )
                for i in range(task["batch_size_actual"]):
                    fitness_list.append(float(fitnesses[i]))
                    constrained_list.append(constrained[i])

        raw_rewards = list(fitness_list)
        _dw = type(self)._composite_fitness.__defaults__[0]
        variances = [_dw * float(np.var(c)) for c in constrained_list]
        fitness_list = self._composite_fitness(fitness_list, constrained_list)

        self.extract_data["pop_rewards"].append(raw_rewards)
        self.extract_data["pop_variances"].append(variances)
        self.extract_data["pop_fitness"].append(list(fitness_list))
        self.extract_data["min_without_var"].append(float(min(raw_rewards)))

        return fitness_list, constrained_list


_mp_worker_ti_lock = None
_mp_worker_framework: Optional[ThickPanelDesignFramework] = None


def _log_mp_event(event: str, batch_idx: int) -> None:
    timing_dir = os.environ.get("FRAMEWORK_MP_TIMING_DIR")
    if not timing_dir:
        return
    os.makedirs(timing_dir, exist_ok=True)
    log_path = os.path.join(timing_dir, f"worker_{os.getpid()}.log")
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"{time.time():.6f}\t{event}\tbatch={batch_idx}\n")


def _init_mp_worker(ti_lock) -> None:
    global _mp_worker_ti_lock, _mp_worker_framework
    _mp_worker_ti_lock = ti_lock
    _mp_worker_framework = None
    cache_root = os.path.join(
        os.environ.get("TEMP", os.environ.get("TMP", ".")),
        "taichi_mp_cache",
        str(os.getpid()),
    )
    os.makedirs(cache_root, exist_ok=True)
    os.environ["TI_OFFLINE_CACHE_FILE_PATH"] = cache_root


def _create_worker_framework(payload: Dict[str, Any]) -> ThickPanelDesignFramework:
    return ThickPanelDesignFramework(
        json_path=payload["json_path"],
        batch_size=payload["batch_size"],
        population_size=payload["population_size"],
        min_thickness=payload["min_thickness"],
        discrete_step=payload["discrete_step"],
        max_offset=payload["max_offset"],
        use_gui=payload["use_gui"],
        symm_mode=payload["symm_mode"],
        algorithm_key=payload["algorithm_key"],
        result_prefix=payload["result_prefix"],
        max_steps=payload["max_steps"],
        fold_angle_step=payload["fold_angle_step"],
        ref_target=payload["ref_target"],
        reuse_simulator=payload["reuse_simulator"],
        n_processes=1,
        _batch_json_dir=os.path.join(tempfile.gettempdir(), "thick_panel_opt", str(os.getpid())),
        _simulator_name_override=payload["simulator_name"],
        _quiet=True,
        _force_cpu=True,
        _reuse_ti_runtime=True,
    )


def _evaluate_worker_payload(
    payload: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    batch_idx = payload["batch_idx"]
    _log_mp_event("eval_start", batch_idx)

    test_delay = os.environ.get("FRAMEWORK_MP_TEST_DELAY")
    if test_delay:
        time.sleep(float(test_delay))

    fitnesses, constrained = _mp_worker_framework.evaluate_batch(
        payload["height_matrix"],
        payload["algo_step"],
    )
    _log_mp_event("eval_end", batch_idx)
    return (
        fitnesses,
        constrained,
        batch_idx,
        payload["batch_size_actual"],
    )


def _ensure_mp_worker_ready(payload: Dict[str, Any]) -> None:
    """Initialize framework and Taichi once per worker process.

    On Windows, concurrent LLVM JIT context construction (inside ti.init)
    triggers IMAGE_REL_AMD64_ADDR32NB relocation errors.  The lock serializes
    only the ti.init() call (~seconds per worker).  Framework creation,
    OrigamiSimulator init, kernel JIT compilation, and all simulation steps
    run fully in parallel after that.
    """
    global _mp_worker_framework

    if _mp_worker_framework is not None:
        return

    _log_mp_event("bootstrap_start", payload["batch_idx"])

    if os.environ.get("FRAMEWORK_MP_DRY_RUN") != "1":
        # Serialize ti.init() only — LLVM JIT context construction is not safe
        # to call concurrently across processes on Windows.
        if _mp_worker_ti_lock is not None:
            with _mp_worker_ti_lock:
                ensure_taichi_init(force_cpu=True)
        else:
            ensure_taichi_init(force_cpu=True)

    # Framework and batch-JSON creation have no Taichi dependency; run concurrently.
    _mp_worker_framework = _create_worker_framework(payload)
    _log_mp_event("bootstrap_done", payload["batch_idx"])


def _run_mp_worker(payload: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Bootstrap worker on first call, then evaluate batch; all workers run concurrently."""
    _ensure_mp_worker_ready(payload)
    return _evaluate_worker_payload(payload)


def evaluate_batch_worker_impl(
    payload: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Evaluate one batch chunk inside a worker process."""
    return _run_mp_worker(payload)


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    return _ALGORITHM_SPECS[resolve_algorithm(name)]