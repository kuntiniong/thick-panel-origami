"""
厚板折纸高度偏移量计算设计框架
Thick Panel Origami Height Offset Computational Design Framework

该框架用于优化设计厚板折纸中各折痕的高度偏移量，使得折纸能够尽可能完全地折叠。
This framework optimizes the height offset of each crease in thick-panel origami to maximize folding degree.

作者/Author: WorkBuddy
日期/Date: 2026-04-22
"""

import json
import numpy as np
import copy
import sys
import os
from typing import List, Dict, Tuple, Optional, Callable
import warnings
from cmaes import CMA
import gc

# 添加PyGamiX-V7到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phys_sim_pd14 import PD_Origami_Simulator as OrigamiSimulator
from phys_sim_pd14 import ti, data_type, use_gpu

class ThickPanelDesignFramework:
    """
    厚板折纸高度偏移量计算设计框架
    
    该框架通过优化算法（进化策略）来设计折痕的高度偏移量，使得厚板折纸能够尽可能完全地折叠。
    
    约束条件/Constraints:
    1. 高度偏移量绝对值至少为min_thickness（默认2mm），保证板具有厚度
    2. 高度偏移量绝对值按discrete_step（默认0.4mm）离散化
    3. 对于同时包含山折痕和谷折痕的板，谷折痕的有符号高度偏移量必须大于山折痕的有符号高度偏移量，且至少相差4mm
    4. 使用OrigamiSimulator评估折叠程度
    
    :param json_path: 输入JSON文件路径 / Input JSON file path
    :param batch_size: 批量评估大小 / Batch size for parallel evaluation
    :param min_thickness: 最小厚度（高度偏移量绝对值的最小值）/ Minimum thickness
    :param discrete_step: 高度偏移量离散化步长 / Discretization step for height offset
    :param max_offset: 最大高度偏移量 / Maximum height offset
    :param use_gui: 是否使用GUI / Whether to use GUI
    """
    
    def __init__(self, 
                 json_path: str,
                 batch_size: int = 64,
                 population_size: int = 256,
                 min_thickness: float = 2.0,
                 discrete_step: float = 0.4,
                 max_offset: float = 50.0,
                 use_gui: bool = False):
        self.json_path = json_path
        self.batch_size = batch_size
        self.population_size = population_size
        self.min_thickness = min_thickness
        self.discrete_step = discrete_step
        self.max_offset = max_offset
        self.use_gui = use_gui
        
        # 读取原始JSON
        self.original_data = self._load_json(json_path)
        
        # 解析几何信息
        self.crease_info = self._parse_crease_info()
        self.num_creases = len(self.crease_info)
        
        # 创建批量仿真用的JSON
        self.batch_json_path = self._create_batch_json()
        
        # 初始化仿真器（只初始化一次）
        self.simulator = None

        self.data = []
        self.extract_data = {
            "gen": [],
            "avg": [],
            "std": [],
            "min": [],
            "num": self.population_size
        }
        
        print(f"[初始化] 厚板折纸设计框架初始化完成")
        print(f"[Init] Thick panel design framework initialized")
        print(f"  - 折痕数量/Number of creases: {self.num_creases}")
        print(f"  - 批量大小/Batch size: {batch_size}")
        print(f"  - 最小厚度/Min thickness: {min_thickness}mm")
        print(f"  - 离散步长/Discrete step: {discrete_step}mm")
        
    def _load_json(self, path: str) -> Dict:
        """加载JSON文件 / Load JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
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
            # 只关注valley(0)和mountain(1)折痕，忽略border(2)
            if crease_type in [0, 1]:
                crease_info.append({
                    "index": i,
                    "type": crease_type,  # 0=valley, 1=mountain
                    "line_index": i,  # 假设line_features和lines一一对应
                    "original_height": feature.get("thick_panel_height", 0.0)
                })
        
        return crease_info
    
    def _create_batch_json(self) -> str:
        """
        创建批量仿真用的JSON文件
        
        将原始折纸结构复制batch_size份，每份在x/y方向上有偏移，避免相互干涉。
        这样可以一次性评估多个候选解。
        
        :return: 生成的JSON文件路径
        """
        batch_data = self._construct_batch_data(self.original_data, self.batch_size)
        
        # 保存到文件
        base_name = os.path.basename(self.json_path).replace('.json', '')
        batch_json_name = f"{base_name}_batch_{self.batch_size}.json"
        batch_json_path = os.path.join(os.path.dirname(self.json_path), batch_json_name)
        
        with open(batch_json_path, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2)
        
        print(f"[批量JSON] 已创建批量仿真文件: {batch_json_path}")
        print(f"[Batch JSON] Batch simulation file created")
        
        return batch_json_path
    
    def _construct_batch_data(self, original_data: Dict, batch_size: int) -> Dict:
        """
        构造批量数据
        
        将原始折纸数据复制batch_size份，在横向(x)和纵向(y)形成网格布局。
        例如：batch_size=4 时，布局为 2x2 网格
              batch_size=6 时，布局为 3x2 或 2x3 网格
              
        :param original_data: 原始JSON数据
        :param batch_size: 批量大小
        :return: 批量数据字典
        """
        batch_data = {
            "kps": [],
            "lines": [],
            "units": [],
            "line_features": [],
            "strings": original_data.get("strings", {"type": [], "id": [], "reverse": []}),
            "P_candidators": original_data.get("P_candidators", {"points": [], "connections": []}),
            "contributions": [],
            "split_num": batch_size
        }
        
        if "crease_angle" in original_data:
            batch_data["crease_angle"] = []

        # 计算原始模型的尺寸
        kps = original_data.get("kps", [])
        if len(kps) > 0:
            xs = [kp[0] for kp in kps]
            ys = [kp[1] for kp in kps]
            width = max(xs) - min(xs) if len(xs) > 1 else 100
            height = max(ys) - min(ys) if len(ys) > 1 else 100
        else:
            width = 100
            height = 100
        
        # 计算网格布局（尽量接近正方形）
        grid_cols = int(np.ceil(np.sqrt(batch_size)))  # 列数
        grid_rows = int(np.ceil(batch_size / grid_cols))  # 行数
        
        # 间距（原始模型尺寸的1.5倍，确保不重叠）
        spacing_x = width * 1.2
        spacing_y = height * 1.2
        
        print(f"[网格布局] Grid layout: {grid_rows} rows x {grid_cols} cols")
        print(f"[间距] Spacing: x={spacing_x:.1f}mm, y={spacing_y:.1f}mm")
        
        # 复制batch_size份，按网格布局排列
        for b in range(batch_size):
            # 计算在网格中的位置（行、列）
            row = b // grid_cols  # 行索引（y方向）
            col = b % grid_cols    # 列索引（x方向）
            
            # 计算偏移量
            offset_x = col * spacing_x
            offset_y = row * spacing_y
            
            # 复制关键点（添加x和y偏移）
            for kp in original_data.get("kps", []):
                batch_data["kps"].append([kp[0] + offset_x, kp[1] + offset_y])
            
            # 复制折痕线
            for line in original_data.get("lines", []):
                batch_data["lines"].append([
                    [line[0][0] + offset_x, line[0][1] + offset_y],
                    [line[1][0] + offset_x, line[1][1] + offset_y]
                ])
            
            # 复制单元
            for unit in original_data.get("units", []):
                new_unit = []
                for point in unit:
                    new_unit.append([point[0] + offset_x, point[1] + offset_y, point[2]])
                batch_data["units"].append(new_unit)
            
            # 复制折痕特征（高度偏移量稍后设置）
            for feature in original_data.get("line_features", []):
                batch_data["line_features"].append(copy.deepcopy(feature))
            
            # 复制贡献值
            for contrib in original_data.get("contributions", []):
                batch_data["contributions"].append(copy.deepcopy(contrib))

            # 复制折叠目标角
            if "crease_angle" in original_data:
                for target in original_data["crease_angle"]:
                    batch_data["crease_angle"].append(copy.deepcopy(target))

        return batch_data
    
    def _discretize_offset(self, offset: float) -> float:
        """
        离散化高度偏移量
        
        将连续的高度偏移量离散化为指定的步长。
        
        :param offset: 连续的高度偏移量
        :return: 离散化后的高度偏移量
        """
        # 计算符号
        sign = 1 if offset >= 0 else -1
        abs_offset = abs(offset)
        
        # 确保至少为min_thickness
        abs_offset = max(abs_offset, self.min_thickness)
        
        # 离散化：找到最近的离散值
        # 离散值为：min_thickness, min_thickness+discrete_step, min_thickness+2*discrete_step, ...
        n_steps = round((abs_offset - self.min_thickness) / self.discrete_step)
        discrete_abs = self.min_thickness + n_steps * self.discrete_step
        
        # 限制最大值
        discrete_abs = min(discrete_abs, self.max_offset)
        
        return sign * discrete_abs
    
    def _apply_constraints(self, offsets: np.ndarray) -> np.ndarray:
        """
        应用约束条件
        
        1. 强制符号：valley(0)→正值，mountain(1)→负值
        2. 离散化
        3. 确保valley折痕的高度偏移量 - mountain折痕的高度偏移量 >= 4mm
        
        :param offsets: 原始高度偏移量数组
        :return: 约束处理后的高度偏移量数组
        """
        constrained = np.copy(offsets)
        
        # Bug修复5：先强制符号，再离散化。
        # CMA-ES 采样出来的值可能正负不符合物理约束，
        # 离散化只保留绝对值，然后加回正确符号。
        for i, info in enumerate(self.crease_info):
            if info["type"] == 0:   # valley → 必须为正
                constrained[i] = abs(constrained[i])
            else:                   # mountain → 必须为负
                constrained[i] = -abs(constrained[i])
        
        # 离散化所有偏移量
        for i in range(len(constrained)):
            constrained[i] = self._discretize_offset(constrained[i])
        
        # 处理山/谷折痕间距约束
        valley_indices = [i for i, info in enumerate(self.crease_info) if info["type"] == 0]
        mountain_indices = [i for i, info in enumerate(self.crease_info) if info["type"] == 1]
        
        if len(valley_indices) > 0 and len(mountain_indices) > 0:
            min_valley = min(constrained[i] for i in valley_indices)
            max_mountain = max(constrained[j] for j in mountain_indices)
            
            if min_valley < max_mountain + 4.0:
                gap = (max_mountain + 4.0) - min_valley
                
                # 将valley折痕上移一半，mountain折痕下移一半
                for i in valley_indices:
                    constrained[i] += gap / 2 + 0.2
                for j in mountain_indices:
                    constrained[j] -= gap / 2 - 0.2
                
                # 重新离散化（符号已固定，此处直接离散化即可）
                for i in range(len(constrained)):
                    constrained[i] = self._discretize_offset(constrained[i])
        
        return constrained

    def _init_ti(self):
        """安全初始化 Taichi，避免重复初始化。
        Safely initialize Taichi to avoid re-init errors."""
        try:
            if use_gpu:
                ti.init(arch=ti.gpu, default_fp=data_type,
                        fast_math=False, advanced_optimization=False, kernel_profiler=True)
            else:
                ti.init(arch=ti.cpu, default_fp=data_type,
                        fast_math=False, advanced_optimization=False,
                        cpu_max_num_threads=1, kernel_profiler=False, verbose=False)
            # ti.init(arch=ti.cpu, default_fp=data_type, fast_math=False, advanced_optimization=False, cpu_max_num_threads=1, kernel_profiler=False, verbose=False)
        except Exception:
            pass

    def _set_heights_in_batch_json(self, height_matrix: np.ndarray):
        """
        在批量JSON中设置高度偏移量
        
        :param height_matrix: 形状为(batch_size, num_creases)的矩阵，每行是一个候选解
        """
        # 重新加载JSON（避免修改已缓存的数据）
        with open(self.batch_json_path, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        line_features = batch_data["line_features"]
        
        # 为每个折纸副本设置高度偏移量
        for b in range(self.batch_size):
            offsets = height_matrix[b]
            
            for c, crease in enumerate(self.crease_info):
                # 计算在line_features中的全局索引
                global_index = b * len(self.original_data.get("line_features", [])) + crease["index"]
                
                if global_index < len(line_features):
                    # 设置高度偏移量
                    line_features[global_index]["thick_panel_height"] = float(offsets[c])
        
        # 保存修改后的JSON
        with open(self.batch_json_path, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, indent=2)
    
    def evaluate_batch(self, height_matrix: np.ndarray, algo_step: int) -> np.ndarray:
        """
        批量评估候选解
        
        使用OrigamiSimulator批量评估多个高度偏移量配置。
        
        :param height_matrix: 形状为(batch_size, num_creases)的矩阵
        :return: 每个候选解的平均折叠百分比数组，形状为(batch_size,)
        """
        # 应用约束
        constrained_matrix = np.zeros_like(height_matrix)
        for i in range(self.batch_size):
            constrained_matrix[i] = self._apply_constraints(height_matrix[i])
        
        # 设置高度偏移量到JSON
        self._set_heights_in_batch_json(constrained_matrix)

        # 运行仿真
        batch_json_name = os.path.basename(self.batch_json_path).replace('.json', '')

        # 初始化仿真器（如果还没有初始化）
        if self.simulator is None:
            self._init_ti()

            self.simulator = OrigamiSimulator(
                origami_name=batch_json_name,
                use_gui=self.use_gui,
                fast=True,           # 使用快速仿真模式 / Use fast simulation mode
            )

            self.simulator.ID = algo_step
        
        # 启动仿真
        self.simulator.start(batch_json_name, 4, thick_mode=1)
        
        # 运行直到稳定
        max_steps = 300  # 最大步数限制（减少以加速测试）(300 / 60 seconds)
        step_count = 0

        self.simulator.initializeRunning()
        self.simulator.enable_add_folding_angle = 0.03141 #单步目标折角增量
        
        while step_count < max_steps and self.simulator.window.running:
            self.simulator.step()
            if self.use_gui:
                self.simulator.render()
            # 检查是否稳定
            if self.simulator.stop():
                self.simulator.outputFigure()
                break
            
            step_count += 1
        
        if step_count == max_steps:
            self.simulator.outputFigure()
        
        # 获取每个折纸的折叠程度
        folding_percentages = self._extract_folding_percentages()
        
        # if self.use_gui:
        self.simulator.window.destroy()
        
        gc.collect() #清除内存残留

        ti.reset()

        self.simulator = None
        
        return folding_percentages, constrained_matrix
    
    def _extract_folding_percentages(self) -> np.ndarray:
        """
        从仿真器中提取每个折纸的折叠百分比
        
        由于批量仿真中所有折纸都在同一个仿真器中，需要根据crease_angle的索引来区分。
        
        :return: 每个折纸的平均折叠百分比数组
        """
        folding_percentages = self.simulator.reward()
        
        return folding_percentages
    
    def optimize(self, 
                 population_size: int = 16,
                 generations: int = 50,
                 sigma_init: float = 5.0,
                 verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        使用CMA-ES算法优化高度偏移量
        
        使用CMA-ES（协方差矩阵自适应进化策略）来优化高度偏移量。
        CMA-ES是一种高效的黑盒优化算法，特别适合连续参数优化。
        
        :param population_size: 种群大小 / Population size (λ)，默认16
        :param generations: 迭代代数 / Number of generations
        :param sigma_init: 初始变异强度 / Initial mutation strength
        :param verbose: 是否打印进度 / Whether to print progress
        :return: (最优高度偏移量, 最佳折叠百分比)
        """
        print("\n" + "="*60)
        print("开始CMA-ES优化高度偏移量 / Starting CMA-ES height offset optimization")
        print("="*60)

        self.data.clear()
        self.extract_data['gen'].clear()
        self.extract_data['avg'].clear()
        self.extract_data['std'].clear()
        self.extract_data['min'].clear()
        
        # 初始化均值向量（根据折痕类型设置初始值）
        # valley折痕(0)应该在上方（正值），mountain折痕(1)应该在下方（负值）
        mean = np.zeros(self.num_creases)
        for i, info in enumerate(self.crease_info):
            if info["type"] == 0:  # valley
                mean[i] = self.min_thickness
            else:  # mountain
                mean[i] = -self.min_thickness
        
        # 设置边界约束
        bounds = np.array([
            [-self.max_offset, self.max_offset] for _ in range(self.num_creases)
        ])
        
        # 初始化CMA-ES优化器
        optimizer = CMA(
            mean=mean,
            sigma=sigma_init,
            bounds=bounds,
            population_size=population_size,
        )
        
        print(f"CMA-ES初始化完成 / CMA-ES initialized")
        print(f"  种群大小/Population size: {population_size}")
        print(f"  初始变异强度/Initial sigma: {sigma_init}")
        print(f"  维度/Dimension: {self.num_creases}")
        
        best_fitness = np.inf   # 残余能量初始设为正无穷（最小化目标）
        best_solution = None
        generation = 0
        
        # CMA-ES进化循环
        while not optimizer.should_stop() and generation < generations:
            self.data.append([])
            generation += 1
            
            # 生成候选解
            solutions = []
            candidate_heights = []
            
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                candidate_heights.append(x)
            
            # 分批评估（每批batch_size个）
            num_batches = (optimizer.population_size + self.batch_size - 1) // self.batch_size
            fitness_list = []
            constrained_list = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, optimizer.population_size)
                batch_size_actual = end_idx - start_idx
                
                # 构建高度矩阵
                height_matrix = np.zeros((self.batch_size, self.num_creases))
                for i in range(batch_size_actual):
                    height_matrix[i] = candidate_heights[start_idx + i]
                
                # 批量评估
                fitnesses, constrained = self.evaluate_batch(height_matrix, (generation - 1) * num_batches * self.batch_size + start_idx)
                self.data[generation - 1] += list(fitnesses)
                
                for i in range(batch_size_actual):
                    fitness_list.append(fitnesses[i])
                    constrained_list.append(constrained[i])
            
            # 准备tell数据
            # Bug修复3（fitness语义）：reward() 返回残余能量（越小 = 折叠越完全）。
            # CMA-ES 的 tell() 约定为「最小化」目标函数，因此直接传入 energy 即可。
            # 原代码将 (x, energy) 传给 tell 再用 argmin+>，方向完全颠倒。
            # 修复：tell 传 energy（最小化），best 追踪 argmin(energy)，用 < 比较。
            for i in range(optimizer.population_size):
                solutions.append((candidate_heights[i], fitness_list[i]))
            
            # 更新CMA-ES状态
            optimizer.tell(solutions)

            array_all_data = np.array(self.data[generation - 1])
            _avg = array_all_data.mean()
            _std = array_all_data.std()
            _min = array_all_data.min()

            self.extract_data['gen'].append(generation - 1)
            self.extract_data['avg'].append(_avg)
            self.extract_data['std'].append(_std)
            self.extract_data['min'].append(_min)
            
            # 更新最佳解：energy 最小的解是最优解
            current_best_idx = np.argmin(fitness_list)
            if fitness_list[current_best_idx] < best_fitness:
                best_fitness = fitness_list[current_best_idx]
                best_solution = np.copy(constrained_list[current_best_idx])
                
                if verbose:
                    print(f"第{generation}代/Gen {generation}: 最优适应度 = {best_fitness:.4f}")
                    print(f"  高度偏移量/Height offsets: {best_solution}")
                    print(f"  当前sigma/Current sigma: {optimizer._sigma:.4f}")
            elif verbose:
                print(f"第{generation}代/Gen {generation}: 最优适应度 = {best_fitness:.4f}")
                print(f"  高度偏移量/Height offsets: {best_solution}")
                print(f"  当前sigma/Current sigma: {optimizer._sigma:.4f}")

            origami_name = os.path.basename(self.batch_json_path).replace('.json', '')
            with open('./physResult/cdf-' + origami_name + '/data.json', 'w', encoding="utf-8") as f:
                json.dump(self.extract_data, f, indent=4)
        
        print("\n" + "="*60)
        print("CMA-ES优化完成 / CMA-ES optimization completed")
        print("="*60)
        print(f"总迭代次数/Total generations: {generation}")
        print(f"最优适应度/Best fitness: {best_fitness:.4f}")
        print(f"最优高度偏移量/Optimal height offsets:")
        for i, info in enumerate(self.crease_info):
            type_name = "Valley" if info["type"] == 0 else "Mountain"
            print(f"  折痕/Crease {i} ({type_name}): {best_solution[i]:.2f}mm")
        
        return best_solution, best_fitness


def main():
    """
    主函数：测试miura-thick案例
    
    已知正确答案约为 [2, -2, -2, -6]（或其倍数）
    """
    print("="*60)
    print("厚板折纸高度偏移量计算设计框架")
    print("Thick Panel Origami Height Offset Design Framework")
    print("="*60)
    
    np.random.seed(42)
    # 设置路径
    json_path = os.path.join(os.path.dirname(__file__), "descriptionData", "mountain-big-new.json")
    
    if not os.path.exists(json_path):
        print(f"错误：找不到文件 {json_path}")
        print(f"Error: File not found {json_path}")
        return
    
    # 参数设置
    BATCH_SIZE = 40  # 每次仿真评估的候选解数量（CMA-ES种群大小）
    POPULATION_SIZE = BATCH_SIZE * 10
    
    # 创建设计框架
    framework = ThickPanelDesignFramework(
        json_path=json_path,
        batch_size=BATCH_SIZE,
        population_size=POPULATION_SIZE,
        min_thickness=2.0, 
        discrete_step=1.0,
        max_offset=30.0,
        use_gui=0
    )
    
    # 运行CMA-ES优化
    best_solution, best_fitness = framework.optimize(
        population_size=POPULATION_SIZE,  # CMA-ES种群大小等于batch_size
        generations=100,       # 最大迭代代数
        sigma_init=10.0,       # 初始变异强度
        verbose=True
    )
    
    # 验证已知答案
    # print("\n" + "="*60)
    # print("验证已知正确答案 / Verifying known correct answer")
    # print("="*60)
    # known_answer = np.array([2.0, -2.0, -2.0, -6.0])
    
    # # 调整known_answer长度以匹配实际折痕数量
    # if len(known_answer) != framework.num_creases:
    #     print(f"注意：已知答案长度({len(known_answer)})与实际折痕数量({framework.num_creases})不匹配")
    #     print(f"Note: Known answer length doesn't match actual crease number")
    # else:
    #     height_matrix = np.tile(known_answer, (framework.batch_size, 1))
    #     fitnesses, constrained = framework.evaluate_batch(height_matrix, 50)
    #     print(f"已知答案的残余能量/Known answer fitness: {fitnesses[0]:.2f}")
    #     print(f"约束后的高度偏移量/Constrained height offsets: {constrained[0]}")
    
    # print("\n" + "="*60)
    # print("程序结束 / Program finished")
    # print("="*60)


if __name__ == "__main__":
    main()
