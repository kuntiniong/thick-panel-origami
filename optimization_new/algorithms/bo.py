
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
from skopt import Optimizer
from skopt.space import Real

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization_new.framework import ThickPanelDesignFramework


class ThickPanelBOFramework(ThickPanelDesignFramework):
    """Thick-panel design framework using Bayesian optimization (scikit-optimize)."""

    algorithm_key = "bo" # config.yml key for this algo
    result_prefix = "bo" # output result folder prefix

    def _build_search_space(self, normalize_by_length: bool = False) -> List[Real]:
        """
        Build independent search dimensions for Bayesian optimization.

        :param normalize_by_length: When True, each dimension's bounds are
            divided by the representative crease's length so the optimizer
            works in length-normalized units.  Pass the returned candidates
            through ``_denormalize_offsets_by_crease_length`` before
            sending them to the simulator.
        """
        lo = self._optimizer_bound_lo()
        hi = self._optimizer_bound_hi()
        if normalize_by_length:
            lengths = self._compute_crease_lengths()
            return [
                Real(
                    lo / max(lengths[self.independent_indices[group_idx]], 1.0),
                    hi / max(lengths[self.independent_indices[group_idx]], 1.0),
                    name=f"group_{group_idx}_crease_{self.independent_indices[group_idx]}",
                )
                for group_idx in range(self.num_independent)
            ]
        return [
            Real(
                lo,
                hi,
                name=f"group_{group_idx}_crease_{self.independent_indices[group_idx]}",
            )
            for group_idx in range(self.num_independent)
        ]

    def optimize(
        self,
        population_size: int = 16,
        generations: int = 50,
        n_initial_points: Optional[int] = None,
        base_estimator: str = "GP",
        acq_func: str = "EI",
        random_state: int = 42,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        使用贝叶斯优化（Gaussian Process）优化高度偏移量。

        :param population_size: 每轮建议并评估的候选解数量（对应 CMA-ES 种群大小）
        :param generations: 优化迭代轮数
        :param n_initial_points: GP 拟合前的随机探索点数，默认等于 population_size
        :param base_estimator: 代理模型类型（"GP", "RF", "ET", "GBRT"）
        :param acq_func: 采集函数（"EI", "LCB", "PI", "gp_hedge" 等）
        :param random_state: 随机种子
        :param verbose: 是否打印进度
        :return: (最优高度偏移量, 最佳残余能量)
        """
        print("\n" + "=" * 60)
        print("开始贝叶斯优化高度偏移量 / Starting Bayesian optimization")
        print("=" * 60)

        if n_initial_points is None:
            n_initial_points = population_size

        self.data.clear()
        self.extract_data["gen"].clear()
        self.extract_data["avg"].clear()
        self.extract_data["std"].clear()
        self.extract_data["min"].clear()
        self.extract_data["min_without_var"].clear()
        self.extract_data["num"] = population_size

        dimensions = self._build_search_space()
        initial_mean = self._build_initial_mean()
        x0 = None
        y0 = None
        total_evaluations = 0
        best_fitness = np.inf
        best_solution = None

        if self.has_initial_offsets():
            fitness_list, constrained_list = self.evaluate_population([initial_mean], 0)
            x0 = [initial_mean.tolist()]
            y0 = [float(fitness_list[0])]
            total_evaluations = 1
            best_fitness = float(fitness_list[0])
            best_solution = np.copy(constrained_list[0])

            self.data.append([best_fitness])
            self.extract_data["gen"].append(-1)
            self.extract_data["avg"].append(best_fitness)
            self.extract_data["std"].append(0.0)
            self.extract_data["min"].append(best_fitness)
            self._record_best_offset(best_solution)
            self.save_extract_data()

        optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator=base_estimator,
            acq_func=acq_func,
            n_initial_points=n_initial_points,
            random_state=random_state,
            x0=x0,
            y0=y0,
        )

        print("贝叶斯优化初始化完成 / Bayesian optimizer initialized")
        print(f"  每轮候选数/Points per iteration: {population_size}")
        print(f"  初始随机点数/Initial random points: {n_initial_points}")
        if self.has_initial_offsets():
            print("  初始种子点/Seeded initial point: framework.initial_offsets")
        print(f"  代理模型/Surrogate model: {base_estimator}")
        print(f"  采集函数/Acquisition function: {acq_func}")
        if self.symm_mode:
            print(
                f"  维度/Dimension: {self.num_independent} independent "
                f"(full creases: {self.num_creases})"
            )
        else:
            print(f"  维度/Dimension: {self.num_creases} (symmetry off)")

        for generation in range(1, generations + 1):
            self.data.append([])

            candidates = optimizer.ask(n_points=population_size)
            candidate_heights = [np.asarray(x, dtype=float) for x in candidates]

            fitness_list, constrained_list = self.evaluate_population(
                candidate_heights,
                total_evaluations,
            )
            self.data[generation - 1] = list(fitness_list)

            optimizer.tell(candidates, fitness_list)
            total_evaluations += population_size

            array_all_data = np.array(self.data[generation - 1])
            self.extract_data["gen"].append(generation - 1)
            self.extract_data["avg"].append(float(array_all_data.mean()))
            self.extract_data["std"].append(float(array_all_data.std()))
            self.extract_data["min"].append(float(array_all_data.min()))

            current_best_idx = int(np.argmin(fitness_list))
            if fitness_list[current_best_idx] < best_fitness:
                best_fitness = fitness_list[current_best_idx]
                best_solution = np.copy(constrained_list[current_best_idx])

            if verbose:
                print(f"第{generation}轮/Iter {generation}: 最优适应度 = {best_fitness:.4f}")
                print(f"  高度偏移量/Height offsets: {best_solution}")
                print(f"  本轮最小/Iter min: {array_all_data.min():.4f}")

            self._record_best_offset(constrained_list[current_best_idx])
            self.save_extract_data()

        print("\n" + "=" * 60)
        print("贝叶斯优化完成 / Bayesian optimization completed")
        print("=" * 60)
        print(f"总迭代轮数/Total iterations: {generations}")
        print(f"总评估次数/Total evaluations: {total_evaluations}")
        print(f"最优适应度/Best fitness: {best_fitness:.4f}")
        print("最优高度偏移量/Optimal height offsets:")
        for i, info in enumerate(self.crease_info):
            type_name = "Valley" if info["type"] == 0 else "Mountain"
            print(f"  折痕/Crease {i} ({type_name}): {best_solution[i]:.2f}mm")

        return best_solution, best_fitness


def main():
    from optimization_new.run import main as run_main

    run_main(default_algorithm="bo")


if __name__ == "__main__":
    main()