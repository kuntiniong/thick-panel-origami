import os
import sys
from typing import Tuple

import numpy as np
from cmaes import CMA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.framework import ThickPanelDesignFramework


class ThickPanelCMAFramework(ThickPanelDesignFramework):
    """Thick-panel design framework using CMA-ES optimization."""

    algorithm_key = "cma-es" # config.yml key for this algo
    result_prefix = "cma-es" # output result folder prefix

    def optimize(
        self,
        population_size: int = 16,
        generations: int = 50,
        sigma_init: float = 5.0,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        使用CMA-ES算法优化高度偏移量

        :param population_size: 种群大小 / Population size (λ)，默认16
        :param generations: 迭代代数 / Number of generations
        :param sigma_init: 初始变异强度 / Initial mutation strength
        :param verbose: 是否打印进度 / Whether to print progress
        :return: (最优高度偏移量, 最佳折叠百分比)
        """
        print("\n" + "=" * 60)
        print("开始CMA-ES优化高度偏移量 / Starting CMA-ES height offset optimization")
        print("=" * 60)

        self.data.clear()
        self.extract_data["gen"].clear()
        self.extract_data["avg"].clear()
        self.extract_data["std"].clear()
        self.extract_data["min"].clear()
        self.extract_data["min_without_var"].clear()

        mean = self._build_initial_mean()

        bounds = np.array(
            [[-self.max_offset, self.max_offset] for _ in range(self.num_independent)]
        )

        optimizer = CMA(
            mean=mean,
            sigma=sigma_init,
            bounds=bounds,
            population_size=population_size,
        )

        print("CMA-ES初始化完成 / CMA-ES initialized")
        print(f"  种群大小/Population size: {population_size}")
        print(f"  初始变异强度/Initial sigma: {sigma_init}")
        if self.symm_mode:
            print(
                f"  维度/Dimension: {self.num_independent} independent "
                f"(full creases: {self.num_creases})"
            )
        else:
            print(f"  维度/Dimension: {self.num_creases} (symmetry off)")

        best_fitness = np.inf
        best_solution = None
        generation = 0

        while not optimizer.should_stop() and generation < generations:
            self.data.append([])
            generation += 1

            candidate_heights = []
            for _ in range(optimizer.population_size):
                candidate_heights.append(optimizer.ask())

            fitness_list, constrained_list = self.evaluate_population(
                candidate_heights,
                (generation - 1) * optimizer.population_size,
            )
            self.data[generation - 1] = list(fitness_list)

            solutions = []
            for i in range(optimizer.population_size):
                solutions.append((candidate_heights[i], fitness_list[i]))

            optimizer.tell(solutions)

            array_all_data = np.array(self.data[generation - 1])
            self.extract_data["gen"].append(generation - 1)
            self.extract_data["avg"].append(array_all_data.mean())
            self.extract_data["std"].append(array_all_data.std())
            self.extract_data["min"].append(array_all_data.min())

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

            self._record_best_offset(constrained_list[current_best_idx])
            self.save_extract_data()

        print("\n" + "=" * 60)
        print("CMA-ES优化完成 / CMA-ES optimization completed")
        print("=" * 60)
        print(f"总迭代次数/Total generations: {generation}")
        print(f"最优适应度/Best fitness: {best_fitness:.4f}")
        print("最优高度偏移量/Optimal height offsets:")
        for i, info in enumerate(self.crease_info):
            type_name = "Valley" if info["type"] == 0 else "Mountain"
            print(f"  折痕/Crease {i} ({type_name}): {best_solution[i]:.2f}mm")

        return best_solution, best_fitness


def main():
    from optimization.run import main as run_main

    run_main(default_algorithm="cma-es")


if __name__ == "__main__":
    main()