import os
import sys
from typing import Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.framework import ThickPanelDesignFramework


class ThickPanelManualFramework(ThickPanelDesignFramework):
    """Evaluate one manually provided set of thick-panel offsets."""

    algorithm_key = "manual"
    result_prefix = "manual"

    def optimize(
        self,
        offsets: Sequence[float],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        print("\n" + "=" * 60)
        print("开始手动偏移量评估 / Starting manual height offset evaluation")
        print("=" * 60)

        self.data.clear()
        self.extract_data["gen"].clear()
        self.extract_data["avg"].clear()
        self.extract_data["std"].clear()
        self.extract_data["min"].clear()
        self.extract_data["min_without_var"].clear()
        self.extract_data["best_offset"].clear()
        self.extract_data["best_offset_normalized"].clear()
        self.extract_data["pop_rewards"].clear()
        self.extract_data["pop_variances"].clear()
        self.extract_data["pop_fitness"].clear()
        self.extract_data["num"] = 1

        manual_offsets = np.asarray(offsets, dtype=float)
        if manual_offsets.ndim != 1:
            raise ValueError("manual.offsets must be a 1D list of numbers")

        if manual_offsets.size == self.num_creases:
            candidate = self._reduce_offsets(
                self._magnitudes_to_optimizer_vars(manual_offsets)
            )
        elif manual_offsets.size == self.num_independent:
            candidate = self._magnitudes_to_optimizer_vars(manual_offsets)
        else:
            raise ValueError(
                "manual.offsets length must match the number of creases "
                f"({self.num_creases}) or independent variables ({self.num_independent}); "
                f"got {manual_offsets.size}"
            )

        fitness_list, constrained_list = self.evaluate_population([candidate], 0)
        best_fitness = float(fitness_list[0])
        best_solution = np.copy(constrained_list[0])

        self.data.append([best_fitness])
        self.extract_data["gen"].append(0)
        self.extract_data["avg"].append(best_fitness)
        self.extract_data["std"].append(0.0)
        self.extract_data["min"].append(best_fitness)
        self._record_best_offset(best_solution)
        self.save_extract_data()

        if verbose:
            print(f"手动偏移量适应度/Manual offset fitness: {best_fitness:.4f}")
            print(f"约束后高度偏移量/Constrained height offsets: {best_solution}")

        print("\n" + "=" * 60)
        print("手动偏移量评估完成 / Manual height offset evaluation completed")
        print("=" * 60)
        print(f"适应度/Fitness: {best_fitness:.4f}")
        print("最终高度偏移量/Final height offsets:")
        for i, info in enumerate(self.crease_info):
            type_name = "Valley" if info["type"] == 0 else "Mountain"
            print(f"  折痕/Crease {i} ({type_name}): {best_solution[i]:.2f}mm")

        return best_solution, best_fitness


def main():
    from optimization.run import main as run_main

    run_main(default_algorithm="manual")


if __name__ == "__main__":
    main()