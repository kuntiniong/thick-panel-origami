import os
import sys
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization_new.framework import ThickPanelDesignFramework


class ThickPanelDEFramework(ThickPanelDesignFramework):
    """Thick-panel design framework using Differential Evolution (DE)."""

    algorithm_key = "de"
    result_prefix = "de"

    @staticmethod
    def _normalize_strategy(strategy: str) -> str:
        key = strategy.lower().replace("-", "").replace("_", "").replace("/", "")
        aliases = {
            "rand1bin": "rand1bin",
            "rand1": "rand1bin",
            "derand1bin": "rand1bin",
            "best1bin": "best1bin",
            "best1": "best1bin",
            "debest1bin": "best1bin",
        }
        normalized = aliases.get(key)
        if normalized is None:
            supported = ", ".join(sorted(set(aliases.values())))
            raise ValueError(f"Unknown DE strategy '{strategy}'. Supported values: {supported}")
        return normalized

    def _clip_to_bounds(self, vectors: np.ndarray) -> np.ndarray:
        bounds = self._build_optimizer_bounds()
        return np.clip(vectors, bounds[:, 0], bounds[:, 1])

    def _init_population(
        self,
        population_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        bounds = self._build_optimizer_bounds()
        population = rng.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(population_size, self.num_independent),
        )
        if self.has_initial_offsets():
            population[0] = self._build_initial_mean()
        return population

    def _mutate_rand1(
        self,
        population: np.ndarray,
        index: int,
        rng: np.random.Generator,
        mutation_factor: float,
    ) -> np.ndarray:
        candidates = [i for i in range(population.shape[0]) if i != index]
        r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
        return population[r1] + mutation_factor * (population[r2] - population[r3])

    def _mutate_best1(
        self,
        population: np.ndarray,
        best_index: int,
        index: int,
        rng: np.random.Generator,
        mutation_factor: float,
    ) -> np.ndarray:
        candidates = [i for i in range(population.shape[0]) if i != index]
        r1, r2 = rng.choice(candidates, size=2, replace=False)
        return population[best_index] + mutation_factor * (population[r1] - population[r2])

    def _crossover_bin(
        self,
        target: np.ndarray,
        mutant: np.ndarray,
        rng: np.random.Generator,
        crossover_prob: float,
    ) -> np.ndarray:
        dim = target.shape[0]
        mask = rng.random(dim) < crossover_prob
        mask[rng.integers(0, dim)] = True
        return np.where(mask, mutant, target)

    def _build_trial_population(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        rng: np.random.Generator,
        mutation_factor: float,
        crossover_prob: float,
        strategy: str,
    ) -> np.ndarray:
        trials = np.zeros_like(population)
        best_index = int(np.argmin(fitness))

        for index in range(population.shape[0]):
            if strategy == "best1bin":
                mutant = self._mutate_best1(
                    population,
                    best_index,
                    index,
                    rng,
                    mutation_factor,
                )
            else:
                mutant = self._mutate_rand1(population, index, rng, mutation_factor)

            mutant = self._clip_to_bounds(mutant)
            trials[index] = self._crossover_bin(
                population[index],
                mutant,
                rng,
                crossover_prob,
            )

        return self._clip_to_bounds(trials)

    def optimize(
        self,
        population_size: int = 16,
        generations: int = 50,
        mutation_factor: float = 0.8,
        crossover_prob: float = 0.9,
        strategy: str = "rand1bin",
        random_state: int = 42,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize height offsets with Differential Evolution (DE/rand/1/bin or DE/best/1/bin).

        :param population_size: Population size (NP)
        :param generations: Number of generations
        :param mutation_factor: Differential weight F
        :param crossover_prob: Crossover probability CR
        :param strategy: Mutation strategy (rand1bin | best1bin)
        :param random_state: Random seed for the DE sampler
        :param verbose: Whether to print progress
        :return: (best height offsets, best fitness)
        """
        strategy_key = self._normalize_strategy(strategy)
        rng = np.random.default_rng(random_state)

        print("\n" + "=" * 60)
        print("开始差分进化优化高度偏移量 / Starting Differential Evolution optimization")
        print("=" * 60)

        self.data.clear()
        self.extract_data["gen"].clear()
        self.extract_data["avg"].clear()
        self.extract_data["std"].clear()
        self.extract_data["min"].clear()
        self.extract_data["min_without_var"].clear()
        self.extract_data["num"] = population_size

        population = self._init_population(population_size, rng)
        fitness_list, constrained_list = self.evaluate_population(
            [population[i] for i in range(population_size)],
            0,
        )
        fitness = np.asarray(fitness_list, dtype=float)
        constrained = [np.copy(vector) for vector in constrained_list]

        best_index = int(np.argmin(fitness))
        best_fitness = float(fitness[best_index])
        best_solution = np.copy(constrained[best_index])

        self.data.append(list(fitness_list))
        self.extract_data["gen"].append(0)
        self.extract_data["avg"].append(float(fitness.mean()))
        self.extract_data["std"].append(float(fitness.std()))
        self.extract_data["min"].append(float(fitness.min()))
        self._record_best_offset(best_solution)
        self.save_extract_data()

        print("差分进化初始化完成 / Differential Evolution initialized")
        print(f"  种群大小/Population size: {population_size}")
        print(f"  变异因子/Mutation factor F: {mutation_factor}")
        print(f"  交叉概率/Crossover probability CR: {crossover_prob}")
        print(f"  策略/Strategy: {strategy_key}")
        if self.has_initial_offsets():
            print("  初始种子个体/Seeded individual: framework.initial_offsets")
        if self.symm_mode:
            print(
                f"  维度/Dimension: {self.num_independent} independent "
                f"(full creases: {self.num_creases})"
            )
        else:
            print(f"  维度/Dimension: {self.num_creases} (symmetry off)")

        total_evaluations = population_size

        for generation in range(1, generations + 1):
            trials = self._build_trial_population(
                population,
                fitness,
                rng,
                mutation_factor,
                crossover_prob,
                strategy_key,
            )

            trial_fitness_list, trial_constrained_list = self.evaluate_population(
                [trials[i] for i in range(population_size)],
                total_evaluations,
            )
            total_evaluations += population_size
            trial_fitness = np.asarray(trial_fitness_list, dtype=float)

            improved = trial_fitness <= fitness
            for index in range(population_size):
                if improved[index]:
                    population[index] = trials[index]
                    fitness[index] = trial_fitness[index]
                    constrained[index] = np.copy(trial_constrained_list[index])

            self.data.append(list(fitness))
            self.extract_data["gen"].append(generation)
            self.extract_data["avg"].append(float(fitness.mean()))
            self.extract_data["std"].append(float(fitness.std()))
            self.extract_data["min"].append(float(fitness.min()))

            current_best_index = int(np.argmin(fitness))
            if fitness[current_best_index] < best_fitness:
                best_fitness = float(fitness[current_best_index])
                best_solution = np.copy(constrained[current_best_index])

            if verbose:
                print(f"第{generation}代/Gen {generation}: 最优适应度 = {best_fitness:.4f}")
                print(f"  高度偏移量/Height offsets: {best_solution}")
                print(f"  本轮改进个体/Improved individuals: {int(improved.sum())}/{population_size}")

            self._record_best_offset(constrained[current_best_index])
            self.save_extract_data()

        print("\n" + "=" * 60)
        print("差分进化优化完成 / Differential Evolution optimization completed")
        print("=" * 60)
        print(f"总迭代次数/Total generations: {generations}")
        print(f"总评估次数/Total evaluations: {total_evaluations}")
        print(f"最优适应度/Best fitness: {best_fitness:.4f}")
        print("最优高度偏移量/Optimal height offsets:")
        for i, info in enumerate(self.crease_info):
            type_name = "Valley" if info["type"] == 0 else "Mountain"
            print(f"  折痕/Crease {i} ({type_name}): {best_solution[i]:.2f}mm")

        return best_solution, best_fitness


def main():
    from optimization_new.run import main as run_main

    run_main(default_algorithm="de")


if __name__ == "__main__":
    main()