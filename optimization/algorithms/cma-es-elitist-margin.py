"""(1+λ)-CMA-ES with Margin optimizer for thick-panel origami height offsets.

All upstream classes from elitist-cmaes-with-margin (Watanabe et al., GECCO 2023)
are inlined here so this file is fully self-contained — no sys.path manipulation.

The elitist variant uses a success-based step-size adaptation (SSA) instead of
cumulative step-size adaptation (CSA), and applies elite preservation: the mean
is updated only when the best offspring improves on the incumbent.

Inlined dependency order:
  _Gaussian → _GaussianSigmaACA
  CMAESwM_elitist

code obtained from https://github.com/shiralab/elitist-cmaes-with-margin
"""

import os
import sys
from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.stats import chi2, norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.framework import ThickPanelDesignFramework

# ---------------------------------------------------------------------------
# Distribution model  (elitist-cmaes-with-margin/cma/util/model.py)
# ---------------------------------------------------------------------------

class _Gaussian:
    """Multivariate Gaussian N(m, sigma^2 * C)."""

    def __init__(self, d, m=None, C=None, minimal_eigenval=1e-30, normalize='None'):
        self.normalize = normalize
        self.d = d
        self.m = m if m is not None else np.zeros(self.d)
        self.C = C if C is not None else np.identity(self.d)
        self.init_C = C
        self.min_eigenval = minimal_eigenval

    def _get_C(self):
        return self.__C

    def _set_C(self, C):
        if self.normalize == 'det':
            C_ = (C + C.T) / 2
            coeff = (np.linalg.det(self.init_C) / np.linalg.det(C_)) ** (1 / len(C))
            self.__C = coeff * C_
        elif self.normalize == 'trace':
            C_ = (C + C.T) / 2
            coeff = np.trace(self.init_C) / np.trace(C_)
            self.__C = coeff * C_
        else:
            self.__C = (C + C.T) / 2
        self.__eigen_decomposition()

    C = property(_get_C, _set_C)

    def sampling(self, lam):
        return np.random.randn(lam, self.d).dot(self.sqrtC.T) + self.m

    def encoding(self, lam, X):
        return X

    def loglikelihood(self, X):
        Z = np.dot((X - self.m), self.invSqrtC.T)
        return (
            -0.5 * (self.d * np.log(2.0 * np.pi) + self.logDetC)
            - 0.5 * np.linalg.norm(Z, axis=1) ** 2
        )

    def terminate_condition(self):
        return (
            (np.min(self.eigvals) < self.min_eigenval)
            | (0 < (np.isinf(self.m).sum() + np.isinf(self.C).sum()))
            | (0 < (np.isnan(self.m).sum() + np.isnan(self.C).sum()))
        )

    def verbose_display(self):
        return ' MinEigVal: %e' % np.min(self.eigvals)

    def __eigen_decomposition(self):
        self.eigvals, self.eigvectors = scipy.linalg.eigh(self.__C, driver='ev')
        B = self.eigvectors
        if np.min(self.eigvals) > 0.:
            D = np.diag(np.sqrt(self.eigvals))
            self.sqrtC = B.dot(D).dot(B.T)
            self.invSqrtC = B.dot(np.diag(np.reciprocal(np.sqrt(self.eigvals)))).dot(B.T)
            self.logDetC = np.log(self.eigvals).sum()
        else:
            print('The minimal eigenvalue became negative!')


class _GaussianSigmaACA(_Gaussian):
    def __init__(self, d, z_space, m=None, C=None, sigma=1., minimal_eigenval=1e-30, normalize='None'):
        super().__init__(d, m=m, C=C, minimal_eigenval=minimal_eigenval, normalize=normalize)
        self.sigma = sigma
        self.zd = len(z_space)
        self.A = np.full(d, 1.)
        lim = (z_space[:, 1:] + z_space[:, :-1]) / 2
        df_a = pd.DataFrame(z_space.T)
        df_li = pd.DataFrame(lim.T)
        self.z_space = df_a.fillna(df_a.max()).values.T
        self.z_lim = df_li.fillna(df_li.max()).values.T
        self.z_lim_low = np.concatenate(
            [self.z_lim.min(axis=1).reshape([self.zd, 1]), self.z_lim], 1
        )
        self.z_lim_up = np.concatenate(
            [self.z_lim, self.z_lim.max(axis=1).reshape([self.zd, 1])], 1
        )
        m_z = m[self.d - self.zd:].reshape([self.zd, 1])
        self.m_z_lim_low = (
            self.z_lim_low
            * np.where(np.sort(np.concatenate([self.z_lim, m_z], 1)) == m_z, 1, 0)
        ).sum(axis=1)
        self.m_z_lim_up = (
            self.z_lim_up
            * np.where(np.sort(np.concatenate([self.z_lim, m_z], 1)) == m_z, 1, 0)
        ).sum(axis=1)
        self.prev_sample = None

    def sampling(self, lam):
        samples = self.sigma * np.random.randn(lam, self.d).dot(self.sqrtC.T) + self.m
        self.prev_sample = samples
        return samples

    def encoding(self, lam, X):
        X = (X - self.m) * self.A + self.m
        num_cont = self.d - self.zd
        X_z = X[:, num_cont:]
        X_z_c = X_z.reshape([lam, self.zd, 1])
        X_z_enc = (
            self.z_space
            * np.where(
                np.sort(np.concatenate([np.tile(self.z_lim, (lam, 1, 1)), X_z_c], 2)) == X_z_c,
                1, 0,
            )
        ).sum(axis=2)
        return np.hstack((X[:, :num_cont], X_z_enc))

    def loglikelihood(self, X):
        Z = np.dot((X - self.m), self.invSqrtC.T) / self.sigma
        return (
            -0.5 * (self.d * np.log(2.0 * np.pi) + self.logDetC)
            - np.log(self.sigma)
            - 0.5 * np.linalg.norm(Z, axis=1) ** 2
        )

    def terminate_condition(self):
        return np.logical_or(
            np.logical_or(
                (self.sigma ** 2) * np.min(self.eigvals) < self.min_eigenval,
                0 < (np.isinf(self.m).sum() + np.isinf(self.C).sum() + np.isinf(self.sigma).sum()),
            ),
            0 < (np.isnan(self.m).sum() + np.isnan(self.C).sum() + np.isnan(self.sigma).sum()),
        )

    def verbose_display(self):
        return (
            ' MinEigVal: %e' % ((self.sigma ** 2) * np.min(self.eigvals))
            + ' Cond: %e' % (np.max(self.eigvals) / np.min(self.eigvals))
            + ' sigma: %e' % (self.sigma ** 2)
        )


# ---------------------------------------------------------------------------
# (1+λ)-CMA-ES with Margin  (elitist-cmaes-with-margin/cma/optimizer/cmaeswm_elitist.py)
# ---------------------------------------------------------------------------

class CMAESwM_elitist:
    """(1+1)-CMA-ES with Margin — elite-preserving variant (Watanabe et al., 2023).

    Uses success-based step-size adaptation (SSA) and updates the covariance and
    mean only when an offspring improves on the incumbent.
    """

    def __init__(
        self,
        d,
        discrete_space,
        sampler,
        m=None,
        C=None,
        sigma=1.,
        minimal_eigenval=1e-30,
        c_cov=None,
        c_p=None,
        damping=None,
        margin=None,
        normalize='None',
        min_problem=True,
        postprocess=False,
        tie_success=True,
        enc_m=True,
    ):
        self.model = _GaussianSigmaACA(
            d, m=m, C=C, sigma=sigma, z_space=discrete_space,
            minimal_eigenval=minimal_eigenval, normalize=normalize,
        )
        self.model_init = _GaussianSigmaACA(
            d, m=m, C=C, sigma=sigma, z_space=discrete_space,
            minimal_eigenval=minimal_eigenval, normalize=normalize,
        )
        self.sampler = sampler
        self.d = d
        self.zd = len(discrete_space)

        self.is_better = (lambda x, y: x < y) if min_problem else (lambda x, y: x > y)
        self.best_X = None
        self.best_eval = None

        # SSA parameters
        self.model.sigma = 1. if sigma is None else sigma
        self.damping = 1. + d / 2. if damping is None else damping
        self.c_p = 1. / 12. if c_p is None else c_p
        self.p_target = 2. / 11.
        self.p_succ = self.p_target

        # CMA parameters
        self.c_cov = 2. / (d ** 2 + 6.) if c_cov is None else c_cov
        self.c_c = 2. / (2. + d)
        self.p_thresh = 0.44
        self.pc = np.zeros(d)
        self.gen_count = 0

        self.margin = margin if margin is not None else 1. / d
        self.postprocess = postprocess
        self.tie_success = tie_success
        self.enc_m = enc_m

    def sampling_model(self):
        return self.model

    def update_step_size(self):
        self.model.sigma *= np.exp(
            (self.p_succ - self.p_target) / (1.0 - self.p_target) / self.damping
        )

    def update(self, X, evals):
        """Update distribution given candidates sorted best-first.

        :param X: continuous samples, shape (lam, d), sorted ascending by fitness
        :param evals: fitness values, shape (lam,), sorted ascending
        """
        self.gen_count += 1

        # initialise on first call
        if self.best_X is None:
            self.best_X = X[0]
            self.best_eval = evals[0]
            self.model.m = X[0]
            return

        lam_succ = self.is_better(evals[0], self.best_eval)
        eq_lam_succ = evals[0] == self.best_eval
        lam_succ = lam_succ or (eq_lam_succ and self.tie_success)

        self.p_succ = (1.0 - self.c_p) * self.p_succ + self.c_p * lam_succ

        if lam_succ:
            self.update_cov((X[0] - self.model.m) / self.model.sigma)
            self.best_X = X[0]
            self.best_eval = evals[0]
            if self.enc_m:
                self.model.m = self.model.encoding(1, X[:1])[0]
            else:
                self.model.m = (X[0] - self.model.m) * self.model.A + self.model.m

        self.update_step_size()
        self.modify_margin()

        if self.postprocess:
            if np.all(self.model.A > 1.):
                min_A = np.min(self.model.A)
                self.model.A /= min_A
                self.model.sigma *= min_A

    def update_cov(self, y):
        if self.p_succ < self.p_thresh:
            self.pc = (1.0 - self.c_c) * self.pc + (self.c_c * (2.0 - self.c_c)) ** 0.5 * y
            self.model.C = (
                (1.0 - self.c_cov) * self.model.C
                + self.c_cov * np.outer(self.pc, self.pc)
            )
        else:
            self.pc = (1.0 - self.c_c) * self.pc
            self.model.C = (
                (1.0 - self.c_cov) * self.model.C
                + self.c_cov * (np.outer(self.pc, self.pc) + self.c_c * (2.0 - self.c_c) * self.model.C)
            )

    def modify_margin(self):
        if self.margin <= 0.:
            return

        num_cont = self.model.d - self.model.zd
        updated_m_int = self.model.m[num_cont:, np.newaxis]
        z_lim_low = np.concatenate(
            [self.model.z_lim.min(axis=1).reshape([self.model.zd, 1]), self.model.z_lim], 1
        )
        z_lim_up = np.concatenate(
            [self.model.z_lim, self.model.z_lim.max(axis=1).reshape([self.model.zd, 1])], 1
        )
        m_z_lim_low = (
            z_lim_low
            * np.where(
                np.sort(np.concatenate([self.model.z_lim, updated_m_int], 1)) == updated_m_int,
                1, 0,
            )
        ).sum(axis=1)
        m_z_lim_up = (
            z_lim_up
            * np.where(
                np.sort(np.concatenate([self.model.z_lim, updated_m_int], 1)) == updated_m_int,
                1, 0,
            )
        ).sum(axis=1)

        z_scale = (self.model.sigma * self.model.A * np.sqrt(np.diag(self.model.C)))[num_cont:]
        updated_m_int = updated_m_int.flatten()
        low_cdf = norm.cdf(m_z_lim_low, loc=updated_m_int, scale=z_scale)
        up_cdf = 1.0 - norm.cdf(m_z_lim_up, loc=updated_m_int, scale=z_scale)
        mid_cdf = 1.0 - (low_cdf + up_cdf)
        edge_mask = np.maximum(low_cdf, up_cdf) > 0.5
        side_mask = ~edge_mask

        C_diag_sq = np.sqrt(np.diag(self.model.C))[num_cont:]

        if np.any(edge_mask):
            modify_mask = np.minimum(low_cdf, up_cdf) < self.margin
            modify_sign = np.sign(self.model.m[num_cont:] - m_z_lim_up)
            m_edge_dist = np.maximum(
                (self.model.m[num_cont:] - m_z_lim_low) * (modify_sign == 1),
                (m_z_lim_up - self.model.m[num_cont:]) * (modify_sign == -1),
            )
            self.model.A[num_cont:] += edge_mask * (
                m_edge_dist / (
                    np.sqrt(chi2.ppf(q=1.0 - 2.0 * self.margin, df=1))
                    * self.model.sigma * C_diag_sq
                )
                - self.model.A[num_cont:]
            )

        low_cdf = np.maximum(low_cdf, self.margin / 2.0)
        up_cdf = np.maximum(up_cdf, self.margin / 2.0)
        denom = low_cdf + mid_cdf + up_cdf - 3.0 * self.margin / 2.0
        excess = 1.0 - low_cdf - up_cdf - mid_cdf
        modified_low_cdf = np.clip(
            low_cdf + excess * (low_cdf - self.margin / 2.0) / denom, 1e-10, 0.5 - 1e-10
        )
        modified_up_cdf = np.clip(
            up_cdf + excess * (up_cdf - self.margin / 2.0) / denom, 1e-10, 0.5 - 1e-10
        )

        chi_low = np.sqrt(chi2.ppf(q=1.0 - 2.0 * modified_low_cdf, df=1))
        chi_up = np.sqrt(chi2.ppf(q=1.0 - 2.0 * modified_up_cdf, df=1))

        self.model.A[num_cont:] += side_mask * (
            (m_z_lim_up - m_z_lim_low) / ((chi_low + chi_up) * self.model.sigma * C_diag_sq)
            - self.model.A[num_cont:]
        )
        self.model.m[num_cont:] += side_mask * (
            (m_z_lim_low * chi_up + m_z_lim_up * chi_low) / (chi_low + chi_up)
            - self.model.m[num_cont:]
        )

    def terminate_condition(self):
        return self.model.terminate_condition()


# ---------------------------------------------------------------------------
# Shim & framework class
# ---------------------------------------------------------------------------

class _SamplerShim:
    """Minimal shim; only needed if the optimizer ever accesses sampler.lam."""

    def __init__(self, lam: int) -> None:
        self.lam = lam


class ThickPanelCMAElitistMarginFramework(ThickPanelDesignFramework):
    """Thick-panel design framework using (1+λ)-CMA-ES with Margin.

    Watanabe et al., "(1+1)-CMA-ES with Margin for Discrete and Mixed-Integer
    Problems", GECCO 2023.  At each generation, λ=population_size candidates
    are evaluated and the best is passed to the elitist update step.  The
    covariance and mean are updated only when the best offspring improves on
    the incumbent; the step-size is adapted every generation via a success
    probability tracker (SSA).
    """

    algorithm_key = "cma-es-elitist-margin"
    result_prefix = "cma-es-elitist-margin"

    def optimize(
        self,
        population_size: int = 16,
        generations: int = 50,
        sigma_init: float = 5.0,
        margin: Optional[float] = None,
        enc_m: bool = True,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        使用(1+λ)-CMA-ES with Margin算法优化高度偏移量

        :param population_size: λ — candidates sampled per generation step
        :param generations: 迭代代数 / Number of generations
        :param sigma_init: 初始步长 / Initial step-size sigma
        :param margin: Margin parameter α; if None defaults to 1/dim
        :param enc_m: Discretize the mean vector after each successful update
        :param verbose: 是否打印进度 / Whether to print progress
        :return: (最优高度偏移量, 最佳折叠百分比)
        """
        print("\n" + "=" * 60)
        print(
            "开始(1+λ)-CMA-ES-Elitist-Margin优化 / "
            "Starting (1+λ)-CMA-ES-Elitist-Margin optimization"
        )
        print("=" * 60)

        self.data.clear()
        self.extract_data["gen"].clear()
        self.extract_data["avg"].clear()
        self.extract_data["std"].clear()
        self.extract_data["min"].clear()
        self.extract_data["min_without_var"].clear()

        # Discrete magnitude grid; sign applied in _apply_constraints.
        all_vals = self._build_discrete_magnitude_values()
        discrete_space = np.tile(all_vals, (self.num_independent, 1))

        # Initial mean: built by the framework
        # uniformly regardless of which algorithm is used.
        mean = self._build_initial_mean()

        dim = self.num_independent
        margin_val = margin if margin is not None else 1.0 / dim
        shim = _SamplerShim(population_size)

        optimizer = CMAESwM_elitist(
            dim,
            discrete_space,
            shim,
            m=mean,
            sigma=sigma_init,
            margin=margin_val,
            enc_m=enc_m,
            min_problem=True,
        )

        print("(1+λ)-CMA-ES-Elitist-Margin初始化完成 / initialized")
        print(f"  λ (population per step): {population_size}")
        print(f"  初始步长/Initial sigma: {sigma_init}")
        print(f"  Margin parameter α: {margin_val:.6f}")
        print(f"  Mean discretization enc_m: {enc_m}")
        if self.symm_mode:
            print(f"  维度/Dimension: {dim} independent (full creases: {self.num_creases})")
        else:
            print(f"  维度/Dimension: {dim}")
        print(f"  离散值数量/Discrete values per variable: {len(all_vals)}")

        best_fitness = np.inf
        best_solution = None
        generation = 0

        while not optimizer.terminate_condition() and generation < generations:
            self.data.append([])
            generation += 1

            model = optimizer.sampling_model()
            X = model.sampling(population_size)               # continuous (pop, dim)
            X_enc = model.encoding(population_size, X)        # discrete-snapped (pop, dim)

            fitness_list, constrained_list = self.evaluate_population(
                list(X_enc),
                (generation - 1) * population_size,
            )
            self.data[generation - 1] = list(fitness_list)

            # Sort best-first; elitist update uses only X[0] / evals[0]
            order = np.argsort(fitness_list)
            optimizer.update(X[order], np.array(fitness_list)[order])

            array_all_data = np.array(self.data[generation - 1])
            self.extract_data["gen"].append(generation - 1)
            self.extract_data["avg"].append(float(array_all_data.mean()))
            self.extract_data["std"].append(float(array_all_data.std()))
            self.extract_data["min"].append(float(array_all_data.min()))

            current_best_idx = int(order[0])
            if fitness_list[current_best_idx] < best_fitness:
                best_fitness = fitness_list[current_best_idx]
                best_solution = np.copy(constrained_list[current_best_idx])

                if verbose:
                    print(
                        f"第{generation}代/Gen {generation}: "
                        f"最优适应度 = {best_fitness:.4f} ✓"
                    )
                    print(f"  高度偏移量/Height offsets: {best_solution}")
                    print(f"  当前sigma/Current sigma: {optimizer.model.sigma:.6f}")
                    print(f"  成功率/p_succ: {optimizer.p_succ:.4f}")
            elif verbose:
                print(
                    f"第{generation}代/Gen {generation}: "
                    f"最优适应度 = {best_fitness:.4f}"
                )
                print(f"  当前sigma/Current sigma: {optimizer.model.sigma:.6f}")
                print(f"  成功率/p_succ: {optimizer.p_succ:.4f}")

            self._record_best_offset(constrained_list[current_best_idx])
            self.save_extract_data()

        print("\n" + "=" * 60)
        print("(1+λ)-CMA-ES-Elitist-Margin优化完成 / optimization completed")
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

    run_main(default_algorithm="cma-es-elitist-margin")


if __name__ == "__main__":
    main()
