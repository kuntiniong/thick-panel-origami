"""CMA-ES with Margin optimizer for thick-panel origami height offsets.

All upstream classes from CMA-ES_with_Margin (Hamano et al., 2022) are inlined
here so this file is fully self-contained — no sys.path manipulation required.

Inlined dependency order:
  CMAParam → CMAWeight → CMAWeightWithNegativeWeights
  Model → Gaussian → GaussianSigmaACA
  BaseOptimizer → CMAESwM

code obtained from https://github.com/EvoConJP/CMA-ES_with_Margin
"""

import copy
import os
import sys
from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.stats import chi2, norm, rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.framework import ThickPanelDesignFramework

# ---------------------------------------------------------------------------
# CMA parameters  (CMA-ES_with_Margin/cma/optimizer/cmaeswm.py :: CMAParam)
# ---------------------------------------------------------------------------

class CMAParam:
    @staticmethod
    def pop_size(dim):
        return 4 + int(np.floor(3 * np.log(dim)))

    @staticmethod
    def mu_eff(lam, weights=None):
        if weights is None and lam < 4:
            weights = CMAWeight(4).w
        if weights is None:
            weights = CMAWeight(lam).w
        w_1 = np.absolute(weights).sum()
        return w_1 ** 2 / weights.dot(weights)

    @staticmethod
    def c_1(dim, mueff):
        return 2.0 / ((dim + 1.3) * (dim + 1.3) + mueff)

    @staticmethod
    def c_mu(dim, mueff, c1=0., alpha_mu=2.):
        return np.minimum(
            1. - c1,
            alpha_mu * (mueff - 2. + 1. / mueff) / ((dim + 2.) ** 2 + alpha_mu * mueff / 2.),
        )

    @staticmethod
    def c_c(dim, mueff):
        return (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)

    @staticmethod
    def c_sigma(dim, mueff):
        return (mueff + 2.0) / (dim + mueff + 5.0)

    @staticmethod
    def damping(dim, mueff):
        return (
            1.0
            + 2.0 * np.maximum(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0)
            + CMAParam.c_sigma(dim, mueff)
        )

    @staticmethod
    def chi_d(dim):
        return np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))


# ---------------------------------------------------------------------------
# Recombination weights  (CMA-ES_with_Margin/cma/util/weight.py)
# ---------------------------------------------------------------------------

class CMAWeight:
    def __init__(self, lam, dim=None, min_problem=True):
        self.lam = lam
        self.dim = dim
        self.min_problem = min_problem
        self.w = np.maximum(
            np.log((self.lam + 1.) / 2.) - np.log(np.arange(self.lam) + 1.),
            np.zeros(self.lam),
        )
        self.w = self.w / self.w.sum() if self.w.sum() != 0 else self.w
        self.weights = np.zeros_like(self.w)

    def __call__(self, evals):
        evals = evals if self.min_problem else -evals
        index = np.argsort(evals)
        self.weights[index] = self.w
        unique_val, count = np.unique(evals, return_counts=True)
        if len(evals) == len(unique_val):
            return self.weights
        for u_val in unique_val[count > 1]:
            dup = np.where(evals == u_val)
            self.weights[dup] = self.weights[dup].mean()
        return self.weights


class CMAWeightWithNegativeWeights:
    def __init__(self, lam, dim, min_problem=True):
        self.lam = lam
        self.dim = dim
        self.min_problem = min_problem
        self.w_prime = np.log((self.lam + 1.) / 2.) - np.log(np.arange(self.lam) + 1.)
        self.mu_eff = (
            np.sum(self.w_prime[self.w_prime > 0]) ** 2
            / np.sum(self.w_prime[self.w_prime > 0] ** 2)
        )
        self.mu_eff_neg = (
            np.sum(self.w_prime[self.w_prime <= 0]) ** 2
            / np.sum(self.w_prime[self.w_prime <= 0] ** 2)
        )
        self.c_1 = CMAParam.c_1(self.dim, self.mu_eff)
        self.c_mu = CMAParam.c_mu(self.dim, self.mu_eff, self.c_1)
        self.alpha_mu_neg = 1 + self.c_1 / self.c_mu
        self.alpha_mu_eff_neg = 1 + (2 * self.mu_eff_neg / (self.mu_eff + 2))
        self.alpha_pos_def_neg = (1 - self.c_1 - self.c_mu) / (self.dim * self.c_mu)
        self.w = np.zeros_like(self.w_prime)
        self.w[self.w_prime >= 0] = (
            self.w_prime[self.w_prime >= 0] / np.abs(self.w_prime[self.w_prime >= 0]).sum()
        )
        self.w[self.w_prime < 0] = self.w_prime[self.w_prime < 0] * min(
            self.alpha_mu_neg, self.alpha_mu_eff_neg, self.alpha_pos_def_neg
        ) / np.abs(self.w_prime[self.w_prime < 0]).sum()
        self.weights = np.zeros_like(self.w_prime)

    def __call__(self, evals):
        evals = evals if self.min_problem else -evals
        index = np.argsort(evals)
        self.weights[index] = self.w
        unique_val, count = np.unique(evals, return_counts=True)
        if len(evals) == len(unique_val):
            return self.weights
        for u_val in unique_val[count > 1]:
            dup = np.where(evals == u_val)
            self.weights[dup] = self.weights[dup].mean()
        return self.weights


# ---------------------------------------------------------------------------
# Distribution model  (CMA-ES_with_Margin/cma/util/model.py)
# ---------------------------------------------------------------------------

class _Model:
    """Abstract base for distribution models."""

    @abstractmethod
    def sampling(self, lam): ...

    @abstractmethod
    def loglikelihood(self, X): ...

    def terminate_condition(self):
        return False


class _Gaussian(_Model):
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

    def sampling(self, lam):
        return self.sigma * np.random.randn(lam, self.d).dot(self.sqrtC.T) + self.m

    def encoding(self, lam, X):
        X = (X - self.m) * self.A + self.m
        num_cont = self.d - self.zd
        X_z = X[:, num_cont:]
        X_z_c = X_z.reshape([lam, self.zd, 1])
        X_z_enc = (
            self.z_space
            * np.where(
                np.sort(np.concatenate([np.tile(self.z_lim, (lam, 1, 1)), X_z_c], 2)) == X_z_c,
                1,
                0,
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


# ---------------------------------------------------------------------------
# CMA-ES with Margin optimizer  (CMA-ES_with_Margin/cma/optimizer/cmaeswm.py)
# ---------------------------------------------------------------------------

class CMAESwM:
    """CMA-ES with Margin (Hamano et al., 2022)."""

    def __init__(
        self,
        d,
        discrete_space,
        weight_func,
        sampler,
        m=None,
        C=None,
        sigma=1.,
        minimal_eigenval=1e-30,
        lam=None,
        c_m=1.,
        c_1=None,
        c_mu=None,
        c_c=None,
        c_sigma=None,
        damping=None,
        alpha_mu=2.,
        margin=None,
        restart=None,
        mean_regenerate_func=None,
        normalize='None',
        reset_margin=False,
        local_restart=None,
    ):
        self.model = _GaussianSigmaACA(
            d, m=m, C=C, sigma=sigma, z_space=discrete_space,
            minimal_eigenval=minimal_eigenval, normalize=normalize,
        )
        self.model_init = _GaussianSigmaACA(
            d, m=m, C=C, sigma=sigma, z_space=discrete_space,
            minimal_eigenval=minimal_eigenval, normalize=normalize,
        )
        self.weight_func = weight_func
        self.sampler = sampler
        self.lam = lam if lam is not None else CMAParam.pop_size(d)
        self.d = d
        self.zd = len(discrete_space)

        self.alpha_mu = alpha_mu
        self.mu_eff = CMAParam.mu_eff(self.lam)
        self.c_1 = CMAParam.c_1(d, self.mu_eff) if c_1 is None else c_1
        self.c_mu = CMAParam.c_mu(d, self.mu_eff, c1=self.c_1, alpha_mu=alpha_mu) if c_mu is None else c_mu
        self.c_c = CMAParam.c_c(d, self.mu_eff) if c_c is None else c_c
        self.c_sigma = CMAParam.c_sigma(d, self.mu_eff) if c_sigma is None else c_sigma
        self.damping = CMAParam.damping(d, self.mu_eff) if damping is None else damping
        self.chi_d = CMAParam.chi_d(d)
        self.c_m = c_m

        self.ps = np.zeros(d)
        self.pc = np.zeros(d)
        self.gen_count = 0

        self.max_restart = restart if restart is not None else 9
        self.restart_count = 0
        self.eval_hist = []
        self.best_eval = np.inf
        self.not_improve_ite = 0
        self.best_evals = []
        self.Tolfun = 1e-12
        self.TolX = 1e-12
        self.maxcond = 1e14
        self.mean_regenerate_func = mean_regenerate_func
        self.reset_margin = reset_margin
        self.local_restart = local_restart
        self.best_candidate = np.zeros(self.d)

        self.margin = margin if margin is not None else 1 / (d * lam)

    def sampling_model(self):
        return self.model

    def update(self, X, evals):
        self.gen_count += 1

        best_eval_t = np.min(evals)
        self.eval_hist.insert(0, best_eval_t)
        if len(self.eval_hist) > 10 + 30 * self.d / self.lam:
            self.eval_hist.pop()
        if self.best_eval > best_eval_t:
            self.best_eval = best_eval_t
            self.best_candidate = X[np.argmin(evals)]
            self.not_improve_ite = 0
        else:
            self.not_improve_ite += 1

        weights = self.weight_func(evals)
        Y = (X - self.model.m) / self.model.sigma
        weights_for_mean = np.zeros_like(weights)
        weights_for_mean[weights > 0] = weights[weights > 0]
        m_diff = self.model.sigma * (weights_for_mean * Y.T).sum(axis=1)

        weights_for_cov = np.zeros_like(weights)
        if np.any(weights < 0):
            weights_for_cov[weights > 0] = weights[weights > 0]
            weights_for_cov[weights < 0] = weights[weights < 0] * self.d / (
                scipy.linalg.norm(np.dot(self.model.invSqrtC, Y[weights < 0].T), axis=0) ** 2
                + 1e-10
            )
        else:
            weights_for_cov = weights
        C_rank_mu = np.dot(weights_for_cov * Y.T, Y) - weights.sum() * self.model.C

        hsig = 1.
        if self.c_1 != 0. or self.damping != np.inf:
            self.ps = (1.0 - self.c_sigma) * self.ps + np.sqrt(
                self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff
            ) * np.dot(self.model.invSqrtC, self.c_m * m_diff / self.model.sigma)
            hsig = (
                1.
                if scipy.linalg.norm(self.ps)
                / np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * self.gen_count + 1))
                < (1.4 + 2.0 / (self.model.d + 1.0)) * self.chi_d
                else 0.
            )
            self.pc = (1.0 - self.c_c) * self.pc + hsig * np.sqrt(
                self.c_c * (2.0 - self.c_c) * self.mu_eff
            ) * self.c_m * m_diff / self.model.sigma
        if self.damping != np.inf:
            self.model.sigma *= np.exp(
                self.c_sigma / self.damping * (scipy.linalg.norm(self.ps) / self.chi_d - 1.0)
            )

        self.model.m = self.model.m + self.c_m * m_diff
        self.model.C = (
            self.model.C
            + (1.0 - hsig) * self.c_1 * self.c_c * (2.0 - self.c_c) * self.model.C
            + self.c_1 * (np.outer(self.pc, self.pc) - self.model.C)
            + self.c_mu * C_rank_mu
        )

        # margin correction
        if self.margin > 0.:
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

            if np.any(edge_mask):
                modify_mask = np.minimum(low_cdf, up_cdf) < self.margin
                modify_sign = np.sign(self.model.m[num_cont:] - m_z_lim_up)
                dist = self.model.sigma * self.model.A[num_cont:] * np.sqrt(
                    chi2.ppf(q=1.0 - 2.0 * self.margin, df=1) * np.diag(self.model.C)[num_cont:]
                )
                self.model.m[num_cont:] += modify_mask * edge_mask * (
                    m_z_lim_up + modify_sign * dist - self.model.m[num_cont:]
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
            C_diag_sq = np.sqrt(np.diag(self.model.C))[num_cont:]

            self.model.A[num_cont:] += side_mask * (
                (m_z_lim_up - m_z_lim_low) / ((chi_low + chi_up) * self.model.sigma * C_diag_sq)
                - self.model.A[num_cont:]
            )
            self.model.m[num_cont:] += side_mask * (
                (m_z_lim_low * chi_up + m_z_lim_up * chi_low) / (chi_low + chi_up)
                - self.model.m[num_cont:]
            )

        # restart check (IPOP) — only active when max_restart >= 0
        flag_count = self.restart_count < self.max_restart
        flag_not_improve = self.not_improve_ite > int(100 + 100 * (self.d ** 1.5) / self.lam)
        flag_min_std_pc = (
            max(max(self.model.sigma * np.sqrt(np.diag(self.model.C))), max(self.model.sigma * self.pc))
            < self.TolX
        )
        flag_equal_hist = (
            len(self.eval_hist) >= 10 + 30 * self.d / self.lam
            and max(self.eval_hist) - min(self.eval_hist) < self.Tolfun
        )
        tmDim = self.gen_count % self.d
        D = np.sort(np.sqrt(np.diag(self.model.C)))[::-1]
        flag_noeffectaxis = all(
            self.model.m == self.model.m + 0.1 * self.model.sigma * D[tmDim] * self.model.eigvectors[:, tmDim]
        )
        flag_noeffectcoord = any(
            self.model.m == self.model.m + 0.2 * self.model.sigma * np.sqrt(np.diag(self.model.C))
        )
        flag_conditioncov = np.linalg.cond(self.model.C) > self.maxcond

        if flag_count and (
            flag_not_improve or flag_min_std_pc or flag_equal_hist
            or flag_noeffectaxis or flag_noeffectcoord or flag_conditioncov
        ):
            self.sampler.lam = 2 * self.sampler.lam
            self.lam = 2 * self.lam
            if isinstance(self.weight_func, CMAWeight):
                self.weight_func = CMAWeight(self.lam, min_problem=True)
            else:
                self.weight_func = CMAWeightWithNegativeWeights(self.lam, self.d, min_problem=True)
            self.mu_eff = CMAParam.mu_eff(self.lam)
            self.c_1 = CMAParam.c_1(self.d, self.mu_eff)
            self.c_mu = CMAParam.c_mu(self.d, self.mu_eff, c1=self.c_1, alpha_mu=self.alpha_mu)
            self.c_c = CMAParam.c_c(self.d, self.mu_eff)
            self.c_sigma = CMAParam.c_sigma(self.d, self.mu_eff)
            self.damping = CMAParam.damping(self.d, self.mu_eff)
            self.model = copy.copy(self.model_init)
            if self.reset_margin:
                self.margin = 1.0 / (self.lam * self.d)
            if self.mean_regenerate_func is not None:
                self.model.m = self.mean_regenerate_func()
            if self.local_restart == 'integer':
                self.model.m[num_cont:] = self.model.encoding(
                    1, self.best_candidate.reshape(1, -1)
                )[0, num_cont:]
            elif self.local_restart == 'all':
                self.model.m = self.model.encoding(1, self.best_candidate.reshape(1, -1))[0]
            self.ps = np.zeros(self.d)
            self.pc = np.zeros(self.d)
            self.model.A = np.full(self.d, 1.)
            self.best_evals.append(self.best_eval)
            self.best_eval = np.inf
            self.not_improve_ite = 0
            self.restart_count += 1

    def terminate_condition(self):
        return self.model.terminate_condition()


# ---------------------------------------------------------------------------
# Shim & framework class
# ---------------------------------------------------------------------------

class _SamplerShim:
    """Minimal shim satisfying CMAESwM's internal sampler reference.

    CMAESwM stores the sampler object only to double its ``lam`` on IPOP
    restarts.  Since we disable restarts (``restart=-1``), this attribute is
    never accessed, but the constructor still expects the argument.
    """

    def __init__(self, lam: int) -> None:
        self.lam = lam


class ThickPanelCMAMarginFramework(ThickPanelDesignFramework):
    """Thick-panel design framework using CMA-ES with Margin optimization.

    CMA-ES with Margin (Hamano et al., 2022) extends vanilla CMA-ES with an
    explicit *margin* mechanism that prevents the search distribution from
    collapsing away from discrete category boundaries.  This makes it
    well-suited for the thick-panel problem where every height offset is
    quantised by ``discrete_step``.
    """

    algorithm_key = "cma-es-margin"
    result_prefix = "cma-es-margin"

    def optimize(
        self,
        population_size: int = 16,
        generations: int = 50,
        sigma_init: float = 5.0,
        margin: Optional[float] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        使用CMA-ES with Margin算法优化高度偏移量

        :param population_size: 种群大小 / Population size (λ)，默认16
        :param generations: 迭代代数 / Number of generations
        :param sigma_init: 初始变异强度 / Initial mutation strength
        :param margin: Margin parameter α; if None defaults to 1/(dim·λ)
        :param verbose: 是否打印进度 / Whether to print progress
        :return: (最优高度偏移量, 最佳折叠百分比)
        """
        print("\n" + "=" * 60)
        print(
            "开始CMA-ES-Margin优化高度偏移量 / "
            "Starting CMA-ES-Margin height offset optimization"
        )
        print("=" * 60)

        self.data.clear()
        self.extract_data["gen"].clear()
        self.extract_data["avg"].clear()
        self.extract_data["std"].clear()
        self.extract_data["min"].clear()
        self.extract_data["min_without_var"].clear()

        # ------------------------------------------------------------------
        # Discrete magnitude grid: [min_thickness, max_offset] stepped by
        # discrete_step. Sign is applied in _apply_constraints before simulation.
        # ------------------------------------------------------------------
        all_vals = self._build_discrete_magnitude_values()
        # shape: (num_independent, num_discrete_values)
        discrete_space = np.tile(all_vals, (self.num_independent, 1))

        # ------------------------------------------------------------------
        # Initial mean: unsigned magnitudes (default min_thickness per crease)
        # ------------------------------------------------------------------
        mean = self._build_initial_mean()

        lam = population_size
        dim = self.num_independent
        margin_val = margin if margin is not None else 1.0 / (dim * lam)

        w_func = CMAWeightWithNegativeWeights(lam, dim, min_problem=True)
        shim = _SamplerShim(lam)

        optimizer = CMAESwM(
            dim,
            discrete_space,
            w_func,
            shim,
            lam=lam,
            m=mean,
            sigma=sigma_init,
            margin=margin_val,
            restart=-1,          # disable IPOP restarts
            minimal_eigenval=1e-30,
        )

        print("CMA-ES-Margin初始化完成 / CMA-ES-Margin initialized")
        print(f"  种群大小/Population size: {lam}")
        print(f"  初始变异强度/Initial sigma: {sigma_init}")
        print(f"  Margin parameter α: {margin_val:.6f}")
        if self.symm_mode:
            print(
                f"  维度/Dimension: {dim} independent "
                f"(full creases: {self.num_creases})"
            )
        else:
            print(f"  维度/Dimension: {dim}")
        print(f"  离散值数量/Discrete values per variable: {len(all_vals)}")

        best_fitness = np.inf
        best_solution = None
        generation = 0

        while not optimizer.terminate_condition() and generation < generations:
            self.data.append([])
            generation += 1

            # Sample continuous candidates, then encode to discrete grid
            model = optimizer.sampling_model()
            X = model.sampling(lam)          # continuous, shape (lam, dim)
            X_enc = model.encoding(lam, X)   # discrete-snapped, shape (lam, dim)

            # evaluate_population expands symmetry and applies constraints
            fitness_list, constrained_list = self.evaluate_population(
                list(X_enc),
                (generation - 1) * lam,
            )
            self.data[generation - 1] = list(fitness_list)

            # CMAESwM.update expects the *continuous* X (not encoded)
            optimizer.update(X, np.array(fitness_list))

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
                    print(
                        f"第{generation}代/Gen {generation}: "
                        f"最优适应度 = {best_fitness:.4f}"
                    )
                    print(f"  高度偏移量/Height offsets: {best_solution}")
                    print(
                        f"  当前sigma/Current sigma: {optimizer.model.sigma:.4f}"
                    )
            elif verbose:
                print(
                    f"第{generation}代/Gen {generation}: "
                    f"最优适应度 = {best_fitness:.4f}"
                )
                print(f"  高度偏移量/Height offsets: {best_solution}")
                print(f"  当前sigma/Current sigma: {optimizer.model.sigma:.4f}")

            self._record_best_offset(constrained_list[current_best_idx])
            self.save_extract_data()

        print("\n" + "=" * 60)
        print("CMA-ES-Margin优化完成 / CMA-ES-Margin optimization completed")
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

    run_main(default_algorithm="cma-es-margin")


if __name__ == "__main__":
    main()
