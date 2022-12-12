import numpy as np
from numba import njit

from src.base_fields import *
from src.fwer import procedure
from src.stats_func import *


@njit(fastmath=True, parallel=True)
def multiple_means_mc_power_analysis(
    sample_size: int,
    sample_mean: float,
    sample_sd: float,
    n_variants: int,
    relative_effect: float,
    alpha: float = ALPHA,
    n_simulation: int = N_SIMULATION,
) -> dict[str, float | np.float_]:
    """Monte Carlo simulation for power analysis for multiple means tests

    Args:
        sample_size (int): Size of the sample
        sample_mean (float): Sample mean
        sample_sd (float): Sample standard deviation
        n_variants (int): Number of variants
        relative_effect (float): Relative effect, or minimum interest of effect
        alpha (float, optional): Type I error rate. Defaults to 0.05.
        n_simulation (int, optional): Number of simulations. Defaults to 2000.

    Returns:
        dict[str, float | np.float_]: Sample size and corresponding statistical power
    """
    n_per_variant = int(np.floor(sample_size / (n_variants + 1)))
    significance_either = np.zeros(shape=n_simulation, dtype=np.bool8)
    significance_all = np.zeros(shape=n_simulation, dtype=np.bool8)

    for i in range(n_simulation):
        p_vals = np.empty(n_variants)
        for j in range(n_variants):
            control_sample = np.random.normal(
                sample_mean, sample_sd, size=n_per_variant
            )
            variant_sample = np.random.normal(
                sample_mean * relative_effect, sample_sd, size=n_per_variant
            )
            p_vals[j] = independent_ttest(control_sample, variant_sample)[1]
        # Hypothesis testing
        tests = procedure.holm_step_down_procedure(p_vals, alpha)

        # Either one is significant or all
        significance_either[i] = np.any(tests)
        significance_all[i] = np.all(tests)
    return {
        "sample_size": sample_size,
        "power_either_significant": np.mean(significance_either),
        "power_all_significant": np.mean(significance_all),
    }


@njit(fastmath=True, parallel=True)
def multiple_proportions_mc_power_analysis(
    sample_size: int | float,
    base_rate: float,
    n_variants: int,
    relative_effect: float,
    alpha: float = ALPHA,
    n_simulation: int = N_SIMULATION,
) -> dict[str, float | np.float_]:
    """Monte Carlo simulation for power analysis for multiple proportions tests

    Args:
        sample_size (int): Size of the sample
        base_rate (float): Base conversion rate
        n_variants (int): Number of variants
        relative_effect (float): Relative effect, or minimum interest of effect
        alpha (float, optional): Type I error rate. Defaults to 0.05.
        n_simulation (int, optional): Number of simulations. Defaults to 2000.

    Returns:
        dict[str, float | np.float_]: Sample size and corresponding statistical power
    """
    n_per_variant = int(np.floor(sample_size / (n_variants + 1)))
    significance_either = np.zeros(shape=n_simulation, dtype=np.bool8)
    significance_all = np.zeros(shape=n_simulation, dtype=np.bool8)

    for i in range(n_simulation):
        p_vals = np.empty(n_variants)
        for j in range(n_variants):
            control_sample = np.random.binomial(1, base_rate, size=n_per_variant)
            variant_sample = np.random.binomial(
                1, base_rate * relative_effect, size=n_per_variant
            )
            p_vals[j] = two_proportions_ztest(
                count=np.array([np.sum(variant_sample), np.sum(control_sample)]),
                nobs=np.array([n_per_variant, n_per_variant]),
            )[1]

        # Hypothesis testing
        tests = procedure.holm_step_down_procedure(p_vals, alpha)

        # Either one is significant or all
        significance_either[i] = np.any(tests)
        significance_all[i] = np.all(tests)
    return {
        "sample_size": sample_size,
        "power_either_significant": np.mean(significance_either),
        "power_all_significant": np.mean(significance_all),
    }
