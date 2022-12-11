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
) -> tuple[int | float, np.number | float, np.number]:
    """Monte Carlo simulation for power analysis for multiple means tests

    Args:
        sample_size (int): Size of the sample
        sample_mean (float): Sample mean
        sample_sd (float): Sample standard deviation
        relative_effect (float): Minimum relative effect
        alpha (float, optional): Type I error rate. Defaults to 0.05.
        n_simulation (int, optional): Number of simulations. Defaults to 2000.

    Returns:
        tuple[int, np.number]: Sample size and corresponding statistical power
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
    return sample_size, np.mean(significance_either), np.mean(significance_all)


@njit(fastmath=True, parallel=True)
def multiple_proportions_mc_power_analysis(
    sample_size: int | float,
    base_rate: float,
    n_variants: int,
    relative_effect: float,
    alpha: float = ALPHA,
    alternative: str = ALTERNATIVE,
    n_simulation: int = N_SIMULATION,
) -> tuple[int | float, np.number | float, np.number]:
    """Monte Carlo simulation for power analysis for multiple proportions tests

    Args:
        sample_size (int): Size of the sample
        sample_mean (float): Sample mean
        sample_sd (float): Sample standard deviation
        relative_effect (float): Minimum relative effect
        alpha (float, optional): Type I error rate. Defaults to 0.05.
        alternative (str, optional): Test type. Defaults to "two-sided".
        n_simulation (int, optional): Number of simulations. Defaults to 2000.

    Returns:
        tuple[int, np.number]: Sample size and corresponding statistical power
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
                alternative=alternative,
            )[1]

        # Hypothesis testing
        tests = procedure.holm_step_down_procedure(p_vals, alpha)

        # Either one is significant or all
        significance_either[i] = np.any(tests)
        significance_all[i] = np.all(tests)
    return sample_size, np.mean(significance_either), np.mean(significance_all)


@nb.njit(fastmath=True)
def two_means_mc_power_analysis(
    sample_size: int,
    sample_mean: float,
    sample_sd: float,
    relative_effect: float,
    alpha: float = ALPHA,
    n_simulation: int = N_SIMULATION,
) -> tuple[int | float, np.number]:
    """Monte Carlo simulation for power analysis for two means

    Args:
        sample_size (int): Size of the sample
        sample_mean (float): Sample mean
        sample_sd (float): Sample standard deviation
        relative_effect (float): Minimum relative effect
        alpha (float, optional): Type I error rate. Defaults to 0.05.
        n_simulation (int, optional): Number of simulations. Defaults to 2000.

    Returns:
        tuple[int, np.number]: Sample size and corresponding statistical power
    """
    control_data = np.random.normal(loc=sample_mean, scale=sample_sd, size=sample_size)
    # Multiply the control data by the relative effect, this will shift the distribution
    # of the variant left or right depending on the direction of the relative effect
    variant_data = control_data * relative_effect
    significance_results = np.empty(n_simulation)

    for i in range(n_simulation):
        # Randomly allocate the sample data to the control and variant
        rv = np.random.binomial(1, 0.5, size=sample_size)
        control_sample = control_data[rv == True]
        variant_sample = variant_data[rv == False]

        # Use Welch's t-test, make no assumptions on tests for equal variances
        test_result = independent_ttest(control_sample, variant_sample)
        # Test for significance
        significance_results[i] = test_result[1] <= alpha
    # The power is the number of times we have a significant result
    # as we are assuming the alternative hypothesis is true
    return sample_size, np.mean(significance_results)


@njit(fastmath=True)
def two_proportions_mc_power_analysis(
    sample_size: int | float,
    base_conversion_rate: float,
    relative_effect: float,
    alpha: float = ALPHA,
    alternative: str = ALTERNATIVE,
    n_simulation: int = N_SIMULATION,
) -> tuple[int | float, np.number]:
    sample_per_variant = int(np.floor(sample_size / 2))

    significance_results = np.zeros(shape=n_simulation, dtype=np.bool8)
    for i in range(n_simulation):
        # # Randomly generate binomial data for variant and control with different
        # success probabilities
        control_sample = np.random.binomial(
            1, base_conversion_rate, size=sample_per_variant
        )
        variant_sample = np.random.binomial(
            1, base_conversion_rate * relative_effect, size=sample_per_variant
        )
        test_result = two_proportions_ztest(
            count=np.array([sum(variant_sample), sum(control_sample)]),
            nobs=np.array([sample_per_variant, sample_per_variant]),
            alternative=alternative,
        )
        significance_results[i] = test_result[1] <= alpha  # Test for significance
    return sample_size, np.mean(significance_results)
