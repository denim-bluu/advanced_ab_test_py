from typing import Any

import numpy as np
from scipy import stats as st
from statsmodels.stats.proportion import proportions_ztest

from src.base_fields import *
from src.fwer import procedure


def multiple_means_mc_power_analysis(
    sample_size: int,
    sample_data: np.ndarray[float, Any],
    n_variants: int,
    relative_effect: float,
    alpha: float = ALPHA,
    alternative: str = ALTERNATIVE,
    n_simulation: int = N_SIMULATION,
) -> tuple[int | float, np.number | float, np.number]:
    """Monte Carlo simulation for power analysis for multiple tests

    Args:
        sample_data (int): Example of the population sample
        sample_size (int): Size of the sample
        n_variants (int): Number of variants
        relative_effect (float): Minimum relative effect
        alpha (float, optional): Type I error rate. Defaults to 0.05.
        alternative (str, optional): Test type. Defaults to "two-sided".
        n_simulation (int, optional): Number of simulations. Defaults to 2000.

    Returns:
        tuple[int, np.number]: Sample size and corresponding statistical power
    """
    if (0.0 > alpha) | (alpha > 1):
        raise ValueError(f"alpha has to be within 1 and 0")
    n_groups = 1 + n_variants

    control_data = sample_data[:sample_size]
    variant_data = sample_data * relative_effect
    significance_either = []
    significance_all = []

    for _ in range(n_simulation):
        p_vals = []

        # Randomly allocate the sample data to the control and variant
        indices = list(range(sample_size))
        np.random.shuffle(indices)
        idx_partitions = [sorted(indices[i::n_groups]) for i in range(n_groups)]

        control_sample = np.array([control_data[j] for j in idx_partitions[0]])
        for i in range(n_variants):
            variant_sample = np.array([variant_data[j] for j in idx_partitions[i + 1]])
            p_vals.append(
                st.ttest_ind(
                    control_sample,
                    variant_sample,
                    alternative=alternative,
                    equal_var=False,
                )[1]
            )

        # Hypothesis testing
        tests = procedure.holm_step_down_procedure(p_vals, alpha)

        # Either one is significant or all
        significance_either.append(any(tests))
        significance_all.append(all(tests))
    return sample_size, np.mean(significance_either), np.mean(significance_all)


def multiple_proportions_mc_power_analysis(
    sample_size: int | float,
    base_rate: np.number,
    n_variants: int,
    relative_effect: float,
    alpha: float = ALPHA,
    alternative: str = ALTERNATIVE,
    n_simulation: int = N_SIMULATION,
) -> tuple[int | float, np.number | float, np.number]:
    """Monte Carlo simulation for power analysis for multiple tests

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
    if (0.0 > alpha) | (alpha > 1):
        raise ValueError(f"alpha has to be within 1 and 0")
    n_per_variant = int(np.floor(sample_size / (n_variants + 1)))
    significance_either = []
    significance_all = []

    for _ in range(n_simulation):
        control_sample = st.binom.rvs(1, base_rate, size=n_per_variant)

        p_vals = []
        for _ in range(n_variants):
            variant_sample = st.binom.rvs(
                1, base_rate * relative_effect, size=n_per_variant
            )
            p_vals.append(
                proportions_ztest(
                    count=[np.sum(variant_sample), np.sum(control_sample)],
                    nobs=[n_per_variant, n_per_variant],
                    alternative=alternative,
                )[1]
            )

        # Hypothesis testing
        tests = procedure.holm_step_down_procedure(p_vals, alpha)

        # Either one is significant or all
        significance_either.append(any(tests))
        significance_all.append(all(tests))
    return sample_size, np.mean(significance_either), np.mean(significance_all)


def two_means_mc_power_analysis(
    sample_size: int | float,
    sample_mean: float,
    sample_sd: float,
    relative_effect: float,
    alpha: float = ALPHA,
    alternative: str = ALTERNATIVE,
    n_simulation: int = N_SIMULATION,
) -> tuple[int | float, np.number]:
    """Monte Carlo simulation for power analysis for two means

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
    if (0.0 > alpha) | (alpha > 1):
        raise ValueError(f"alpha has to be within 1 and 0")

    control_data = np.asarray(
        st.norm.rvs(loc=sample_mean, scale=sample_sd, size=sample_size)
    )
    # Multiply the control data by the relative effect, this will shift the distribution
    # of the variant left or right depending on the direction of the relative effect
    variant_data = control_data * relative_effect
    significance_results = []

    for _ in range(n_simulation):
        # Randomly allocate the sample data to the control and variant
        rv = st.binom.rvs(1, 0.5, size=sample_size)
        control_sample = control_data[rv == True]
        variant_sample = variant_data[rv == False]

        # Use Welch's t-test, make no assumptions on tests for equal variances
        test_result = st.ttest_ind(
            control_sample, variant_sample, alternative=alternative, equal_var=False
        )
        # Test for significance
        significance_results.append(test_result[1] <= alpha)
    # The power is the number of times we have a significant result
    # as we are assuming the alternative hypothesis is true
    return sample_size, np.mean(significance_results)


def two_proportions_mc_power_analysis(
    sample_size: int | float,
    base_conversion_rate: float,
    relative_effect: float,
    alpha: float = ALPHA,
    alternative: str = ALTERNATIVE,
    n_simulation: int = N_SIMULATION,
) -> tuple[int | float, np.number]:
    if (0.0 > base_conversion_rate) | (base_conversion_rate > 1):
        raise ValueError(f"Base Conversion Rate has to be within 1 and 0")

    if (0.0 > alpha) | (alpha > 1):
        raise ValueError(f"alpha has to be within 1 and 0")

    sample_per_variant = int(np.floor(sample_size / 2))
    significance_results = []
    for _ in range(n_simulation):
        # # Randomly generate binomial data for variant and control with different
        # success probabilities
        control_sample = np.array(
            st.binom.rvs(1, base_conversion_rate, size=sample_per_variant)
        )
        variant_sample = np.array(
            st.binom.rvs(
                1, base_conversion_rate * relative_effect, size=sample_per_variant
            )
        )
        test_result = proportions_ztest(
            count=[sum(variant_sample), sum(control_sample)],
            nobs=[sample_per_variant, sample_per_variant],
            alternative=alternative,
        )
        significance_results.append(test_result[1] <= alpha)  # Test for significance
    return sample_size, np.mean(significance_results)
