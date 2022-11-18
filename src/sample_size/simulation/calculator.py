import numpy as np
from scipy import stats as st
from statsmodels.stats.proportion import proportions_ztest
from src.base_fields import *
from dataclasses import dataclass


@dataclass
class MeanPowerAnalysisInput:
    sample_mean: float
    sample_sd: float
    relative_effect: float


def multiple_means_mc_power_analysis(
    sample_size: int | float,
    analysis_inputs: list[MeanPowerAnalysisInput],
    alpha: float = ALPHA,
    alternative: str = ALTERNATIVE,
    n_simulation: int = N_SIMULATION,
) -> tuple[int | float, np.number | float, np.number]:
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

    control_data_ls = []
    variant_data_ls = []

    for inp in analysis_inputs:
        control_data = np.asarray(
            st.norm.rvs(loc=inp.sample_mean, scale=inp.sample_sd, size=sample_size)
        )
        control_data_ls.append(control_data)

        # Multiply the control data by the relative effect, this will shift the distribution
        # of the variant left or right depending on the direction of the relative effect
        variant_data = control_data * inp.relative_effect
        variant_data_ls.append(variant_data)

    significance_either = []
    significance_all = []

    for _ in range(n_simulation):
        # Randomly allocate the sample data to the control and variant
        rv = st.binom.rvs(1, 0.5, size=sample_size)
        test_results = []
        for control, variant in zip(control_data_ls, variant_data_ls):
            control_sample = control[rv == True]
            variant_sample = variant[rv == False]
            test_results.append(
                st.ttest_ind(
                    control_sample,
                    variant_sample,
                    alternative=alternative,
                    equal_var=False,
                )[1]
            )
        # Multiple test correction
        # Use Holm correction
        # Due to performance issues on statsmodels.stats.multitest.multipletests,
        # Retrieved a snippet from the package
        n_pvals = len(test_results)
        sortind = np.argsort(test_results)
        notreject = test_results > alpha / np.arange(len(test_results), 0, -1)
        nr_index = np.nonzero(notreject)[0]
        if nr_index.size == 0:
            # nonreject is empty, all rejected
            notrejectmin = len(test_results)
        else:
            notrejectmin = np.min(nr_index)
        notreject[notrejectmin:] = True
        reject = ~notreject
        pvals_corrected_raw = test_results * np.arange(n_pvals, 0, -1)
        pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)
        if pvals_corrected is not None:  # not necessary anymore
            pvals_corrected[pvals_corrected > 1] = 1

        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[sortind] = pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[sortind] = reject

        # Either one is significant or all
        significance_either.append(any(reject_))
        significance_all.append(all(reject_))
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
    n_variant=N_VARIANT,
    alpha: float = ALPHA,
    alternative: str = ALTERNATIVE,
    n_simulation: int = N_SIMULATION,
) -> tuple[int | float, np.number]:
    if (0.0 > base_conversion_rate) | (base_conversion_rate > 1):
        raise ValueError(f"Base Conversion Rate has to be within 1 and 0")

    if (0.0 > alpha) | (alpha > 1):
        raise ValueError(f"alpha has to be within 1 and 0")

    sample_per_variant = int(np.floor(sample_size / (n_variant + 1)))
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
