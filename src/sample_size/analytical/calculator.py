import numpy as np
from scipy import stats as st
from numpy import typing as npt
from src.base_fields import *


def two_mean_sample_size_calculate(
    base_mean: int | float,
    variance: int | float,
    relative_change: float | npt.NDArray[np.number],
    alpha: float = ALPHA,
    power: float = 1 - BETA,
    alternative: str = ALTERNATIVE,
) -> float | npt.NDArray[np.number]:
    """Sample size for detecting a difference between the means of two samples for
    each group

    Example:
        IF:
            Confidence level: 95%
            Power: 80%
            Hypothetical distance: 2
            Population variance: 100
            Two sided : True
        THEN:
            Sample size: 392.44 (or 393)

    Args:
        base_mean (float): Mean of the null hypothesis
        relative_change (float | np.ndarray[float, Any]): The relative effect that will
            be detected (1-β)% of the time, aka minimum detectable effect
        variance (float): population variance, or sample variance if population variance
            is unknown
        alpha (float, optional): Type I error rate. Defaults to 0.05.
        power (float, optional): 1 - Type II error rate. Defaults to 0.8.

    Returns:
        float | np.ndarray[float, Any]: number of sample size required
    """
    if (0.0 > alpha) | (alpha > 1):
        raise ValueError(f"alpha has to be within 1 and 0")
    if (0.0 > power) | (power > 1):
        raise ValueError(f"beta has to be within 1 and 0")
    if alternative == "two-sided":
        alpha = alpha / 2
    z_alpha = st.norm.ppf(alpha)
    z_beta = st.norm.ppf(1 - power)
    treatment_rate = base_mean * relative_change
    d = treatment_rate - base_mean
    return (z_alpha + z_beta) ** 2 * 2 * variance / d**2


def two_proportion_sample_size_calculate(
    base_rate: float,
    relative_change: float,
    ratio: float,
    alpha: float = ALPHA,
    power: float = 1 - BETA,
) -> float:
    if (0.0 > alpha) | (alpha > 1):
        raise ValueError(f"alpha has to be within 1 and 0")
    if (0.0 > power) | (power > 1):
        raise ValueError(f"beta has to be within 1 and 0")
    if (0.0 > ratio) | (ratio > 1):
        raise ValueError(f"ratio has to be within 1 and 0")

    z_alpha = st.norm.ppf(alpha / 2)
    z_beta = st.norm.ppf(1 - power)
    treatment_rate = base_rate * (1 + relative_change)
    avg_rate = ((ratio * treatment_rate) + base_rate) / (1 + ratio)

    return (
        (
            (z_alpha * np.sqrt(avg_rate * (1 - avg_rate) * (1 + (1 / ratio))))
            + (
                z_beta
                * np.sqrt(
                    (base_rate * (1 - base_rate))
                    + (treatment_rate * (1 - treatment_rate))
                )
            )
        )
        ** 2
    ) / ((treatment_rate - base_rate) ** 2)
