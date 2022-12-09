import numpy as np
import math
from scipy import stats as st
from typing import Sequence
from numpy import typing as npt
from numba import njit


def shift_array(
    arr: np.ndarray, num: int, fill_value: float | int = np.nan
) -> np.ndarray:
    """Shift the array

    Args:
        arr (np.ndarray): pre-shift array
        num (int): Number of shift
        fill_value (float | int, optional): Default fill value. Defaults to np.nan.

    Returns:
        np.ndarray: Shifted array with the filled values
    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def pooled_stdv(
    stdv1: float | int, n1: float | int, stdv2: float | int, n2: float | int
) -> float | int:
    """Pool standard deviation calculation

    Args:
        stdv1 (float | int): Standard deviation of group 1
        n1 (float | int): Number of sample for group 1
        stdv2 (float | int): Standard deviation of group 2
        n2 (float | int): Number of sample for group 2

    Returns:
        float | int: Pooled standard deviation
    """
    return np.sqrt(((n1 - 1) * stdv1**2 + (n2 - 1) * stdv2**2) / (n1 + n2 - 2))


def pooled_stde(
    stdv1: float | int, n1: float | int, stdv2: float | int, n2: float | int
) -> float | int:
    """Pool standard error

    Args:
        stdv1 (float | int): Standard deviation of group 1
        n1 (float | int): Number of sample for group 1
        stdv2 (float | int): Standard deviation of group 2
        n2 (float | int): Number of sample for group 2

    Returns:
        float | int: Pooled standard error
    """
    return pooled_stdv(stdv1, n1, stdv2, n2) * np.sqrt(1.0 / n1 + 1.0 / n2)


def ci_two_mean_difference(
    mean1: float | int,
    stdv1: float | int,
    n1: float | int,
    mean2: float | int,
    stdv2: float | int,
    n2: float | int,
    interval: Sequence[float] = (0.025, 0.975),
) -> dict[float, npt.NDArray[np.number]]:
    mean_diff = mean1 - mean2
    dof = n1 + n2 - 2
    stde = pooled_stde(stdv1, n1, stdv2, n2)

    # If the sample sizes are larger, that is both n1, n2 > 30, then use z-table.
    dist_func = st.norm.ppf if (n1 > 30) and (n2 > 30) else st.t.ppf

    return dict(
        [
            (
                round(p, 5),
                np.array(mean_diff + dist_func(p / 100.0, df=dof) * stde, dtype=float),
            )
            for p in interval
        ]
    )


@njit(fastmath=True)
def erf(x):
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y  # erf(-x) = -erf(x)


@njit(fastmath=True)
def norm_cdf(x):
    return 0.5 * (1 + erf(x / math.sqrt(2)))


@njit(fastmath=True)
def norm_pdf(x):
    return math.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)


@njit(fastmath=True)
def inverse_normal_cdf(
    p: float, mu: float = 0, sigma: float = 1, tol: float = 1e-05
) -> float:
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p)
    low_z = -10.0  # standard normal is 0 at cdf(-10)
    high_z = 10.0  # standard normal is 0 at cdf(10)
    midz = 0.0
    # Use binary search to find the desired value
    while high_z - low_z > tol:
        midz = (high_z + low_z) / 2
        pmid = norm_cdf(midz)

        if p > pmid:
            low_z = midz
        else:
            high_z = midz
    return round(midz, 2)


@njit(fastmath=True)
def t_pdf(x, df):
    return math.gamma((df + 1) / 2) / (
        math.sqrt(math.pi * df)
        * math.gamma(df / 2)
        * (1 + x**2 / df) ** ((df + 1) / 2)
    )


@njit(fastmath=True)
def unbiased_variance(x: npt.NDArray[np.number]) -> float:
    m = sum(x) / len(x)
    return sum([(xi - m) ** 2 for xi in x]) / (len(x) - 1)


@njit(fastmath=True)
def independent_ttest(x1, x2):
    # calculate means
    x1_bar, x2_bar = np.mean(x1), np.mean(x2)
    n1, n2 = len(x1), len(x2)
    dof1, dof2 = n1 - 1, n2 - 1

    var_x1, var_x2 = unbiased_variance(x1), unbiased_variance(x2)

    # pooled sample variance
    pool_var = ((dof1 * var_x1) + (dof2 * var_x2)) / (dof1 + dof2)

    # standard error
    std_error = np.sqrt(pool_var * (1.0 / n1 + 1.0 / n2))

    # calculate t statistics
    tstat = (x1_bar - x2_bar) / std_error
    return tstat, 2 * (1 - st.t.cdf(tstat, dof1 + dof2, 0, 1))


@njit(fastmath=True)
def factorial(n):
    x = 1
    for i in range(1, n + 1):
        x *= i
    return x


@njit(fastmath=True)
def combination(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))


@njit(fastmath=True)
def binompmf(k, n, p):
    return combination(n, k) * (p**k) * ((1 - p) ** (n - k))


@njit(fastmath=True)
def two_proportions_ztest(
    count, nobs, value=0.0, alternative="two-sided", prop_var=False
):
    count = np.asarray(count)
    nobs = np.asarray(nobs)

    prop = count / nobs

    diff = prop[0] - prop[1] - value

    p_pooled = np.sum(count) * 1.0 / np.sum(nobs)

    nobs_fact = np.sum(1.0 / nobs)
    if prop_var:
        p_pooled = prop_var
    var_ = np.float64(p_pooled) * np.float64(1 - p_pooled) * np.float64(nobs_fact)
    std_diff = np.sqrt(var_)
    return _zstat_generic2(diff, std_diff, alternative)


@njit(fastmath=True)
def one_proportions_ztest(
    count, nobs, value=0.0, alternative="two-sided", prop_var=False
):
    count = np.asarray(count)
    nobs = np.asarray(nobs) * np.ones_like(count)

    prop = count / nobs
    if value == 0.0:
        raise ValueError("value must be provided for a 1-sample test")

    diff = prop - value

    p_pooled = np.sum(count) * 1.0 / np.sum(nobs)

    nobs_fact = np.sum(1.0 / nobs)
    if prop_var:
        p_pooled = prop_var
    var_ = np.float64(p_pooled) * np.float64(1 - p_pooled) * np.float64(nobs_fact)
    std_diff = np.sqrt(var_)
    return _zstat_generic2(diff, std_diff, alternative)


@njit(fastmath=True)
def _zstat_generic2(value, std, alternative):
    zstat = value / std
    if alternative in ["two-sided", "2-sided", "2s"]:
        pvalue = (1 - norm_cdf(np.abs(zstat))) * 2
    elif alternative in ["larger", "l"]:
        pvalue = 1 - norm_cdf(zstat)
    elif alternative in ["smaller", "s"]:
        pvalue = norm_cdf(zstat)
    else:
        raise ValueError("invalid alternative")
    return zstat, pvalue
