import math
from typing import Sequence

import numba as nb
import numpy as np
from numpy import typing as npt
from scipy import stats as st

# TODO: Add documentatiaon


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


@nb.njit(parallel=True, fastmath=True)
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


@nb.njit(parallel=True, fastmath=True)
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
                np.array(mean_diff + dist_func(p / 100.0, df=dof) * stde),
            )
            for p in interval
        ]
    )


@nb.njit(fastmath=True)
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


@nb.njit(fastmath=True)
def norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / math.sqrt(2)))


@nb.njit(fastmath=True)
def norm_pdf(x: float) -> float:
    return math.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)


@nb.njit(fastmath=True)
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


@nb.njit(fastmath=True)
def unbiased_variance(x: npt.NDArray[np.number]) -> float:
    m = sum(x) / len(x)
    return sum([(xi - m) ** 2 for xi in x]) / (len(x) - 1)


@nb.njit(fastmath=True)
def independent_ttest(
    x1: npt.NDArray[np.float_], x2: npt.NDArray[np.float_]
) -> tuple[float, float]:
    """Calculate T-test for the means of two independent samples"""
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
    return tstat, 2 * (1 - t_cdf(tstat, dof1 + dof2))


@nb.njit(fastmath=True)
def factorial(n: int) -> int:
    x = 1
    for i in range(1, n + 1):
        x *= i
    return x


@nb.njit(fastmath=True)
def combination(n: int, k: int) -> float:
    return factorial(n) / (factorial(k) * factorial(n - k))


@nb.njit(fastmath=True)
def binompmf(k, n, p):
    return combination(n, k) * (p**k) * ((1 - p) ** (n - k))


@nb.njit(fastmath=True)
def two_proportions_ztest(
    count: npt.NDArray[np.number],
    nobs: npt.NDArray[np.number],
    value: float = 0.0,
    prop_var: bool = False,
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
    zstat = diff / std_diff
    return zstat, (1 - norm_cdf(np.abs(zstat))) * 2


@nb.njit(fastmath=True)
def ztest(
    x1: npt.NDArray[np.number],
    x2: None | npt.NDArray[np.number] = None,
    value: float | int = 0,
    ddof: float = 1.0,
):
    nobs1 = x1.shape[0]
    x1_mean = x1.mean()
    x1_var = x1.var()
    if x2 is not None:
        nobs2 = x2.shape[0]
        x2_mean = x2.mean()
        x2_var = x2.var()
        var_pooled = nobs1 * x1_var + nobs2 * x2_var
        var_pooled /= nobs1 + nobs2 - 2 * ddof
        var_pooled *= 1.0 / nobs1 + 1.0 / nobs2
    else:
        var_pooled = x1_var / (nobs1 - ddof)
        x2_mean = 0

    std_diff = np.sqrt(var_pooled)
    zstat = (x1_mean - x2_mean - value) / std_diff
    return zstat, (1 - norm_cdf(np.abs(zstat))) * 2


@nb.njit(fastmath=True)
def quad_trap(f, xmin, xmax, args=(), n=100):
    """Compute a definite integral"""
    h = (xmax - xmin) / n
    integral = h * (f(xmin, args) + f(xmax, args)) / 2
    for k in range(n):
        xk = (xmax - xmin) * k / n + xmin
        integral = integral + h * f(xk, args)
    return integral


@nb.njit(fastmath=True)
def quad_trap_alt(f, xmin, xmax, npoints=10):
    """Alternative form of computing a definite integral"""
    area = 0
    x = np.linspace(xmin, xmax, npoints)
    n = len(x)
    dx = x[1] - x[0]
    for k in range(1, n):
        area += (f(x[k - 1]) + f(x[k])) * dx / 2
    return area


# TODO: T-stats function with big dof return NaN due to the Gamma function.
# Need to find the approximation? alternative?
@nb.njit(nb.float64(nb.float64, nb.float64), fastmath=True)
def t_cdf_integrand(x, df):
    return (
        math.gamma((df + 1) / 2)
        / math.gamma(df / 2)
        * 1
        / math.sqrt(df * math.pi)
        * 1
        / (1 + x**2 / df) ** ((df + 1) / 2)
    )


@nb.njit(fastmath=True)
def incompbeta(a: float, b: float, x: float) -> float:
    """Incomplete Beta Function
    https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    """
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        lbeta = (
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + a * math.log(x)
            + b * math.log(1 - x)
        )
        if x < (a + 1) / (a + b + 2):
            return math.exp(lbeta) * contfractbeta(a, b, x) / a
        else:
            return 1 - math.exp(lbeta) * contfractbeta(b, a, 1 - x) / b


@nb.njit(nb.float64(nb.float64, nb.float64), fastmath=True)
def beta(a: float, b: float) -> float:
    """Beta function
    https://en.wikipedia.org/wiki/Beta_function
    """

    beta = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    return beta


@nb.njit(nb.float64(nb.float64, nb.float64), fastmath=True)
def _beta(a: float, b: float) -> float:
    """Alternative form of Beta function
    https://en.wikipedia.org/wiki/Beta_function
    """
    beta = math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    return beta


@nb.njit(fastmath=True)
def contfractbeta(a: float, b: float, x: float, ITMAX: int = 200) -> float:
    """Evaluates the continued fraction form of the incomplete Beta function"""

    EPS = 3.0e-7
    bm = az = am = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab * x / qap

    for i in range(ITMAX + 1):
        em = float(i + 1)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((qap + tem) * (a + tem))
        app = ap + d * az
        bpp = bp + d * bz
        aold = az
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1.0
        if abs(az - aold) < (EPS * abs(az)):
            return az
    print(
        "a or b too large or given ITMAX too small for computing incomplete beta"
        " function."
    )
    return np.nan


@nb.njit(fastmath=True)
def t_cdf(x: float, df: float) -> float:
    """Student 's t-distribution cumulative distribution function"""
    return 1 - 0.5 * incompbeta(df / 2, 0.5, df / (x**2 + df))


@nb.njit(nb.float64(nb.float64, nb.float64), fastmath=True)
def _t_cdf(x: float, df: float) -> float:
    """Alternative form of Student 's t-distribution cumulative distribution function"""
    return quad_trap(t_cdf_integrand, -10, x, args=(df))


@nb.njit(fastmath=True)
def t_pdf(x: float, df: float) -> float:
    """Student's t-distribution probability density function"""
    return (1 / (np.sqrt(df) * beta(0.5, df / 2))) / (1 + x**2 / df) ** ((df + 1) / 2)


@nb.njit(fastmath=True)
def _t_pdf(x: float, df: float) -> float:
    """Alternative form of Student's t-distribution probability density function"""
    return math.gamma((df + 1) / 2) / (
        math.sqrt(math.pi * df)
        * math.gamma(df / 2)
        * (1 + x**2 / df) ** ((df + 1) / 2)
    )
