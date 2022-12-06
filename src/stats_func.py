import numpy as np
from scipy import stats as st
from typing import Sequence
from numpy import typing as npt


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
