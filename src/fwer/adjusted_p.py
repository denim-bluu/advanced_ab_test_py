import numpy as np
from numpy import typing as npt
import numba as nb


@nb.njit(fastmath=True)
def bonferroni_alpha(m: int, alpha: float = 0.05) -> npt.NDArray[np.number]:
    return np.asarray([alpha / m])


@nb.njit(fastmath=True)
def sidak_alpha(m: int, alpha: float = 0.05) -> npt.NDArray[np.number]:
    return np.asarray([1 - (1 - alpha) ** (1 / (k + 1)) for k in range(m)])


@nb.njit(fastmath=True)
def holm_hochberg_alpha(m: int, alpha: float = 0.05) -> npt.NDArray[np.number]:
    return np.asarray([alpha / (m - (k + 1) + 1) for k in range(m)])
