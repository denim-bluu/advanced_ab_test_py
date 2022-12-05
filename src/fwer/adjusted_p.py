import numpy as np
from numpy import typing as npt


def bonferroni_alpha(m: int, alpha: float = 0.05) -> npt.NDArray[np.number]:
    return np.asarray([alpha / m])


def sidak_alpha(m: int, alpha: float = 0.05) -> npt.NDArray[np.number]:
    return 1 - (1 - alpha) ** (1 / m)


def holm_hochberg_alpha(m: int, alpha: float = 0.05) -> npt.NDArray[np.number]:
    return np.asarray([alpha / (m - (k + 1) + 1) for k in range(m)])
