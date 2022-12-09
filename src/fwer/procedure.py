import numpy as np
from numpy import typing as npt
from src.fwer import adjusted_p as ap
from numba import njit

# TODO: Write documentatiaon


@njit
def bonferroni_procedure(
    p_vals: list[float], alpha: float = 0.05
) -> npt.NDArray[np.bool_]:
    _p_vals, m = np.asarray(p_vals), len(p_vals)
    return _p_vals < ap.bonferroni_alpha(m, alpha)


@njit
def sidak_procedure(p_vals: list[float], alpha: float = 0.05) -> npt.NDArray[np.bool_]:
    p, m = np.asarray(p_vals), len(p_vals)
    return p < ap.sidak_alpha(m, alpha)


@njit
def holm_step_down_procedure(
    p_vals: list[float], alpha: float = 0.05
) -> npt.NDArray[np.bool_]:
    _p_vals, m = np.asarray(p_vals), len(p_vals)
    idx = _p_vals.argsort()
    test = _p_vals[idx] > ap.holm_hochberg_alpha(m, alpha)
    r = m - np.sum(test)  # Minimum ordered index k that suffices the test above
    reject = np.zeros(len(_p_vals), dtype=np.bool8)
    reject[idx[0:r]] = True
    return reject


@njit
def hochberg_step_up_procedure(
    p_vals: list[float], alpha: float = 0.05
) -> npt.NDArray[np.bool_]:
    _p_vals, m = np.asarray(p_vals), len(p_vals)
    idx = _p_vals.argsort()
    test = _p_vals[idx] <= ap.holm_hochberg_alpha(m, alpha)
    r = np.sum(test)  # Largest ordered index k that suffices the test above
    reject = np.zeros(len(_p_vals), dtype=np.bool8)
    reject[idx[0:r]] = True
    return reject
