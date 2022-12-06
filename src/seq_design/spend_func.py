import numpy as np
from numpy import typing as npt
from scipy import stats as st
from strenum import StrEnum

from src.base_fields import ALPHA
from src.stats_func import shift_array


class SpendOptions(StrEnum):
    """Type of spending functions

    Args:
        StrEnum (_type_): _description_
    """

    OBF = "obrien_fleming"
    POCOCK = "pocock"
    KIM_DEMETS = "kim_demets"
    HWANG_SHIH_DECANI = "hwang_shih_decani"


def obrien_fleming(
    t: float | npt.NDArray[np.number], alpha: float = ALPHA
) -> npt.NDArray[np.number]:
    """Lan-DeMets O'Brien-Fleming approximation

    Args:
        t (float | npt.NDArray[np.number]): proportion of sample size/information
        alpha (float, optional): Type I OR II error. Defaults to 0.05.

    Returns:
        npt.NDArray[np.number]: Alpha spending functions
    """
    z = -st.norm.ppf(alpha / 2)
    return 2 * np.asarray(1 - st.norm.cdf((z / np.sqrt(t))))


def pocock(
    t: float | npt.NDArray[np.number], alpha: float = ALPHA
) -> npt.NDArray[np.number]:
    """Lan-DeMets Pocock approximation

    Args:
        t (float | npt.NDArray[np.number]): proportion of sample size/information
        alpha (float, optional): Type I OR II error. Defaults to 0.05.

    Returns:
        npt.NDArray[np.number]: Alpha spending functions
    """
    return np.asarray(alpha * np.log(1 + (np.exp(1) - 1) * t))


def kim_demets(
    t: float | npt.NDArray[np.number], alpha: float = ALPHA
) -> npt.NDArray[np.number]:
    """Kim-DeMets (power) Spending Function

    Args:
        t (float | npt.NDArray[np.number]): proportion of sample size/information
        alpha (float, optional): Type I OR II error. Defaults to 0.05.

    Returns:
        npt.NDArray[np.number]: Alpha spending functions
    """
    return np.asarray(alpha * t**3)


def hwang_shih_decani(
    t: float | npt.NDArray[np.number], gamma: float | int = -2, alpha: float = ALPHA
) -> npt.NDArray[np.number]:
    """Hwang-Shih-DeCani Spending Function

    Args:
        t (float | npt.NDArray[np.number]): proportion of sample size/information
        gamma (float | int, optional): Gamma parameter. Defaults to -2.
        alpha (float, optional): Type I OR II error. Defaults to 0.05.

    Returns:
        npt.NDArray[np.number]: Alpha spending functions
    """
    return np.where(
        gamma == 0,
        alpha * t,
        alpha * (1 - np.exp(-t * gamma)) / (1 - np.exp(-gamma)),
    )


def spend_function(adjusted_alpha: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """Calculate alpha spending function

    Args:
        adjusted_alpha (npt.NDArray[np.number]): Array of adjusted alpha

    Returns:
        npt.NDArray[np.number]: Alpha spending function
    """
    return adjusted_alpha - shift_array(adjusted_alpha, 1, 0.0)
