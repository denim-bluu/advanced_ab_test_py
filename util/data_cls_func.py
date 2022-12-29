import numpy as np
from numpy import typing as npt


def floating_array(array: np.ndarray | list) -> npt.NDArray[np.float_]:
    """Create a numpy array with replacement of nan or None values to 0.0"""
    return np.asarray(array, dtype=float)
