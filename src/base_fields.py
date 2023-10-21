import numpy as np
from attrs import define, field, asdict
from numpy import typing as npt
from scipy import stats as st
import pandas as pd

from src.stats_func import poooled_p, proportions_pooled_stde
from util.data_cls_func import floating_array

ALPHA = 0.05
BETA = 0.2
N_SIMULATION = 2000
ALTERNATIVE = "two-sided"
N_VARIANT = 1
N_STAGE = 3


@define
class PowerAnalysisResult:
    sample_size: npt.NDArray[np.float_]
    power: npt.NDArray[np.float_]

    def convert_to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_size": self.sample_size,
                "power": self.power,
            }
        )
