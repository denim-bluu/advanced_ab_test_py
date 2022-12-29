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


@define
class VariantTestMetric:
    variant_id: str
    days_since_assignment: npt.NDArray[np.int_] = field(converter=floating_array)
    n_samples: npt.NDArray[np.float_] = field(converter=floating_array)
    n_converted: npt.NDArray[np.float_] = field(converter=floating_array)
    fixed_sample_size: int | float
    cvr: npt.NDArray[np.float_] = field(init=False)
    sample_proportion: npt.NDArray[np.float_] = field(init=False)

    def __attrs_post_init__(self):
        if not (
            self.days_since_assignment.shape
            == self.n_samples.shape
            == self.n_converted.shape
        ):
            raise ValueError("All arrays must have the same length")
        self.cvr = self.n_converted / self.n_samples
        self.sample_proportion = self.n_samples / self.fixed_sample_size


@define
class TestMetric:
    control: VariantTestMetric
    treatment: VariantTestMetric
    pooled_p: npt.NDArray[np.float_] = field(init=False)
    pooled_stderr: npt.NDArray[np.float_] = field(init=False)
    z_score: npt.NDArray[np.float_] = field(init=False)
    p_value: npt.NDArray[np.float_] = field(init=False)
    ci_lower: npt.NDArray[np.float_] = field(init=False)
    ci_upper: npt.NDArray[np.float_] = field(init=False)

    def __attrs_post_init__(self):
        if self.control.variant_id != self.treatment.variant_id:
            self.pooled_p = poooled_p(
                self.treatment.n_converted,
                self.treatment.n_samples,
                self.control.n_converted,
                self.control.n_samples,
            )
            self.pooled_stderr = proportions_pooled_stde(
                self.treatment.n_converted,
                self.treatment.n_samples,
                self.control.n_converted,
                self.control.n_samples,
            )
            self.z_score = (self.treatment.cvr - self.control.cvr) / self.pooled_stderr
            self.p_value = st.norm.sf(abs(self.z_score))
            self.ci_lower = self.treatment.cvr - 1.96 * self.pooled_stderr
            self.ci_upper = self.treatment.cvr + 1.96 * self.pooled_stderr
        else:
            self.pooled_p = np.array([None] * len(self.control.n_converted))
            self.pooled_stderr = np.array([None] * len(self.control.n_converted))
            self.z_score = np.array([None] * len(self.control.n_converted))
            self.p_value = np.array([None] * len(self.control.n_converted))
            self.ci_lower = np.array([None] * len(self.control.n_converted))
            self.ci_upper = np.array([None] * len(self.control.n_converted))
