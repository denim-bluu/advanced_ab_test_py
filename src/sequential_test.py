import numpy as np
import pandas as pd
import pandas_gbq as gbq
from numpy import typing as npt

from src.fwer import procedure
from src.seq_design import boundary as bd
from src.base_fields import VariantTestMetric, TestMetric


def retrieve_experiment_df(experiment_id: str, metric_names: list[str]) -> pd.DataFrame:
    query = f""" 
    --beginsql
    SELECT
        t1.experiment_metric_name,
        t1.variant_id,
        t1.days_since_assignment,
        t1.converting_users_cum AS n_converted,
        t1.users_counted_total AS n_samples,
        t2.experiment_sample_size AS fixed_sample_size,
        t2.min_users_in_variant AS min_users_in_variant,
    FROM `prod.experiment_conversions`  AS t1
    LEFT JOIN `prod.experiments_metadata` AS t2
    ON t1.experiment_id = t2.experiment_id
    WHERE 
    t1.experiment_id = '{experiment_id}'
    AND t1.experiment_metric_name IN ({str([id for id in metric_names])[1:-1]})
    ORDER BY 1,2,3,4
    --endsql
    """
    df = gbq.read_gbq(query, project_id="monzo-analytics")
    if df is None or isinstance(df, pd.Series):
        raise ValueError("No data found.")
    if "control" not in df["variant_id"].unique():
        raise ValueError("Control variant must exist within the variant IDs")
    return df


def get_variant_metrics(
    df: pd.DataFrame, sample_size_per_variant: int | float | None = None
) -> dict[str, VariantTestMetric]:
    variant_metics: dict[str, VariantTestMetric] = {}
    variants = df["variant_id"].unique()
    for i in variants:
        _df = df.loc[df["variant_id"] == i]
        if not sample_size_per_variant:
            fixed_sample_size = int(_df["fixed_sample_size"].max()) / len(variants)
        else:
            fixed_sample_size = sample_size_per_variant
        variant_metics[i] = VariantTestMetric(
            variant_id=i,
            days_since_assignment=_df["days_since_assignment"].to_numpy(),
            n_samples=_df["n_samples"].to_numpy(),
            n_converted=_df["n_converted"].to_numpy(),
            fixed_sample_size=fixed_sample_size,
        )
    return variant_metics


def create_test_metric(
    variant_metics: dict[str, VariantTestMetric]
) -> dict[str, TestMetric]:
    test_metrics: dict[str, TestMetric] = {}
    for i in variant_metics.keys():
        test_metrics[i] = TestMetric(
            control=variant_metics["control"], treatment=variant_metics[i]
        )
    return test_metrics


def significance_test(test_metrics: dict[str, TestMetric]):
    significance: dict[str, dict[str, npt.NDArray[np.bool_]]] = {}
    for alpha in [0.025, 0.05, 0.1, 0.15]:
        significance[str(alpha)] = {
            k: procedure.holm_step_down_procedure(v.p_value, alpha=alpha)
            for k, v in test_metrics.items()
            if k != "control"
        }
    significance_df = pd.DataFrame.from_dict(
        {
            (i, j): significance[i][j]
            for i in significance.keys()
            for j in significance[i].keys()
        },
    ).T
    significance_df = significance_df.swaplevel()
    significance_df = significance_df.sort_index().T
    significance_df.index.name = "days_since_assignment"

    def color(val):
        if val:
            color = "green"
        else:
            color = "red"
        return f"background-color: {color}"

    return significance_df.style.applymap(color)


def create_summary_statistics(test_metrics: dict[str, TestMetric]) -> pd.DataFrame:
    dfs = []
    metric_name = [
        "n_samples",
        "sample_proportion",
        "n_converted",
        "cvr",
        "p",
        "stderr",
        "z_score",
        "p_value",
    ]
    for k, v in test_metrics.items():
        mult_index = pd.MultiIndex.from_product([[k], metric_name])
        dfs.append(
            pd.DataFrame(
                [
                    v.treatment.n_samples,
                    v.treatment.sample_proportion,
                    v.treatment.n_converted,
                    v.treatment.cvr,
                    v.pooled_p,
                    v.pooled_stderr,
                    v.z_score,
                    v.p_value,
                ],
                index=mult_index,
            ).T
        )
    output = pd.concat(dfs, axis=1)
    output.index.name = "days_since_assignment"
    return output.dropna(axis=1)

