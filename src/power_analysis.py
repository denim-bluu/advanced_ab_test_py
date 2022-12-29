from functools import partial
from typing import Iterable

from joblib import Parallel, delayed, cpu_count
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.base_fields import PowerAnalysisResult
from src.sample_size.simulation.calculator import *


def run_multiple_means_power_analysis_per_scenario(
    sample_sizes: Iterable,
    relative_effects: Iterable,
    sample_mean: float,
    sample_sd: float,
    n_variants: int = 1,
    alpha: float = ALPHA,
    n_simulation: int = N_SIMULATION,
    all_significant: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run multiple iterations of multiple means power analysis

    Args:
        sample_sizes (Iterable): Sequence of sample sizes to iterate
        relative_effects (Iterable): Sequence of relative effect sizes to iterate
        sample_size (int): Size of the sample
        sample_mean (float): Sample mean
        sample_sd (float): Sample standard deviation
        n_variants (int): Number of variants
        relative_effect (float): Relative effect, or minimum interest of effect
        alpha (float, optional): Type I error rate. Defaults to 0.05.
        n_simulation (int, optional): Number of simulations. Defaults to 2000.
        all_significant (bool, optional): Multiple variants test criteria. Defaults to False.

    Returns:
        dict[str, pd.DataFrame]: Dictionary of power analysis results for each MDE
    """
    output = {k: pd.DataFrame() for k in relative_effects}
    for re in relative_effects:
        fn = partial(
            multiple_means_mc_power_analysis,
            sample_mean=sample_mean,
            sample_sd=sample_sd,
            relative_effect=re,
            n_variants=n_variants,
            alpha=alpha,
            n_simulation=n_simulation,
            all_significant=all_significant,
        )
        result = Parallel(n_jobs=cpu_count())(delayed(fn)(i) for i in sample_sizes)
        sample, power = np.array(result).T
        output[str(re)] = PowerAnalysisResult(sample, power).convert_to_df()
    return output


def run_multiple_proportions_power_analysis_per_scenario(
    sample_sizes: Iterable,
    relative_effects: Iterable,
    base_rate: float,
    n_variants: int = 1,
    alpha: float = ALPHA,
    n_simulation: int = N_SIMULATION,
    all_significant: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run multiple iterations of multiple means power analysis

    Args:
        sample_sizes (Iterable): Sequence of sample sizes to iterate
        relative_effects (Iterable): Sequence of relative effect sizes to iterate
        sample_size (int): Size of the sample
        base_rate (float): Base conversion rate
        n_variants (int): Number of variants
        relative_effect (float): Relative effect, or minimum interest of effect
        alpha (float, optional): Type I error rate. Defaults to 0.05.
        n_simulation (int, optional): Number of simulations. Defaults to 2000.
        all_significant (bool, optional): Multiple variants test criteria. Defaults to False.

    Returns:
        dict[str, pd.DataFrame]: Dictionary of power analysis results for each MDE
    """
    output: dict[str, pd.DataFrame] = {}
    for re in relative_effects:
        fn = partial(
            multiple_proportions_mc_power_analysis,
            base_rate=base_rate,
            n_variants=n_variants,
            relative_effect=re,
            alpha=alpha,
            n_simulation=n_simulation,
            all_significant=all_significant,
        )
        result = Parallel(n_jobs=cpu_count())(delayed(fn)(i) for i in sample_sizes)

        sample, power = np.array(result).T
        output[str(re)] = PowerAnalysisResult(sample, power).convert_to_df()
    return output


def calculate_required_experiment_duration(
    power_analysis_result: dict[str, pd.DataFrame],
    weekly_runrate: int | float,
    target_power: float = 1 - BETA,
) -> pd.DataFrame:
    """Calculate the required experiment duration given the weekly run rate

    Args:
        power_analysis_result (dict[str, pd.DataFrame]): Dictionary of power analysis results for each MDE
        weekly_runrate (int | float): Weekly run rate
        target_power (float, optional): Target Power. Defaults to 1-BETA.

    Returns:
        pd.DataFrame: Table with the required experiment duration per MDE
    """
    output = []
    for k, _df in power_analysis_result.items():
        df = _df.copy()
        df["beyond_target_power"] = df["power"] >= target_power
        fixed_size = df["sample_size"].loc[df["beyond_target_power"] > 0].iloc[0]
        output.append([k, fixed_size, fixed_size / weekly_runrate])
    return pd.DataFrame(
        output, columns=["mde", "fixed_sample_size", "duration_week"]
    ).round(2)


def visualise_power_analysis(
    power_analysis_result: dict[str, pd.DataFrame],
    current_sasmple_size: int | float | None = None,
    target_power: float = 1 - BETA,
) -> go.Figure:
    """Visualise the multiple iterations of power analysis results

    Args:
        power_analysis_result (dict[str, pd.DataFrame]): Dictionary of power analysis results for each MDE
        current_sasmple_size (int | float | None, optional): Current sample size collected. Defaults to None.
        target_power (float, optional): Target Power. Defaults to 1-BETA.

    Returns:
        go.Figure: Visulisation of the multiple iterations of power analysis results
    """
    palette = px.colors.qualitative.Plotly
    fig = go.Figure()
    fig.update_layout(
        title_text="Power Analysis",
        xaxis_title="Sample Size",
        yaxis_title="Power",
        legend_title="Legend",
    )
    fig.add_hline(
        y=target_power,
        line_width=1,
        line_dash="dash",
        line_color="red",
        annotation_text=f"{round(target_power*100, 2)}% Power Target",
        annotation_position="top left",
        annotation=dict(font_size=10),
    )
    for i, (k, df) in enumerate(power_analysis_result.items()):
        x = df["sample_size"].to_numpy()
        y1 = df["power"].to_numpy()
        y2 = np.where(df["power"] >= target_power, df["power"], np.nan)
        poly = np.polyfit(x, y1, 3)
        poly_y = np.minimum(np.poly1d(poly)(x), 1.0)
        if current_sasmple_size:
            fig.add_vline(
                x=current_sasmple_size,
                line_dash="dot",
                line_width=0.1,
                line_color="green",
            )
            fig.add_vrect(
                x0=0.0,
                x1=current_sasmple_size,
                annotation_text=f"Current Sample Size {current_sasmple_size}",
                annotation_position="bottom left",
                fillcolor="green",
                opacity=0.05,
                line_width=0,
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y1,
                mode="markers",
                name=f"MDE: {k}",
                marker=dict(color=palette[i]),
                opacity=0.5,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y2,
                mode="markers",
                name=f"MDE: {k}, Power >= {round(target_power*100, 2)}%",
                marker=dict(color=palette[i], symbol="star"),
                opacity=1.0,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=poly_y,
                mode="lines",
                marker=dict(color=palette[i]),
                showlegend=False,
            )
        )
    return fig
