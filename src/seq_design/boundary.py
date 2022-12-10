"""Group sequential designs using error spending functions

Reference:
    
- Chang MN, Hwang I, Shih WJ. Group sequential designs using both type I and type II 
error probability spending functions. Communications in Statistics - Theory Methods, 
1998. 27:1322–1339
- Since I could not get the access to the literature above, got indirect reference from:
    - Wu, J., & Li, Y. (2020). Group sequential design for historical control trials 
    using error spending functions. Journal of biopharmaceutical statistics, 30(2), 
    351–363.
"""
from dataclasses import dataclass

from matplotlib import pyplot as plt
from src.base_fields import *

import numpy as np
from numpy import typing as npt
from scipy import stats as st
from scipy.optimize import brentq as root
from statsmodels.sandbox.distributions.extras import mvnormcdf

from src.seq_design import spend_func as spend
from util.stdout import suppress_stdout


@dataclass
class SDBoundary:
    upper: npt.NDArray[np.number]
    lower: npt.NDArray[np.number]
    ts: npt.NDArray[np.number]


def fx1(x, ub, covm, tprob) -> float:
    kn = np.shape(ub)
    lb = np.repeat(-np.inf, kn)
    upper = np.append(ub, [np.inf], axis=0)
    lower = np.append(lb, [x], axis=0)
    umu = np.zeros_like(kn[0] + 1)
    with suppress_stdout():
        pmv = mvnormcdf(upper=upper, lower=lower, mu=umu, cov=covm)
    return tprob - pmv


def fx2(x, lb, eta_m, ts, covm, tprob) -> float:
    kn = np.shape(lb)
    ub = np.repeat(np.inf, kn)
    upper = np.append(ub, [x], axis=0)
    lower = np.append(lb, [-np.inf], axis=0)
    lmu = eta_m * np.sqrt(ts[0 : kn[0] + 1])
    with suppress_stdout():
        pmv = mvnormcdf(upper=upper, lower=lower, mu=lmu, cov=covm)
    return tprob - pmv


def sequential_design(
    k: int = N_STAGE,
    alpha: float = ALPHA,
    beta: float = BETA,
    option: spend.SpendOptions = spend.SpendOptions.OBF,
) -> SDBoundary:
    if option == spend.SpendOptions.OBF:
        alpha_spend_func = spend.obrien_fleming
    elif option == spend.SpendOptions.POCOCK:
        alpha_spend_func = spend.pocock
    elif option == spend.SpendOptions.KIM_DEMETS:
        alpha_spend_func = spend.kim_demets
    elif option == spend.SpendOptions.HWANG_SHIH_DECANI:
        alpha_spend_func = spend.hwang_shih_decani
    else:
        raise ValueError(f"Alpha Spend Function: {option} is not supported.")

    eta_0 = st.norm.ppf(1 - alpha) + st.norm.ppf(1 - beta)
    eta_1 = np.sqrt(2) * eta_0
    eta_m = (eta_0 + eta_1) / 2

    ts = np.linspace(1, k, num=k) / k
    alpha_1 = spend.spend_function(alpha_spend_func(ts, alpha=alpha))
    beta_1 = spend.spend_function(alpha_spend_func(ts, alpha=beta))

    covmat = np.empty(shape=(k, k))
    for i in range(0, k):
        for j in range(0, k):
            covmat[i, j] = np.minimum(ts[i], ts[j]) / np.sqrt(ts[i] * ts[j])
    ub, lb = np.zeros(k), np.zeros(k)
    ub[0] = st.norm.ppf(1 - alpha_1[0])
    for i in range(1, k):
        ubi = ub[0:i]
        args = (ubi, covmat[0 : (i + 1), 0 : (i + 1)], alpha_1[i])
        ub[i] = root(fx1, -10, 10, args=args)
    flag2 = 0
    while True:
        flag = 0
        eta_m = (eta_0 + eta_1) / 2
        lb[0] = st.norm.ppf(beta_1[0]) + eta_m * np.sqrt(ts[0])
        if lb[0] > ub[0]:
            eta_1 = eta_m
        else:
            for i in range(1, k):
                lbi = lb[0:i]
                cov = covmat[0 : (i + 1), 0 : (i + 1)]
                args = (lbi, eta_m, ts, cov, beta_1[i])
                lb[i] = root(fx2, -10, 10, args=args)
                if lb[i] > ub[i]:
                    flag = 1
                    break
            if flag == 1:
                    eta_1 = eta_m
            else:
                lb[k - 1] = ub[k - 1]
                pv = np.empty_like(lb)
                pv[0] = st.norm.cdf(lb[0], loc=eta_m * np.sqrt(ts[0]))
                for i in range(1, k):
                    upper = np.append(ub[0:i], [lb[i]], axis=0)
                    lower = np.append(lb[0:i], [-np.inf], axis=0)
                    lmu = eta_m * np.sqrt(ts[0 : i + 1])
                    covm = covmat[0 : (i + 1), 0 : (i + 1)]
                    with suppress_stdout():
                        pv[i] = mvnormcdf(upper=upper, lower=lower, mu=lmu, cov=covm)
                beta_k = sum(pv)
                if beta_k < beta:
                    eta_1 = eta_m
                else:
                    eta_0 = eta_m
                if abs(beta - beta_k) < 1e-05:
                    flag2 = 1
                    break
        if flag2 == 1:
                    break

    return SDBoundary(upper=ub, lower=lb, ts=ts)


def vis_sequential_design(boundary: SDBoundary) -> None:
    x, ub, lb = boundary.ts, boundary.upper, boundary.lower
    _, ax = plt.subplots()
    ax.plot(x, ub, label="Upper Bound")
    ax.plot(x, lb, label="Lower Bound")
    ax.set_xlabel("Sample / Information Proportion", fontsize=14)
    ax.set_ylabel("Z-score", fontsize=14)
    plt.title(f"Group Sequential Method for early stopping (stage={len(x)})")
    for index in range(len(x)):
        ax.text(x[index], ub[index], round(ub[index], 2), size=12)
        ax.text(x[index], lb[index], round(lb[index], 2), size=12)
    plt.legend()
    plt.show()
