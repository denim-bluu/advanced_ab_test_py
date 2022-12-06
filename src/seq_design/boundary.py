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
import pandas as pd
from matplotlib import pyplot as plt
from src.base_fields import *

import numpy as np
from numpy import typing as npt
from scipy import stats as st
from scipy.optimize import brentq as root
from statsmodels.sandbox.distributions.extras import mvnormcdf

from src.seq_design import spend_func as spend
from util.stdout import blockprint


@dataclass
class SDBoundary:
    upper: npt.NDArray[np.number]
    lower: npt.NDArray[np.number]
    eta_m: float
    ts: npt.NDArray[np.number]


def fx1(x, ub, covm, tprob) -> float:
    kn = np.shape(ub)
    lb = np.repeat(-np.inf, kn)
    upper = np.append(ub, [np.inf], axis=0)
    lower = np.append(lb, [x], axis=0)
    umu = np.zeros_like(kn[0] + 1)
    blockprint()
    pmv = mvnormcdf(upper=upper, lower=lower, mu=umu, cov=covm)
    return tprob - pmv


def fx2(x, lb, eta_m, ts, covm, tprob) -> float:
    kn = np.shape(lb)
    ub = np.repeat(np.inf, kn)
    upper = np.append(ub, [x], axis=0)
    lower = np.append(lb, [-np.inf], axis=0)
    lmu = eta_m * np.sqrt(ts[0 : kn[0] + 1])
    blockprint()
    pmv = mvnormcdf(upper=upper, lower=lower, mu=lmu, cov=covm)
    return tprob - pmv


def find_bound(
    alpha: float = ALPHA,
    beta: float = BETA,
    k: int = N_STAGE,
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
    tij = np.insert(ts, 0, 0.0, axis=0)
    alpha1 = alpha_spend_func(ts, alpha=alpha)
    beta_1 = alpha_spend_func(ts, alpha=beta)

    covmat = np.empty(shape=(k + 1, k + 1))
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            covmat[i, j] = np.minimum(tij[i], tij[j]) / np.sqrt(tij[i] * tij[j])
    ub, lb = np.zeros(k), np.zeros(k)
    ub[0] = st.norm.ppf(1 - alpha1[0])
    for i in range(1, k):
        ubi = ub[0:i]
        args = (ubi, covmat[1 : (i + 2), 1 : (i + 2)], alpha1[i] - alpha1[i - 1])
        ub[i] = root(fx1, -10, 10, args=args)
    ctn = 0
    while True:
        ctn += 1
        flag = 0
        eta_m = (eta_0 + eta_1) / 2
        lb[0] = st.norm.ppf(beta_1[0]) + eta_m * np.sqrt(ts[0])
        if lb[0] > ub[0]:
            eta_1 = eta_m
        else:
            for i in range(1, k):
                lbi = lb[0:i]
                cov = covmat[1 : (i + 2), 1 : (i + 2)]
                args = (lbi, eta_m, ts, cov, beta_1[i] - beta_1[i - 1])
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
                    covm = covmat[1 : (i + 2), 1 : (i + 2)]
                    blockprint()
                    pv[i] = mvnormcdf(upper=upper, lower=lower, mu=lmu, cov=covm)
                beta_k = sum(pv)
                if beta_k < beta:
                    eta_1 = eta_m
                else:
                    eta_0 = eta_m
                if abs(beta - beta_k) < 1e-05:
                    flag = 2
        if flag == 2:
            break

    return SDBoundary(upper=ub, lower=lb, eta_m=eta_m, ts=ts)


def sequential_design(
    k: int,
    alpha: float = ALPHA,
    beta: float = BETA,
    option: spend.SpendOptions = spend.SpendOptions.OBF,
) -> SDBoundary:
    find = find_bound(alpha=alpha, beta=beta, k=k, option=option)
    ctn = 0
    while True:
        ctn += 1
        find = find_bound(alpha=alpha, beta=beta, k=k, option=option)
        if ctn > 5:
            break
    return SDBoundary(upper=find.upper, lower=find.lower, eta_m=find.eta_m, ts=find.ts)


def vis_sequential_design(boundary: SDBoundary) -> None:
    x, ub, lb = boundary.ts, boundary.upper, boundary.lower
    _, ax = plt.subplots()
    ax.plot(x, ub, label="Upper Bound")
    ax.plot(x, lb, label="Lower Bound")
    ax.set_xlabel("Sample Proportion per stage", fontsize=14)
    ax.set_ylabel("Z-score", fontsize=14)
    plt.title(f"Group Sequential Method for early stopping (stage={len(x)})")
    for index in range(len(x)):
        ax.text(x[index], ub[index], round(ub[index], 2), size=12)
        ax.text(x[index], lb[index], round(lb[index], 2), size=12)
    plt.legend()
    plt.show()
