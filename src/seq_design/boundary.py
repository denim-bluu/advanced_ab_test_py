"""Group sequential designs using error spending functions

Reference:
    Wu, J., & Li, Y. (2020). Group sequential design for historical control trials using
    error spending functions. Journal of biopharmaceutical statistics, 30(2), 351–363.
"""
from dataclasses import dataclass

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
    etam: float
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


def fx2(x, lb, etam, ts, covm, tprob) -> float:
    kn = np.shape(lb)
    ub = np.repeat(np.inf, kn)
    upper = np.append(ub, [x], axis=0)
    lower = np.append(lb, [-np.inf], axis=0)
    lmu = etam * np.sqrt(ts[0 : kn[0] + 1])
    blockprint()
    pmv = mvnormcdf(upper=upper, lower=lower, mu=lmu, cov=covm)
    return tprob - pmv


def find_bound(
    d2, d1, alpha, beta, k, option: spend.SpendOptions = spend.SpendOptions.OBF
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

    eta0 = st.norm.ppf(1 - alpha) + st.norm.ppf(1 - beta)
    eta1 = np.sqrt(2) * eta0
    etam = (eta0 + eta1) / 2

    ## This is for HCT
    # ti =np.linspace(1, k, num=k) / k
    # r = d2 / d1
    # ts = (1 + r) * ti / (1 + r * ti)

    ts = np.linspace(1, k, num=k) / k
    tij = np.insert(ts, 0, 0.0, axis=0)
    alpha1 = alpha_spend_func(ts, alpha=alpha)
    beta1 = alpha_spend_func(ts, alpha=beta)

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
        etam = (eta0 + eta1) / 2
        lb[0] = st.norm.ppf(beta1[0]) + etam * np.sqrt(ts[0])
        if lb[0] > ub[0]:
            eta1 = etam
        else:
            for i in range(1, k):
                lbi = lb[0:i]
                cov = covmat[1 : (i + 2), 1 : (i + 2)]
                args = (lbi, etam, ts, cov, beta1[i] - beta1[i - 1])
                lb[i] = root(fx2, -10, 10, args=args)
                if lb[i] > ub[i]:
                    flag = 1
                    break
            if flag == 1:
                eta1 = etam
            else:
                lb[k - 1] = ub[k - 1]
                pv = np.empty_like(lb)
                pv[0] = st.norm.cdf(lb[0], loc=etam * np.sqrt(ts[0]))
                for i in range(1, k):
                    upper = np.append(ub[0:i], [lb[i]], axis=0)
                    lower = np.append(lb[0:i], [-np.inf], axis=0)
                    lmu = etam * np.sqrt(ts[0 : i + 1])
                    covm = covmat[1 : (i + 2), 1 : (i + 2)]
                    blockprint()
                    pv[i] = mvnormcdf(upper=upper, lower=lower, mu=lmu, cov=covm)
                betak = sum(pv)
                if betak < beta:
                    eta1 = etam
                else:
                    eta0 = etam
                if abs(beta - betak) < 1e-05:
                    flag = 2
        if flag == 2:
            break

    return SDBoundary(upper=ub, lower=lb, etam=etam, ts=ts)


def sequential_design(
    k: int,
    alpha: float = 0.05,
    beta: float = 0.1,
    delta: int | float = 1 / 1.75,
    d1: float | int = 65,
    option: spend.SpendOptions = spend.SpendOptions.OBF,
) -> SDBoundary:
    z_alpha = st.norm.ppf(1 - alpha)
    z_power = st.norm.ppf(1 - beta)
    temp1 = np.exp(np.sqrt(1 / d1 * (z_alpha + z_power) ** 2))
    temp2 = np.exp(-np.sqrt(1 / d1 * (z_alpha + z_power) ** 2))
    if (delta < temp1) & (delta > temp2):
        raise ValueError(f"Delta must be greater than {temp1} or less than {temp2}")
    d2start = np.ceil((np.log(delta) ** 2 / (z_alpha + z_power) ** 2 - 1 / d1) ** (-1))
    find = find_bound(d2=d2start, d1=d1, alpha=alpha, beta=beta, k=k, option=option)
    ctn = 0
    while True:
        ctn += 1
        d2 = np.ceil(1 / ((np.log(delta) ** 2) / (find.etam**2) - (1 / d1)))
        find = find_bound(d2=d2, d1=d1, alpha=alpha, beta=beta, k=k, option=option)
        if ctn > 5:
            break
    return SDBoundary(upper=find.upper, lower=find.lower, etam=find.etam, ts=find.ts)


# # Example
# d2 = 48.0
# d1 = 65
# alpha = 0.05
# beta = 0.1
# k = 3
# sequential_design(3, alpha=0.05, beta=0.1, delta=1 / 1.75, d1=65)
