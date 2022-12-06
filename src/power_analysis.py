import numpy as np
from scipy import stats as st
from src.sample_size.simulation.calculator import *
import multiprocess as mp
from functools import partial
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

sample_base_rate = 0.3852
sample_data = st.binom.rvs(1, sample_base_rate, size=900000)
sample_sizes = range(5000, 200000 + 1, 2500)  # Sample sizes we will test over
relative_effect = 1.03

alpha = 0.05
sims = 3000
alternative = "two-sided"


pool = mp.Pool(mp.cpu_count() - 1)
result = pool.map(
    partial(
        multiple_proportions_mc_power_analysis,
        base_rate=np.mean(sample_data),
        n_variants=2,
        relative_effect=relative_effect,
        alpha=0.05,
        n_simulation=sims,
        alternative=alternative,
    ),
    tqdm(
        np.array(sample_sizes),
        total=len(sample_sizes),
        desc="🚀 Running simulations",
        colour="red",
    ),
)


result = pd.DataFrame(result, columns=["Sample Size", "Power for any", "Power for all"])

power_col = "Power for any"
x = result["Sample Size"]
y = result[power_col]

poly = np.polyfit(x, y, 3)
poly_y = np.minimum(np.poly1d(poly)(x), 1.0)
plt.style.use("ggplot")
plt.figure(figsize=[12, 6])

fig, ax = plt.subplots()
ax.scatter(x, y, color="red", alpha=0.5, label="MC Simulation")
ax.axhline(y=0.8, color="blue", linestyle="--", label="Power Target")
ax.plot(x, poly_y, color="red")
ax.set_xlabel("Sample Size Required (2 groups)", fontsize=14)
ax.set_ylabel("Power", fontsize=14)
plt.legend()

target_id = min(
    range(len(result[power_col])), key=lambda i: abs(result[power_col][i] - 0.8)
)
print("Minimum sample required for 80% power", result["Sample Size"][target_id])
