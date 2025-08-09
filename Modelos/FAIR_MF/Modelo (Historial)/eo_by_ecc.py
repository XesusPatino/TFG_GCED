import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NO_SPLITS = 200

avg_eo_by_ecc = np.full((NO_SPLITS, 19), np.nan)

avg_overall_eo = np.zeros(NO_SPLITS)

avg_rmse_by_ecc_m = np.full((NO_SPLITS, 19), np.nan)
avg_rmse_by_ecc_f = np.full((NO_SPLITS, 19), np.nan)

xx = None

for split_idx in range(0, NO_SPLITS):
    df = pd.read_csv(
        f"eo_by_ecc_results/eo_by_ecc_split_{split_idx+1}.csv", index_col=0
    )

    avg_eo_by_ecc[split_idx] = df["eo"].values
    avg_rmse_by_ecc_m[split_idx] = df["rmse_m"].values
    avg_rmse_by_ecc_f[split_idx] = df["rmse_f"].values

    xx = df.index

    with open(f"eo_by_ecc_results/overall_eo_split_{split_idx+1}.txt") as f:
        overall_eo = float(f.read().strip())
        avg_overall_eo[split_idx] = overall_eo

std_overall_eo = np.nanstd(avg_overall_eo)
avg_overall_eo = np.nanmean(avg_overall_eo)


std_eo_by_ecc = np.nanstd(avg_eo_by_ecc, axis=0)
avg_eo_by_ecc = np.nanmean(avg_eo_by_ecc, axis=0)


print(avg_overall_eo, std_overall_eo)
print(avg_overall_eo)

plt.plot(xx, avg_eo_by_ecc, color="purple")
plt.fill_between(
    xx,
    avg_eo_by_ecc - std_eo_by_ecc,
    avg_eo_by_ecc + std_eo_by_ecc,
    color="purple",
    alpha=0.1,
)
plt.axhline(avg_overall_eo, color="red", linestyle="--")
plt.fill_between(
    xx,
    np.full_like(xx, avg_overall_eo) - std_overall_eo,
    np.full_like(xx, avg_overall_eo) + std_overall_eo,
    color="red",
    alpha=0.05,
)
plt.axhline(0, color="black", linestyle="-")
plt.xlabel("Eccentricity")
plt.ylabel("Average EO")

plt.ylim(-4 * avg_overall_eo, 4 * avg_overall_eo)

plt.show()

avg_rmse_by_ecc_m = np.nanmean(avg_rmse_by_ecc_m, axis=0)
avg_rmse_by_ecc_f = np.nanmean(avg_rmse_by_ecc_f, axis=0)

std_rmse_by_ecc_m = np.nanstd(avg_rmse_by_ecc_m, axis=0)
std_rmse_by_ecc_f = np.nanstd(avg_rmse_by_ecc_f, axis=0)

plt.plot(xx, avg_rmse_by_ecc_m, color="blue", label="M", alpha=0.75)
plt.fill_between(
    xx,
    avg_rmse_by_ecc_m - std_rmse_by_ecc_m,
    avg_rmse_by_ecc_m + std_rmse_by_ecc_m,
    color="blue",
    alpha=0.1,
)

plt.plot(xx, avg_rmse_by_ecc_f, color="purple", label="F", alpha=0.75)
plt.fill_between(
    xx,
    avg_rmse_by_ecc_f - std_rmse_by_ecc_f,
    avg_rmse_by_ecc_f + std_rmse_by_ecc_f,
    color="purple",
    alpha=0.1,
)

plt.xlabel("Eccentricity")
plt.ylabel("Average Error")
plt.legend()

plt.show()
