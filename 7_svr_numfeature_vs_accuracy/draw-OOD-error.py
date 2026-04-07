import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import shutil

shutil.copy('../1_svr_with_18_features/18_performance.csv','.')
shutil.copy('../1_svr_with_18_features/pred_18_feature_values.csv','.')

RAW_DIR = './'
feature_list = list(range(2, 12)) + [18]

records = []
for i in feature_list:
    # ---- (A) Read RMSE Prediction Set (for C0 mean ± std) ----
    perf_path = os.path.join(RAW_DIR, f"{i}_performance.csv")
    perf_df = pd.read_csv(perf_path)
    pred_row = perf_df[perf_df["Metric"] == "RMSE Prediction set"].iloc[0]
    rmse_pred_mean = float(pred_row["Mean"])
    rmse_pred_std  = float(pred_row["Std"])

    pred_mat_path = os.path.join(RAW_DIR, f"pred_{i}_feature_values.csv")
    M = pd.read_csv(pred_mat_path, header=None).values  # expected shape ~ (21, 30)

    # Use rows 1..20 (20 trials). If shape differs, clip to available up to 20 trial rows.
    trials_pred = M[1:21, :] if M.shape[0] >= 21 else M[1:, :]
    if trials_pred.shape[0] > 20:
        trials_pred = trials_pred[:20, :]

    per_sample_std = np.nanstd(trials_pred, axis=0, ddof=0)  # shape (n_samples,)

    error_deviation = float(np.nanmean(per_sample_std))

    records.append({
        "features": i,
        "rmse_pred_mean": rmse_pred_mean,
        "rmse_pred_std":  rmse_pred_std,
        "error_deviation": error_deviation,
    })

# Build dataframe for plotting
df_raw = pd.DataFrame(records).sort_values("features").reset_index(drop=True)

# Print summary information
print("Summary of Performance Results:")
print("#features | RMSE Prediction (Mean ± Std) | Deviation")
for _, row in df_raw.iterrows():
    print(f"{row['features']:>8} | {row['rmse_pred_mean']:.3f} ± {row['rmse_pred_std']:.3f} | {row['error_deviation']:.3f}")

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(df_raw["features"], df_raw["rmse_pred_mean"],
         marker='s', color='C0', linestyle='-', markersize=15, zorder=2)
ax1.errorbar(df_raw["features"], df_raw["rmse_pred_mean"], yerr=df_raw["rmse_pred_std"],
             marker='s', capsize=6, color='C0', linestyle='-', zorder=2)


ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

ax2 = ax1.twinx()
ax2.plot(df_raw["features"], df_raw["error_deviation"],
         marker='x', markersize=15, color='C1', linestyle='--')

ax1.tick_params(axis='both', which='major', labelsize=20)  # Left axis
ax2.tick_params(axis='y', which='major', labelsize=20)     # Right axis
ax2.spines['right'].set_color('C1')
ax2.tick_params(axis='y', colors='C1')
ax2.yaxis.label.set_color('C1')

ax1.set_ylim([0.15, 0.65])
ax2.set_ylim([0.06, 0.18])
ax1.set_ylim([0.30, 0.70])

plt.tight_layout()

plt.savefig('figure-OOD.png', dpi=125)

plt.show()

