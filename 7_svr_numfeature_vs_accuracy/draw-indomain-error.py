import pandas as pd
import matplotlib.pyplot as plt
import os

import shutil

shutil.copy('../1_svr_with_18_features/18_performance.csv','.')

RAW_DIR = './'
feature_list = list(range(2, 12)) + [18]

records = []
for i in feature_list:
    fpath = os.path.join(RAW_DIR, f"{i}_performance.csv")
    dfi = pd.read_csv(fpath)

    # Extract the required rows
    test_row = dfi[dfi["Metric"] == "RMSE Test"].iloc[0]
    train_row = dfi[dfi["Metric"] == "RMSE Train"].iloc[0]

    records.append({
        "features": i,
        "rmse_test_mean": float(test_row["Mean"]),
        "rmse_test_std": float(test_row["Std"]),
        "rmse_train_mean": float(train_row["Mean"]),
        "gen_gap": float(test_row["Mean"]) - float(train_row["Mean"]),
    })

df = pd.DataFrame(records).sort_values("features").reset_index(drop=True)

# Print summary information
print("Summary of Performance Results:")
print("#features | RMSE Test (Mean ± Std) | Generalization Gap")
for _, row in df.iterrows():
    print(f"{row['features']:>8} | {row['rmse_test_mean']:.3f} ± {row['rmse_test_std']:.3f} | {row['gen_gap']:.3f}")

fig, ax1 = plt.subplots(figsize=(12, 6))

# C0: Test RMSE with error bars
ax1.plot(df["features"], df["rmse_test_mean"], marker='s', color='C0',
         linestyle='-', markersize=15, zorder=2)
ax1.errorbar(df["features"], df["rmse_test_mean"], yerr=df["rmse_test_std"],
             marker='s', capsize=6, color='C0', linestyle='-', zorder=2)

ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# C1: Generalization gap
ax2 = ax1.twinx()
ax2.plot(df["features"], df["gen_gap"], marker='x', label='Generalization Gap',
         color='C1', markersize=15, linestyle='--')

# Axis formatting
ax1.tick_params(axis='both', which='major', labelsize=20)  # Left axis
ax2.tick_params(axis='y', which='major', labelsize=20)     # Right axis
ax2.spines['right'].set_color('C1')
ax2.tick_params(axis='y', colors='C1')
ax2.yaxis.label.set_color('C1')

ax1.set_ylim([0.15, 0.65])
ax2.set_ylim([0, 0.08])

plt.tight_layout()

plt.savefig('figure-indomain.png', dpi=125)

plt.show()

