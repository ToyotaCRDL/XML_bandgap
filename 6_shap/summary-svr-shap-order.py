import pandas as pd
import numpy as np

# ---------------------------
# 1) Load data (PFI + SHAP)
# ---------------------------
file_path_pfi = '../5_pfi/svr-pfi-11feature.csv'
pfi_df = pd.read_csv(file_path_pfi, header=None)

pfi_mean = pfi_df.mean()
pfi_std  = pfi_df.std()

head_list = [
    r'$E_\mathrm{g}^\mathrm{PBE}$', r'$E_\mathrm{coh}$', r'$\overline{|n|}$',
    r'$\overline{Z}$', r'$\overline{r}$', r'$\overline{\chi}$',
    r'$\sigma(|n|)$', r'$\sigma(p)$', r'$\sigma(m)$', r'$\sigma(r)$', r'$\sigma(\chi)$'
]

# Actual feature column numbers corresponding to head_list order
feature_cols = [1, 4, 5, 6, 9, 10, 12, 14, 15, 16, 17]

# ---------------------------
# 2) Load SHAP (20 runs, absolute mean)
# ---------------------------
arb = []
for i in range(20):
    d = pd.read_csv(f'./shap_values_{i}.csv', header=None).iloc[:, 1:]
    d.columns = head_list
    arb.append(abs(d).mean().values)

arb2 = np.array(arb)
shap_mean = np.mean(arb2, axis=0)
shap_std  = np.std(arb2, axis=0)

# ---------------------------
# 3) Sort by average of PFI mean and SHAP mean
# ---------------------------
avg_vals = (pfi_mean.to_numpy() + shap_mean) / 2.0
sorted_indices = np.argsort(-avg_vals)

# ---------------------------
# 4) Make ranked table
# ---------------------------
rows = []
for rank, idx in enumerate(sorted_indices, start=1):
    rows.append({
        "Rank": rank,
        "Column": feature_cols[idx],
        "Feature": head_list[idx],
        "PFI_mean": pfi_mean.iloc[idx],
        "PFI_std": pfi_std.iloc[idx],
        "SHAP_mean": shap_mean[idx],
        "SHAP_std": shap_std[idx],
        "Average_(PFI+SHAP)/2": avg_vals[idx]
    })

result_df = pd.DataFrame(rows)

# ---------------------------
# 5) Print results
# ---------------------------
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

print(result_df.to_string(index=False))

# ---------------------------
# 6) Print sorted column-number list
# ---------------------------
sorted_feature_cols = [feature_cols[idx] for idx in sorted_indices]

print("\nSorted feature column numbers:")
print(sorted_feature_cols)

# ---------------------------
# 7) Save results
# ---------------------------
result_df.to_csv("pfi_shap_ranked.csv", index=False)
