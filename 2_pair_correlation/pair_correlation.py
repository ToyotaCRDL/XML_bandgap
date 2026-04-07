import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Column headers
head_list = [
    r'$E_\mathrm{g}^\mathrm{GW}$', r'$E_\mathrm{g}^\mathrm{PBE}$', r'$E_\mathrm{g}^\mathrm{mBJ}$',
    '$V$', r'$E_\mathrm{coh}$', r'$\overline{|n|}$', r'$\overline{Z}$', r'$\overline{p}$',
    r'$\overline{m}$', r'$\overline{r}$', r'$\overline{\chi}$', r'$\overline{I}$',
    r'$\sigma(|n|)$', r'$\sigma(Z)$', r'$\sigma(p)$', r'$\sigma(m)$', r'$\sigma(r)$',
    r'$\sigma(\chi)$', r'$\sigma(I)$'
]

# Load data
filename = "../GWgap_predictor_data_PRB_93_115104_2016.csv"
csvframe = pd.read_csv(filename)
variables = csvframe.iloc[:, 0:]
variables.columns = head_list

def plot_combined_correlation(dataframe):
    pearson_corr = dataframe.corr(method='pearson')
    spearman_corr = dataframe.corr(method='spearman')

    # Masks for triangles
    mask_lower = np.triu(np.ones_like(pearson_corr, dtype=bool))
    mask_upper = np.tril(np.ones_like(spearman_corr, dtype=bool))

    # Plot
    plt.figure(figsize=(12, 12))

    # Pearson (lower)
    sns.heatmap(
        pearson_corr,
        mask=mask_lower,
        annot=True,
        cmap='icefire',
        fmt='.2f',
        linewidths=.5,
        cbar=False,
        vmin=-1, vmax=1,
        xticklabels=dataframe.columns,
        yticklabels=dataframe.columns
    )

    # Spearman (upper)
    sns.heatmap(
        spearman_corr,
        mask=mask_upper,
        annot=True,
        cmap='icefire',
        fmt='.2f',
        linewidths=.5,
        cbar_kws={"shrink": .8, "ticks": [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
        vmin=-1, vmax=1,
        xticklabels=dataframe.columns,
        yticklabels=dataframe.columns
    )

    plt.title('Combined Pearson (Lower) and Spearman (Upper) Correlation')
    plt.savefig('correl-18-with-GW.png', dpi=150)
    plt.show()

# --- Compute correlation matrices (Pearson & Spearman) ---
pearson_corr = variables.corr(method='pearson')
spearman_corr = variables.corr(method='spearman')

# --- Save each correlation matrix as CSV ---
pearson_corr.round(6).to_csv('correl-18-with-GW-pearson.csv', encoding='utf-8')
spearman_corr.round(6).to_csv('correl-18-with-GW-spearman.csv', encoding='utf-8')

# --- Rank feature pairs by average correlation strength ---
def rank_feature_pairs(pearson_df: pd.DataFrame, spearman_df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    features = pearson_df.columns.tolist()
    rows = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            p = pearson_df.loc[f1, f2]
            s = spearman_df.loc[f1, f2]
            mean_abs = (abs(p) + abs(s)) / 2.0
            mean_signed = (p + s) / 2.0
            rows.append({
                "feature_1": f1,
                "feature_2": f2,
                "pearson": p,
                "spearman": s,
                "mean_abs": mean_abs,
                "mean_signed": mean_signed
            })
    pair_df = pd.DataFrame(rows)
    # Sort by mean_abs descending (strongest correlation magnitude first)
    pair_df = pair_df.sort_values(by="mean_abs", ascending=False).reset_index(drop=True)
    return pair_df.head(top_k), pair_df

# Build ranking and print top 15
top15, all_pairs = rank_feature_pairs(pearson_corr, spearman_corr, top_k=20)

# Print to console (readable formatting)
print("\nTop 20 feature pairs by mean(|Pearson|, |Spearman|):")
print(top15[["feature_1", "feature_2", "pearson", "spearman", "mean_abs", "mean_signed"]]
      .round(4).to_string(index=False))

# Optionally save the top list and the full pair table
top15.round(6).to_csv("top15_pairs_by_mean_abs.csv", index=False, encoding="utf-8")
all_pairs.round(6).to_csv("all_pairs_by_mean_abs.csv", index=False, encoding="utf-8")

# --- Plot combined heatmap (unchanged visual) ---
plot_combined_correlation(variables)

