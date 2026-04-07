import os
import itertools
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# ============================================================
# Settings
# ============================================================

ALPHA = 0.01          # 99% significance level
CY = 20               # number of random splits
TEST_SIZE = 0.25

TRAIN_CSV = "../GWgap_predictor_data_PRB_93_115104_2016.csv"
OOD_CSV = "../GWgap_OOD_40.csv"
PRISTINE_PERF_CSV = "../1_svr_with_18_features/18_performance.csv"

START_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

GAMMA_RANGE = 10.0 ** np.arange(-6, 6)
COST_RANGE = 10.0 ** np.arange(-5, 5)
EPSILON_RANGE = [0.001, 0.005, 0.01]


# ============================================================
# Utility functions
# ============================================================

def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def safe_spearman(x, y):
    return pd.Series(x).corr(pd.Series(y), method="spearman")


def pair_score_from_rp_rs(rp, rs):
    return 0.5 * (abs(rp) + abs(rs))


def extract_rmse_test_trials(perf_csv_path):
    """
    Extract RMSE Test trial values from a formatted performance CSV.

    Expected row format:
    RMSE Test, mean, std, trial_0, trial_1, ..., trial_19
    """
    df = pd.read_csv(perf_csv_path, header=None)

    rmse_row = None
    for i in range(len(df)):
        first_cell = str(df.iloc[i, 0]).strip()
        if first_cell == "RMSE Test":
            rmse_row = df.iloc[i]
            break

    if rmse_row is None:
        raise ValueError(f"'RMSE Test' row not found in {perf_csv_path}")

    trials = pd.to_numeric(rmse_row.iloc[3:], errors="coerce").dropna().to_numpy(dtype=float)

    if len(trials) == 0:
        raise ValueError(f"No RMSE Test trial values found in {perf_csv_path}")

    return trials


def compute_target_correlations(csvframe, target, feature_col):
    x = csvframe.iloc[:, feature_col]
    rp = safe_pearson(x, target)
    rs = safe_spearman(x, target)
    score = pair_score_from_rp_rs(rp, rs)
    return rp, rs, score


def compute_pair_ranking(csvframe, current_order, out_csv=None):
    rows = []

    for f1, f2 in itertools.combinations(current_order, 2):
        x1 = csvframe.iloc[:, f1]
        x2 = csvframe.iloc[:, f2]

        rp = safe_pearson(x1, x2)
        rs = safe_spearman(x1, x2)
        score = pair_score_from_rp_rs(rp, rs)

        rows.append({
            "feature_1": f1,
            "feature_2": f2,
            "pair_rp": rp,
            "pair_rs": rs,
            "pair_score_abs_avg": score
        })

    pair_df = pd.DataFrame(rows).sort_values(
        by="pair_score_abs_avg", ascending=False
    ).reset_index(drop=True)

    if out_csv is not None:
        pair_df.to_csv(out_csv, index=False)

    return pair_df


def compute_vif_table(csvframe, feature_order):
    X = csvframe.iloc[:, feature_order]
    X_const = add_constant(X)

    vif_list = []
    for i in range(1, X_const.shape[1]):  # skip constant
        vif_value = variance_inflation_factor(X_const.values, i)
        vif_list.append(vif_value)

    return pd.DataFrame({
        "feature": feature_order,
        "VIF": vif_list
    })


def compare_to_pristine_one_sided(pristine_rmse, candidate_rmse):
    """
    One-sided paired t-test for stopping rule.

    H0: mean(candidate - pristine) <= 0
    H1: mean(candidate - pristine) > 0

    If p < ALPHA, the reduced model is significantly worse than pristine.
    """
    t_stat, p_value = ttest_rel(candidate_rmse, pristine_rmse, alternative="greater")

    candidate_mean = np.mean(candidate_rmse)
    pristine_mean = np.mean(pristine_rmse)

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "significant_increase": bool(p_value < ALPHA),
        "candidate_mean": candidate_mean,
        "pristine_mean": pristine_mean
    }


# ============================================================
# SVR runner for one feature set
# ============================================================

def run_svr_for_feature_set(csvframe, target, csvframe2, target2, feature_order, outdir, case_label):
    os.makedirs(outdir, exist_ok=True)

    feature_order2 = [x + 2 for x in feature_order]

    variables = csvframe.iloc[:, feature_order]
    variables2 = csvframe2.iloc[:, feature_order2]

    rmse_tst = np.zeros(CY)
    mae_tst = np.zeros(CY)
    rmse_tr = np.zeros(CY)
    mae_tr = np.zeros(CY)
    cost = np.zeros(CY)
    gamma = np.zeros(CY)
    eps = np.zeros(CY)

    R_sq_tr = np.zeros(CY)
    R_sq = np.zeros(CY)

    rmse_pred = np.zeros(CY)
    mae_pred = np.zeros(CY)
    R_sq_pred = np.zeros(CY)

    pred_result_collect = []
    pred_result_collect2 = []
    pred_result_collect.append(target2)
    pred_result_collect2.append(target2)

    train_idx_list = []
    test_idx_list = []

    print("------------------------------------------------------------")
    print("Running SVR for case:", case_label)
    print("Feature order:", feature_order)
    print("------------------------------------------------------------")

    for i in range(CY):
        print("cycle # :", i)

        idx = np.arange(len(variables))
        tr_idx, tst_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=i)

        tr_x = variables.iloc[tr_idx]
        tst_x = variables.iloc[tst_idx]
        tr_y = target.iloc[tr_idx]
        tst_y = target.iloc[tst_idx]

        train_idx_list.append(np.array(tr_idx, dtype=int))
        test_idx_list.append(np.array(tst_idx, dtype=int))

        pred_y = target2
        pred_x = variables2

        scaler = StandardScaler()
        tr_x_scaled = scaler.fit_transform(tr_x)
        tst_x_scaled = scaler.transform(tst_x)
        pred_x_scaled = scaler.transform(pred_x)

        param_grid = {
            "gamma": GAMMA_RANGE,
            "C": COST_RANGE,
            "epsilon": EPSILON_RANGE
        }

        grid_search = GridSearchCV(
            svm.SVR(),
            param_grid,
            cv=10,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        grid_search.fit(tr_x_scaled, tr_y)

        best_params = grid_search.best_params_
        gamma[i] = best_params["gamma"]
        eps[i] = best_params["epsilon"]
        cost[i] = best_params["C"]

        svr = svm.SVR(kernel="rbf", gamma=gamma[i], C=cost[i], epsilon=eps[i])
        svr.fit(tr_x_scaled, tr_y)

        tr_pred = svr.predict(tr_x_scaled)
        tst_pred = svr.predict(tst_x_scaled)
        pred_pred = svr.predict(pred_x_scaled)

        rmse_tr[i] = np.sqrt(mean_squared_error(tr_y, tr_pred))
        rmse_tst[i] = np.sqrt(mean_squared_error(tst_y, tst_pred))
        mae_tr[i] = mean_absolute_error(tr_y, tr_pred)
        mae_tst[i] = mean_absolute_error(tst_y, tst_pred)

        r_tr = safe_pearson(tr_y, tr_pred)
        r_tst = safe_pearson(tst_y, tst_pred)
        r_pred = safe_pearson(pred_y, pred_pred)

        R_sq_tr[i] = r_tr ** 2 if not np.isnan(r_tr) else np.nan
        R_sq[i] = r_tst ** 2 if not np.isnan(r_tst) else np.nan
        rmse_pred[i] = np.sqrt(mean_squared_error(pred_y, pred_pred))
        mae_pred[i] = mean_absolute_error(pred_y, pred_pred)
        R_sq_pred[i] = r_pred ** 2 if not np.isnan(r_pred) else np.nan

        pred_result_collect.append(pred_pred - pred_y)
        pred_result_collect2.append(pred_pred)

        tr_pd = pd.DataFrame({"tr_y": tr_y, "tr_pred": tr_pred})
        tst_pd = pd.DataFrame({"tst_y": tst_y, "tst_pred": tst_pred})

        tr_pd.to_csv(os.path.join(outdir, case_label + "_tr_" + str(i) + ".csv"), index=False)
        tst_pd.to_csv(os.path.join(outdir, case_label + "_tst_" + str(i) + ".csv"), index=False)

#        pd.DataFrame({"train_idx": tr_idx}).to_csv(
#            os.path.join(outdir, case_label + "_train_idx_" + str(i) + ".csv"), index=False
#        )
#        pd.DataFrame({"test_idx": tst_idx}).to_csv(
#            os.path.join(outdir, case_label + "_test_idx_" + str(i) + ".csv"), index=False
#        )

    train_idx_df = pd.DataFrame({f"cycle_{i}": train_idx_list[i] for i in range(CY)})
    test_idx_df = pd.DataFrame({f"cycle_{i}": test_idx_list[i] for i in range(CY)})

    train_idx_df.to_csv(os.path.join(outdir, case_label + "_train_indices_cycles.csv"), index=False)
    test_idx_df.to_csv(os.path.join(outdir, case_label + "_test_indices_cycles.csv"), index=False)

    hyper_df = pd.DataFrame({
        "cycle": np.arange(CY, dtype=int),
        "gamma": gamma,
        "C": cost,
        "epsilon": eps
    })
    hyper_df.to_csv(os.path.join(outdir, case_label + "_hyperparams.csv"), index=False)

    pd.DataFrame(pred_result_collect, columns=None).to_csv(
        os.path.join(outdir, case_label + "_feature_errors.csv"), index=False, header=False
    )
    pd.DataFrame(pred_result_collect2, columns=None).to_csv(
        os.path.join(outdir, case_label + "_feature_values.csv"), index=False, header=False
    )

    metrics = {
        "RMSE Train": rmse_tr,
        "RMSE Test": rmse_tst,
        "MAE Train": mae_tr,
        "MAE Test": mae_tst,
        "R^2 Train": R_sq_tr,
        "R^2 Test": R_sq,
        "RMSE Prediction set": rmse_pred,
        "MAE Prediction set": mae_pred,
        "R^2 Prediction set": R_sq_pred,
    }

    rows = []
    for metric_name, values in metrics.items():
        row = {
            "Metric": metric_name,
            "Mean": np.mean(values),
            "Std": np.nanstd(values),
        }
        for j, v in enumerate(values):
            row[f"trial_{j}"] = v
        rows.append(row)

    perf_df = pd.DataFrame(rows)
    perf_df.to_csv(os.path.join(outdir, case_label + "_performance.csv"), index=False)

    return {
        "feature_order": feature_order,
        "rmse_tst": rmse_tst.copy(),
        "rmse_tst_mean": np.mean(rmse_tst),
        "rmse_tst_std": np.nanstd(rmse_tst),
        "rmse_tr": rmse_tr.copy(),
        "rmse_pred": rmse_pred.copy(),
        "mae_tst": mae_tst.copy(),
        "R_sq": R_sq.copy(),
        "performance_csv": os.path.join(outdir, case_label + "_performance.csv")
    }


# ============================================================
# Pair evaluation
# ============================================================

def evaluate_pair(
    csvframe,
    target,
    csvframe2,
    target2,
    pristine_rmse,
    current_order,
    pair_feature_1,
    pair_feature_2,
    pair_rank,
    n_folder
):
    pair_prefix = f"pair{pair_rank+1}_{pair_feature_1}_{pair_feature_2}"

    pair_rp = safe_pearson(csvframe.iloc[:, pair_feature_1], csvframe.iloc[:, pair_feature_2])
    pair_rs = safe_spearman(csvframe.iloc[:, pair_feature_1], csvframe.iloc[:, pair_feature_2])

    feat1_target_rp, feat1_target_rs, feat1_target_score = compute_target_correlations(
        csvframe, target, pair_feature_1
    )
    feat2_target_rp, feat2_target_rs, feat2_target_score = compute_target_correlations(
        csvframe, target, pair_feature_2
    )

    order_remove_1 = [x for x in current_order if x != pair_feature_1]
    order_remove_2 = [x for x in current_order if x != pair_feature_2]

    vif_result_remove_1 = compute_vif_table(csvframe, order_remove_1)
    vif_result_remove_2 = compute_vif_table(csvframe, order_remove_2)

    max_vif_remove_1 = vif_result_remove_1["VIF"].max()
    max_vif_remove_2 = vif_result_remove_2["VIF"].max()

    vif_result_remove_1.to_csv(
        os.path.join(n_folder, f"{pair_prefix}_vif_remove_{pair_feature_1}.csv"),
        index=False
    )
    vif_result_remove_2.to_csv(
        os.path.join(n_folder, f"{pair_prefix}_vif_remove_{pair_feature_2}.csv"),
        index=False
    )

    print("============================================================")
    print("Evaluating pair:", pair_feature_1, pair_feature_2)
    print("Pair rank:", pair_rank + 1)
    print("Pair rp:", pair_rp)
    print("Pair rs:", pair_rs)
    print("============================================================")

    result_remove_1 = run_svr_for_feature_set(
        csvframe, target, csvframe2, target2,
        order_remove_1,
        n_folder,
        pair_prefix + "_remove_" + str(pair_feature_1)
    )

    result_remove_2 = run_svr_for_feature_set(
        csvframe, target, csvframe2, target2,
        order_remove_2,
        n_folder,
        pair_prefix + "_remove_" + str(pair_feature_2)
    )

    rmse_if_remove_1 = result_remove_1["rmse_tst"]
    rmse_if_remove_2 = result_remove_2["rmse_tst"]

    mean_rmse_if_remove_1 = np.mean(rmse_if_remove_1)
    std_rmse_if_remove_1 = np.nanstd(rmse_if_remove_1)

    mean_rmse_if_remove_2 = np.mean(rmse_if_remove_2)
    std_rmse_if_remove_2 = np.nanstd(rmse_if_remove_2)

    # two-sided paired t-test for deciding which feature to remove
    t_stat_pair, p_value_pair = ttest_rel(rmse_if_remove_1, rmse_if_remove_2)

    if p_value_pair < ALPHA:
        if mean_rmse_if_remove_1 < mean_rmse_if_remove_2:
            final_removed_feature = pair_feature_1
            final_retained_feature = pair_feature_2
            final_rmse = rmse_if_remove_1
            final_mean_rmse = mean_rmse_if_remove_1
            final_std_rmse = std_rmse_if_remove_1
            final_feature_order = order_remove_1
            final_max_vif = max_vif_remove_1
            decision_reason = "Significant pairwise difference: remove the feature whose exclusion gives lower RMSE."
        else:
            final_removed_feature = pair_feature_2
            final_retained_feature = pair_feature_1
            final_rmse = rmse_if_remove_2
            final_mean_rmse = mean_rmse_if_remove_2
            final_std_rmse = std_rmse_if_remove_2
            final_feature_order = order_remove_2
            final_max_vif = max_vif_remove_2
            decision_reason = "Significant pairwise difference: remove the feature whose exclusion gives lower RMSE."
    else:
        if feat1_target_score < feat2_target_score:
            final_removed_feature = pair_feature_1
            final_retained_feature = pair_feature_2
            final_rmse = rmse_if_remove_1
            final_mean_rmse = mean_rmse_if_remove_1
            final_std_rmse = std_rmse_if_remove_1
            final_feature_order = order_remove_1
            final_max_vif = max_vif_remove_1
            decision_reason = "No significant pairwise difference: remove the feature with smaller target correlation score."
        else:
            final_removed_feature = pair_feature_2
            final_retained_feature = pair_feature_1
            final_rmse = rmse_if_remove_2
            final_mean_rmse = mean_rmse_if_remove_2
            final_std_rmse = std_rmse_if_remove_2
            final_feature_order = order_remove_2
            final_max_vif = max_vif_remove_2
            decision_reason = "No significant pairwise difference: remove the feature with smaller target correlation score."

    # one-sided paired t-test for stopping rule
    pristine_cmp = compare_to_pristine_one_sided(pristine_rmse, final_rmse)

    summary_df = pd.DataFrame([{
        "pair_rank": pair_rank + 1,
        "feature_1": pair_feature_1,
        "feature_2": pair_feature_2,
        "pair_rp": pair_rp,
        "pair_rs": pair_rs,
        "pair_score_abs_avg": pair_score_from_rp_rs(pair_rp, pair_rs),

        "feature_1_target_rp": feat1_target_rp,
        "feature_1_target_rs": feat1_target_rs,
        "feature_1_target_score": feat1_target_score,
        "max_VIF_if_remove_feature_1": max_vif_remove_1,
        "rmse_test_if_remove_feature_1_mean": mean_rmse_if_remove_1,
        "rmse_test_if_remove_feature_1_std": std_rmse_if_remove_1,

        "feature_2_target_rp": feat2_target_rp,
        "feature_2_target_rs": feat2_target_rs,
        "feature_2_target_score": feat2_target_score,
        "max_VIF_if_remove_feature_2": max_vif_remove_2,
        "rmse_test_if_remove_feature_2_mean": mean_rmse_if_remove_2,
        "rmse_test_if_remove_feature_2_std": std_rmse_if_remove_2,

        "pair_t_stat_two_sided": t_stat_pair,
        "pair_p_value_two_sided": p_value_pair,

        "final_removed_feature": final_removed_feature,
        "final_retained_feature": final_retained_feature,
        "decision_reason": decision_reason,

        "pristine_t_stat_one_sided": pristine_cmp["t_stat"],
        "pristine_p_value_one_sided": pristine_cmp["p_value"],
        "pristine_significant_increase": pristine_cmp["significant_increase"],
        "candidate_mean_rmse": pristine_cmp["candidate_mean"],
        "pristine_mean_rmse": pristine_cmp["pristine_mean"],
        "accepted_for_next_step": (not pristine_cmp["significant_increase"])
    }])

    summary_df.to_csv(
        os.path.join(n_folder, f"{pair_prefix}_summary.csv"),
        index=False
    )

    print("FINAL SUMMARY FOR THIS PAIR")
    print("Compared pair :", pair_feature_1, "vs", pair_feature_2)
    print("[1] Pair correlation")
    print("rp =", pair_rp)
    print("rs =", pair_rs)
    print("[2] Target correlations")
    print("Feature", pair_feature_1, ": rp =", feat1_target_rp, ", rs =", feat1_target_rs, ", score =", feat1_target_score)
    print("Feature", pair_feature_2, ": rp =", feat2_target_rp, ", rs =", feat2_target_rs, ", score =", feat2_target_score)
    print("[3] Max VIF in each reduced model")
    print("If feature", pair_feature_1, "is removed: max VIF =", max_vif_remove_1)
    print("If feature", pair_feature_2, "is removed: max VIF =", max_vif_remove_2)
    print("[4] Pairwise RMSE comparison")
    print("If feature", pair_feature_1, "is removed: RMSE =", mean_rmse_if_remove_1, "+/-", std_rmse_if_remove_1)
    print("If feature", pair_feature_2, "is removed: RMSE =", mean_rmse_if_remove_2, "+/-", std_rmse_if_remove_2)
    print("[5] Pairwise paired t-test (two-sided)")
    print("t-statistic =", t_stat_pair)
    print("p-value =", p_value_pair)
    print("[6] Decision within pair")
    print("Removed feature  :", final_removed_feature)
    print("Retained feature :", final_retained_feature)
    print("Reason           :", decision_reason)
    print("[7] Comparison to pristine 18-feature model (one-sided)")
    print("candidate mean RMSE =", pristine_cmp["candidate_mean"])
    print("pristine mean RMSE  =", pristine_cmp["pristine_mean"])
    print("t-statistic         =", pristine_cmp["t_stat"])
    print("p-value             =", pristine_cmp["p_value"])
    print("significant increase? =", pristine_cmp["significant_increase"])
    print("accepted for next step? =", (not pristine_cmp["significant_increase"]))
    print("============================================================")

    return {
        "pair_rank": pair_rank + 1,
        "pair_feature_1": pair_feature_1,
        "pair_feature_2": pair_feature_2,
        "pair_rp": pair_rp,
        "pair_rs": pair_rs,
        "pair_score_abs_avg": pair_score_from_rp_rs(pair_rp, pair_rs),
        "pair_t_stat": t_stat_pair,
        "pair_p_value": p_value_pair,
        "feature_1_target_rp": feat1_target_rp,
        "feature_1_target_rs": feat1_target_rs,
        "feature_1_target_score": feat1_target_score,
        "feature_2_target_rp": feat2_target_rp,
        "feature_2_target_rs": feat2_target_rs,
        "feature_2_target_score": feat2_target_score,
        "max_vif_remove_1": max_vif_remove_1,
        "max_vif_remove_2": max_vif_remove_2,
        "mean_rmse_remove_1": mean_rmse_if_remove_1,
        "std_rmse_remove_1": std_rmse_if_remove_1,
        "mean_rmse_remove_2": mean_rmse_if_remove_2,
        "std_rmse_remove_2": std_rmse_if_remove_2,
        "final_removed_feature": final_removed_feature,
        "final_retained_feature": final_retained_feature,
        "decision_reason": decision_reason,
        "final_feature_order": final_feature_order,
        "final_rmse": final_rmse.copy(),
        "final_mean_rmse": final_mean_rmse,
        "final_std_rmse": final_std_rmse,
        "final_max_vif": final_max_vif,
        "pristine_t_stat": pristine_cmp["t_stat"],
        "pristine_p_value": pristine_cmp["p_value"],
        "pristine_significant_increase": pristine_cmp["significant_increase"],
        "accepted_for_next_step": (not pristine_cmp["significant_increase"])
    }


# ============================================================
# Main iterative procedure
# ============================================================

def main():
    csvframe = pd.read_csv(TRAIN_CSV)
    target = csvframe.iloc[:, 0]

    csvframe2 = pd.read_csv(OOD_CSV)
    target2 = csvframe2.iloc[:, 2]

    pristine_rmse = extract_rmse_test_trials(PRISTINE_PERF_CSV)

    current_order = START_ORDER.copy()

    history_rows = []
    stop_reason = None
    stop_result = None
    iteration = 0

    while True:
        iteration += 1
        n_features = len(current_order)
        n_folder = f"{n_features}_features"
        os.makedirs(n_folder, exist_ok=True)

        print("\n\n############################################################")
        print("Iteration:", iteration)
        print("Current number of features:", n_features)
        print("Current feature order:", current_order)
        print("############################################################")

        if n_features <= 2:
            stop_reason = "Stopped because the current feature set has 2 or fewer features."
            break

        pair_ranking_csv = os.path.join(n_folder, f"pair_ranking_{n_features}.csv")
        pair_df = compute_pair_ranking(csvframe, current_order, out_csv=pair_ranking_csv)

        if len(pair_df) == 0:
            stop_reason = "Stopped because no feature pairs are available."
            break

        top_k = min(2, len(pair_df))
        accepted = False

        for pair_rank in range(top_k):
            pair_feature_1 = int(pair_df.iloc[pair_rank]["feature_1"])
            pair_feature_2 = int(pair_df.iloc[pair_rank]["feature_2"])

            result = evaluate_pair(
                csvframe,
                target,
                csvframe2,
                target2,
                pristine_rmse,
                current_order,
                pair_feature_1,
                pair_feature_2,
                pair_rank,
                n_folder
            )

            history_rows.append({
                "iteration": iteration,
                "n_features_before": n_features,
                "current_feature_order_before": str(current_order),
                "pair_rank": result["pair_rank"],
                "pair_feature_1": result["pair_feature_1"],
                "pair_feature_2": result["pair_feature_2"],
                "pair_score_abs_avg": result["pair_score_abs_avg"],
                "pair_t_stat_two_sided": result["pair_t_stat"],
                "pair_p_value_two_sided": result["pair_p_value"],
                "final_removed_feature": result["final_removed_feature"],
                "final_retained_feature": result["final_retained_feature"],
                "candidate_feature_order_after_removal": str(result["final_feature_order"]),
                "final_mean_rmse": result["final_mean_rmse"],
                "final_std_rmse": result["final_std_rmse"],
                "pristine_t_stat_one_sided": result["pristine_t_stat"],
                "pristine_p_value_one_sided": result["pristine_p_value"],
                "pristine_significant_increase": result["pristine_significant_increase"],
                "accepted_for_next_step": result["accepted_for_next_step"]
            })

            pd.DataFrame(history_rows).to_csv("feature_elimination_history.csv", index=False)

            if result["pristine_significant_increase"]:
                stop_reason = (
                    "Stopped because the reduced model showed a significant RMSE increase "
                    "relative to the pristine 18-feature model."
                )
                stop_result = result

                # DO NOT rollback. Current feature set is the final model.
                print("Stopping: reduced model is significantly worse than pristine.")
                print("Final model remains the current feature set:", current_order)
                accepted = False
                break

            if result["accepted_for_next_step"]:
                current_order = result["final_feature_order"]
                accepted = True

                print("Accepted removal of feature", result["final_removed_feature"])
                print("Proceeding to next iteration with", len(current_order), "features.")
                break

        if stop_reason is not None:
            break

        if not accepted:
            stop_reason = "Stopped because no candidate pair was accepted."
            break

    print("\n\n============================================================")
    print("GLOBAL FINAL SUMMARY")
    print("============================================================")
    print("Stop reason        :", stop_reason)
    print("Final feature set  :", current_order)
    print("Final n_features   :", len(current_order))
    print("History CSV        : feature_elimination_history.csv")
    print("============================================================")


if __name__ == "__main__":
    main()
