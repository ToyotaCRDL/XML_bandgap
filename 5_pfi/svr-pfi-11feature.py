import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Load the CSV file
filename = "../GWgap_predictor_data_PRB_93_115104_2016.csv"
csvframe = pd.read_csv(filename)
target = csvframe.iloc[:, 0]
#variables = csvframe.iloc[:, 1:]

variables = csvframe.iloc[:, [1,4,5,6,9,10,12,14,15,16,17]]

# Define parameters
cy = 20
#pfi_scores = np.zeros(variables.shape[1])
rmse_tst = np.zeros(cy)
mae_tst = np.zeros(cy)
rmse_tr = np.zeros(cy)
cv_tr = np.zeros(cy)
mae_tr = np.zeros(cy)
cost = np.zeros(cy)
gamma = np.zeros(cy)
nsv = np.zeros(cy)
eps = np.zeros(cy)
R_sq = np.zeros(cy)
pfi_scores = np.zeros([cy,variables.shape[1]])


# Load Boston housing dataset as an example

# Function to calculate permutation feature importance
def handcraft_PFI(model, X, y, baseline_score, n_permutations=1, random_state=None):
    np.random.seed(random_state)
#    baseline_score = metric(y, model.predict(X))
    importances = np.zeros(X.shape[1])

    for feature in range(X.shape[1]):
        permuted_scores = []
        original_column = X[:, feature].copy()

        for _ in range(n_permutations):
            # Permute the values of the feature
            X[:, feature] = np.random.permutation(original_column)
            permuted_score = np.sqrt(mean_squared_error(y, model.predict(X)))
            permuted_scores.append(permuted_score)

        # Calculate the importance as the decrease in performance
        importances[feature] = np.mean(permuted_scores) - baseline_score


        # Reset the feature column to its original values
        X[:, feature] = original_column

    return importances


# Hyperparameter ranges
gamma_range = 10.0**np.arange(-6, 6)
cost_range = 10.0**np.arange(-5, 5)
epsilon_range = [0.001, 0.005, 0.01]

for i in range(cy):
    # Split the data into training and testing sets
    tr_x, tst_x, tr_y, tst_y = train_test_split(variables, target, test_size=0.25, random_state=i)

    # Standardization
    scaler = StandardScaler()
    tr_x_scaled = scaler.fit_transform(tr_x)
    tst_x_scaled = scaler.transform(tst_x)

    # Perform grid search for SVR hyperparameters
    param_grid = {
        'gamma': gamma_range,
        'C': cost_range,
        'epsilon': epsilon_range
    }
    grid_search = GridSearchCV(svm.SVR(), param_grid, cv=10, scoring='neg_mean_squared_error',n_jobs = -1)
    grid_search.fit(tr_x_scaled, tr_y)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    gamma[i] = best_params['gamma']
    eps[i] = best_params['epsilon']
    cost[i] = best_params['C']

    # Train SVR model with best hyperparameters
    svr = svm.SVR(kernel='rbf', gamma=gamma[i], C=cost[i], epsilon=eps[i])
    svr.fit(tr_x_scaled, tr_y)

    # Make predictions on train and test sets
    tr_pred = svr.predict(tr_x_scaled)
    tst_pred = svr.predict(tst_x_scaled)

    # Calculate performance metrics
    rmse_tr[i] = np.sqrt(mean_squared_error(tr_y, tr_pred))
    rmse_tst[i] = np.sqrt(mean_squared_error(tst_y, tst_pred))
    mae_tr[i] = mean_absolute_error(tr_y, tr_pred)
    mae_tst[i] = mean_absolute_error(tst_y, tst_pred)

    corr_matrix = np.corrcoef(tst_y, tst_pred)
    corr = corr_matrix[0,1]
    R_sq[i] = corr**2

    # Calculate permutation feature importances
    feature_importance = handcraft_PFI(svr, tst_x_scaled, tst_y, rmse_tst[i], n_permutations=10, random_state=2*i)


    # Print or use the PFI feature importances
    print("Permutation Feature Importance (PFI) for Iteration", i)
    print("basescore of rmse_tst", rmse_tst[i]) 
    for j, imp in enumerate(feature_importance):
        pfi_scores[i,j] = imp

# Print or use the results as needed
print("Gamma:", gamma)
print("Epsilon:", eps)
print("Cost:", cost)
print("RMSE Train:", rmse_tr, np.mean(rmse_tr), '+-', np.std(rmse_tr))
print("RMSE Test:", rmse_tst, np.mean(rmse_tst), '+-', np.std(rmse_tst))
print("MAE Train:", mae_tr, np.mean(mae_tr), '+-', np.std(rmse_tr))
print("MAE Test:", mae_tst, np.mean(mae_tst), '+-', np.std(mae_tst))
print("R^2 Test:", R_sq, np.nanmean(R_sq), '+-', np.nanstd(R_sq))
print("PFI Scores:", pfi_scores)
print("PFI mean: ",np.mean(pfi_scores,axis=0))
pd.DataFrame(pfi_scores,columns=None).to_csv('svr-pfi-11feature.csv',index=False,header=False)
