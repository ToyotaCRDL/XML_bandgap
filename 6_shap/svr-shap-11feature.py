import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import shap  # Import the SHAP library
import matplotlib.pyplot as plt

head_list = [r'$E_\mathrm{g}^\mathrm{PBE}$',r'$E_\mathrm{coh}$',r'$\overline{|n|}$',r'$\overline{Z}$',r'$\overline{r}$',r'$\overline{\chi}$',r'$\sigma(|n|)$',r'$\sigma(p)$',r'$\sigma(m)$',r'$\sigma(r)$',r'$\sigma(\chi)$']


# Load the CSV file
filename = "../GWgap_predictor_data_PRB_93_115104_2016.csv"
csvframe = pd.read_csv(filename)
target = csvframe.iloc[:, 0]
variables = csvframe.iloc[:, [1,4,5,6,9,10,12,14,15,16,17]]

variables.columns = head_list
#print (variables.columns)

# Define parameters
cy = 20
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

    tr_x_scaled = pd.DataFrame(tr_x_scaled)
    tst_x_scaled = pd.DataFrame(tst_x_scaled)

    tr_x_scaled.columns = head_list
    tst_x_scaled.columns = head_list

    # Perform grid search for SVR hyperparameters
    param_grid = {
        'gamma': gamma_range,
        'C': cost_range,
        'epsilon': epsilon_range
    }
    grid_search = GridSearchCV(svm.SVR(), param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs = -1)
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

    corr_matrix = np.corrcoef(tst_y, tst_pred)
    corr = corr_matrix[0,1]
    R_sq[i] = corr**2


    # Calculate performance metrics
    rmse_tr[i] = np.sqrt(mean_squared_error(tr_y, tr_pred))
    rmse_tst[i] = np.sqrt(mean_squared_error(tst_y, tst_pred))
    mae_tr[i] = mean_absolute_error(tr_y, tr_pred)
    mae_tst[i] = mean_absolute_error(tst_y, tst_pred)
    nsv[i] = np.sum(svr.n_support_)

    # Wrap the SVR model in a lambda function
    model_with_predict = lambda x: svr.predict(x)

    # Create a SHAP explainer
    explainer = shap.KernelExplainer(model_with_predict, tr_x_scaled,seed=i)
    shap_values = explainer.shap_values(tst_x_scaled, nsamples=100)

#    print ('shap_values',shap_values)

#    printout_shap = pd.DataFrame(np.concatenate((tst_pred.reshape(len(tst_pred),1),shap_values.values),axis=1))
    printout_shap = pd.DataFrame(np.concatenate((tst_pred.reshape(len(tst_pred),1),shap_values),axis=1))
    pd.DataFrame(printout_shap).to_csv('shap_values_'+str(i)+'.csv',index=None,header=None)
    pd.DataFrame(tst_x).to_csv('tst_x_'+str(i)+'.csv',index=None,header=None)


# Print or use the results as needed
print("Gamma:", gamma)
print("Epsilon:", eps)
print("Cost:", cost)
print("RMSE Train:", rmse_tr, np.mean(rmse_tr), '+-', np.std(rmse_tr))
print("RMSE Test:", rmse_tst, np.mean(rmse_tst), '+-', np.std(rmse_tr))
print("MAE Train:", mae_tr, np.mean(mae_tr),'+-', np.std(rmse_tr))
print("MAE Test:", mae_tst, np.mean(mae_tst),'+-', np.std(rmse_tr))
print("R^2 Test:", R_sq, np.nanmean(R_sq),'+-', np.std(rmse_tr))

