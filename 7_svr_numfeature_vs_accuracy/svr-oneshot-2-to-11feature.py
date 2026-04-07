import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Load the CSV file


start_num = 2
order =   [1, 17, 10, 5, 14, 6, 9, 4, 12, 15, 16]


order2 = [x+2 for x in order]

filename = "../GWgap_predictor_data_PRB_93_115104_2016.csv" 
csvframe = pd.read_csv(filename) #,header=None)
target = csvframe.iloc[:, 0]     ### column of target variables

### for prediction
filename2 = "../GWgap_OOD_40.csv"
csvframe2 = pd.read_csv(filename2)
target2 = csvframe2.iloc[:, 2]


pred_result_collect = []

for s in range(start_num,len(order)+1):
   variables = csvframe.iloc[:, order[0:s]]     ### columns for features, check column number from csv file
   variables2 = csvframe2.iloc[:, order2[0:s]]

   print (str(s), 'features --------')
   print ('variables : ',variables)
   # Define parameters
   cy = 20    ### Number of random data selections (training + validation) : test

   rmse_tst = np.zeros(cy)
   mae_tst = np.zeros(cy)
   rmse_tr = np.zeros(cy)
   cv_tr = np.zeros(cy)
   mae_tr = np.zeros(cy)
   cost = np.zeros(cy)
   gamma = np.zeros(cy)
   nsv = np.zeros(cy)
   eps = np.zeros(cy)

   R_sq_tr = np.zeros(cy)
   R_sq = np.zeros(cy)

   rmse_pred = np.zeros(cy)
   mae_pred = np.zeros(cy)
   R_sq_pred = np.zeros(cy)
   pred_result_collect = [] ; pred_result_collect2 = []
   pred_result_collect.append(target2) ; pred_result_collect2.append(target2)

   train_idx_list = [] ; test_idx_list = [] ; # for printing index

   # Hyperparameter ranges
   gamma_range = 10.0**np.arange(-6, 6)
   cost_range = 10.0**np.arange(-5, 5)
   epsilon_range = [0.001, 0.005, 0.01]


   print ('gamma_range : ',gamma_range)
   print ('cost_range : ',cost_range)
   print ('epsilon_range : ',epsilon_range)

   for i in range(cy):

     print ('cycle # : ',i,'-----------------')
     # Split the data into training and testing sets
#     idx = np.arange(len(variables))
#     tr_x, tst_x, tr_y, tst_y = train_test_split(variables, target, test_size=0.25, random_state=i)
#     tr_x, tst_x, tr_y, tst_y = train_test_split(variables, target, idx,test_size=0.25, random_state=i)

     idx = np.arange(len(variables))
     tr_idx, tst_idx = train_test_split(idx, test_size=0.25, random_state=i)
     tr_x = variables.iloc[tr_idx]
     tst_x = variables.iloc[tst_idx]
     tr_y = target.iloc[tr_idx]
     tst_y = target.iloc[tst_idx]

     train_idx_list.append(np.array(tr_idx,dtype=int)) ; test_idx_list.append(np.array(tst_idx,dtype=int)) ### for print index

     pred_y = target2 ; pred_x = variables2   ## for prediction

     # Standardization
     scaler = StandardScaler()
     tr_x_scaled = scaler.fit_transform(tr_x)
     tst_x_scaled = scaler.transform(tst_x)

     pred_x_scaled = scaler.transform(pred_x)   ## for prediction

     # Perform grid search for SVR hyperparameters
     param_grid = {
        'gamma': gamma_range,
        'C': cost_range,
        'epsilon': epsilon_range
     }
     grid_search = GridSearchCV(svm.SVR(), param_grid, cv=10, scoring='neg_mean_squared_error',n_jobs=-1)
     grid_search.fit(tr_x_scaled, tr_y)

     # Get the best hyperparameters
     best_params = grid_search.best_params_
     gamma[i] = best_params['gamma']
     eps[i] = best_params['epsilon']
     cost[i] = best_params['C']


     # Train SVR model with best hyperparameters
     svr = svm.SVR(kernel='rbf', gamma=gamma[i], C=cost[i], epsilon=eps[i])
     svr.fit(tr_x_scaled, tr_y)


     print ('bestparam (gamma, cost, eps) : ',best_params['gamma'], best_params['C'], best_params['epsilon'])


     # Make predictions on train and test sets
     tr_pred = svr.predict(tr_x_scaled)
     tst_pred = svr.predict(tst_x_scaled)

     pred_pred = svr.predict(pred_x_scaled)   ## for prediction



     # Calculate performance metrics
     rmse_tr[i] = np.sqrt(mean_squared_error(tr_y, tr_pred))
     rmse_tst[i] = np.sqrt(mean_squared_error(tst_y, tst_pred))
     mae_tr[i] = mean_absolute_error(tr_y, tr_pred)
     mae_tst[i] = mean_absolute_error(tst_y, tst_pred)


     corr_matrix_tr = np.corrcoef(tr_y, tr_pred)
     corr_tr = corr_matrix_tr[0,1]
     R_sq_tr[i] = corr_tr**2

     corr_matrix = np.corrcoef(tst_y, tst_pred)
     corr = corr_matrix[0,1]
     R_sq[i] = corr**2

     # prediction

     rmse_pred[i] = np.sqrt(mean_squared_error(pred_y, pred_pred))
     mae_pred[i] = mean_absolute_error(pred_y, pred_pred)
     R_sq_pred[i] = (np.corrcoef(pred_y, pred_pred)[0,1])**2


     pred_result_collect.append(pred_pred - pred_y)
     pred_result_collect2.append(pred_pred)


     tr_pd = pd.DataFrame({'tr_y':tr_y,'tr_pred':tr_pred}) #,columns=None)
     tst_pd = pd.DataFrame({'tst_y':tst_y,'tst_pred':tst_pred}) #,columns=None)

     tr_pd.to_csv('svr_'+str(s)+'_features_tr_'+str(i)+'.csv',index=False) #,header=False)
     tst_pd.to_csv('svr_'+str(s)+'_features_tst_'+str(i)+'.csv',index=False) #,header=False)


#     pd.DataFrame({"train_idx": tr_idx}).to_csv('svr_'+str(s)+'_train_idx_'+str(i)+'.csv',index=False) 
#     pd.DataFrame({"test_idx": tst_idx}).to_csv('svr_'+str(s)+'_test_idx_'+str(i)+'.csv',index=False) 

# ======

   train_idx_df = pd.DataFrame(
    {f"cycle_{i}": train_idx_list[i] for i in range(cy)}
   )
   test_idx_df = pd.DataFrame(
    {f"cycle_{i}": test_idx_list[i] for i in range(cy)}
   )

   train_idx_df.to_csv(f"svr_{s}_train_indices_cycles.csv", index=False)
   test_idx_df.to_csv(f"svr_{s}_test_indices_cycles.csv", index=False)

   hyper_df = pd.DataFrame({
        "cycle": np.arange(cy, dtype=int),
        "gamma": gamma,
        "C": cost,
        "epsilon": eps
   })
   hyper_df.to_csv(f"svr_{s}_hyperparams.csv", index=False)

### =====

   print ('------------------------------------------')
   print (str(s)+'-features result---')
   print ('------------------------------------------')

   # Print or use the results as needed
   print(str(s),"features, Gamma:", gamma)
   print(str(s),"features, Epsilon:", eps)
   print(str(s),"features, Cost:", cost)
   print(str(s),"features, RMSE Train:", rmse_tr, np.mean(rmse_tr),u"\u00B1",np.nanstd(rmse_tr))
   print(str(s),"features, RMSE Test:", rmse_tst, np.mean(rmse_tst),u"\u00B1",np.nanstd(rmse_tst))
   print(str(s),"features, MAE Train:", mae_tr, np.mean(mae_tr),u"\u00B1",np.nanstd(mae_tr))
   print(str(s),"features, MAE Test:", mae_tst, np.mean(mae_tst),u"\u00B1",np.nanstd(mae_tst))
   print(str(s),"features, R^2 Train:", R_sq_tr, np.nanmean(R_sq_tr),u"\u00B1",np.nanstd(R_sq_tr))
   print(str(s),"features, R^2 Test:", R_sq, np.nanmean(R_sq), u"\u00B1",np.nanstd(R_sq))

   print(str(s),"features, RMSE Prediction set:", rmse_pred, np.mean(rmse_pred),u"\u00B1",np.nanstd(rmse_pred))
   print(str(s),"features, MAE Prediction set:", mae_pred, np.mean(mae_pred),u"\u00B1",np.nanstd(mae_pred))
   print(str(s),"features, R^2 Prediction set:", R_sq_pred, np.nanmean(R_sq_pred),u"\u00B1",np.nanstd(R_sq_pred))
   pd.DataFrame(pred_result_collect,columns=None).to_csv('pred_'+str(s)+'_feature_errors.csv',index=False,header=False)
   pd.DataFrame(pred_result_collect2,columns=None).to_csv('pred_'+str(s)+'_feature_values.csv',index=False,header=False)


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
    for i, v in enumerate(values):
        row[f"trial_{i}"] = v
    rows.append(row)
    pd.DataFrame(rows).to_csv(str(s)+'_performance.csv',index=False)
