# -*- coding: utf-8 -*-
"""
Point forecasting, electricity prices

@author: akylas.stratigakos@minesparis.psl.eu"""

import pickle
import os, sys
import gurobipy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
import scipy.sparse as sp
import time
import itertools
cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

# import from forecasting libraries

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from utility_functions import * 
from FDR_regressor import *
from QR_regressor import *

def retrain_model(X, Y, testX, target_col, fix_col, Gamma, base_loss = 'l2'):
    ''' Retrain model withtout missing features
        returns a list models and corresponding list of missing features'''
    # all combinations of missing features for up to Gamma features missing
    combinations = [list(item) for sublist in [list(itertools.combinations(range(len(target_col)), gamma)) for gamma in range(1,Gamma+1)] for item in sublist]
    # first instance is without missing features
    combinations.insert(0, [])
    models = []
    predictions = []
    for i,v in enumerate(combinations):
        
        # find columns not missing 
        temp_col = [col for col in target_col if col not in v]
        temp_X = X[:,temp_col+fix_col]
        temp_test_X = testX[:,temp_col+fix_col]
        
        # retrain model without missing features
        if base_loss == 'l2':
            lr = LinearRegression(fit_intercept = True)
            lr.fit(temp_X, Y)
        elif base_loss == 'l1':
            lr = QR_regressor()
            lr.fit(temp_X, Y)
            
        models.append(lr)
        predictions.append(lr.predict(temp_test_X).reshape(-1))
    
    predictions = np.array(predictions).T
    
    return models, predictions, combinations

def eval_predictions(pred, target, metric = 'mae'):
    if metric == 'mae':
        return np.mean(np.abs(pred-target))
    elif metric == 'rmse':
        return np.sqrt(np.square(pred-target).mean())
    elif metric == 'mape':
        return np.mean(np.abs(pred-target)/target)

def params():
    ''' Set up the experiment parameters'''

    params = {}
    params['scale'] = True
    params['train'] = True # If True, then train models, else tries to load previous runs
    params['parallel'] = False # If True, then trees are grown in parallel
    params['save'] = False # If True, then saves models and results
    params['impute'] = True # If True, apply mean imputation for missing features
    params['percentage'] = [.05, .10, .20, .50]  # percentage of corrupted datapoints
    params['iterations'] = 5 # per pair of (n_nodes,percentage)
    params['quantile'] = .5    
    return params

# IEEE plot parameters (not sure about mathfont)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
#%%
price_df = pd.read_csv(cd+'\\data\\prices_data.csv', index_col=0, parse_dates=True)

pred_col = ['Net Load Forecast', 'System Margin', 'DA Price_24', 'DA Price_144', 'DA Price_168', 
            'Day', 'Hour', 'Month']

config = params()

Predictors = price_df[pred_col]
Y = price_df['DA Price'].to_frame()

#Predictors['Month'] = np.sin(np.deg2rad(Predictors['Month']*360/12))
#Predictors['Day'] = np.sin(np.deg2rad(Predictors['Day']*360/7))
#Predictors['Hour'] = np.sin(np.deg2rad(Predictors['Hour']*360/7))

#Predictors['Month'] = Predictors['Month'].astype(str)
#Predictors['Day'] = Predictors['Day'].astype(str)
#Predictors['Hour'] = Predictors['Hour'].astype(str)

target_scaler = MinMaxScaler()
pred_scaler = MinMaxScaler()
#%%
start = '2017-01-01'
split = '2018-01-01'

if config['scale']:
    trainY = target_scaler.fit_transform(Y[start:split].values)
    testY = target_scaler.transform(Y[split:])
    Target = Y[split:]
    
    trainPred = pred_scaler.fit_transform(Predictors[start:split])
    testPred = pred_scaler.transform(Predictors[split:])
else:
    trainY = Y[start:split].values
    testY = Y[split:].values
    Target = Y[split:]
    
    trainPred = Predictors[start:split].values
    testPred = Predictors[split:].values

#trainPred = np.column_stack((trainPred, np.ones(len(trainPred))))
#testPred = np.column_stack((testPred, np.ones(len(testPred))))

#%%%% Linear models: linear regression, ridge, lasso 

# Hyperparameter tuning with by cross-validation
param_grid = {"alpha": [10**pow for pow in range(-5,6)]}

ridge = GridSearchCV(Ridge(fit_intercept = True), param_grid)
ridge.fit(trainPred, trainY)

lasso = GridSearchCV(Lasso(fit_intercept = True), param_grid)
lasso.fit(trainPred, trainY)

lr = LinearRegression(fit_intercept = True)
lr.fit(trainPred, trainY)

if config['scale']:
    lr_pred = target_scaler.inverse_transform(lr.predict(testPred).reshape(-1,1))
    lasso_pred = target_scaler.inverse_transform(lasso.predict(testPred).reshape(-1,1))
    ridge_pred = target_scaler.inverse_transform(ridge.predict(testPred).reshape(-1,1))
else:
    lr_pred= lr.predict(testPred).reshape(-1,1)
    lasso_pred = lasso.predict(testPred).reshape(-1,1)
    ridge_pred = ridge.predict(testPred).reshape(-1,1)
    
print('LR: ', eval_point_pred(lr_pred.reshape(-1,1), Target.values, digits=2))
print('Lasso: ', eval_point_pred(lasso_pred.reshape(-1,1), Target.values, digits=2))
print('Ridge: ', eval_point_pred(ridge_pred.reshape(-1,1), Target.values, digits=2))

#%%%% Least absolute deviations (LAD)

lad_model = QR_regressor(quantile = 0.5)    
lad_model.fit(trainPred, trainY)
            
print(eval_point_pred(target_scaler.inverse_transform(lad_model.predict(testPred).reshape(-1,1)), Target.values))

#%%%%% Random Forest model (scaling is not required)

rf = ExtraTreesRegressor(n_estimators = 300)
rf.fit(trainPred, Y[start:split])
print(eval_point_pred(rf.predict(testPred).reshape(-1,1), Target.values))

#%%%%% Feature-deletion robust regression: train one model per value of \Gamma

target_pred = ['Net Load Forecast', 'System Margin', 'DA Price_24', 'DA Price_144', 'DA Price_168']
fixed_pred = ['Day', 'Hour', 'Month']
target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
fix_col = [np.where(Predictors.columns == c)[0][0] for c in fixed_pred]

FDR_models = []
K_parameter = [0, 1, 2, 3, 4, 5]

#%%
if config['train']:
    FDR_models = []
    for K in [1]:
        print('K = ', K)
        fdr = FDR_regressor(K = K, quant = 0.5)
        fdr.fit(trainPred, trainY, target_col, fix_col, verbose=-1, solution = 'affine')

        fdr_pred = target_scaler.inverse_transform(fdr.predict(testPred).reshape(-1,1))
        print('FDR: ', eval_point_pred(fdr_pred, Target.values, digits=2))
        
        FDR_models.append(fdr)
    
    if config['save']:
        with open(cd + '\\trained-models\\' +'det-fdr-prices.pickle', 'wb') as handle:
            pickle.dump(FDR_models, handle)
else:
    with open(cd + '\\trained-models\\' +'det-fdr-prices.pickle', 'rb') as handle:    
            FDR_models = pickle.load(handle)

#%%%%%%%%% Retrain without missing features (Tawn, Browell)

retrain_models, retrain_pred, retrain_comb = retrain_model(trainPred, trainY, testPred, 
                                                           target_col, fix_col, max(K_parameter), base_loss='l1')

#%%%%%%%%% Validation: benchmark w feature deletion over whole test set
percentage_obs = [1]
n_feat = len(target_col)
n_test_obs = len(testY)
num_del_feat = np.arange(n_feat+1)
iterations = 5
error_metric = 'mae'
imputation_values = trainPred.mean(0)

models = ['BASE', 'BASE-RIDGE', 'BASE-LASSO', 'LAD',  'RF', 'BASE-retrain'] + ['FDRR_'+str(k) for k in K_parameter]
if config['impute']:
    labels = ['LS$^{imp}$', 'LS$_{\ell_2}^{imp}$', 'LS$_{\ell_1}^{imp}$', 'LAD', 'RF$^{imp}$', 'RETRAIN'] + ['FDRR$^'+str(k)+'$' for k in K_parameter]
    #labels = ['BASE$^{imp}$', 'BASE$_{\ell_2}^{imp}$', 'BASE$_{\ell_1}^{imp}$', 'RF$^{imp}$', 'RETRAIN'] + ['FDRR$^'+str(k)+'$' for k in K_parameter]
else:
    labels = ['BASE', 'BASE$_{\ell_2}$', 'BASE$_{\ell_1}$', 'LAD', 'RF', 'RETRAIN'] + ['FDRR$^'+str(k)+'$' for k in K_parameter]
    
mae_df = pd.DataFrame(data = [], columns = models+['percentage', 'feat_del', 'iteration'])

# supress warning
pd.options.mode.chained_assignment = None
run_counter = 0

for iter_ in range(iterations):
    for m_feat in num_del_feat:
        for perc in percentage_obs:
            # initialize row in dataframe
            mae_df = mae_df.append({'percentage':perc}, ignore_index=True)                
            mae_df['feat_del'][run_counter] = m_feat
            mae_df['iteration'][run_counter] = iter_
                        
            # indices with missing features
            missing_ind = np.random.choice(np.arange(0, n_test_obs), size = int(perc*n_test_obs), replace = False)
            # missing features for each indice            
            missing_feat = [np.random.choice(np.arange(0, len(target_col)), size = m_feat, replace = False) for i in range(perc*n_test_obs)]
            missing_feat = np.array(missing_feat).reshape(int(perc*n_test_obs), m_feat)
            
            # create missing data with 0s or mean imputation
            miss_X = testPred.copy()
            imp_X = testPred.copy()
            for ind, j in zip(missing_ind, missing_feat):
                miss_X[ind, j] = 0 
                imp_X[ind, j] = imputation_values[j]
            if config['impute'] != True:
                imp_X = miss_X.copy()
            print("HERE")
            print(imp_X.shape)
            print(Target.shape)
            # Retrain model
            f_retrain_pred = retrain_pred[:,0:1].copy()
            if m_feat > 0:
                for j, ind in enumerate(missing_ind):
                    temp_feat = np.sort(missing_feat[j])
                    temp_feat = list(temp_feat)                    
                    # find position in combinations list
                    j_ind = retrain_comb.index(temp_feat)
                    f_retrain_pred[ind] = retrain_pred[ind, j_ind]                

            if config['scale']:
                f_retrain_pred = target_scaler.inverse_transform(f_retrain_pred)            
            retrain_mae = eval_predictions(f_retrain_pred.reshape(-1,1), Target.values, metric=error_metric)

            # Base vanilla model
            base_pred = lr.predict(imp_X).reshape(-1,1)
            if config['scale']:
                base_pred = target_scaler.inverse_transform(base_pred)            
            base_mae = eval_predictions(base_pred.reshape(-1,1), Target.values, metric=error_metric)
            
            # LASSO
            lasso_pred = lasso.predict(imp_X).reshape(-1,1)
            if config['scale']:
                lasso_pred = target_scaler.inverse_transform(lasso_pred)    
            lasso_mae = eval_predictions(lasso_pred, Target.values, metric= error_metric)

            # RIDGE
            l2_pred = ridge.predict(imp_X).reshape(-1,1)
            if config['scale']:
                l2_pred = target_scaler.inverse_transform(l2_pred)    
            l2_mae = eval_predictions(l2_pred, Target.values, metric= error_metric)

            # LAD
            lad_pred = lad_model.predict(imp_X).reshape(-1,1)
            if config['scale']:
                lad_pred = target_scaler.inverse_transform(lad_pred)    
            lad_mae = eval_predictions(lad_pred, Target.values, metric= error_metric)

            # Random Forest
            rf_pred = rf.predict(imp_X).reshape(-1,1)
            rf_mae = eval_predictions(rf_pred, Target.values, metric= error_metric)

            mae_df['BASE'][run_counter] = base_mae
            mae_df['BASE-retrain'][run_counter] = retrain_mae
            mae_df['BASE-LASSO'][run_counter] = lasso_mae
            mae_df['BASE-RIDGE'][run_counter] = l2_mae
            mae_df['LAD'][run_counter] = lad_mae
            mae_df['RF'][run_counter] = rf_mae

            # FDR predictions
            for i, k in enumerate(K_parameter):
                fdr_pred = FDR_models[i].predict(miss_X).reshape(-1,1)
                # Robust
                if config['scale']:
                    fdr_pred = target_scaler.inverse_transform(fdr_pred)
                                    
                fdr_mae = eval_predictions(fdr_pred, Target.values, metric='mae')
                mae_df['FDRR_'+str(k)][run_counter] = fdr_mae

            run_counter += 1


std_bar = mae_df.groupby(['feat_del'])[models].std()
fig,ax = plt.subplots(constrained_layout = True)
mae_df.groupby(['feat_del'])[models].mean().plot(kind='bar', ax=ax, rot = 0, 
                                                 yerr=std_bar, legend=False)
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_xlabel('# of deleted features')

#%% Plot: point forecast accuracy versus deleted features

color_list = ['black', 'black', 'gray', 'tab:cyan','tab:green',
         'tab:blue', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive']

marker = ['o', '2', '^','h', 'd', '1', '+', 's', 'v', '*', '^', 'p']
base_colors = plt.cm.tab20c( list(np.arange(3)))
fdr_colors = plt.cm.tab20c([8,9,10, 12, 13, 14])
colors = list(base_colors) + ['tab:brown'] + ['tab:orange'] + ['black'] + list(fdr_colors) 

models_to_plot = ['BASE-retrain'] + ['FDRR_'+str(k) for k in K_parameter]
labels = ['LS', 'LS-${\ell_2}$', 'LS-${\ell_1}$', 'LAD','RF', 'RETRAIN'] + ['FDRR$('+str(k)+')$' for k in K_parameter]

fig,ax = plt.subplots(constrained_layout = True)
for i, m in enumerate(models):
    if m not in models_to_plot: continue
    if m=='BASE-retrain':
        style = 'dashed'
    else: style='solid'
    
    x_val = mae_df['feat_del'].unique()
    values = mae_df.groupby(['feat_del'])[m].mean()
    std = mae_df.groupby(['feat_del'])[m].std()
    plt.plot(x_val, values, marker = marker[i], color = colors[i],
             markersize = 4, label = labels[i], linewidth = 1, linestyle = style)
plt.xticks(x_val, x_val.astype(int))
plt.xlabel('# of deleted features')
plt.ylabel('MAE (EUR/MWh)')
plt.legend(loc='upper left', ncol=2, fontsize = 6)
plt.ylim(0,60)
if config['save']: 
    if config['impute']: 
        plt.savefig(cd+'\\figures\\det-imp-price.pdf')
    else: 
        plt.savefig(cd+'\\figures\\det-price.pdf')
plt.show()
#%% Accuracy graph v2: merge all FDRRs in a single line
temp_df = mae_df.copy()
temp_df['FDRR'] = temp_df[['FDRR_'+str(k) for k in K_parameter]].min(axis=1)

fig,ax = plt.subplots(constrained_layout = True)

plot_models = ['BASE', 'BASE-RIDGE', 'BASE-LASSO', 'LAD', 'RF', 'BASE-retrain', 'FDRR']
if config['impute']:
    plot_labels = ['LS$^{imp}$', 'LS$_{\ell_2}^{imp}$', 'LS$_{\ell_1}^{imp}$', 'RF$^{imp}$', 'RETRAIN', 'FDRR']
    #plot_labels = ['BASE$^{imp}$', 'BASE$_{\ell_2}^{imp}$', 'BASE$_{\ell_1}^{imp}$', 'RF$^{imp}$', 'RETRAIN', 'FDRR']
else:
    plot_labels = ['BASE', 'BASE$_{\ell_2}$', 'BASE$_{\ell_1}$', 'RF', 'RETRAIN', 'FDRR']
    
plot_labels = ['LS', 'LS-${\ell_2}$', 'LS-${\ell_1}$', 'LAD', 'RF', 'RETRAIN', 'FDRR']

for i, m in enumerate(plot_models):
    if m=='BASE-retrain':
        style = 'dashed'
    else: style='solid'

    x_val = temp_df['feat_del'].unique()
    values = temp_df.groupby(['feat_del'])[m].mean()
    std = temp_df.groupby(['feat_del'])[m].std()
    # main plot
    plt.plot(x_val, values, marker = marker[i], color = colors[i], label = plot_labels[i], linestyle = style,
             markersize = 5, linewidth = 1)

plt.xticks(x_val, x_val.astype(int))
plt.xlabel('# of deleted features')
plt.ylabel('ΜΑΕ (EUR/MWh)')
plt.legend(loc='upper left', fontsize = 6, ncol=1)
if config['save']: 
    if config['impute']:
        plt.savefig(cd+'\\figures\\det-imp-prices-v2.pdf')
    else:
        plt.savefig(cd+'\\figures\\det-prices-v2.pdf')
plt.show()

#%%
# save results
if config['save']:
    output = mae_df.copy()
    if config['impute']:
        filename = 'imp-prices'
    else: 
        filename = 'prices'
        
    output.to_csv(cd+'\\results\\det-'+filename+'.csv')
    table_output = output.copy().groupby(['feat_del'])[models].mean().round(2).transpose()
    table_output.index = labels
    table_output.to_csv(cd+'\\results\\table-det-v2-'+filename+'.csv')

#%%
# Plot estimated coefficients

fig, ax = plt.subplots(constrained_layout = True, figsize = (3.5,2))

for i, m in enumerate(models[6:]):
    plt.plot(np.arange(trainPred.shape[1]+1),np.array(list(FDR_models[i].coef_)+list(FDR_models[i].bias_)),
             marker = marker[i+6], label= '$\Gamma='+str(K_parameter[i])+'$', color = colors[i+6])
    
plt.legend()
plt.xticks(range(Predictors.shape[1]+1), list(Predictors.columns)+['Bias'], rotation=45)
plt.vlines(n_feat-1+0.5, -0.1, 0.6, linestyle = 'dashed', color = 'black')
#plt.ylim(-0.1, .45)
plt.ylabel('Coeff. magnitude')
#plt.xlabel('Features')
plt.legend(fontsize=6, ncol=2)
if config['save']: plt.savefig(cd+'\\figures\\price_coef.pdf')
plt.show()

#%%% Learned coefficient/ bar plot

fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout = True, figsize = (3.5, 1.2))

### Bar plot
step = 0.275
Gamma = [0, 1, 2]
c = ['tab:orange', 'tab:red', 'tab:brown', 'tab:blue', 'tab:green']
hatches = ['--','||','\\']

for j, g in enumerate(Gamma):
    bars = plt.bar(np.arange(trainPred.shape[1]+1) + g*step, 
            np.array(list(FDR_models[g].coef_)+list(FDR_models[g].bias_)), width = step, label= '$\Gamma='+str(K_parameter[g])+'$', color = colors[6+g*2], 
            edgecolor = 'black',)

    for patch in bars.patches:
       patch.set_hatch(hatches[j])

plt.xticks(np.arange(Predictors.shape[1]+1)+step-(1-len(Gamma)%2)*0.5*step, ['F.' + str(t) for t in range(1,Predictors.shape[1]+2)] , rotation=0)
plt.vlines(n_feat+step+0.33+0.175, -0.05, 0.45, linestyle = 'dashed', color = 'black', linewidth = .5)
plt.legend(ncol=1, fontsize=6)
plt.ylabel('Coeff. magnitude')
plt.show()
if config['save']: fig.savefig(cd+'\\figures\\price_coef.pdf')
#%%
######## Learned coefficients plot: bar plot with table

gs_kw = dict(width_ratios=[1], height_ratios=[2,1])

fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout = True, figsize = (3.5, 1.5 + .5), 
                         gridspec_kw = gs_kw)


### Bar plot
plt.sca(axes[0])

step = 0.275

Gamma = [0, 1, 2]

c = ['tab:orange', 'tab:red', 'tab:brown', 'tab:blue', 'tab:green']
hatches = ['--','||','\\']

for j, g in enumerate(Gamma):
    bars = plt.bar(np.arange(trainPred.shape[1]+1) + g*step, 
            np.array(list(FDR_models[g].coef_)+list(FDR_models[g].bias_)), width = step, label= '$\Gamma='+str(K_parameter[g])+'$', color = colors[6+g*2], 
            edgecolor = 'black',)

    for patch in bars.patches:
       patch.set_hatch(hatches[j])

#plt.bar(np.arange(trainPred.shape[1]+1) + step, np.array(list(FDR_models[1].coef_)+list(FDR_models[1].bias_)), width = 0.3, label= '$\Gamma='+str(K_parameter[1])+'$', color = colors[8], 
#        edgecolor = 'black',)

plt.xticks(np.arange(Predictors.shape[1]+1)+step-(1-len(Gamma)%2)*0.5*step, ['F.' + str(t) for t in range(1,Predictors.shape[1]+2)] , rotation=0)
plt.vlines(n_feat+step+0.33+0.175, -0.05, 0.45, linestyle = 'dashed', color = 'black', linewidth = .5)
plt.legend(ncol=1, fontsize=6)
plt.ylabel('Coeff. magnitude')

### Table legend
text_data = np.array([['F-' +str(j) for j in range(1,10)]+ list(Predictors.columns)+['Bias']]).reshape(2,9).T
text_data = []
for i in range(1,10):    
    text_data.append('F.'+str(i)+': ' + [list(Predictors.columns)+['Bias']][0][i-1] )
text_data.append('')
text_data = np.array(text_data).reshape(2,5).T

the_table = axes[1].table(cellText=text_data, loc='best', fontsize=6, cellLoc='left', edges='open', 
                          colLabels = ['Features in $\mathcal{J}$', 'Features in $\mathcal{C}$'],colLoc='left')

axes[1].axis('off')
axes[1].grid(False)

plt.show()
if config['save']: fig.savefig(cd+'\\figures\\price_coef.pdf')

#%%%%%% Uniform feature deletion: delete the same feature for all observations in test set

percentage_obs = [1]
n_feat = len(target_col)
num_del_feat = np.arange(n_feat+1)
iterations = 5

mae_df = pd.DataFrame(data = np.zeros((len(target_col), len(models))), columns = models, index = Predictors.columns[target_col])

# Supress warning
pd.options.mode.chained_assignment = None
run_counter = 0
n_test_obs = len(testY)

for run_counter, ind in enumerate(target_col):
    feat_name = Predictors.columns[ind]

    miss_X = testPred.copy()
    miss_X[:,ind] = 0

    imp_X = testPred.copy()
    imp_X[:,ind] = trainPred[:,ind].mean()
    if config['impute'] != True:
        imp_X = miss_X.copy()
        
    # Retrain model
    j_ind = retrain_comb.index(ind)
    f_retrain_pred = retrain_pred[:, j_ind].reshape(-1,1)
    if config['scale']:
        f_retrain_pred = target_scaler.inverse_transform(f_retrain_pred)
    f_retrain_mae = eval_predictions(f_retrain_pred.reshape(-1,1), Target.values)


    # Base vanilla model
    base_pred = lr.predict(imp_X).reshape(-1,1)
    if config['scale']:
        base_pred = target_scaler.inverse_transform(base_pred)
    base_mae = eval_predictions(base_pred.reshape(-1,1), Target.values, metric = error_metric)

    # LAD model
    lad_pred = lad_model.predict(imp_X).reshape(-1,1)
    if config['scale']:
        lad_pred = target_scaler.inverse_transform(lad_pred)
    lad_mae = eval_predictions(lad_pred.reshape(-1,1), Target.values, metric = error_metric)
    
    # Lasso
    lasso_pred = lasso.predict(imp_X).reshape(-1,1)
    if config['scale']:
        lasso_pred = target_scaler.inverse_transform(lasso_pred)

    lasso_mae = eval_predictions(lasso_pred.reshape(-1,1), Target.values, metric = error_metric)
        
    # Ridge model
    l2_pred = ridge.predict(imp_X).reshape(-1,1)
    if config['scale']:
        l2_pred = target_scaler.inverse_transform(l2_pred)
    l2_mae = eval_predictions(l2_pred.reshape(-1,1), Target.values, metric = error_metric)
    
    # Random Forest
    rf_pred = rf.predict(imp_X)
    rf_mae = eval_predictions(rf_pred.reshape(-1,1), Target.values, metric = error_metric)

    mae_df['BASE'][run_counter] = base_mae
    mae_df['LAD'][run_counter] = lad_mae
    mae_df['BASE-retrain'][run_counter] = f_retrain_mae
    mae_df['BASE-LASSO'][run_counter] = lasso_mae
    mae_df['BASE-RIDGE'][run_counter] = l2_mae
    mae_df['RF'][run_counter] = rf_mae

    # FDR model
    for i, k in enumerate(K_parameter):
        fdr_pred = FDR_models[i].predict(miss_X).reshape(-1,1)
        if config['scale']:
            fdr_pred = target_scaler.inverse_transform(fdr_pred)
        
        fdr_mae = eval_predictions(fdr_pred, Target.values)
        mae_df['FDRR_'+str(k)][run_counter] = fdr_mae

#%%
from itertools import cycle, islice

fig,ax = plt.subplots(constrained_layout = True, figsize = (3.5, 2))

my_colors = list(islice(cycle(colors), None, mae_df.shape[1]))

patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
models_to_plot =['BASE', 'BASE-RIDGE', 'BASE-LASSO', 'LAD', 'RF', 'BASE-retrain','FDRR_0','FDRR_1']
labels_to_plot =['LS', 'LS-${\\ell_2}$', 'LS-${\\ell_1}$', 'LAD', 'RF', 'RETRAIN', 'FDRR$(0)$', 'FDRR$(1)$']    

bar = mae_df[models_to_plot].plot(kind='bar', ax=ax, color=my_colors, rot=45, 
                             width = 0.7, ylim=[4, 26])

bars = ax.patches
hatches = ''.join(h*len(mae_df) for h in 'x/oO.-*')

#for bar, hatch in zip(bars, hatches):
#    print(hatch)
#    bar.set_hatch(hatch)
    
ax.set_ylabel('MAE (EUR/MWh)')
#ax.set_xlabel('Deleted feature')
plt.legend(labels_to_plot, ncol=2, fontsize=6)
if config['save']: 
    if config['impute']:
        plt.savefig(cd+'\\figures\\uniform_deletion-imp-prices.pdf')
    else:
        plt.savefig(cd+'\\figures\\uniform_deletion-prices.pdf')

#%%%%% Joint coef. accuracy plot

gs_kw = dict(width_ratios=[1], height_ratios=[1, 1, .3])

fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout = True, figsize = (3.5, 1.5 + 1.5 + .25), 
                         gridspec_kw = gs_kw)


# Learned coefficients plot
plt.sca(axes[1])

'''
ind = 6
for i, m in enumerate(models[ind:]):
    plt.plot(np.arange(trainPred.shape[1]+1),np.array(list(FDR_models[i].coef_)+list(FDR_models[i].bias_)),
             marker = marker[ind+i], linewidth=1, label= '$\Gamma='+str(K_parameter[i])+'$', color = colors[ind+i])
plt.xticks(range(Predictors.shape[1]+1), ['F.' + str(t) for t in range(1,Predictors.shape[1]+2)] , rotation=0)
plt.vlines(n_feat-1+0.5, -0.1, 0.6, linestyle = 'dashed', color = 'black')
plt.legend(ncol=2, fontsize=6)
plt.ylabel('Coeff. magnitude')
'''

step = 0.15

plt.bar(np.arange(trainPred.shape[1]+1) - step, np.array(list(FDR_models[0].coef_)+list(FDR_models[0].bias_)), width = 0.3, label= '$\Gamma='+str(K_parameter[0])+'$', color = colors[6], 
        edgecolor = 'black',)
plt.bar(np.arange(trainPred.shape[1]+1) + step, np.array(list(FDR_models[1].coef_)+list(FDR_models[1].bias_)), width = 0.3, label= '$\Gamma='+str(K_parameter[1])+'$', color = colors[8], 
        edgecolor = 'black',)

#plt.scatter(np.arange(trainPred.shape[1]+1),np.array(list(FDR_models[0].coef_)+list(FDR_models[0].bias_)), marker = 's', linewidth=1, label= '$\Gamma='+str(K_parameter[0])+'$', color = colors[6])
#plt.scatter(np.arange(trainPred.shape[1]+1),np.array(list(FDR_models[1].coef_)+list(FDR_models[1].bias_)), marker = 'd', linewidth=1, label= '$\Gamma='+str(K_parameter[1])+'$', color = colors[8])

plt.xticks(range(Predictors.shape[1]+1), ['F.' + str(t) for t in range(1,Predictors.shape[1]+2)] , rotation=0)
plt.vlines(n_feat-1+0.5, -0.05, 0.45, linestyle = 'dashed', color = 'black')
plt.legend(ncol=1, fontsize=6)
plt.ylabel('Coeff. magnitude')


# Barplots for uniform feature deletion
plt.sca(axes[0])

models_to_plot =[ 'BASE-retrain','FDRR_0','FDRR_1']
labels = ['LS', 'LS-${\ell_2}$', 'LS-${\ell_1}$', 'LAD','RF', 'RETRAIN'] + ['FDRR$('+str(k)+')$' for k in K_parameter]

#models_to_plot =['BASE', 'BASE-RIDGE', 'BASE-LASSO', 'LAD', 'RF', 'BASE-retrain','FDRR_0','FDRR_1']
colors_to_plot = [colors[i] for i,m in enumerate(models) if m in models_to_plot]
labels_to_plot = [labels[i] for i,m in enumerate(models) if m in models_to_plot]
markers_to_plot = [marker[i] for i,m in enumerate(models) if m in models_to_plot]
colors_to_plot[-1] = colors[8]

my_colors = list(islice(cycle(colors_to_plot), None, mae_df.shape[1]))

bar = mae_df[models_to_plot].plot(kind='bar', ax=axes[0], color=my_colors, edgecolor = 'black', rot=45, 
                             width = 0.75, ylim=[4, 26],)

bars = axes[1].patches
hatches = ''.join(h*len(mae_df) for h in 'x/oO.-*')

# axis details
#axes[1].annotate('Performance with \n a single missing feature', (0.05, 0.85), color = 'black', 
#                    xycoords = 'axes fraction', fontsize = 8,
#               bbox=dict(facecolor='none', edgecolor='black', boxstyle='square'))

#for bar, hatch in zip(bars, hatches):
#    print(hatch)
#    bar.set_hatch(hatch)
plt.xticks(range(0,5), ['F.' + str(t) for t in range(1,6)] , rotation=0)

axes[0].set_ylabel('MAE (EUR/MWh)')
#ax.set_xlabel('Deleted feature')
axes[0].legend(labels_to_plot, ncol=1, fontsize=6)

props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
textstr = '''f1: Net Load Forecast, f2: System Margin 
            \nf3: DA Price_24, f4: DA Price_144
            \nf5: DA Price_168, f6: Day
            \nf7: Hour, f8: Bias'''
            
#fig.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
#        verticalalignment='top',  bbox=props)

text_data = np.array([['F-' +str(j) for j in range(1,10)]+ list(Predictors.columns)+['Bias']]).reshape(2,9).T
text_data = []
for i in range(1,10):    
    text_data.append('F.'+str(i)+': ' + [list(Predictors.columns)+['Bias']][0][i-1] )
text_data.append('')
text_data = np.array(text_data).reshape(2,5).T

the_table = axes[2].table(cellText=text_data, loc='best', fontsize=6, cellLoc='left', edges='horizontal', 
                          colLabels = ['Features in $\mathcal{J}$', 'Features in $\mathcal{C}$'])
axes[2].axis('off')
axes[2].axis('tight')
plt.show()
if config['save']: fig.savefig(cd+'\\figures\\joint_coef_feat_deletion.pdf')

#%%%%% Varying the number of observations with missing values at test set

percentage_missing = [0, 0.01, .05, .1, .25, .5]
n_feat = len(target_col)
iterations = 10
models = ['BASE-retrain', 'BASE', 'BASE-RIDGE', 'BASE-LASSO', 'LAD', 'RF', 'FDRR'] 
labels = ['BASE-R', 'BASE', 'BASE$_{\ell_2}$', 'BASE$_{\ell_1}$', 'LAD', 'RF', 'FDRR']

perfomance_df = pd.DataFrame(data = [], columns = models+['percentage', 'iteration'])

# Supress warning
pd.options.mode.chained_assignment = None
run_counter = 0
n_test_obs = len(testY)

for iter_ in range(iterations):
    for perc in percentage_missing:
        
        # index of observations with missing features
        missing_ind = np.random.choice(np.arange(0, n_test_obs), size = int(perc*n_test_obs), replace = False)
        # number of missing features per observations
        miss_feat_observation = np.random.choice(np.arange(1, n_feat+1), size = int(perc*n_test_obs), replace = True)
        # missing features for each indice (different number of features per each case)
        missing_feat = [np.random.choice(np.arange(0, len(target_col)), size = miss_feat_observation[i], replace = False) for i in range(int(perc*n_test_obs))]

        # create missing data with 0s and mean imputation 
        miss_X = testPred.copy()
        imp_X = testPred.copy()
        for ind, j in zip(missing_ind, missing_feat):
            miss_X[ind, j] = 0 
            imp_X[ind, j] = imputation_values[j]
        
        if config['impute'] != True:
            imp_X = miss_X.copy()
        
        # initiate row
        perfomance_df = perfomance_df.append({'percentage':perc}, ignore_index=True)                
        perfomance_df['iteration'][run_counter] = iter_

        # Retrain model
        f_retrain_pred = retrain_pred[:,0:1].copy()
        if perc > 0:
            for j, ind in enumerate(missing_ind):
                temp_feat = np.sort(missing_feat[j])
                temp_feat = list(temp_feat)   
                # find position in combinations list
                j_ind = retrain_comb.index(temp_feat)
                f_retrain_pred[ind] = retrain_pred[ind, j_ind]
            
        if config['scale']:
            f_retrain_pred = target_scaler.inverse_transform(f_retrain_pred)            
        retrain_mae = eval_predictions(f_retrain_pred.reshape(-1,1), Target.values, metric = error_metric)


        # Base vanilla model
        base_pred = lr.predict(imp_X).reshape(-1,1)
        if config['scale']:
            base_pred = target_scaler.inverse_transform(base_pred)            
        base_mae = eval_predictions(base_pred.reshape(-1,1), Target.values, metric = error_metric)

        # LAD model
        lad_pred = lad_model.predict(imp_X).reshape(-1,1)
        if config['scale']:
            lad_pred = target_scaler.inverse_transform(lad_pred)            
        lad_mae = eval_predictions(lad_pred.reshape(-1,1), Target.values, metric = error_metric)
        
        # LASSO
        lasso_pred = lasso.predict(imp_X).reshape(-1,1)
        if config['scale']:
            lasso_pred = target_scaler.inverse_transform(lasso_pred)    
        lasso_mae = eval_predictions(lasso_pred, Target.values, metric = error_metric)

        # RIDGE
        l2_pred = ridge.predict(imp_X).reshape(-1,1)
        if config['scale']:
            l2_pred = target_scaler.inverse_transform(l2_pred)    
        l2_mae = eval_predictions(l2_pred, Target.values, metric = error_metric)

        # Random Forest
        rf_pred = rf.predict(imp_X).reshape(-1,1)
        rf_mae = eval_predictions(rf_pred, Target.values, metric = error_metric)

        perfomance_df['BASE'][run_counter] = base_mae
        perfomance_df['LAD'][run_counter] = lad_mae
        perfomance_df['BASE-retrain'][run_counter] = retrain_mae
        perfomance_df['BASE-LASSO'][run_counter] = lasso_mae
        perfomance_df['BASE-RIDGE'][run_counter] = l2_mae
        perfomance_df['RF'][run_counter] = rf_mae

        # FDR predictions (select the appropriate model for each case)
        fdr_predictions = []
        for i, k in enumerate(K_parameter):
            fdr_pred = FDR_models[i].predict(miss_X).reshape(-1,1)
            # Robust
            if config['scale']:
                fdr_pred = target_scaler.inverse_transform(fdr_pred)
            fdr_predictions.append(fdr_pred.reshape(-1))
        fdr_predictions = np.array(fdr_predictions).T
        
        # Use only the model with the appropriate K
        final_fdr_pred = fdr_predictions[:,0]
        for j, ind in enumerate(missing_ind):
            final_fdr_pred[ind] = fdr_predictions[ind, miss_feat_observation[j]]
                
        fdr_mae = eval_predictions(final_fdr_pred.reshape(-1,1), Target.values, metric = error_metric)
        perfomance_df['FDRR'][run_counter] = fdr_mae

        run_counter += 1

#%%
std_bar = perfomance_df.groupby(['percentage'])[models].std()
fig,ax = plt.subplots(constrained_layout = True)
perfomance_df.groupby(['percentage'])[models].mean().plot(kind='bar', ax=ax, rot = 0, 
                                                 yerr=std_bar, legend=True)
ax.set_ylabel('MAE (EUR/MWh)')
ax.set_xlabel('% of observations')
#%%
if config['save']:
    perfomance_df.to_csv(cd+'\\results\\imp-prices_sensitivity_analysis.csv')
    if config['impute']:
        (perfomance_df.groupby(['percentage']).mean()[models]).round(2).to_csv(cd+'\\results\\imp-prices_mean_additional.csv')
        (perfomance_df.groupby(['percentage']).std()[models]).round(2).to_csv(cd+'\\results\\imp-prices_std_additional.csv')
    else:
        (perfomance_df.groupby(['percentage']).mean()[models]).round(2).to_csv(cd+'\\results\\prices_mean_additional.csv')
        (perfomance_df.groupby(['percentage']).std()[models]).round(2).to_csv(cd+'\\results\\prices_std_additional.csv')

#%%
fig, ax = plt.subplots(constrained_layout = True)
for i, m in enumerate(models):    
    if (m == 'BASE')or(m == 'RF')or(m == 'FDRR'):
        x_val = perfomance_df.groupby(['percentage'])[m].mean().values
        #plt.errorbar(perfomance_df['percentage'].unique(), perfomance_df.groupby(['percentage'])[m].mean().values, yerr=std_bar[m])
        plt.plot(perfomance_df['percentage'].unique(), perfomance_df.groupby(['percentage'])[m].mean().values, label = labels[i], color = color_list[i])
        plt.fill_between(perfomance_df['percentage'].unique(), x_val-std_bar[m], x_val+std_bar[m], alpha = 0.15, color = color_list[i])    
plt.legend()
plt.ylabel('MAE (EUR/MWh)')
plt.xlabel('% of observations')
plt.xticks(percentage_missing, (np.array(percentage_missing)*100).astype(int))
plt.show()        