#Checking for intraday trading the difference in having market data in the dataset. 

# -*- coding: utf-8 -*-
"""
ID forecasting/ trading in dual price w missing data
Smart4RES data

"""
#%%
import pickle
import os, sys
import gurobipy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import itertools
import random

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

# import from forecasting libraries

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
#from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import seaborn as sns 

os.chdir(os.path.dirname(os.getcwd()))

from utility_functions import * 
from FDR_regressor import *
from Feature_driven_reg import * 

os.chdir(cd)


sns.set() 

#%%
# IEEE plot parameters (not sure about mathfont)
#plt.rcParams['figure.dpi'] = 600
#plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
#plt.rcParams['font.size'] = 8
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'Times New Roman'
#plt.rcParams["mathtext.fontset"] = 'dejavuserif'

def eval_dual_predictions(pred, target, cost_up, cost_down):
    ''' Returns expected (or total) trading cost under dual loss function (quantile loss)'''
    error = target.reshape(-1)-pred.reshape(-1)
    print(error.shape)
    total_cost = (-cost_up*error[error<0]).sum() + (cost_down*error[error>0]).sum()
    return (1/len(target))*total_cost


def eval_trades(pred, target,cost_up,cost_down):
    ''' Returns expected (or total) trading cost under dual loss function (quantile loss)'''
    error = target.reshape(-1)-pred.reshape(-1)
    maskup = error < 0
    maskdown = error >0 
    total_cost = (-cost_up*error)*maskup + (cost_down*error)*maskdown
    return total_cost

def eval_trading_dual(pred, target, cost_up, cost_down):
    ''' Returns expected (or total) trading cost under dual loss function (quantile loss)'''
    
    error = target.reshape(-1) - pred.reshape(-1)
    total_cost = (-(cost_up*error)[error<0]).sum() + ((cost_down*error)[error>0]).sum()
    
    return (1/len(target))*total_cost

def projection(pred, ub = 1, lb = 0):
    'Projects to feasible set'
    pred[pred>ub] = ub
    pred[pred<lb] = lb
    return pred

def eval_predictions(pred, target, metric = 'mae'):
    error = target.reshape(-1)-pred.reshape(-1)
    if metric == 'mae':
        return np.mean(np.abs(error))
    elif metric == 'rmse':
        return np.sqrt(np.square(error).mean())
    elif metric == 'mape':
        return np.mean(np.abs(error)/target)

def create_Marketlags(df, min_lag, max_lag):
    ''' Supervised learning set for ID forecasting with lags'''
    #min_lag = 1
    #max_lag = min_lag + 4 # 4 steps back
    p_df = df.copy()

    # Create supervised set
    pred_col = []
    for feature in p_df.columns:
        for lag in range(min_lag, max_lag):
            p_df[feature+'_'+str(lag)] = p_df[feature].shift(lag)
            pred_col.append(feature+'_'+str(lag))
    
    Predictors = p_df[pred_col]
    
    return  Predictors

def create_IDsupervised(target_col, df, min_lag, max_lag):
    ''' Supervised learning set for ID forecasting with lags'''
    #min_lag = 1
    #max_lag = min_lag + 4 # 4 steps back
    lead_time_name = '-' + target_col + '_t'+str(min_lag)
    p_df = df.copy()

    # Create supervised set
    pred_col = []
    for park in p_df.columns:
        for lag in range(min_lag, max_lag):
            p_df[park+'_'+str(lag)] = p_df[park].shift(lag)
            pred_col.append(park+'_'+str(lag))
    
    Predictors = p_df[pred_col]
    Y = p_df[target_col].to_frame()
    
    return Y, Predictors, pred_col

def create_feat_matrix(df, min_lag, max_lag):
    ''' Supervised learning set for ID forecasting with lags'''
    #min_lag = 1
    #max_lag = min_lag + 4 # 4 steps back
    p_df = df.copy()

    # Create supervised set
    pred_col = []
    for park in p_df.columns:
        for lag in range(min_lag, max_lag):
            p_df[park+'_'+str(lag)] = p_df[park].shift(lag)
            pred_col.append(park+'_'+str(lag))
    
    Predictors = p_df[pred_col]
    return Predictors

def projection(pred, ub = 1, lb = 0):
    'Projects to feasible set'
    pred[pred>ub] = ub
    pred[pred<lb] = lb
    return pred

def params():
    ''' Set up the experiment parameters'''
    params = {}
    params['scale'] = False #Data is already normalized for capacity
    params['train'] = True # If True, then train models, else tries to load previous runs
    params['save'] = False # If True, then saves models and results
    params['impute'] = True # If True, apply mean imputation for missing features
    params['cap'] = False # If True, apply dual constraints for capacity (NOT YET IMPLEMENTED)

    params['store_folder'] = 'ID_results' # folder to save stuff (do not change)
    params['max_lag'] = 3
    params['min_lag'] = 1  #!!! do not change this for the moment
    params['fixed_cost'] =False
    # Penalties for imbalance cost (only for fixed prices)
    params['pen_up'] = 4 
    params['pen_down'] = 3 
    # Parameters for numerical experiment
    #!!!!!!! To be changed with dates, not percentage
    #params['percentage_split'] = .75
    #params['start_date'] = '2019-01-01'
    #params['split_date'] = '2019-03-01'
    #params['end_date'] = '2020-01-01'
    
    params['start_date'] = '2018-10-01' # start of train set
    params['split_date'] = '2020-01-01' # end of train set/start of test set
    params['end_date'] = '2020-05-01'# end of test set
    
    return params

#%% Load data at turbine level, aggregate to park level
config = params()

target_scaler = MinMaxScaler()
pred_scaler = MinMaxScaler()

power_df = pd.read_csv(f'{cd}\\data\\smart4res_data\\wind_power_clean_30min.csv', index_col = 0, parse_dates = True)
metadata_df = pd.read_csv(f'{cd}\\data\\smart4res_data\\wind_metadata.csv', index_col=0, parse_dates = True)
market_df = pd.read_csv(f'{cd}\\trading-data\\Market_Data_processed.csv',index_col = 0, parse_dates = True)
# scale between [0,1]/ or divide by total capacity
power_df = (power_df - power_df.min(0))/(power_df.max() - power_df.min())
park_ids = list(power_df.columns)

cols = market_df.columns

trans_market_df = pred_scaler.fit_transform(market_df)
trans_market_df = pd.DataFrame(trans_market_df, columns= cols, index= market_df.index)
market_predictors = ['Volume','Pen_up','Pen_down']
#%%
#Either pick a park to forecast or assume average of all parks. 
target_park = 'p_1257'

# number of lags back to consider
min_lag = config['min_lag']
max_lag = config['max_lag']
Y, Predictors, pred_col = create_IDsupervised(target_park, power_df, min_lag, max_lag)

lead_time_name = '-'+target_park[0]+'_t'+str(min_lag)
market_feat_lagged = create_Marketlags(trans_market_df, min_lag, max_lag)

market_feat_lagged['Pen_up_ewm_1'] = market_feat_lagged['Pen_up_1'].ewm(alpha=.9).mean()
market_feat_lagged['Pen_down_ewm_1'] = market_feat_lagged['Pen_down_1'].ewm(alpha=.9).mean()
market_feat_lagged['Pen_up_ewm_2'] = market_feat_lagged['Pen_up_2'].ewm(alpha=.9).mean()
market_feat_lagged['Pen_down_ewm_2'] = market_feat_lagged['Pen_down_2'].ewm(alpha=.9).mean()

#%%
#Features for different models
Power_Predictors = Predictors.columns 
ALL_Predictors = Predictors.columns.to_list()+market_feat_lagged.columns.to_list()
Pen_Predictors = Predictors.columns.to_list()+['Pen_up_ewm_1']+['Pen_down_ewm_1']

#Create main dataset
#%%
Predictors = pd.concat([Predictors, market_feat_lagged], axis = 1).dropna()

Full_Predictors = Predictors.columns

start = Predictors.index[0]
split = '2020-01-01'
end =  Predictors.index[-1]
#%%
if config['fixed_cost'] == False: 
    #Variable costs
    #config['pen_up']= np.mean(market_df['Pen_up'][start:split])
    #config['pen_down'] = np.mean(market_df['Pen_down'][start:split])

    config['pen_up']= market_df['Pen_up'][start:split][market_df['Pen_up'][start:split]>0].mean()
    config['pen_down']= market_df['Pen_down'][start:split][market_df['Pen_down'][start:split]>0].mean()
    
    cost_up_train = market_df['Pen_up'][start:split]
    cost_down_train = market_df['Pen_down'][start:split]
    cost_up_test = market_df['Pen_up'][split:end]
    cost_down_test = market_df['Pen_up'][split:end]

#Create training sets 
Predictors['Pen_up_1_pos'] = Predictors['Pen_up_1']
Predictors['Pen_down_1_pos'] = Predictors['Pen_down_1']

Predictors['Pen_up_1_pos'][Predictors['Pen_up_1_pos'] == 0] = np.nan
Predictors['Pen_up_1_pos'] = Predictors['Pen_up_1_pos'].pad()

Predictors['Pen_down_1_pos'][Predictors['Pen_down_1_pos'] == 0] = np.nan
Predictors['Pen_down_1_pos'] = Predictors['Pen_down_1_pos'].pad()


Predictors = Predictors.fillna(method='bfill')

Predictors['Pen_up_ewm_1'] = Predictors['Pen_up_1_pos'].ewm(alpha=.9).mean()
Predictors['Pen_down_ewm_1'] = Predictors['Pen_down_1_pos'].ewm(alpha=.9).mean()

Predictors['Day'] = Predictors.index.weekday
Predictors['Hour'] = Predictors.index.weekday
Predictors['Minute'] = Predictors.index.weekday

trainPred = Predictors[start:split]
testPred = Predictors[split:end]

#Drop current timestep
trainY = Y[start:split].values
testY = Y[split:end].values
Target = Y[split:end]

plt.plot(trainY)



#%%%%% Regresion: train models for standard forecasting

param_grid = {"alpha": [10**pow for pow in range(-5,2)]}
    
FD50 = Feature_driven_reg( 2, 2, alpha = 0)
FD50.fit(trainPred[Power_Predictors], trainY)

lr = LinearRegression(fit_intercept = True)
lr.fit(trainPred[Power_Predictors], trainY)

lasso = GridSearchCV(Lasso(fit_intercept = True, max_iter = 3000), param_grid)
lasso.fit(trainPred[Power_Predictors], trainY)

rf = ExtraTreesRegressor(n_estimators = 500)
rf.fit(trainPred[Power_Predictors], trainY)

#%% Predictions and evaluation of MAE

persistence_pred = np.zeros(len(testY))
persistence_pred[0] = trainY[-1]
persistence_pred[1:] = testY[:-1,0]

FD50_NWP_pred = FD50.predict(testPred[Power_Predictors])
lasso_NWP_pred = lasso.predict(testPred[Power_Predictors])
lr_NWP_pred = lr.predict(testPred[Power_Predictors])
rf_NWP_pred = rf.predict(testPred[Power_Predictors])

#Error results (MAE)
Persistence_mae =  eval_predictions(persistence_pred,testY)
FD50_mae =  eval_predictions(FD50_NWP_pred,testY)
lr_NWP_mae = eval_predictions(lr_NWP_pred,testY)
lasso_mae =  eval_predictions(lasso_NWP_pred,testY)
rf_NWP_mae = eval_predictions(rf_NWP_pred,testY)

mae_vec =np.array([Persistence_mae,FD50_mae,lasso_mae,rf_NWP_mae,lr_NWP_mae])
mae_df = pd.DataFrame(data = 100*mae_vec, index = ['Persistence','FD50', 'Linear Regression','Linear Regression L1','Random Forest'], columns = ['MAE'])


#%%%%%%%%%%% Trading: train considering also penalties

# Persistence: offer the last known value
persistence_pred = np.zeros(len(testY))
persistence_pred[0] = trainY[-1]
persistence_pred[1:] = testY[:-1,0]

opt_quant = config['pen_down']/(config['pen_up'] + config['pen_down'])
#%%
#Do boxplot comparison of model training with in sample mean and variable cost: 
FD_base = Feature_driven_reg( pen_up = cost_up_train.mean(), pen_down = cost_down_train.mean(), alpha = 0)
FD_base.fit(trainPred[Power_Predictors], trainY)


Time_Predictors = list(Power_Predictors)
#%%
FD_var = Feature_driven_reg( pen_up = cost_up_train.values, pen_down = cost_down_train.values, alpha = 0)
FD_var.fit(trainPred[Time_Predictors],trainY)

#%%
#Predict out of sample 
FD_base_pred = FD_base.predict(testPred[Power_Predictors])
FD_var_pred = FD_var.predict(testPred[Power_Predictors])

#Get individual penalties 
base = eval_trades(FD_base_pred,testY,cost_up_test.values,cost_down_test.values)
var = eval_trades(FD_var_pred,testY,cost_up_test.values,cost_down_test.values)

df = pd.DataFrame(np.array([base,var]).T,columns = ["In Sample Mean", "Variable Prices"])
#%%
#Create boxplot 
s = sns.boxplot(df[(df > 0).any(axis=1)],
    notch=True, showcaps=False,
    showfliers = False,
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "coral"})
if config['save']: 
    plt.savefig(cd+'\\ID\\Figures\\InsampleBox.pdf')
    df.mean().to_csv(cd+'\\ID\\Results\\MeanBox.csv')    

#%%

# Feature-driven/ prices are set at in-sample mean
FD_base = Feature_driven_reg( pen_up = config['pen_up'], pen_down = config['pen_down'], alpha = 0)
FD_base.fit(trainPred[Power_Predictors], trainY)

# Feature-driven/ estimates the 50th quantile
FD50 = Feature_driven_reg( 2, 2, alpha = 0)
FD50.fit(trainPred[Power_Predictors], trainY)
#%%
# Feature-driven/ variable prices/ includes additional features
#!!!!! do feature engineering here
#Base 


Time_Predictors = list(Power_Predictors) + ['Hour', 'Minute']

FD_time = Feature_driven_reg( pen_up = cost_up_train.values, pen_down = cost_down_train.values, alpha = 0)
FD_time.fit(trainPred[Time_Predictors], trainY)

#Do variable price with penalty information: 
Pen_Predictors = list(Power_Predictors) + ['Pen_up_1_pos','Pen_down_1_pos']+ ['Hour', 'Minute']
FD_pen = Feature_driven_reg( pen_up = cost_up_train.values.mean(), pen_down = cost_down_train.values.mean(), alpha = 0)
FD_pen.fit(trainPred[Pen_Predictors], trainY)

#Do one with variable other market features 
Mark_Predictors = list(Power_Predictors) + ['Volume_1','Volume_2','Margin_1','Margin_2']

FD_mark = Feature_driven_reg( pen_up = cost_up_train.values, pen_down = cost_down_train.values, alpha = 0)
FD_mark.fit(trainPred[Mark_Predictors], trainY)

#Random forest on power feautres 
rf_pow = ExtraTreesRegressor(n_estimators = 300)
rf_pow.fit(trainPred[Power_Predictors],trainY.reshape(-1))
#Random forest on all features 
rf_mark = ExtraTreesRegressor(n_estimators = 300)
rf_mark.fit(trainPred[ALL_Predictors], trainY.reshape(-1))

#Models 
TradingModels = ['Persistence','FD50','FD-Base Feat','FD-Time Feat', 'FD-Pen Feat', 'FD-Mark Feat', 'RF','RF Mark Feat']
FD50_pred = projection(FD50.predict(testPred[Power_Predictors]))
FD_base_pred = projection(FD_base.predict(testPred[Power_Predictors]))

# forecast trading decisions for each model/ project to feasible set
FD_time_pred = projection(FD_time.predict(testPred[Time_Predictors]))
FD_pen_pred = projection(FD_pen.predict(testPred[Pen_Predictors]))
FD_mark_pred = projection(FD_mark.predict(testPred[Mark_Predictors]))
#RF predictions 
Scenarios = np.array([rf_pow.estimators_[tree].predict(testPred[Power_Predictors]).reshape(-1) for tree in range(len(rf_pow.estimators_))]).T
rf_pow_pred = np.quantile(Scenarios, opt_quant, axis = 1).T

Scenarios = np.array([rf_mark.estimators_[tree].predict(testPred[ALL_Predictors]).reshape(-1) for tree in range(len(rf_mark.estimators_))]).T
rf_mark_pred = np.quantile(Scenarios, opt_quant, axis = 1).T
#%%    
#Evaluate trading decisions: 
persistence_cost =  eval_trading_dual(persistence_pred, testY, cost_up_test.values, cost_down_test)
FD50_cost        =  eval_trading_dual(FD50_pred, testY, cost_up_test.values, cost_down_test.values)
FD_base_cost     = eval_trading_dual(FD_base_pred, testY, cost_up_test.values, cost_down_test.values)
FD_time_cost     = eval_trading_dual(FD_time_pred, testY, cost_up_test.values, cost_down_test.values)
FD_pen_cost      = eval_trading_dual(FD_pen_pred, testY, cost_up_test.values, cost_down_test.values)
FD_mark_cost     = eval_trading_dual(FD_mark_pred, testY, cost_up_test.values, cost_down_test.values)
rf_pow_cost      = eval_trading_dual(rf_pow_pred, testY, cost_up_test.values, cost_down_test.values)
rf_mark_cost     = eval_trading_dual(rf_mark_pred, testY, cost_up_test.values, cost_down_test.values)
#%%
Penalties_var_price = [persistence_cost,FD50_cost,FD_base_cost
    ,FD_time_cost
    ,FD_pen_cost
    ,FD_mark_cost
    ,rf_pow_cost
    ,rf_mark_cost]
cost_df = pd.DataFrame(Penalties_var_price, columns =['Mean Trading Cost'], index = TradingModels)
print(cost_df)

#Get standard deviations: 
persistence_std  = np.std(eval_trades(persistence_pred, testY, cost_up_test.values, cost_down_test))
FD50_std         = np.std(eval_trades(FD50_pred, testY, cost_up_test.values, cost_down_test.values))
FD_base_std     = np.std(eval_trades(FD_base_pred, testY, cost_up_test.values, cost_down_test.values))
FD_time_std     = np.std(eval_trades(FD_time_pred, testY, cost_up_test.values, cost_down_test.values))
FD_pen_std      = np.std(eval_trades(FD_pen_pred, testY, cost_up_test.values, cost_down_test.values))
FD_mark_std     = np.std(eval_trades(FD_mark_pred, testY, cost_up_test.values, cost_down_test.values))
rf_pow_std      = np.std(eval_trades(rf_pow_pred, testY, cost_up_test.values, cost_down_test.values))
rf_mark_std     = np.std(eval_trades(rf_mark_pred, testY, cost_up_test.values, cost_down_test.values))

Penalties_std = [persistence_std,FD50_cost,FD_base_std
    ,FD_time_std
    ,FD_pen_std
    ,FD_mark_std
    ,rf_pow_std
    ,rf_mark_std]
std_df = pd.DataFrame(Penalties_std, columns =['Standard Deviation'], index = TradingModels)

if config['save']:
        df = cost_df
        df = pd.concat([df,std_df],axis = 1)
        df.to_csv(f'{cd}\\ID\\Results\\Variable_costs_results.csv')
#%%    
persistence_std  =eval_trades(persistence_pred, testY, cost_up_test.values, cost_down_test)
FD50_std         =eval_trades(FD50_pred, testY, cost_up_test.values, cost_down_test.values)
FD_base_std     = eval_trades(FD_base_pred, testY, cost_up_test.values, cost_down_test.values)
FD_time_std     = eval_trades(FD_time_pred, testY, cost_up_test.values, cost_down_test.values)
FD_pen_std      = eval_trades(FD_pen_pred, testY, cost_up_test.values, cost_down_test.values)
FD_mark_std     = eval_trades(FD_mark_pred, testY, cost_up_test.values, cost_down_test.values)
rf_pow_std      = eval_trades(rf_pow_pred, testY, cost_up_test.values, cost_down_test.values)
rf_mark_std     = eval_trades(rf_mark_pred, testY, cost_up_test.values, cost_down_test.values)
Penalties_gen = [persistence_std.values,FD50_std,FD_base_std
    ,FD_time_std
    ,FD_pen_std
    ,FD_mark_std
    ,rf_pow_std
    ,rf_mark_std]
Penalties_gen = np.transpose(Penalties_gen)
gen_df = pd.DataFrame(Penalties_gen, columns =cost_df.index)

#%%

sns.barplot(gen_df)
plt.xticks(rotation = 45)
plt.tight_layout()
plt.ylabel("Mean Penalty [€/MW]")
plt.savefig(cd+'\\ID\\Figures\\OutOfSamplePrelimID.pdf')

#So, apparently the base models do  a better job. 


# %%

#Make cumulative trading plots for Persistence, Fd-Base feat with mean and RF 
#Compare for l1 regularization 
FD_l1 = Feature_driven_reg(pen_up = cost_up_train.mean(), pen_down = cost_down_train.mean(), alpha = lasso.best_estimator_.alpha)
FD_l1.fit(trainPred[ALL_Predictors], trainY)
FD_l1_pred = projection(FD_l1.predict(testPred[ALL_Predictors]))
FD_l1_cost =eval_trading_dual(FD_l1_pred, testY, cost_up_test.values, cost_down_test)

#%%
#Plot predictions vs. test data 
df_lines = pd.DataFrame(np.array([FD50_pred,FD_pen_pred,FD_l1_pred,rf_pow_pred,testY.reshape(-1)]).T, columns = ['FD50','FD feat','FD l1','RF','Y'])


# %%
sns.lineplot(data = df.iloc[0:100])

# %%
