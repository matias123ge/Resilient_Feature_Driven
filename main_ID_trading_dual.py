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

from utility_functions import * 
from FDR_regressor import *
from Feature_driven_reg import * 
import seaborn as sns 
sns.set() 


# IEEE plot parameters (not sure about mathfont)
#plt.rcParams['figure.dpi'] = 600
#plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
#plt.rcParams['font.size'] = 8
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'Times New Roman'
#plt.rcParams["mathtext.fontset"] = 'dejavuserif'

def eval_dual_predictions(pred, target, cost_up, cost_down):
    ''' Returns expected (or total) trading cost under dual loss function (quantile loss)'''
    
    error = target.reshape(-1) - pred.reshape(-1)
    total_cost = (-(cost_up*error)[error<0]).sum() + ((cost_down*error)[error>0]).sum()

    return (1/len(target))*total_cost

def projection(pred, ub = 1, lb = 0):
    'Projects to feasible set'
    pred[pred>ub] = ub
    pred[pred<lb] = lb
    return pred

#Hyperparameter tuning for feature driven model
def simple_CV(regvalues, X,y, k, penup,pendown): 
    penalty = [] 
    for val in regvalues: 
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.33, random_state = 42, shuffle = False)
        temp_error = []
        for i in range(k):  
            Mod = Feature_driven_reg(pen_up = penup, pen_down = pendown , alpha = val)
            Mod.fit(X_train,y_train)
            temp_error.append(eval_dual_predictions(Mod.predict(X_valid), y_valid, cost_up = config['pen_up'], cost_down = config['pen_down']))
        penalty.append(np.mean(temp_error))
    penalty = np.array(penalty)
    regval = regvalues[penalty.argmin()]
    return regval

def eval_predictions(pred, target, metric = 'mae'):
    if metric == 'mae':
        return np.mean(np.abs(pred-target))
    elif metric == 'rmse':
        return np.sqrt(np.square(pred-target).mean())
    elif metric == 'mape':
        return np.mean(np.abs(pred-target)/target)

def get_next_term(t_s):
    return random.choices([0,1], t_s)[0]

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
def make_chain(t_m, start_term, n):
    ''' Simulates block missingness with Markov Chains/ transition matrix
        t_m: transition matrix. First row controls the non-missing, second row controls the missing data.
        Row-wise sum must be 1. Example: the average block of missing data has length 10 steps.
        Then set the second row as: t_m[1] = [0.1, 0.9]'''
    if  isinstance(t_m, pd.DataFrame):
        t_m = t_m.copy().values
    chain = [start_term]
    for i in range(n-1):
        chain.append(get_next_term(t_m[chain[-1]]))
    return np.array(chain)

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

def params():
    ''' Set up the experiment parameters'''
    params = {}
    params['train'] = False # If True, then train models, else tries to load previous runs
    params['save'] = True # If True, then saves models and results
    params['impute'] = True # If True, apply mean imputation for missing features
    params['cap'] = False # If True, apply dual constraints for capacity (NOT YET IMPLEMENTED)
    params['trainReg'] = False #Determine best value of regularization for given values. 
    params['fixed_costs'] =False
    params['store_folder'] = 'ID-case' # folder to save stuff (do not change)
    params['max_lag'] = 3
    params['min_lag'] = 1  #!!! do not change this for the moment
    params['train_prices'] = False
    # Penalties for imbalance cost (only for fixed prices)
    params['pen_up'] = 4 
    params['pen_down'] = 3 
    params['NrRuns'] = 10 # of crossvalidation runs
    params['Regval'] = 0.001 #Previous best value of regularization used if not training
    # Parameters for numerical experiment
    #!!!!!!! To be changed with dates, not percentage
    #params['percentage_split'] = .75
    params['start_date'] = '2019-01-08' # start of train set
    params['split_date'] = '2020-01-01' # end of train set/start of test set
    params['end_date'] = '2020-05-01'# end of test set
    
    params['percentage'] = [.05, .10, .20, .50]  # percentage of corrupted datapoints
    params['iterations'] = 5 # per pair of (n_nodes,percentage)
    params['scale'] = False
    return params

#%% Load data at turbine level, aggregate to park level
config = params()

power_df = pd.read_csv(f'{cd}\\data\\smart4res_data\\wind_power_clean_30min.csv', index_col = 0,parse_dates = True)
metadata_df = pd.read_csv(f'{cd}\\data\\smart4res_data\\wind_metadata.csv', index_col=0,parse_dates = True)
market_df = pd.read_csv(f'{cd}\\trading-data\\Market_Data_processed.csv',index_col = 0,parse_dates = True)
# scale between [0,1]/ or divide by total capacity
power_df = (power_df - power_df.min(0))/(power_df.max() - power_df.min())
park_ids = list(power_df.columns)
# transition matrix to generate missing data/ estimated from training data (empirical estimation)
P = np.array([[0.999, 0.001], [0.241, 0.759]])

plt.figure(constrained_layout = True)
plt.scatter(x=metadata_df['Long'], y=metadata_df['Lat'])
plt.show()

#%%
#Either pick a park to forecast or assume average of all parks. 
target_park = 'p_1257' 
park_names = ['p_1003', 'p_1088', 'p_1257' ,'p_1475']#, 'p_1815', 'p_1825','p_1937', 'p_2137', 'p_2204', 'p_2275', 'p_2292', 'p_2419', 'p_2472']
power_df = power_df[park_names]

# number of lags back to consider
min_lag = config['min_lag']
max_lag = config['max_lag']
Y, Predictors, pred_col = create_IDsupervised(target_park, power_df, min_lag, max_lag)
lead_time_name = '-'+target_park[0]+'_t'+str(min_lag)
#Apply feature engineering: 
lead_time_name = '-'+target_park[0]+'_t'+str(min_lag)
market_feat_lagged = create_Marketlags(market_df, min_lag, max_lag)
market_feat_lagged['Pen_up_ewm_1'] = market_feat_lagged['Pen_up_1'].ewm(alpha=.9).mean()
market_feat_lagged['Pen_down_ewm_1'] = market_feat_lagged['Pen_down_1'].ewm(alpha=.9).mean()
market_feat_lagged['Pen_up_ewm_2'] = market_feat_lagged['Pen_up_2'].ewm(alpha=.9).mean()
market_feat_lagged['Pen_down_ewm_2'] = market_feat_lagged['Pen_down_2'].ewm(alpha=.9).mean()


#%%
#Features for different models
Power_Predictors = Predictors.columns 
ALL_Predictors = Predictors.columns.to_list()+market_feat_lagged.columns.to_list()
Pen_Predictors = Predictors.columns.to_list()+['Pen_up_ewm_1']+['Pen_down_ewm_1']

Predictors = pd.concat([Predictors, market_feat_lagged], axis = 1).dropna()
#Create main dataset
#%%

Full_Predictors = Predictors.columns

start = Predictors.index[0]
split = '2020-01-01'
end =  Predictors.index[-1]




#%%
if config['fixed_costs'] == False: 
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

Pen_Predictors = list(Power_Predictors) + ['Pen_up_1_pos','Pen_down_1_pos']+ ['Hour', 'Minute']

#Choose best values from Prelim study
Predictors = Predictors[Pen_Predictors]

trainPred = Predictors[start:split]
testPred = Predictors[split:end]

#Drop current timestep
trainY = Y[start:split].values
testY = Y[split:end].values
Target = Y[split:end]



#%%%% Quantile models: Feature driven, featuredriven l1 crossvalidation 

# Hyperparameter tuning with by cross-validation
param_grid = {"alpha": [10**pow for pow in range(-8,1)]}

ridge = GridSearchCV(Ridge(fit_intercept = True, max_iter = 500), param_grid)
ridge.fit(trainPred, trainY)

lasso = GridSearchCV(Lasso(fit_intercept = True, max_iter = 500), param_grid)
lasso.fit(trainPred, trainY)

lr = LinearRegression(fit_intercept = True)
lr.fit(trainPred, trainY)


lr_pred= lr.predict(testPred).reshape(-1,1)
lasso_pred = lasso.predict(testPred).reshape(-1,1)
ridge_pred = ridge.predict(testPred).reshape(-1,1)

persistence_pred = Target.values[:-config['min_lag']]
persistence_pred = np.insert(persistence_pred, 0, trainY[-1]).reshape(-1,1)
persistence_mae = eval_point_pred(persistence_pred, Target.values, digits=4)[1]
print('Persistence: ', eval_point_pred(persistence_pred, Target.values, digits=4))
print('LR: ', eval_point_pred(lr_pred.reshape(-1,1), Target.values, digits=4))
print('Lasso: ', eval_point_pred(lasso_pred.reshape(-1,1), Target.values, digits=4))
print('Ridge: ', eval_point_pred(ridge_pred.reshape(-1,1), Target.values, digits=4))

#%%
# check forecasts visually
plt.plot(Target[:60].values)
plt.plot(lr_pred[:60])
plt.plot(lasso_pred[:60])
plt.plot(persistence_pred[:60])
plt.show()


#Configure price option: 
if config['fixed_costs'] == False: 

    config['pen_up']= market_df['Pen_up'][start:split][market_df['Pen_up'][start:split]>0].mean()
    config['pen_down']= market_df['Pen_down'][start:split][market_df['Pen_down'][start:split]>0].mean()
    
    cost_up_train = market_df['Pen_up'][start:split]
    cost_down_train = market_df['Pen_down'][start:split]
    cost_up_test = market_df['Pen_up'][split:end]
    cost_down_test = market_df['Pen_up'][split:end]



#### FEATURE DRIVEN MODELS 
#%%
#Train feature driven model (Basic): 

case_folder = config['store_folder']
output_file_name = f'{cd}\\{case_folder}\\trained-models\\{target_park}_FeatureDriven_{max_lag}shortver.pickle'
if config['train']:
    FD50 = Feature_driven_reg(2,2,fit_intercept = True, alpha = 0)
    FD50.fit(trainPred, trainY)
   
    if config['save']:
        with open(output_file_name, 'wb') as handle:
            pickle.dump(FD50, handle)
else:
    with open(output_file_name, 'rb') as handle:    
            FD50 = pickle.load(handle)
#%%
#Train feature driven model on penalties

case_folder = config['store_folder']
output_file_name = f'{cd}\\{case_folder}\\trained-models\\{target_park}_FeatureDriven_{max_lag}shortver.pickle'
if config['train']:
    FD_Pen = Feature_driven_reg(cost_up_train.mean(),cost_down_train.mean(),fit_intercept = True, alpha = 0)
    FD_Pen.fit(trainPred, trainY)
   
    if config['save']:
        with open(output_file_name, 'wb') as handle:
            pickle.dump(FD_Pen, handle)
else:
    with open(output_file_name, 'rb') as handle:    
            FD_Pen = pickle.load(handle)


#Regularized model with variable price and all features
case_folder = config['store_folder']
output_file_name = f'{cd}\\{case_folder}\\trained-models\\{target_park}_FeatureDrivenL1_{max_lag}shortver.pickle'
if config['train'] ==False:
    FDl1 = Feature_driven_reg( pen_up = cost_up_train.mean(), pen_down = cost_down_train.mean(), alpha = 0.00001)#lasso.best_estimator_.alpha)
    FDl1.fit(trainPred, trainY)
   
    if config['save']:
        with open(output_file_name, 'wb') as handle:
            pickle.dump(FDl1, handle)
else:
    with open(output_file_name, 'rb') as handle:    
            FDl1 = pickle.load(handle)



#################FDR MODELS 
#%%%%% FDDR-R: train one model per value of \Gamma
#Gamma is only the parks available for deletion!! 
park_ids = list(power_df.columns.values)
target_pred = [] 
for i in range(len(park_ids)): 
    targ  = [col for col in Predictors.columns if park_ids[i] in col]
    target_pred.append(targ)

target_pred = [item for subitem in target_pred for item in subitem]
fixed_pred = [item for item in Predictors.columns if item not in target_pred]
target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
fix_col = [np.where(Predictors.columns == c)[0][0] for c in fixed_pred]
K_parameter = np.arange(0, len(target_pred)+1)
#%%
case_folder = config['store_folder']
output_file_name = f'{cd}\\{case_folder}\\trained-models\\{target_park}_fdrr-r_{max_lag}shortver.pickle'

if config['train'] :
    FDRR_R_models = []
    for K in K_parameter:
        print('Gamma: ', K)
        
        fdr = FDR_regressor(K = K, p_up = cost_up_train.mean(),p_down = cost_down_train.mean())
        fdr.fit(trainPred, trainY, target_col, fix_col, verbose=-1, solution = 'reformulation')  
        
        fdr_pred = fdr.predict(testPred).reshape(-1,1)

        print('FDR: ', eval_point_pred(fdr_pred, Target.values, digits=2))
        FDRR_R_models.append(fdr)
    
    if config['save']:
        with open(output_file_name, 'wb') as handle:
            pickle.dump(FDRR_R_models, handle)
else:
    with open(output_file_name, 'rb') as handle:    
            FDRR_R_models = pickle.load(handle)

#%%%%% FDDR-AAR: train one model per value of \Gamma

output_file_name = f'{cd}\\{case_folder}\\trained-models\\{target_park}_fdrr-aar_{max_lag}shortver.pickle'

if config['train']:
    FDRR_AAR_models = []
    for K in K_parameter:
        print('Gamma: ', K)
        
        fdr = FDR_regressor(K = K, p_up = cost_up_train.mean(), p_down = cost_down_train.mean())
        fdr.fit(trainPred, trainY, target_col, fix_col, verbose=-1, solution = 'affine')  
        
        fdr_pred = fdr.predict(testPred).reshape(-1,1)

        print('FDR: ', eval_point_pred(fdr_pred, Target.values, digits=2))
        FDRR_AAR_models.append(fdr)
    
    if config['save']:
        with open(output_file_name, 'wb') as handle:
            pickle.dump(FDRR_AAR_models, handle)
else:
    with open(output_file_name, 'rb') as handle:    
            FDRR_AAR_models = pickle.load(handle)

#            
###### RANDOM FOREST MODELS 
rf = ExtraTreesRegressor(n_estimators = 500)
rf.fit(trainPred, trainY)
opt_quant = cost_down_train.mean()/(cost_down_train.mean()+cost_up_train.mean())
#%%%%%%%%% Varying the number of missing observations/ persistence imputation
    
n_feat = len(target_col)
n_test_obs = len(testY)
iterations = 5
error_metric = 'mae'


#percentage = [0, .001, .005, .01, .05, .1]
percentage = [0, .001, .005, .01, .05, .1]
# transition matrix to generate missing data
P = np.array([[.999, .001], [0.241, 0.759]])

models = ['BASE', 'BASE-RIDGE', 'BASE-LASSO', 'RF','FDRR-R', 'FDRR-AAR']
#labels = ['BASE', 'BASE$_{\ell_2}$', 'BASE$_{\ell_1}$','FDRR-R', 'FDRR-AAR']

mae_df = pd.DataFrame(data = [], columns = models+['iteration', 'percentage'])
n_series = 4
# supress warning
pd.options.mode.chained_assignment = None
run_counter = 0


#series_missing = [c + str('_1') for c in park_ids]
series_missing = park_ids
#series_missing_col = [pred_col.index(series) for series in series_missing]

imputation = 'persistence'
mean_imput_values = trainPred.mean(0)

#%%
miss_ind = make_chain(np.array([[.95, .05], [0.05, 0.95]]), 0, len(testPred))

s = pd.Series(miss_ind)
block_length = s.groupby(s.diff().ne(0).cumsum()).transform('count')
check_length = pd.DataFrame()
check_length['Length'] = block_length[block_length.diff()!=0]
check_length['Missing'] = miss_ind[block_length.diff()!=0]
check_length.groupby('Missing').mean()

#%%
#if config['save']:
#    if config['impute']:
#        (mae_df.groupby(['percentage']).mean()[models_to_plot]).round(2).to_csv(cd+'\\results\\id-wind\\imp-id-wind_mean_additional'+lead_time_name+'.csv')
#        (mae_df.groupby(['percentage']).std()[models_to_plot]).round(2).to_csv(cd+'\\results\\id-wind\\imp-id-wind_std_additional'+lead_time_name+'.csv')
#    else:
#        (mae_df.groupby(['percentage']).mean()[models_to_plot]).round(2).to_csv(cd+'\\results\\id-wind\\id-wind_mean_additional'+lead_time_name+'.csv')
#        (mae_df.groupby(['percentage']).std()[models_to_plot]).round(2).to_csv(cd+'\\results\\id-wind\\id-wind_std_additional'+lead_time_name+'.csv')
#
#%%%%%%%%%%%%% The rest are similar to the DA forecasting 
#Trading, evaluate based on trading loss. 

models = ['Persistence','RF', 'Feature-Driven','Feature-Driven-L1','FDRR-R','FDRR-A']
labels = ['Persistence','RF', 'Feature-Driven','Feature-Driven-L1','FDRR-R','FDRR-A']

trade_df = pd.DataFrame(data = [], columns = models+['percentage missing', 'iteration'])

#Mn_feat = len(target_col)
n_test_obs = len(testY)
iterations = 5
park_ids = list(power_df.columns.values)

percentage = [0, .001, .005, .01, .05, .1]
#percentage = [0, .01, .05, .1, .5, 1]
# transition matrix to generate missing data
P = np.array([[.999, .001], [0.241, 0.759]])


#labels = ['BASE', 'BASE$_{\ell_2}$', 'BASE$_{\ell_1}$','FDRR-R', 'FDRR-AAR']

n_series = 4
# supress warning
pd.options.mode.chained_assignment = None
run_counter = 0


#series_missing = [c + str('_1') for c in park_ids]
series_missing = park_ids
#series_missing_col = [pred_col.index(series) for series in series_missing]

imputation = 'mean'
mean_imput_values = trainPred.mean(0)

#%%
miss_ind = make_chain(np.array([[.95, .05], [0.05, 0.95]]), 0, len(testPred))

s = pd.Series(miss_ind)
block_length = s.groupby(s.diff().ne(0).cumsum()).transform('count')
check_length = pd.DataFrame()
check_length['Length'] = block_length[block_length.diff()!=0]
check_length['Missing'] = miss_ind[block_length.diff()!=0]
check_length.groupby('Missing').mean()

#%%
for perc in percentage:
    for iter_ in range(iterations):
        
        P = np.array([[1-perc, perc], [0.05, 0.95]])

        # generate missing data
        #miss_ind = np.array([make_chain(P, 0, len(testPred)) for i in range(len(target_col))]).T
        miss_ind = np.zeros((len(testPred), len(series_missing)))
        for j in range(len(series_missing)):
            miss_ind[:,j] = make_chain(P, 0, len(testPred))

        mask_ind = miss_ind==1
        
        if run_counter%iterations==0:
            print('Percentage of missing values: ', mask_ind.sum()/mask_ind.size)
            percentages = (mask_ind.sum()/mask_ind.size)*100

        # Predictors w missing values
        miss_X = power_df[split:end].copy()[series_missing]
        miss_X[mask_ind] = np.nan
        
        miss_X = create_feat_matrix(miss_X, config['min_lag'], config['max_lag'])
        
        # Predictors w missing values
        miss_X_zero = miss_X.copy()
        miss_X_zero = miss_X_zero.fillna(0)
        
        # Predictors w mean imputation
        if config['impute'] != True:
            imp_X = miss_X_zero.copy()
        else:
            imp_X = miss_X.copy()
            # imputation with persistence or mean            
            if imputation == 'persistence':
                pers = imp_X.copy()
                persistence_pred = pers.fillna(method = 'ffill').fillna(method = 'bfill')[target_park+'_1']
                imp_X = miss_X.copy()
                imp_X = imp_X.fillna(method = 'ffill').fillna(method = 'bfill')

                
                #for j in series_missing:
                #    imp_X[mask_ind[:,j],j] = imp_X[mask_ind[:,j],j+1]
                    
            elif imputation == 'mean':
                    dic = dict(zip(Predictors.columns,mean_imput_values))
                    pers = imp_X.copy()
                    persistence_pred = pers.fillna(method = 'ffill').fillna(method = 'bfill')[target_park+'_1']
                    imp_X = imp_X.fillna(dic)
        


        persistence_pred[0] = trainY[-1]
        imp_X = pd.concat([imp_X,Predictors[split:end][fixed_pred]], axis =1)
        miss_X = pd.concat([miss_X,Predictors[split:end][fixed_pred]],axis = 1)
        miss_X_zero = pd.concat([miss_X_zero,Predictors[split:end][fixed_pred]],axis = 1)
        # initialize empty dataframe
        temp_df = pd.DataFrame(data = [round(percentages,3)], columns = ['percentage missing'])
        temp_df['iteration'] = iter_
        #Persistence pred 
        

        persistence_mae = eval_dual_predictions(persistence_pred.values, Target.values, cost_up_test,cost_down_test)
        #Mean bid 
        FD50_pred = projection(FD50.predict(imp_X).reshape(-1,1))
        FD50_mae = eval_dual_predictions(FD50_pred, Target.values, cost_up_test,cost_down_test)
        #Feature driven 
        FD_pen_pred = projection(FD_Pen.predict(imp_X).reshape(-1,1))
        FD_pen_mae = eval_dual_predictions(FD_pen_pred, Target.values, cost_up_test,cost_down_test)
        
        #Feature driven l1
        FDl1_pred = projection(FDl1.predict(imp_X).reshape(-1,1))
        FDl1_mae = eval_dual_predictions(FDl1_pred, Target.values, cost_up_test,cost_down_test)
        
        #Random Forest 
        Scenarios = np.array([rf.estimators_[tree].predict(imp_X).reshape(-1) for tree in range(len(rf.estimators_))]).T
        rfpred = np.quantile(Scenarios, opt_quant, axis = 1).T
        rf_pred = projection(rfpred.reshape(-1,1))
        rf_mae = eval_dual_predictions(rf_pred, Target.values, cost_up_test,cost_down_test)
        #### FDRR-R (select the appropriate model for each case)
        fdr_r_predictions = []
        for i, k in enumerate(K_parameter):
            fdr_pred = FDRR_R_models[i].predict(miss_X_zero).reshape(-1,1)
            fdr_pred = projection(fdr_pred)
            # Robust
            fdr_r_predictions.append(fdr_pred.reshape(-1))
        fdr_r_predictions = np.array(fdr_r_predictions).T
        # Use only the model with the appropriate K
        final_fdr_r_pred = fdr_r_predictions[:,0]
        if perc>0:
            for j, ind in enumerate(mask_ind):
                n_miss_feat = miss_X.isna().values[j].sum()
                final_fdr_r_pred[j] = fdr_r_predictions[j, n_miss_feat]
        final_fdr_r_pred = final_fdr_r_pred.reshape(-1,1)
        final_fdr_r_pred = projection(final_fdr_r_pred)
        fdr_r_mae = eval_dual_predictions(final_fdr_r_pred, Target.values, cost_up_test,cost_down_test)

        ### FDRR-AAR (select the appropriate model for each case)
        fdr_aar_predictions = []
        for i, k in enumerate(K_parameter):
            fdr_pred = FDRR_AAR_models[i].predict(miss_X_zero).reshape(-1,1)
            fdr_pred = projection(fdr_pred)
            # Robust
            fdr_aar_predictions.append(fdr_pred.reshape(-1))
        fdr_aar_predictions = np.array(fdr_aar_predictions).T
        
        # Use only the model with the appropriate K
        final_fdr_aar_pred = fdr_aar_predictions[:,0]
        
        if perc>0:
            for j, ind in enumerate(mask_ind):
                n_miss_feat = miss_X.isna().values[j].sum()
                final_fdr_aar_pred[j] = fdr_aar_predictions[j, n_miss_feat]
        final_fdr_aar_pred = final_fdr_aar_pred.reshape(-1,1)
        final_fdr_aar_pred = projection(final_fdr_aar_pred)
        fdr_aar_mae = eval_dual_predictions(final_fdr_aar_pred,Target.values, cost_up_test,cost_down_test)
        
        #temp_df['BASE'] = [lr_mae]        
        #temp_df['BASE-LASSO'] = [lasso_mae]
        #temp_df['BASE-RIDGE'] = [l2_mae] 
        temp_df['Persistence'] = persistence_mae  
        temp_df['FD50'] = [FD50_mae]
        temp_df['Feature-Driven'] = [FD_pen_mae]
        temp_df['Feature-Driven-L1'] = [FDl1_mae] 
        temp_df['RF'] = [rf_mae]            
        temp_df['FDRR-R'] = fdr_r_mae
        temp_df['FDRR-A'] = fdr_aar_mae
        #temp_df['FDRR-PWL'] = fdr_pwl_mae

        #temp_df['FDRR-CL'] = fdr_cl_mae
        
        trade_df = pd.concat([temp_df,trade_df])
        run_counter += 1
#%%
color_list = ['black', 'black', 'gray', 'tab:cyan','tab:green',
         'tab:blue', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive']
marker = ['o', '2', '^', 'd', '1', '+', 's', 'v', '*', '^', 'p']

marker = ['o', '2', '^', 'd', '1', '+', 's', 'v', '*', '^', 'p']
base_colors = plt.cm.tab20c( list(np.arange(3)))
fdr_colors = plt.cm.tab20c([8,9,10, 12, 13, 14])
colors = list(base_colors) + ['tab:orange'] + ['black'] + list(fdr_colors) 
std_trade_bar = [] 
meantradevals = []
for m in models: 
    std_trade_bar.append( trade_df.groupby(['percentage missing'])[m].std())
    meantradevals.append(np.array(trade_df.groupby(['percentage missing'])[m].mean()))
meantrade_df = pd.DataFrame(np.array(meantradevals).T,columns = models, index = np.unique(trade_df['percentage missing']))


#mae_df.groupby(['percentage'])[models_to_plot].mean().plot(kind='bar', ax=ax, rot = 0, 
#                                                 yerr=std_bar, legend=True)
fig, ax  = plt.subplots(constrained_layout = True)

fig = meantrade_df.plot(kind='bar', ax=ax, rot = 0, yerr=std_trade_bar, legend=True)
ax.set_ylabel("Trading Cost [€/MWh]")
ax.set_xlabel("Percentage Missing [%]")
plt.savefig(cd+'\\ID-case\\figures\\ShortDataMean.pdf')

#%%
# save results
if config['save']:
    output =trade_df.copy()
    if config['impute']:
        filename = 'imp-id-wind-Trading' + lead_time_name
    else: 
        filename = 'id-wind-Trading' + lead_time_name
        
    output.to_csv(cd+'\\ID-case\\results\\Persistence3_lag-'+filename+'.csv')
    table_output = output.copy().groupby(['percentage missing'])[models].mean().round(3).transpose()
    table_output.index = labels
    table_output.to_csv(cd+'\\ID-case\\results\\Persistence3_lag_mean-'+filename+'.csv')

fig, ax = plt.subplots(constrained_layout = True, figsize = (10,6))
for i, m in enumerate(K_parameter):
    plt.plot(np.append(FDRR_R_models[i].bias_,FDRR_R_models[i].coef_),label= '$\Gamma='+str(K_parameter[i])+'$')
plt.legend()
plt.xticks(range(len(Pen_Predictors)+1),list(np.append('Intercept',np.array(Pen_Predictors))), rotation=90)
#plt.vlines(n_feat-1+0.5, -0.1, 0.6, linestyle = 'dashed', color = 'black')
#plt.ylim(-0.1, .45)
plt.xlabel('Feature')
plt.ylabel('Coeff. magnitude')
plt.title("Reformulation")
plt.legend(fontsize=6, ncol=2)
if config['save']: plt.savefig(cd+'\\ID-case\\figures\\trade_coef_reform_smallID.pdf')
plt.show()
#%%

fig, ax = plt.subplots(constrained_layout = True, figsize = (10,6))
for i, m in enumerate(K_parameter):
    plt.plot(np.append(FDRR_AAR_models[i].bias_,FDRR_AAR_models[i].coef_),label= '$\Gamma='+str(K_parameter[i])+'$')
plt.legend()
plt.xticks(range(len(Pen_Predictors)+1), list(np.append('Intercept',Pen_Predictors)), rotation=90)
#plt.vlines(n_feat-1+0.5, -0.1, 0.6, linestyle = 'dashed', color = 'black')
#plt.ylim(-0.1, .45)
plt.xlabel('Feature')
plt.ylabel('Coeff. magnitude')
plt.title("Affine")
plt.legend(fontsize=6, ncol=2)
if config['save']: plt.savefig(cd+'\\ID-case\\figures\\trade_coef_affine_smallID.pdf')
plt.show()

#%%
fig, ax = plt.subplots(constrained_layout = True, figsize = (10,6))
plt.plot(FD_Pen.coef_,label = "Feature Driven")
plt.plot(FDl1.coef_,label = "Feature Driven L1")
plt.legend()
plt.xticks(range(len(Pen_Predictors)), list(Pen_Predictors), rotation=90)
#plt.vlines(n_feat-1+0.5, -0.1, 0.6, linestyle = 'dashed', color = 'black')
#plt.ylim(-0.1, .45)
plt.xlabel('Feature')
plt.ylabel('Coeff. magnitude')
plt.title("Feature Driven Models")
plt.legend(fontsize=6, ncol=2)
if config['save']: plt.savefig(cd+'\\ID-case\\figures\\trade_coef_FD_smallID.pdf')
plt.show()
# %%
