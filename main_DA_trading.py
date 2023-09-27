# -*- coding: utf-8 -*-
"""
DA trading with missing data/ dual price balancing market/ fixed or actual prices (user selects)
Renewable trading: ERM model with Linear Decision Rules
Robust re-formulation to deal with feature deletion
"""
#%%
import pickle
import os, sys

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

import seaborn as sns
import sklearn
import gurobipy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from FDR_group_regressor import *
from QR_regressor import *
from Feature_driven_reg import *
from classSAA import * 
from HiddenPrinter import *
from sklearn.preprocessing import MinMaxScaler

from utility_functions import *
sns.set()


def eval_dual_predictions(pred, target, cost_up, cost_down):
    ''' Returns expected (or total) trading cost under dual loss function (quantile loss)'''
    error = target.reshape(-1)-pred.reshape(-1)
    total_cost = (-cost_up*error[error<0]).sum() + (cost_down*error[error>0]).sum()

    return (1/len(target))*total_cost

def eval_trading_dual(pred, target, cost_up, cost_down):
    ''' Returns expected (or total) trading cost under dual loss function (quantile loss)'''
    
    error = target.reshape(-1) - pred.reshape(-1)
    error = np.repeat(error, 2)
    total_cost = (-(cost_up*error)[error<0]).sum() + ((cost_down*error)[error>0]).sum()

    return (1/len(target))*total_cost

def projection(pred, ub = 1, lb = 0):
    'Projects to feasible set'
    pred[pred>ub] = ub
    pred[pred<lb] = lb
    return pred

def resilient_crossval_fixed_costs(x, y, p_up, p_down, group_col, l1_weights, num_runs = 10): 
    ''' Resilient-oriented cross-validation algorithm
        - x,y: features, target variable / only training data
        - p_up, p_down: cost of upward/downward regularions
        - group_col: groups of columns that can be deleted
        - l1_weights: grid of candidate l1 weights (controls regularization penalty)
        - K: '''
        
    # maximum number of missing features
    K_max = len(group_col)
    # Create train and validation set
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x, y, test_size = 0.33, random_state = 42, shuffle = False)


    # Create copies of validation data set with missing data  
    val_datasets = []
    for k in range(num_runs):       
       # Sample number of missing groups
       randomsize = np.random.randint(0, K_max)
       
       # Sample which groups are deleted
       groupchoice = np.random.choice( np.arange(0, K_max), size = randomsize , replace = False)
       
       if groupchoice.size == 0: 
           miss_X = X_valid.copy()
       else:
           miss_X = X_valid.copy()
           getindexes = []
           for i in range(len(groupchoice)): 
               getindexes.append(group_col[groupchoice[i]])

           for j in range(len(getindexes)):
               miss_X[:,getindexes[j]] = 0   
        
       val_datasets.append(miss_X)

    # Train models for all values of alpha in l1_weights
    lasso_models = []
    for i, temp_alpha in enumerate(l1_weights):
        model = Feature_driven_reg(p_up, p_down, alpha = temp_alpha)
        model.fit(X_train,y_train)        
        lasso_models.append(model)
        
    # Check expected validation error over all copies
    val_error_mean = []
    val_error_std = []
    for model in lasso_models:
        
        model_error = np.zeros(num_runs)
        
        for j, temp_X in enumerate(val_datasets):
            # forecast
            temp_pred = model.predict(temp_X)
            temp_pred = projection(temp_pred)
            temp_error = eval_dual_predictions(temp_pred, y_valid, p_up, p_down)
            model_error[j] = temp_error
            
        val_error_mean.append( model_error.mean() )    
        val_error_std.append( model_error.std() )    
        
        
    val_error_mean = np.array(val_error_mean)
    val_error_std = np.array(val_error_std)
    
    best_ind = np.where(val_error_mean == val_error_mean.min())[0][0]
    best_alpha = l1_weights[best_ind]
    
    print('Best alpha: ', best_alpha)        
    return best_alpha, val_error_mean, val_error_std

def resilient_crossval_actual_costs(x, y, p_up, p_down, group_col, l1_weights, num_runs = 10): 
    ''' Resilient-oriented cross-validation algorithm
        - x,y: features, target variable / only training data
        - p_up, p_down: cost of upward/downward regularions [vectors]
        - group_col: groups of columns that can be deleted
        - l1_weights: grid of candidate l1 weights (controls regularization penalty)
        - K: '''
    # maximum number of missing features
    K_max = len(group_col)
    # Create train and validation set/ both for features-target and costs
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x, y, test_size = int(len(y)*0.4//1), random_state = 42, shuffle = False)
    p_up_train, p_up_valid, p_down_train, p_down_valid = sklearn.model_selection.train_test_split(p_up, p_down, test_size = int(len(p_up)*0.4//1), random_state = 42, shuffle = False)
    # Create copies of validation data set with missing data  
    imputation_values = X_train.mean(1)
    val_datasets = []
    for k in range(num_runs):       
       # Sample number of missing groups
       randomsize = np.random.randint(0, K_max)
       
       # Sample which groups are deleted
       groupchoice = np.random.choice( np.arange(0, K_max), size = randomsize , replace = False)
       
       if groupchoice.size == 0: 
           miss_X = X_valid.copy()
           imp_X = X_valid.copy()

       else:
           miss_X = X_valid.copy()
           imp_X = X_valid.copy()
           getindexes = []
           for i in range(len(groupchoice)): 
               getindexes.append(group_col[groupchoice[i]])

           for j in range(len(getindexes)):
               miss_X[:,getindexes[j]] = 0   
               imp_X[:,getindexes[j]] = imputation_values[getindexes[j]]
       val_datasets.append(imp_X)

    # Train models for all values of alpha in l1_weights
    lasso_models = []
    for i, temp_alpha in enumerate(l1_weights):
        model = Feature_driven_reg(p_up_train.mean(), p_down_train.mean(), alpha = temp_alpha)
        model.fit(X_train,y_train)        
        lasso_models.append(model)
        
    # Check expected validation error over all copies
    val_error_mean = []
    val_error_std = []
    for model in lasso_models:
        
        model_error = np.zeros(num_runs)
        
        for j, temp_X in enumerate(val_datasets):
            # forecast
            temp_pred = model.predict(temp_X)
            temp_pred = projection(temp_pred)
            temp_error = eval_trading_dual(temp_pred, y_valid, p_up_valid, p_down_valid)
            model_error[j] = temp_error
            
        val_error_mean.append( model_error.mean() )    
        val_error_std.append( model_error.std() )    
        
        
    val_error_mean = np.array(val_error_mean)
    val_error_std = np.array(val_error_std)
    
    best_ind = np.where(val_error_mean == val_error_mean.min())[0][0]
    best_alpha = l1_weights[best_ind]
    
    print('Best alpha: ', best_alpha)        
    return best_alpha, val_error_mean, val_error_std

def retrain_model(X, Y, testX, p_up, p_down, group_col, fix_col, Gamma, base_loss = 'l1'):
    ''' Retrain model without missing features
        The base learner is a feature driven model (takes as input p_up, p_down)
        returns a list models and corresponding list of missing features'''
    # all combinations of missing groups for up to Gamma features missing
    combinations = [list(item) for sublist in [list(itertools.combinations(range(len(group_col)), gamma)) for gamma in range(1,Gamma+1)] for item in sublist]
    # first instance is without missing features
    combinations.insert(0, [])
    group_col = np.array(group_col)
    index = group_col.ravel().tolist()
    models = []
    predictions = []
    for i,v in enumerate(combinations):
        if len(v) == 0:
            temp_X = X 
            temp_test_X = testX
        else:  
            #Get indexes of remaning columns 
            nonfeatures = group_col[v].ravel().tolist()
            del_col = np.delete(index,nonfeatures)
            temp_col = np.append(del_col,fix_col) 
            temp_X = X[:,temp_col]
            temp_test_X = testX[:,temp_col]
        # retrain model without missing features
        feat_dr_model = Feature_driven_reg(pen_up = p_up, pen_down = p_down)
        feat_dr_model.fit(temp_X, Y)
        models.append(feat_dr_model)
        predictions.append(feat_dr_model.predict(temp_test_X).reshape(-1))
    predictions = np.array(predictions).T
    
    return models, predictions, combinations

#%%

def params():
    ''' Set up the experiment parameters'''

    params = {}
    params['scale'] = True
    params['train'] = False # If True, then train models, else tries to load previous runs
    params['save'] = True # If True, then saves models and results
    #params['K value'] = 5 #Define budget of uncertainty value
    params['impute'] = True # If True, apply mean imputation for missing features
    params['impute_peripheral'] = False # If True impute value from other available features, if no features available do mean imputation 
    if params['impute'] == params['impute_peripheral']: 
        print("Imputing options are equivalent")
        raise(ValueError(params['impute']))
    params['trainReg'] = True #Determine best value of regularization for given values. 
    
    # Penalties for imbalance cost (only for fixed prices)
    params['fixed_costs'] = False
    params['pen_up'] = 4 # used only if fixed_costs == True
    params['pen_down'] = 3 # used only if fixed_costs == True
    
    #Train regularization for feature driven model? 

    # Parameters for numerical experiment
    #!!!!!!! To be changed with dates, not percentage
    #params['percentage_split'] = .75
    
    params['data_source'] = 'smart4res' # smart4res or regions

    if params['fixed_costs']:
        params['store_folder'] = 'DA-fixed-case\\'+params['data_source'] # folder to save stuff (do not change)
    else:
        params['store_folder'] = 'DA-case\\'+params['data_source'] # folder to save stuff (do not change)
        
    
    #params['NWP_number'] = 5 #Number of NWP gridpoints to be considered
    #params['Nr_Feat'] = 0 #Number of features per gridpouint

    params['start_date'] = '2019-01-08' # start of train set
    params['split_date'] = '2020-01-01' # end of train set/start of test set
    params['end_date'] = '2020-05-01'# end of test set
    
    params['percentage'] = [.05, .10, .20, .50]  # percentage of corrupted datapoints
    params['iterations'] = 5 # per pair of (n_nodes,percentage)
    params['Alpha'] = 0.01 #Regularization parameter for quantile regression. 
 
    return params

#%% Load data
config = params()

#%%
#Load power production and market data
market_df = pd.read_csv(f'{cd}\\trading-data\\Market_Data_processed.csv', index_col=0, parse_dates=True)
power_df = pd.read_csv(f'{cd}\\trading-data\\VPP_Data_processed.csv', index_col=0, parse_dates=True)["Norm_Power"]

if config['data_source'] == 'regions':
    power_df = pd.read_csv(cd+'\\trading-data\\VPP_Data_processed.csv', index_col=0, parse_dates=True)["Norm_Power"]
elif config['data_source'] == 'smart4res':
    power_df = pd.read_csv(f'{cd}\\data\\smart4res_data\\wind_power_clean_30min.csv', index_col = 0, parse_dates=True)
    # aggregation, normalize by capacity, upscale to 1h
    power_df["Norm_Power"] = power_df.sum(1)/120_000
    power_df = power_df["Norm_Power"]
    metadata_df = pd.read_csv(f'{cd}\\data\\smart4res_data\\wind_metadata.csv', index_col=0)
    
if config['fixed_costs'] == False:
    joint_df = pd.concat([power_df, market_df], axis = 1)
    power_df = joint_df.copy()["Norm_Power"]
    market_df = joint_df.copy()[market_df.columns]

# select NWP grid points
if config['data_source'] == 'regions':
    filenames = ['champs_vert_nwp', 'couturelle_nwp', 'epivent_nwp', 
                 'joncels_nwp', 'petit_terroir__nwp']

    # Load NWP data from different points, concat in a single dataframe
    group_df = []
    
    for i, f in enumerate(filenames):
        temp_df = pd.read_csv( cd+'\\data\\NWPData\\'+f +'.csv', index_col=0, parse_dates=True)
        #Add radian wind direction, squared and cubed wind terms
        temp_df['WindDirection'] = np.sin(np.deg2rad(temp_df["WindDirection"]))
        temp_df['wind_speed_squared'] = temp_df['WindSpeed']**2
        temp_df['wind_speed cubed'] = temp_df['WindSpeed']**3
        new_col_names = []
        
        temp_df.columns
        for c in temp_df.columns:
            new_col_names.append(c+'_'+str(i))
        temp_df.columns = new_col_names
        
        group_df.append(temp_df)

    #config['Nr_Feat'] = len(temp_df.columns)

    nwp_df = pd.concat(group_df, 1)
    config['NWP_number'] = len(filenames)
    config['num_feat_per_point'] = len(temp_df.columns)

elif config['data_source'] == 'smart4res':
    
    park_names = ['p_1003', 'p_1088', 'p_1257', 'p_1475', 'p_1815', 'p_1825']
                  #,'p_1937', 'p_2137', 'p_2204', 'p_2275', 'p_2292', 'p_2419', 'p_2472']

    nwp_df = pd.read_csv( cd+'\\data\\smart4res_data\\nwp_predictions.csv', header=[0, 1], index_col = 0, parse_dates=True).resample('30min').interpolate()
    nwp_df = nwp_df[park_names]
    
    config['NWP_number'] = len(park_names)
    config['num_feat_per_point'] = len(nwp_df[park_names[0]].columns)

# Merge everything the same dates
joint_df = pd.concat([nwp_df, power_df, market_df], axis = 1).dropna()

# Create dataframe 

vpp_df= pd.concat([power_df, nwp_df], axis =1,join = 'inner')

#Add diurnal patterns as fixed indexes 
vpp_df['diurnal1'] = np.sin(2*np.pi*(vpp_df.index.hour+1)/24)
vpp_df['diurnal2'] = np.cos(2*np.pi*(vpp_df.index.hour+1)/24)
vpp_df['diurnal3'] = np.sin(4*np.pi*(vpp_df.index.hour+1)/24)
vpp_df['diurnal4'] = np.cos(4*np.pi*(vpp_df.index.hour+1)/24)

# Joint market data on the same dates
market_df = joint_df.copy()[market_df.columns]

# Hourly resolution
vpp_df = vpp_df.resample('1h').mean()

# Drop NA
vpp_df = vpp_df.dropna(axis = 0, how = 'any')

#%%%%% Create supervised learning sets

## Market parameters

# check if experiment w fixed costs or actual data
if config['fixed_costs']:
    # user-defined costs
    pen_up = config['pen_up']
    pen_down = config['pen_down']

else:
    # estimates of expected (forecast) costs from data    
    pen_up = market_df[:config['split_date']]['Pen_up'].mean()
    pen_down = market_df[:config['split_date']]['Pen_down'].mean()

    # realized costs in train/test set
    pen_up_train = market_df[config['start_date']:config['split_date']]['Pen_up']
    pen_down_train = market_df[config['start_date']:config['split_date']]['Pen_down']
    
    pen_up_test = market_df[config['split_date']:config['end_date']]['Pen_up']
    pen_down_test = market_df[config['split_date']:config['end_date']]['Pen_down']

opt_quant = pen_down/(pen_down + pen_up)


### features and target

# features to include as predictors
features =  [c for c in vpp_df.columns if c !='Norm_Power']

# check with small training set
n_check_obs = -1

xtrain = vpp_df[features][config['start_date']:config['split_date']]
ytrain = vpp_df['Norm_Power'][config['start_date']:config['split_date']]

xtest = vpp_df[features][config['split_date']:config['end_date']]
ytest = vpp_df['Norm_Power'][config['split_date']:config['end_date']]
    
feat_scaler = MinMaxScaler()

xtrain_sc = feat_scaler.fit_transform(xtrain)
xtest_sc = feat_scaler.transform(xtest)

#%% Set-up parameters for missing data


num_nwp_grid = config['NWP_number']
num_feat_per_grid = config['num_feat_per_point']
K_parameter = np.arange(num_nwp_grid+1)

# Fixed and target features by name
fixed_pred = ['diurnal1', 'diurnal2', 'diurnal3', 'diurnal4']
target_pred = [f for f in features if f not in fixed_pred]

# Fixed and target features by column index
#!!!!! sometimes I work with lists instead of numpys
group_col = [np.arange(c*num_feat_per_grid, (c+1)*num_feat_per_grid) for c in range(num_nwp_grid)]
fix_col = [features.index(c) for c in fixed_pred]

#%%%%%%%%%%%%% Train models

##### FDRR-R: reformulation

case_folder = config['store_folder']
output_file_name = f'{cd}\\{case_folder}\\trained-models\\fdrr-r-models.pickle'

#% FDRR-Reformulation
if config['train'] == True: 
    #Train the first model 
    FDR_models_reform = []
    for K in K_parameter:

        print('K = ', K)
        print("Training Reform....")
        fdr = FDR_group_regressor(K = K, p_up = pen_up, p_down = pen_down)
        fdr.fit(xtrain_sc, ytrain, group_col, fix_col, verbose=0, 
                solution = 'reformulation')

        #Create method for evaluating solution 
        FDR_models_reform.append(fdr)
    if config['save']:
        with open(output_file_name, 'wb') as handle:
            pickle.dump(FDR_models_reform, handle)
else:
    with open(output_file_name, 'rb') as handle:    
            FDR_models_reform = pickle.load(handle)

#%%
##### FDRR-AAR: reformulation

output_file_name = f'{cd}\\{case_folder}\\trained-models\\fdrr-aar-models.pickle'

if config['train'] == True: 
    #Train the first model 
    FDR_models_affine = []
    for K in K_parameter:
        print("K = {}".format(K))
        print("Training affine...")
        fdr = FDR_group_regressor(K = K, p_up = pen_up, p_down = pen_down)
        fdr.fit(xtrain_sc, ytrain, group_col, fix_col, verbose=-1, solution = 'affine', 
                alpha = 0)
        #Create method for evaluating solution 
        FDR_models_affine.append(fdr)
    if config['save']:
        with open(output_file_name, 'wb') as handle:
            pickle.dump(FDR_models_affine, handle)
else:
    with open(output_file_name, 'rb') as handle:    
            FDR_models_affine = pickle.load(handle)

# sanity check: inspect coefficients
t = 1
plt.plot(FDR_models_reform[t].coef_, label = 'FDRR-R')
plt.plot(FDR_models_affine[t].coef_, label = 'FDRR-AAR')
plt.xticks(np.arange(xtrain.shape[1]), xtrain.columns, rotation = 45)
plt.legend(fontsize = 6)
plt.show()
#%%
# SAA: only considers in-sample mean, ignores feastures

output_file_name = f'{cd}\\{case_folder}\\trained-models\\saa-model.pickle'

if config['train'] == True: 

    saa = Feature_driven_reg( pen_up = pen_up, pen_down = pen_down, alpha = 0)
    saa.fit(xtrain_sc, ytrain, fit_saa = True)
    if config['save']:
        with open(output_file_name, 'wb') as handle:
            pickle.dump(saa, handle)
else:
    with open(output_file_name, 'rb') as handle:    
            saa = pickle.load(handle)

#%%
# Feature-driven model (no regularization)
output_file_name = f'{cd}\\{case_folder}\\trained-models\\feat-driven-model.pickle'

if config['train'] == True: 
    Fdr = Feature_driven_reg( pen_up = pen_up, pen_down = pen_down, alpha = 0)
    Fdr.fit(xtrain_sc, ytrain)
    print("Feat-driven coef")
    plt.plot(Fdr.coef_)
    plt.show()
    print(Fdr.coef_)
    print("Fdr PREDICT ON TRAIN")
    print(np.max(Fdr.predict(xtrain_sc)))
    if config['save']:
        with open(output_file_name, 'wb') as handle:
            pickle.dump(Fdr, handle)
else:
    with open(output_file_name, 'rb') as handle:    
            Fdr = pickle.load(handle)

#%%
output_file_name = f'{cd}\\{case_folder}\\trained-models\\rf.pickle'
if config['train'] == True: 
    rf = ExtraTreesRegressor(n_estimators = 500)
    rf.fit(xtrain_sc, ytrain)
    if config['save']:
        with open(output_file_name, 'wb') as handle:
            pickle.dump(rf, handle)
else:
    with open(output_file_name, 'rb') as handle:    
            rf = pickle.load(handle)

#%% RETRAIN: trains a feature-driven model for each combination of missing features 

retrain_models, retrain_pred, retrain_comb = retrain_model(xtrain_sc, ytrain.values, xtest_sc,
                                                           pen_up, pen_down, group_col, fix_col, max(K_parameter), base_loss='l1')

#%% L1 regularized feature-driven models

# Feature-driven w l1 regularization w/ resilient-oriented crossvalidation (**proposed**)

output_file_name = f'{cd}\\{case_folder}\\trained-models\\feat-driven-L1-model.pickle'

#Train with optimized regularization value. 
#Do validation for regularization parameter 
#Previous best value R = 0.01

#Set possible regularization constants
list_alpha = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1, 5])

if config['trainReg']:
    
    if config['fixed_costs']:
        best_aplha, val_errors, _ = resilient_crossval_fixed_costs(xtrain_sc, ytrain.values, pen_up, pen_down, group_col, list_alpha, 10_000)
    else:
        best_aplha, val_errors, _ = resilient_crossval_actual_costs(xtrain_sc, ytrain.values, pen_up_train.values, pen_down_train.values, group_col, list_alpha, 10_000)
    
    plt.plot(list_alpha, val_errors)
    plt.xscale('log')
    plt.ylabel("Penalty")
    plt.xlabel("Regularization Constant α")
    plt.show()
    
    Fdr_l1 = Feature_driven_reg( pen_up = pen_up, pen_down = pen_down, alpha = best_aplha)
    Fdr_l1.fit(xtrain_sc, ytrain.values)
    
    if config['save']:
        with open(output_file_name, 'wb') as handle:
            pickle.dump(Fdr_l1, handle)
            
else:
    with open(output_file_name, 'rb') as handle:    
            Fdr_l1 = pickle.load(handle)

#%%%%% Evaluate results out-of-sample
percentage_obs = [0, 0.05, 0.1, 0.25, 0.5, 1]
n_group = len(group_col)
n_test_obs = len(ytest)
num_del_group = np.arange(len(group_col)+1)
iterations = 10

#Create values that just bid the mean: 
Fdr50 = Feature_driven_reg(0.5,0.5,alpha = 0)
Fdr50.fit(xtrain_sc,ytrain.values)


imputation_values = xtrain.mean(0)

models = ['SAA','RF', 'Feature Driven', 'Feature Driven-CV','BASE-retrain'] + ['FDRR_reform'+str(k) for k in K_parameter] + ['FDRR_affine'+str(k) for k in K_parameter]
if config['impute']:
    labels = ['SAA$^{imp}','RF^{imp}' ,'Feature Driven$^{imp}$', 'Feature Driven-CV','Base Retrain'] + ['FDRRref$^'+str(k)+'$' for k in K_parameter] +['FDRRaff$^'+str(k)+'$' for k in K_parameter]
    #labels = ['BASE$^{imp}$', 'BASE$_{\ell_2}^{imp}$', 'BASE$_{\ell_1}^{imp}$', 'RF$^{imp}$', 'RETRAIN'] + ['FDRR$^'+str(k)+'$' for k in K_parameter]
else:
    labels = ['SAA','RF','Feature Driven', 'Feature Driven-CV', 'BASE-retrain'] + ['FDRRref$^'+str(k)+'$' for k in K_parameter]+['FDRRaff$^'+str(k)+'$' for k in K_parameter]
    
mae_df = pd.DataFrame(data = [], columns = models+['feat_del', 'iteration'])

# supress warning
pd.options.mode.chained_assignment = None
run_counter = 0
#%%
for iter_ in range(iterations):
    for m_feat in num_del_group:
        print("m_feat: ",m_feat)
        #OBS!!! Percentage should not change from 1 currently
        for perc in percentage_obs:
            print("percentage: ",perc)
            
            temp_df = pd.DataFrame(data = [perc], columns = ['percentage'])

            #temp_df = pd.concat([pd.DataFrame(), ])temp_df.append({'percentage':perc}, ignore_index=True)
            temp_df['feat_del'] = [m_feat]
            temp_df['iteration'] = iter_
            
            # indices with missing features
            missing_ind = np.random.choice(np.arange(0, n_test_obs), size = int(perc*n_test_obs), replace = False)
            # missing groups of features per indice            
            missing_group = [np.random.choice(np.arange(0, len(group_col)), size = m_feat, replace = False) for i in range(int(perc*n_test_obs))]
            missing_group = np.array(missing_group).reshape(int(perc*n_test_obs), m_feat)
            
            # missing features per index
            missing_feat = []
            for i in range(len(missing_group)):
                temp = []
                for group in missing_group[i]:
                    temp = temp + list(group_col[group])
                missing_feat.append(temp)
            missing_feat = np.array(missing_feat).reshape(int(perc*n_test_obs), m_feat*num_feat_per_grid)
            #Create matrix containing all averages for all missing indexes for that feautre 

            # create missing data            
            miss_X = xtest_sc.copy()
            imp_X = xtest_sc.copy()
            periph_X = xtest_sc.copy() 
            if m_feat > 0:
                for ind, j in zip(missing_ind, missing_feat):
                    miss_X[ind, j] = 0 
                    imp_X[ind, j] = imputation_values[j]
                    #periph_X[ind,j] = 
            
            if config['impute'] != True:
                imp_X = miss_X.copy()
            elif config['impute_peripheral']:
                imp_X = periph_X.copy()

            #Get retrain values 
            
            f_retrain_pred = retrain_pred[:,0:1].copy()
            if m_feat > 0:
                for j, ind in enumerate(missing_ind):
                    temp_group = np.sort(missing_group[j])
                    temp_group = list(temp_group)                    
                    # find position in combinations list
                    j_ind = retrain_comb.index(temp_group)                    
                    f_retrain_pred[ind] = retrain_pred[ind, j_ind]    

            Scenarios = np.array([rf.estimators_[tree].predict(imp_X).reshape(-1) for tree in range(len(rf.estimators_))]).T
            rf_pred = np.quantile(Scenarios, opt_quant, axis = 1).T
            rf_pred = projection(rf_pred)

            saa_pred = saa.predict(imp_X)    

            #Feature-driven
            #if config['scale']:
            #    F_pred = target_scaler.inverse_transform(lasso_pred)    
            Feature_pred = Fdr.predict(imp_X)
            Feature_pred = projection(Feature_pred)
            

            #Feature-driven L1
            #if config['scale']:
            #    F_pred = target_scaler.inverse_transform(lasso_pred)    
            
            feat_dr_l1_pred = Fdr_l1.predict(imp_X)
            feat_dr_l1_pred = projection(feat_dr_l1_pred)
            
            #Meanvals fit 
            Fdr50_pred = Fdr50.predict(imp_X)
            Fdr50_pred = projection(Fdr50_pred)

            # evaluate predictions/ trading decisions
            
            if config['fixed_costs']:
                mean_mae = eval_dual_predictions(Fdr50_pred.reshape(-1), ytest.values, pen_up, pen_down)
                retrain_mae = eval_dual_predictions(f_retrain_pred.reshape(-1), ytest.values, pen_up, pen_down)
                SAA_mae = eval_dual_predictions(saa_pred, ytest.values, pen_up, pen_down)
                feat_dr_l1_mae = eval_dual_predictions(feat_dr_l1_pred, ytest.values, pen_up, pen_down)
                F_mae = eval_dual_predictions(Feature_pred, ytest.values, pen_up, pen_down)
                rf_mae = eval_dual_predictions(rf_pred,ytest.values, pen_up, pen_down)
            else:
                mean_mae = eval_trading_dual(Fdr50_pred.reshape(-1), ytest.values, pen_up_test, pen_down_test)
                retrain_mae = eval_trading_dual(f_retrain_pred.reshape(-1), ytest.values, pen_up_test, pen_down_test)
                SAA_mae = eval_trading_dual(saa_pred, ytest.values, pen_up_test, pen_down_test)
                feat_dr_l1_mae = eval_trading_dual(feat_dr_l1_pred, ytest.values, pen_up_test, pen_down_test)
                F_mae = eval_trading_dual(Feature_pred, ytest.values, pen_up_test, pen_down_test)
                rf_mae = eval_trading_dual(rf_pred,ytest.values, pen_up_test, pen_down_test)
            temp_df['Mean Bid'] = mean_mae
            temp_df['SAA'] = SAA_mae
            temp_df['RF'] = rf_mae
            temp_df['Feature Driven'] = F_mae
            temp_df['Feature Driven-CV'] = feat_dr_l1_mae
            temp_df['BASE-retrain'] = retrain_mae
            
            #mae_df['SAA'][run_counter] = SAA_mae
            #mae_df['Feature Driven'][run_counter] = F_mae
            #mae_df['Feature Driven-L1'][run_counter] = feat_dr_l1_mae
            #mae_df['BASE-retrain'][run_counter] = retrain_mae

            # FDRR-AAR predictions
            
            for i, k in enumerate(K_parameter):
                fdr_pred = FDR_models_affine[i].predict(miss_X)
                fdr_pred = projection(fdr_pred)
            
                # Robust
                #if config['scale']:
                #    fdr_pred = target_scaler.inverse_transform(fdr_pred)
                                    
                fdr_mae = eval_dual_predictions(fdr_pred, ytest.values, pen_up, pen_down)
                #mae_df['FDRR_affine'+str(k)][run_counter] = fdr_mae
                
                if config['fixed_costs']:
                    fdr_mae = eval_dual_predictions(fdr_pred, ytest.values, pen_up, pen_down)
                else:
                    fdr_mae = eval_trading_dual(fdr_pred, ytest.values, pen_up_test, pen_down_test)

                temp_df['FDRR_affine'+str(k)] = fdr_mae
            
            # FDRR-R predictions
            for i, k in enumerate(K_parameter):
                fdr_pred = FDR_models_reform[i].predict(miss_X)
                fdr_pred = projection(fdr_pred)

                # Robust
                #if config['scale']:
                #    fdr_pred = target_scaler.inverse_transform(fdr_pred)
                                    
                if config['fixed_costs']:
                    fdr_mae = eval_dual_predictions(fdr_pred, ytest.values, pen_up, pen_down)
                else:
                    fdr_mae = eval_trading_dual(fdr_pred, ytest.values, pen_up_test, pen_down_test)

                #mae_df['FDRR_reform'+str(k)][run_counter] = fdr_mae
                temp_df['FDRR_reform'+str(k)] = fdr_mae
            
            mae_df = pd.concat([temp_df, mae_df], axis = 0)
            
            run_counter += 1

print("Finished successfully, lets make some plots")

if config['save']: 
    mae_df.to_csv(f'{cd}\\{case_folder}\\results\\DA_trading_results.csv', index = False)
  
#%%
models_to_plot = ['SAA','RF', 'Feature Driven', 'Feature Driven-CV'] +  ['FDRR_reform'+str(k) for k in K_parameter] + ['FDRR_affine'+str(k) for k in K_parameter]
percentage_to_plot = 1
std_bar = mae_df[mae_df['percentage']== percentage_to_plot].groupby(['feat_del'])[models_to_plot].std()
fig,ax = plt.subplots(constrained_layout = True)
mae_df[mae_df['percentage'] == percentage_to_plot].groupby(['feat_del'])[models_to_plot].mean().plot(kind='bar', ax=ax, rot = 0, 
                                                 yerr=std_bar, legend=False)
fig.legend(fontsize=6, ncol=2)
ax.set_ylabel('Mean Penalty [€/MWh]')
ax.set_xlabel('# of deleted groups')
if config['save']: 
    fig.savefig(f'{cd}\\{case_folder}\\figures\\DA-Mean-Penalty-missing.pdf')

#%% Plot: point forecast accuracy versus deleted features

color_list = ['black', 'black', 'gray', 'tab:cyan','tab:green',
         'tab:blue', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive']

marker = ['o', '2', '^','h', 'd', '1', '+', 's', 'v', '*', '^', 'p','+', '1', '2', '3','4','x']
base_colors = plt.cm.tab20c( list(np.arange(3)))
fdr_colors = plt.cm.tab20c([8, 9, 10, 12, 13, 14])
colors = list(base_colors) + ['tab:brown'] + ['tab:orange'] + ['black'] + list(fdr_colors) 

models_to_plot = ['Mean Bid','SAA','RF','Feature Driven', 'Feature Driven-CV','BASE-retrain'] +  ['FDRR_affine'+str(k) for k in K_parameter]
labels = ['Mean Bid','SAA','RF','Feature Driven', 'Feature Driven-CV','BASE-retrain'] + ['FDRR_affine$('+str(k)+')$' for k in K_parameter] +['FDRR_reform('+str(k)+')$' for k in K_parameter]

fig,ax = plt.subplots(constrained_layout = True)
for i, m in enumerate(models_to_plot):
    if m not in models_to_plot: continue
    if m=='BASE-retrain':
        style = 'dashed'
    else: style='solid'
    x_val = mae_df['feat_del'].unique()
    x_val = np.sort(x_val)
    
    values = mae_df[mae_df['percentage']== percentage_to_plot].groupby(['feat_del'])[m].mean()
    std = mae_df[mae_df['percentage']==percentage_to_plot].groupby(['feat_del'])[m].std()
    plt.plot(x_val, values, marker = marker[i], #color = colors[i],
             markersize = 4, label = labels[i], linewidth = 1, linestyle = style)
#plt.xticks(x_val, x_val.astype(int))
    plt.xlabel('# of deleted groups')
    plt.ylabel('Penalty (EUR/MWh)')
plt.legend(loc='upper left')
plt.legend(fontsize=6, ncol=2)
#plt.ylim(0,4)
if config['save']: 
    plt.savefig(f'{cd}\\{case_folder}\\figures\\DA-Penalty-affine-missing.pdf')
plt.show()


#%% Accuracy graph v2: merge all FDRRs in a single line
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 4
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
temp_df = mae_df.copy()
temp_df['FDRR-AAR'] = temp_df[['FDRR_affine'+str(k) for k in K_parameter]].min(axis=1)
temp_df['FDRR-R'] = temp_df[['FDRR_reform'+str(k) for k in K_parameter]].min(axis=1)

fig,ax = plt.subplots(constrained_layout = True)

plot_models = ['Mean Bid','SAA','RF','Feature Driven','BASE-retrain', 'FDRR-R']#, 'FDRR-AAR']    
plot_labels = ['Mean Bid','SAA','RF', 'Feature Driven', 'RETRAIN', 'FDRR-R']#, 'FDRR-AAR']

color_list = ['tab:blue', 'tab:cyan','tab:green',
              'black', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive']

for i, m in enumerate(plot_models):
    if m=='BASE-retrain':
        style = 'dashed'
    else: style='solid'

    x_val = np.sort(temp_df['feat_del'].unique())
    values = temp_df[temp_df['percentage']==percentage_to_plot].groupby(['feat_del'])[m].mean()
    std = temp_df[temp_df['percentage']==percentage_to_plot].groupby(['feat_del'])[m].std()
    # main plot
    plt.plot(x_val, values, marker = marker[i], #color = color_list[i],
             label = plot_labels[i], linestyle = style,
             markersize = 5, linewidth = 1)

#plt.xticks(x_val, x_val.astype(int))
plt.xlabel('# of Deleted Groups of Features')
plt.ylabel('Mean Penalty (EUR/MWh)')
plt.legend(loc='upper left', fontsize = 6, ncol=1)
#plt.xticks(x_val, x_val.astype(int))
plt.xlabel('# of Deleted Groups of Features')
plt.ylabel('Mean Penalty (EUR/MWh)')
plt.legend(loc='upper left', fontsize = 6, ncol=1)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
if config['save']: 
    plt.savefig(f'{cd}\\{case_folder}\\figures\\DA-accuracy-missing.pdf')
plt.show()
#%%
#Sensitivity plot: 
temp_df = mae_df.copy()
temp_df['FDRR-AAR'] = temp_df[['FDRR_affine'+str(k) for k in K_parameter]].min(axis=1)
temp_df['FDRR-R'] = temp_df[['FDRR_reform'+str(k) for k in K_parameter]].min(axis=1)

plot_models = ['SAA','RF','Feature Driven', 'Feature Driven-CV','BASE-retrain', 'FDRR-R', 'FDRR-AAR']    
plot_labels = ['SAA','RF', 'FeatDr', 'FeatDr-CV', 'RETRAIN', 'FDRR-R', 'FDRR-AAR']

color_list = ['tab:blue', 'tab:cyan','tab:green',
              'black', 'tab:brown', 'tab:purple','tab:red', 'tab:orange', 'tab:olive']
std_bar1 = temp_df.groupby(['percentage'])['SAA','RF','Feature Driven', 'Feature Driven-CV','BASE-retrain'].std()
std_bar2 = temp_df.groupby(['percentage'])['FDRR-R', 'FDRR-AAR'].std()
std_bar = pd.concat([std_bar1,std_bar2],ignore_index=False,axis = True)
fig,ax = plt.subplots(constrained_layout = True)
newdf1 = temp_df.groupby(['percentage'])['SAA','RF','Feature Driven', 'Feature Driven-CV','BASE-retrain'].mean()
newdf2 = temp_df.groupby(['percentage'])['FDRR-R', 'FDRR-AAR'].mean()
finaldf = pd.concat([newdf1,newdf2],ignore_index=False,axis = True)

finaldf.plot(kind='bar', ax=ax, rot = 0, 
                                                 yerr=std_bar, legend=False)
fig.legend(fontsize=6, ncol=2)
ax.set_ylabel('Mean Penalty [€/MWh]')
ax.set_xlabel('% of observations deleted')

if config['save']: 
    plt.savefig(f'{cd}\\{case_folder}\\figures\\DA-sensitivity.pdf')
plt.show()
#%%
fig, ax = plt.subplots(constrained_layout = True, figsize = (10,6))
for i, m in enumerate(K_parameter):
    plt.plot(np.append(FDR_models_reform[i].bias_,FDR_models_reform[i].coef_),label= '$\Gamma='+str(K_parameter[i])+'$')
plt.legend()
plt.xticks(range(len(features)+1), list(np.append('Intercept',features)), rotation=90)
#plt.vlines(n_feat-1+0.5, -0.1, 0.6, linestyle = 'dashed', color = 'black')
#plt.ylim(-0.1, .45)
plt.xlabel('Feature')
plt.ylabel('Coeff. magnitude')
plt.title("Reformulation")
plt.legend(fontsize=6, ncol=2)
if config['save']: plt.savefig(cd+'\\figures\\trade_coef_reform_small.pdf')
plt.show()


fig, ax = plt.subplots(constrained_layout = True, figsize = (10,6))
for i, m in enumerate(K_parameter):
    plt.plot(np.append(FDR_models_affine[i].bias_,FDR_models_affine[i].coef_),label= '$\Gamma='+str(K_parameter[i])+'$')
plt.legend()
plt.xticks(range(len(features)+1), list(np.append('Intercept',features)), rotation=90)
#plt.vlines(n_feat-1+0.5, -0.1, 0.6, linestyle = 'dashed', color = 'black')
#plt.ylim(-0.1, .45)
plt.xlabel('Feature')
plt.ylabel('Coeff. magnitude')
plt.title("Affine")
plt.legend(fontsize=6, ncol=2)
if config['save']: plt.savefig(cd+'\\figures\\trade_coef_affine_small.pdf')
plt.show()

fig, ax = plt.subplots(constrained_layout = True, figsize = (10,6))
plt.plot(Fdr.coef_,label = "Feature Driven")
plt.plot(Fdr_l1.coef_,label = "Feature Driven L1")
plt.legend()
plt.xticks(range(len(features)), list(features), rotation=90)
#plt.vlines(n_feat-1+0.5, -0.1, 0.6, linestyle = 'dashed', color = 'black')
#plt.ylim(-0.1, .45)
plt.xlabel('Feature')
plt.ylabel('Coeff. magnitude')
plt.title("Feature Driven Models")
plt.legend(fontsize=6, ncol=2)
if config['save']: plt.savefig(cd+'\\figures\\trade_coef_FD_small.pdf')
plt.show()

#%%
#sns.boxplot([market_df[market_df['Pen_down'] > 0]['Pen_down'],market_df[market_df['Pen_up'] > 0]['Pen_up']], labels = ["Pen Down","Pen UP"])
#plt.yscale('symlog')
#plt.ylabel("Logged values")
#if config['save']: plt.savefig(cd+'\\figures\\Pen_Up_Box.pdf')
#plt.show()

sns.violinplot(data = market_df[market_df[['Pen_up','Pen_down']]>0][['Pen_up','Pen_down']],scale= "count")
#plt.yscale('symlog')
plt.ylabel("Penalty values")
config['save']: plt.savefig(cd+'\\figures\\Pen_Up_Box.pdf')
plt.show()


# %%
