#Preliminary study of DA trading 


# -*- coding: utf-8 -*-
"""
Prelininary DA study for features
"""
#%%
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
from itertools import combinations

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)

# import from forecasting libraries

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from CVaR_Reg import*
import seaborn as sns 

os.chdir(os.path.dirname(os.getcwd()))

from utility_functions import * 
from FDR_regressor import *
from Feature_driven_reg import * 
os.chdir(cd)


sns.set() 

# IEEE plot parameters (not sure about mathfont)
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['figure.dpi'] = 600
#plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
#plt.rcParams['font.size'] = 8
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'Times New Roman'
#plt.rcParams["mathtext.fontset"] = 'dejavuserif'
def eval_trades(pred, target,cost_up,cost_down):
    ''' Returns expected (or total) trading cost under dual loss function (quantile loss)'''
    error = target.reshape(-1)-pred.reshape(-1)
    maskup = error < 0
    maskdown = error >0 
    total_cost = (-cost_up*error)*maskup + (cost_down*error)*maskdown
    return total_cost

def eval_var_trades(pred, target,cost_up,cost_down):
    ''' Returns expected (or total) trading cost under dual loss function (quantile loss)'''
    error = target.reshape(-1)-pred.reshape(-1)
    maskup = error < 0
    maskdown = error >0 
    total_cost = (-cost_up*error)*maskup + (cost_down*error)*maskdown
    return total_cost

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

def find_weights(rf_model, trainX, testX):
    ''' estimates weights for weighted SAA'''
    #Step 1: Estimate weights for weighted SAA
    Leaf_nodes = rf_model.apply(trainX) # nObs*nTrees: a_ij shows the leaf node for observation i in tree j
    Index = rf_model.apply(testX) # Leaf node for test set
    nTrees = rf_model.n_estimators
    Weights = np.zeros(( len(testX), len(trainX) ))
    
    #print(Weights.shape)
    #Estimate sample weights
    print('Retrieving weights...')
    for i in range(len(testX)):
        #New query point
        x0 = Index[i:i+1, :]
        #Find observations in terminal nodes/leaves (all trees)
        obs = 1*(x0.repeat(len(trainX), axis = 0) == Leaf_nodes)
        #Cardinality of leaves
        cardinality = np.sum(obs, axis = 0).reshape(-1,1).T.repeat(len(trainX), axis = 0)
        #Update weights
        Weights[i,:] = (obs/cardinality).sum(axis = 1)/nTrees
    return Weights

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

#%%

def params():
    ''' Set up the experiment parameters'''

    params = {}
    params['save'] = True # If True, then saves models and results
    params['trainReg'] = True #Determine best value of regularization for given values. 
    
    # Penalties for imbalance cost (only for fixed prices)
    params['fixed_costs'] = False
    params['pen_up'] = 4 # used only if fixed_costs == True
    params['pen_down'] = 3 # used only if fixed_costs == True
    
    #Show CVaR plots? 
    params['CVAR-Plotting'] =False
    # Parameters for numerical experiment
    params['data_source'] = 'smart4res' 

    if params['fixed_costs']:
        params['store_folder'] = 'DA'# folder to save stuff (do not change)
    else:
        params['store_folder'] = 'DA-case'# folder to save stuff (do not change)
        
    
    #params['NWP_number'] = 5 #Number of NWP gridpoints to be considered
    #params['Nr_Feat'] = 0 #Number of features per gridpouint

    params['start_date'] = '2019-08-01' # start of train set
    params['split_date'] = '2020-01-01' # end of train set/start of test set
    params['end_date'] = '2020-05-01'# end of test set
    
    params['percentage'] = [.05, .10, .20, .50]  # percentage of corrupted datapoints
    params['iterations'] = 5 # per pair of (n_nodes,percentage)
    params['Alpha'] = 0.01 #Regularization parameter for quantile regression. 
 
    return params

#%% Load data
config = params()

#Load power production and market data
market_df = pd.read_csv(f'{cd}\\trading-data\\Market_Data_processed.csv', index_col=0, parse_dates=True)
power_df = pd.read_csv(f'{cd}\\trading-data\\VPP_Data_processed.csv', index_col=0, parse_dates=True)["Norm_Power"]

#Load power production and market data
market_df = pd.read_csv(cd+'\\trading-data\\Market_Data_processed.csv', index_col=0, parse_dates=True)

power_df = pd.read_csv(f'{cd}\\data\\smart4res_data\\wind_power_clean_30min.csv', index_col = 0, parse_dates=True)
# aggregation, normalize by capacity, upscale to 1h
power_df["Norm_Power"] = power_df.sum(1)/120_000
power_df = power_df["Norm_Power"]
metadata_df = pd.read_csv(f'{cd}\\data\\smart4res_data\\wind_metadata.csv', index_col=0)

if config['fixed_costs'] == False:
    joint_df = pd.concat([power_df, market_df], axis = 1)
    power_df = joint_df.copy()["Norm_Power"]
    market_df = joint_df.copy()[market_df.columns]
#%%
# select NWP grid points    
park_names = ['p_1003', 'p_1088', 'p_1257', 'p_1475', 'p_1815', 'p_1825',
                'p_1937', 'p_2137', 'p_2204', 'p_2275', 'p_2292', 'p_2419', 'p_2472']

marketcols = market_df.columns 
market_predictors = ['DA Price_48', 'DA Price_96', 'DA Price_336', 'Penalty_96', 'Volume_96', 'Margin', 'Net Load Forecast']

nwp_df = pd.read_csv( cd+'\\data\\smart4res_data\\nwp_predictions.csv', header=[0, 1], index_col = 0, parse_dates=True).resample('30min').interpolate()
nwp_df = nwp_df[park_names]

config['NWP_number'] = len(park_names)
config['num_feat_per_point'] = len(nwp_df[park_names[0]].columns)

# Merge everything the same dates
joint_df = pd.concat([nwp_df, power_df, market_df], axis = 1).dropna()

# Create dataframes, one with diurnal patterns and squared/cubed windterms one without 
vpp_df= pd.concat([power_df, nwp_df], axis =1,join = 'inner')

basedf = vpp_df.copy()
basedf = basedf[basedf.columns.drop(list(basedf.filter(regex='wspeed_2')))]
basedf = basedf[basedf.columns.drop(list(basedf.filter(regex='wspeed_3')))]


#Add diurnal patterns as fixed indexes 
joint_df['diurnal1'] = np.sin(2*np.pi*(joint_df.index.hour+1)/24)
joint_df['diurnal2'] = np.cos(2*np.pi*(joint_df.index.hour+1)/24)
joint_df['diurnal3'] = np.sin(4*np.pi*(joint_df.index.hour+1)/24)
joint_df['diurnal4'] = np.cos(4*np.pi*(joint_df.index.hour+1)/24)

vpp_df['diurnal1'] = np.sin(2*np.pi*(vpp_df.index.hour+1)/24)
vpp_df['diurnal2'] = np.cos(2*np.pi*(vpp_df.index.hour+1)/24)
vpp_df['diurnal3'] = np.sin(4*np.pi*(vpp_df.index.hour+1)/24)
vpp_df['diurnal4'] = np.cos(4*np.pi*(vpp_df.index.hour+1)/24)


# Joint market data on the same dates
market_df = joint_df.copy()[market_df.columns]

joint_df = joint_df.resample('1h').mean()
# Hourly resolution
vpp_df = vpp_df.resample('1h').mean()

#%%
# Drop NA
joint_df = joint_df.dropna(axis = 0, how = 'any')
vpp_df = vpp_df.dropna(axis = 0, how = 'any')

market_df = joint_df.copy()[market_df.columns]

#Base scenarios NWP 
vpp_features = [c for c in vpp_df.columns if c != 'Norm_Power']
#NWP scenario with feature engineering
vpp_base_features = [c for c in basedf.columns if c!= 'Norm_Power']
#NWP and Market with feature engineering
vpp_market_features = vpp_features+market_predictors
#We are ready to investigate the data, and compare different modelling approaches 

# Features for random forest: based features and calendar variables
rf_features = vpp_base_features + ['Day', 'Hour', 'Minute']
#Heatmap of the original data
#%%
#Show correlation plots of market 
fig, ax  =plt.subplots( nrows=1, ncols=1 )
ax = sns.heatmap(joint_df[market_predictors].corr(), annot=True,linewidth=0.75, cmap = "crest")
ax.set_xlabel("")
ax.set_ylabel("")
fig.tight_layout()
if config['save']: 
    fig.savefig(f'{cd}\\DA\\Figures\\Market_heatmap.pdf',bbox_inches="tight")
plt.show()

#%%
#Show correlation plot of parks
fig, ax  =plt.subplots( nrows=1, ncols=1 )
ax = sns.heatmap(joint_df[vpp_base_features[0:(2*(config['num_feat_per_point']-2))]].corr(), annot=True,linewidth=0.75, cmap = "crest")
ax.set_xlabel("")
ax.set_ylabel("")
fig.tight_layout()
if config['save']: 
    fig.savefig(f'{cd}\\DA\\Figures\\NWP_heatmap.pdf',bbox_inches="tight")
plt.show()


#%%
#Penalty configuration 
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

#Calculate optimal quantile for sanity check 
opt_quant = pen_down/(pen_down + pen_up)

# features to include as predictors

# Data size configution
n_check_obs = -1

#Call transform
feat_scaler = MinMaxScaler()
#Create testing and training dataset for normal data
xtrain = joint_df[config['start_date']:config['split_date']]
ytrain = joint_df['Norm_Power'][config['start_date']:config['split_date']]

xtest = joint_df[config['split_date']:config['end_date']]
ytest = vpp_df['Norm_Power'][config['split_date']:config['end_date']]
xtrain_sc = feat_scaler.fit_transform(xtrain)
xtest_sc = feat_scaler.transform(xtest)

#Convert to dataframes 
#%%
xtrain_sc  = pd.DataFrame(xtrain_sc,index = xtrain.index, columns = xtrain.columns)
xtest_sc  = pd.DataFrame(xtest_sc,index = xtest.index, columns = xtrain.columns)

# %%
#CVAR calibration
if config["CVAR-Plotting"]: 
    Testk = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    Testalpha = [0.05,0.25,0.5,0.75,0.95]
    CVARK50= []
    CVARK75 = []
    CVARK95 = [] 
    VaRK95  = []
    VaRALPHA95 = [] 
    CVARAlpha95 = []
    CVARalph50 = []
    CVARalph75 = []
    CVARalph95 = []  
    for i in range(len(Testk)): 
        Fdr_CVaR =  CVaR_reg( pen_up = pen_up, pen_down = pen_down, k =Testk[i],alpha =0.5 )
        Fdr_CVaR.fit(xtrain_sc[vpp_features], ytrain)
        CVARK50.append(Fdr_CVaR.objective_)
        Fdr_CVaR =  CVaR_reg( pen_up = pen_up, pen_down = pen_down, k =Testk[i],alpha =0.75 )
        Fdr_CVaR.fit(xtrain_sc[vpp_features], ytrain)
        CVARK75.append(Fdr_CVaR.objective_)
        Fdr_CVaR =  CVaR_reg( pen_up = pen_up, pen_down = pen_down, k =Testk[i],alpha =0.95 )
        Fdr_CVaR.fit(xtrain_sc[vpp_features], ytrain)
        CVARK95.append(Fdr_CVaR.objective_)
        VaRK95.append(Fdr_CVaR.VaR[0])


    for i in range(len(Testalpha)): 
        Fdr_CVaR =  CVaR_reg( pen_up = pen_up, pen_down = pen_down, k = 0.25,alpha =Testalpha[i] )
        Fdr_CVaR.fit(xtrain_sc[vpp_features], ytrain)
        CVARalph50.append(Fdr_CVaR.objective_)
        Fdr_CVaR =  CVaR_reg( pen_up = pen_up, pen_down = pen_down, k =0.5,alpha =Testalpha[i] )
        Fdr_CVaR.fit(xtrain_sc[vpp_features], ytrain)
        CVARalph75.append(Fdr_CVaR.objective_)
        Fdr_CVaR =  CVaR_reg( pen_up = pen_up, pen_down = pen_down, k =0.75,alpha  = Testalpha[i] )
        Fdr_CVaR.fit(xtrain_sc[vpp_features], ytrain)
        CVARalph95.append(Fdr_CVaR.objective_)
        VaRALPHA95.append(Fdr_CVaR.VaR[0])
    #Show bid distriburion and get efficient frontier
    # 
    if config['fixed_costs'] == False: 
        pen_up = np.minimum(pen_up,np.quantile(pen_up,0.80))
        pen_down = np.minimum(pen_down,np.quantile(pen_down,0.80)) 
    Fdr_CVaR =  CVaR_reg( pen_up = pen_up, pen_down = pen_down, k =0.5,alpha = 0.95 )
    Fdr_CVaR.fit(xtrain_sc[vpp_features], ytrain)

    Fdr =  Feature_driven_reg( pen_up = pen_up, pen_down = pen_down, alpha=0)
    Fdr.fit(xtrain_sc[vpp_features], ytrain)


    predictsCVaR = Fdr_CVaR.predict(xtest_sc[vpp_features])
    predictsCVaR =  projection(predictsCVaR)
    predictsFdr = Fdr.predict(xtest_sc[vpp_features])
    predictsFdr =  projection(predictsFdr)
    if config['fixed_costs'] == False: 
        CVaR_pen = eval_var_trades(predictsCVaR,ytest.values,pen_up_test,pen_down_test)
        Fdr_pen = eval_var_trades(predictsFdr,ytest.values,pen_up_test,pen_down_test)
    else:
        CVaR_pen = eval_var_trades(predictsCVaR,ytest.values,pen_up,pen_down)
        Fdr_pen = eval_var_trades(predictsFdr,ytest.values,pen_up,pen_down)
    #Sensitivity plots
    sns.set() 
    plt.plot(CVARK50,Testk)
    plt.plot(CVARK75,Testk)
    plt.plot(CVARK95,Testk)
    plt.legend(["α = 0.5","α = 0.75","α = 0.95"],ncol =3)
    plt.title("CVaR plot, sensitivity to k-value")
    plt.xlabel("Cost [€/MWh]")
    plt.ylabel("K-value")
    if config['save']: 
        plt.savefig(f'{cd}\\DA\\Figures\\KsensPlot.pdf',bbox_inches="tight")


    plt.plot(CVARalph50,Testalpha)
    plt.plot(CVARalph75,Testalpha)
    plt.plot(CVARalph95,Testalpha)
    plt.legend(["k = 0","k=0.5" ," k = 1"],ncol= 3)
    plt.title("CVaR plot, sensitivity to probability")
    plt.xlabel("Cost [€/MWh]")
    plt.ylabel("Probability level α")
    if config['save']: 
        plt.savefig(f'{cd}\\DA\\Figures\\AsensPlot.pdf',bbox_inches="tight")
    plt.show()
    #Make plot of histogram of out of sample costs 
    bins = np.linspace(0, 2, 20)
    #plt.hist(powervalDD05,bins,alpha=0.5,label = "CVaR")
    plt.hist(CVaR_pen[CVaR_pen != 0],bins,alpha=0.5,label = "CVaR")
    #plt.hist(powerdiffDD,bins,alpha=0.25,label = "Datadriven")
    plt.hist(Fdr_pen[ Fdr_pen != 0],bins,alpha=0.5,label = "Feature Driven")
    plt.title("Distribution of penalties incurred")
    plt.xlabel("Penalty Values")
    plt.ylabel("Nr. of Observations")
    plt.legend(loc = 'upper right',ncol= 1)
    if config['save']: 
        plt.savefig(f'{cd}\\DA\\Figures\\PenaltyDistCvaR.pdf',bbox_inches="tight")
    plt.show()

    plt.scatter(VaRK95,CVARK95)
    plt.title("Efficient Frontier, α = 0.95 ")
    plt.xlabel("CVaR [€/MWh]")
    plt.ylabel("Total Cost [€/MWh]")

    if config['save']: 
        plt.savefig(f'{cd}\\DA\\Figures\\EfficientFront_FD.pdf',bbox_inches="tight")
    plt.show()

#Lets train models and compare 
'''
POSSIBLE SCENARIOS AND CORRESPONDING MODELS: 

Fixed price: 
Random forest
Feature driven 50th (VPP dataset)
Feature driven 50th (VPP dataset with feature engineering)
Feature Driven 50th l1 regularized

Var price: 
SAA 
Random forest
Feature driven VPP dataset with feature engineering   
Feature driven VPP/Market dataset w. feature engineering
CVaR Program 
Feature Driven VPP l1 regularized 
'''
#%%
if config['fixed_costs']: 
    models = ["FD", "FD-Feature Engineered","RF","FD-l1"]
    #train relevant models 
    #Avg bid (no feature engineering )
    Fdrbase50 = Feature_driven_reg(2,2,0)
    Fdrbase50.fit(xtrain_sc[vpp_base_features],ytrain)
    Fdrbase50Pred = Fdrbase50.predict(xtest_sc[vpp_base_features])
    #Avg bid (feature engineering)
    Fdr50 = Feature_driven_reg(2,2,0)
    Fdr50.fit(xtrain_sc[vpp_features].values,ytrain.values)
    Fdr50Pred = Fdr50.predict(xtest_sc[vpp_features])
    print("HERE MOTHERFUCKER")
    
    rf_predictors = vpp_base_features = [c for c in basedf.columns if c!= 'Norm_Power']
    rf = ExtraTreesRegressor(n_estimators = 1000)
    rf.fit(xtrain[rf_features], ytrain)
    #!!!! fix weighted SAA
    Scenarios = np.array([rf.estimators_[tree].predict(xtest[rf_features]).reshape(-1) for tree in range(len(rf.estimators_))]).T
    rfpred = np.quantile(Scenarios, opt_quant, axis = 1).T
    #Regularized
    param_grid = {"alpha": [10**pow for pow in range(-10,10)]}
    lasso = GridSearchCV(Lasso(fit_intercept = True, max_iter = 1_000), param_grid)
    lasso.fit(xtrain_sc[vpp_features], ytrain) 
    FdrReg = Feature_driven_reg(2,2,alpha = lasso.best_params_['alpha'])
    FdrReg.fit(xtrain_sc[vpp_features],ytrain)
    FdrRegPred = FdrReg.predict(xtest_sc[vpp_features])
    #Get individual trade differences for each sample and average penalty
    #Base 
    FdrBase50Vals = eval_trades(Fdrbase50Pred,ytest.values,pen_up,pen_down)
    #Base feature engineered 
    Fdr50Vals = eval_trades(Fdr50Pred,ytest.values,pen_up,pen_down)
    #Random forest
    rfVals =  eval_trades(rfpred,ytest.values,pen_up,pen_down)
    #Regularized: 
    FdrRegvals=  eval_trades(FdrRegPred,ytest.values,pen_up,pen_down)
    values_df = pd.DataFrame([],columns = models)
    values_df.iloc[:,0] = FdrBase50Vals 
    values_df.iloc[:,1] = Fdr50Vals 
    values_df.iloc[:,2] = rfVals 
    values_df.iloc[:,3] = FdrRegvals

    #Make barplot of mean trade penalty with 95% confidence interval
    fig,ax  = plt.subplots(ncols = 1, nrows =1) 
    ax  = sns.barplot(data = values_df)
    ax.set_ylabel("Mean Penalty (€/MWh)")
    ax.set_xticklabels(models)
    if config['save']:
        ax.figure.savefig(f'{cd}\\DA\\Figures\\Fixed_cost_bar.pdf',bbox_inches="tight")
    plt.show()
    #Make cumulative value plots: 
    models_to_plot = ["FD", "FD-Feature Engineered","RF"]
    for i in models_to_plot: 
        plt.plot(range(len(values_df.index)),np.cumsum(values_df[i].values))
    plt.legend(models_to_plot,ncol=1)
    plt.title("Model Comparisons")
    plt.xlabel("Nr. Samples")
    plt.ylabel("Cumulative penalty cost (€/MWh)")
    if config['save']:
        plt.savefig(f'{cd}\\DA\\Figures\\Fixed_cost_cumulative.pdf',bbox_inches="tight")
    plt.show()
    if config['save']:
        df = values_df.mean()
        df = pd.concat([df,values_df.std()],axis = 1)
        df.to_csv(f'{cd}\\DA\\Results\\Fixed_costs_results.csv')
else: 
    models = ["SAA","FD","FD Market Features","RF","RF Market Features","FD-L1","CVaR"]
    #train relevant models 
    Saa  = Feature_driven_reg(pen_up,pen_down,0)
    Saa.fit(xtrain_sc[vpp_features].values,ytrain.values, fit_saa = True )
    SaaPred = Saa.predict(xtest_sc[vpp_features].values)
    SaaPred = projection(SaaPred)
    #Feature Driven Feature Engineered 
    Fdr = Feature_driven_reg(pen_up,pen_down,0)
    Fdr.fit(xtrain_sc[vpp_features].values,ytrain.values)
    FdrPred = Fdr.predict(xtest_sc[vpp_features].values)
    FdrPred = projection(FdrPred)
    #Feature Driven Feature Engineered 
    Fdr = Feature_driven_reg(pen_up,pen_down,0)
    Fdr.fit(xtrain_sc[vpp_features].values,ytrain.values)
    FdrPred = Fdr.predict(xtest_sc[vpp_features].values)
    FdrPred = projection(FdrPred)
    #Feature Driven Market Features + Engineered
    FdrMarket = Feature_driven_reg(pen_up,pen_down,0)
    FdrMarket.fit(xtrain_sc[vpp_market_features].values,ytrain.values)
    FdrMarketPred = FdrMarket.predict(xtest_sc[vpp_market_features].values)
    FdrMarketPred = projection(FdrMarketPred)

    #Random Forest 
    rf = ExtraTreesRegressor(n_estimators = 300)
    rf.fit(xtrain[vpp_features], ytrain)
    Scenarios = np.array([rf.estimators_[tree].predict(xtest[vpp_features]).reshape(-1) for tree in range(len(rf.estimators_))]).T
    rfpred = np.quantile(Scenarios, opt_quant, axis = 1).T
    rfpred = projection(rfpred)

    #Random forest Market features 
    rfmarket = ExtraTreesRegressor(n_estimators = 300)
    rfmarket.fit(xtrain_sc[vpp_market_features], ytrain)
    Scenarios = np.array([rfmarket.estimators_[tree].predict(xtest_sc[vpp_market_features]).reshape(-1) for tree in range(len(rfmarket.estimators_))]).T
    rfmarketpred = np.quantile(Scenarios, opt_quant, axis = 1).T
    rfmarketpred = projection(rfmarketpred)
    #Regularized l1 with market features? 
    param_grid = {"alpha": [10**pow for pow in range(-10,10)]}
    lasso = GridSearchCV(Lasso(fit_intercept = True, max_iter = 1_000), param_grid)
    lasso.fit(xtrain_sc[vpp_market_features].values, ytrain.values) 
    FdrReg = Feature_driven_reg(pen_up,pen_down,alpha = lasso.best_params_['alpha'])
    FdrReg.fit(xtrain_sc[vpp_market_features].values,ytrain.values)
    FdrRegPred = FdrReg.predict(xtest_sc[vpp_market_features].values)
    FdrRegPred =  projection(FdrRegPred)

    #CVar program (sensitivity plots further down? )
    Fdr_CVaR =  CVaR_reg( pen_up = pen_up, pen_down = pen_down, k =0.9,alpha =0.50)
    Fdr_CVaR.fit(xtrain_sc[vpp_features].values, ytrain.values)
    predictsCVaR = Fdr_CVaR.predict(xtest_sc[vpp_features].values)
    predictsCVaR =  projection(predictsCVaR)

    #Get individual trade differences for each sample and average penalty
    #Saa 
    SaaVals = eval_var_trades(SaaPred,ytest.values,pen_up_test,pen_down_test)
    #Feature driven 
    FDVals = eval_var_trades(FdrPred,ytest.values,pen_up_test,pen_down_test)
    #Feature driven market features 
    FDMarketVals = eval_var_trades(FdrMarketPred,ytest.values,pen_up_test,pen_down_test)
    #Random forest
    rfVals = eval_var_trades(rfpred,ytest.values,pen_up_test,pen_down_test)
    #Random forest market features 
    rfmarketVals = eval_var_trades(rfmarketpred,ytest.values,pen_up_test,pen_down_test)
    #Regularized l1 with market features 
    FdrRegVals = eval_var_trades(FdrRegPred,ytest.values,pen_up_test,pen_down_test)

    CVaRVals = eval_var_trades(predictsCVaR,ytest.values,pen_up_test,pen_down_test)
    #Create dataframe of values
    values_varpens_df = pd.DataFrame([],columns = models)
    values_varpens_df.iloc[:,0] = SaaVals 
    values_varpens_df.iloc[:,1] = FDVals
    values_varpens_df.iloc[:,2] = FDMarketVals
    values_varpens_df.iloc[:,3] = rfVals
    values_varpens_df.iloc[:,4] = rfmarketVals
    values_varpens_df.iloc[:,5] = FdrRegVals
    values_varpens_df.iloc[:,6] = CVaRVals
    #Mean plot of predictions 
    fig,ax  = plt.subplots(ncols = 1, nrows =1) 
    ax = sns.barplot(data = values_varpens_df)
    ax.set_ylabel("Mean Penalty (€/MWh)")
    ax.set_xticklabels(models, rotation=80)
    if config['save']:
        fig.savefig(f'{cd}\\DA\\Figures\\Var_cost_bar.pdf',bbox_inches="tight")
    plt.show()
    #Models to plot: 
    models_to_plot = ['SAA','FD','RF Market Features', 'CVaR']
    for i in models_to_plot: 
        plt.plot(range(len(values_varpens_df.index[0:100])),np.cumsum(values_varpens_df[i].values[0:100]))
    plt.legend(models_to_plot,ncol=1)
    plt.title("Model Comparisons")
    plt.xlabel("Nr. Samples")
    plt.ylabel("Cumulative penalty cost (€/MWh)")
    if config['save']:
        plt.savefig(f'{cd}\\DA\\Figures\\Var_cost_cumulative.pdf',bbox_inches="tight")
    plt.show()
    if config['save']:
        df = values_varpens_df.mean()
        df = pd.concat([df,values_varpens_df.std()],axis = 1)
        df.to_csv(f'{cd}\\DA\\Results\\Variable_costs_results.csv')

# %%
#Compare in sample mean vs individual 
pen_up_train = market_df[config['start_date']:config['split_date']]['Pen_up']
pen_down_train = market_df[config['start_date']:config['split_date']]['Pen_down']
FDR1 = Feature_driven_reg(pen_up = pen_up_train.mean(), pen_down = pen_down_train.mean() )
FDR1.fit(xtrain_sc,ytrain,fit_saa = True)
FDR2 = Feature_driven_reg(pen_up = pen_up_train.values.reshape(-1), pen_down = pen_down_train.values.reshape(-1))
FDR2.fit(xtrain_sc,ytrain,fit_saa = True)

meanvalspen = FDR1.predict(xtest_sc)
varvalspen = FDR2.predict(xtest_sc)
#%%
meanvals = eval_trades(meanvalspen,ytest.values,pen_up_test.values,pen_down_test.values)
varvals = eval_trades(varvalspen,ytest.values,pen_up_test.values,pen_down_test.values)

plt.plot(range(len(meanvals)),np.cumsum(meanvals),linestyle ='dotted',linewidth =2)
plt.plot(range(len(varvals)),np.cumsum(varvals),linewidth = 1)
plt.legend(["In Sample Mean","Variable Price"],ncol=1)
plt.title("Model Comparisons")
plt.xlabel("Nr. Samples")
plt.ylabel("Cumulative penalty cost (€/MWh)")
plt.savefig(f'{cd}\\DA\\Figures\\MeanvsVar.pdf')
#%%
#rf = ExtraTreesRegressor(n_estimators = 1000)
#rf.fit(xtrain[rf_features], ytrain)
#!!!! fix weighted SAA
#Scenarios = np.array([rf.estimators_[tree].predict(xtest[rf_features]).reshape(-1) for tree in range(len(rf.estimators_))]).T
#
#y_sort = np.sort(ytrain.copy().values.reshape(-1))
#ind_sort = np.argsort(ytrain.copy().values.reshape(-1))
#
#w = find_weights(rf, xtrain[rf_features], xtest[rf_features])
#
#rf_bid = []
#
#for i in range(len(xtest)):
#    w0 = w[i:i+1]
#    # analytical solution for the newsvendor problem (weighted SAA)
#    y_bid = y_sort[np.where(w0[:,ind_sort].cumsum() > opt_quant)[0][0]]
#    
#    rf_bid.append(y_bid)
#rf_bid = np.array(rf_bid)
#
##%%
#y_sort = np.sort(ytrain.copy().values.reshape(-1))
#ind_sort = np.argsort(ytrain.copy().values.reshape(-1))
#
#
## for an out-of-sample observation x0
#x0 = xtest[rf_features][50:51]
## find weights of train observations
#w0 = find_weights(rf, xtrain[rf_features], x0)
## analytical solution for the newsvendor problem (weighted SAA)
#
#y_bid = y_sort[np.where(w0[:,ind_sort].cumsum() > quant)[0][0]]
#
##%%    
#print('Optimizing Prescriptions...')
##Check that weigths are correct (numerical issues)
#assert( all(Weights.sum(axis = 1) >= 1-10e-4))
#assert( all(Weights.sum(axis = 1) <= 1+10e-4))
#Prescription = []#np.zeros((testX.shape[0],1))
#for i in range(len(testX)):
#    if i%250 == 0:
#        print('Observation ', i) 
#    mask = Weights[i]>0
#    _, temp_prescription = opt_problem(trainY[mask], weights = Weights[i][mask], prescribe = True, **self.decision_kwargs)
#    #Prescription[i] = temp_prescription     
#    Prescription.append(temp_prescription)  
#       
#
##%%