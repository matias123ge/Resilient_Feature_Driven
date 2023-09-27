# -*- coding: utf-8 -*-
"""
Generic quantile regression

"""

#Import Libraries
import numpy as np
import gurobipy as gp
import time
import scipy.sparse as sp

class QR_regressor(object):
  '''
  Generic quantile regression using gurobi, resembles the sklearn format.
      '''
  def __init__(self, price_up, price_down,  alpha = 0, fit_intercept = True):
    # define target quantile, penalization term, and whether to include intercept
    self.price_up =price_up
    self.price_down = price_down 
    self.fit_intercept = fit_intercept
    self.alpha = alpha
    
  def fit(self, X, Y, verbose = -1):

    n_train_obs = len(Y)
    n_feat = X.shape[1]

    # target quantile and robustness budget
    p_up   = self.price_up
    p_down = self.price_down
    alpha = self.alpha
    m = gp.Model()
    if verbose == -1:
        m.setParam('OutputFlag', 0)
    else:
        m.setParam('OutputFlag', 1)
        
    print('Setting up GUROBI model...')
    
    ### Problem variables
    # main variables
    fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fitted')
    coef = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
    bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
    # aux variables
    loss = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')        
    aux = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
    
    ### Constraints
    # fitted values
    m.addConstr( fitted == X@coef + np.ones((n_train_obs,1))@bias)
    print("Shape of fit: ",fitted.shape)
    print("Shape of y: ",Y.shape)
    # linearize loss for each sample
    m.addConstr( loss >= -p_up*(Y.reshape(-1) - fitted))
    m.addConstr( loss >= p_down*(Y.reshape(-1) - fitted))
    
    # l1 regularization penalty
    m.addConstr( aux >= coef)
    m.addConstr( aux >= -coef)

    ### Objective
    m.setObjective((1/n_train_obs)*loss.sum() + alpha*aux.sum(), gp.GRB.MINIMIZE)
    
    print('Solving the problem...')
    
    m.optimize()
    self.coef_ = coef.X
    self.bias_ = bias.X
    self.cpu_time = m.Runtime
    return 
    
  def predict(self, X):
    predictions = X@self.coef_ + self.bias_
    return np.array(predictions)