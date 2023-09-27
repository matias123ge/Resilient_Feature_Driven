# -*- coding: utf-8 -*-
"""
Feature Deletion Robust regression with group of variables

"""


#Import Libraries
import numpy as np
import pandas as pd
import itertools
import gurobipy as gp
import time
import scipy.sparse as sp
import matplotlib.pyplot as plt
from QR_regressor import *

class FDR_group_regressor(object):
  '''Initialize the Feature Deletion Robust Regression
  
  Paremeters:
      quant: estimated quantile
      K: number of groups of features that are missing at each sample/ budget of robustness (integer). Special cases:
              - K = 0: standard regression with l1 loss
              - K = len(target_col): all coefficients set to zero, only fit on remaining features.
      target_col: index of columns that can be deleted
      fix_col: index of columns that can be deleted
      approx: select the type of approximation for the robust counterpart problem
          'reformulation': different features are missing at each sample, pessimistic case. 
                          Interpreration: different features missing at different samples, see [2].
          'affine': affinely adjustable robust counterpart, less pessimistic, see [1].
      fit_lb: lower bound on predictions (ignore)
      
      References:
          [1] Gorissen, B. L., & Den Hertog, D. (2013). Robust counterparts of inequalities containing 
           sums of maxima of linear functions. European Journal of Operational Research, 227(1), 30-43.
          [2] Globerson, Amir, and Sam Roweis. "Nightmare at test time: robust learning by feature deletion." 
          Proceedings of the 23rd international conference on Machine learning. 2006.
      '''
  def __init__(self, K = 2, p_up = 0.5, p_down = 0.5, feat_cluster = False):

    self.K = K
    self.p_down = p_down
    self.p_up = p_up
    self.feat_cluster = feat_cluster
    self.solve_multiple = False
        
  def fit(self, X, Y, group_col, fix_col, fit_lb = True, verbose = -1, solution = 'reformulation',alpha = 0):
    '''group col: list of lists, each sublist includes the indexes of each group that is deleted at once'''
    
    total_n_feat = X.shape[1]
    n_train_obs = len(Y)
    if fit_lb == True:
        fit_lower_bound = 0
    else:
        fit_lower_bound = -gp.GRB.INFINITY
    #target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
    #fix_col = [np.where(Predictors.columns == c)[0][0] for c in fixed_pred]
    n_feat = sum([len(g) for g in group_col])
    n_groups = len(group_col)
    all_target_cols = [item for sublist in group_col for item in sublist]
    # number of feat per group (need to scale K accordingly)
    f_per_group = [] 
    K = self.K
    for i in range(K): 
        f_per_group.append(len(group_col[i]))
    f_per_group = sum(f_per_group)
    K_feat = f_per_group
    print("K_feat here: ", K_feat)
    
    ### Create GUROBI model
    m = gp.Model()
    if verbose == -1:
        m.setParam('OutputFlag', 0)
    else:
        m.setParam('OutputFlag', 1)
    
    
    # If data are pandas, transform to numpys
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.core.series.Series):
        X = X.copy().values        

    if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.core.series.Series):
        Y = Y.copy().values        

    # transform to 1D vector
    Y = Y.reshape(-1)

    if (Y==0).all():
        print('Y = 0: skip training')
        self.coef_ = 0
        self.bias_ = 0
        return

    print('Setting up GUROBI model...')

    if K == 0:
        fdr_model = QR_regressor(alpha = alpha, price_up=self.p_up,price_down=self.p_down)    
        fdr_model.fit(X, Y)
        
        self.coef_ = fdr_model.coef_
        self.bias_ = fdr_model.bias_
        self.cpu_time = fdr_model.cpu_time
        
    elif K>0:
        if solution == 'reformulation':
            # Create incidence matrix 
            M = []
            for group in group_col:
                for i in range(len(group[:-1])):
                    temp = np.zeros(n_feat)
                    temp[group[i]] = 1
                    temp[group[i+1]] = -1        
                    M.append(temp)
            M = np.array(M)
            # Different features can be deleted at different samples
            # variables
            fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = fit_lower_bound, name = 'fitted')
            bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
            cost = m.addMVar(1 , vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'cost')
            d = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'residual')
            loss = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
                
            # Dual variables
            ell_up = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_up = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            t_up = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph_aux')
            pi_up = m.addMVar((n_train_obs, M.shape[0]), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

            ell_down = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_down = m.addMVar(X.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            t_down = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph_aux')
            pi_down = m.addMVar((n_train_obs, M.shape[0]), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
                
            # Linear Decision Rules: different set of coefficients for each group
            coef = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
            fix_coef = m.addMVar(len(fix_col), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fixed_coef')
        
            # check to avoid degenerate solutions
            if self.K == n_groups: 
                m.addConstr( coef == 0 )

            start = time.time()

            # Dual Constraints\new
            m.addConstrs( mu_up + ell_up[:,j] + (M.T@pi_up.T)[j,:] >= -X[:,all_target_cols[j]]*coef[j] for j in range(len(all_target_cols)))
            m.addConstr( t_up == K_feat*mu_up + ell_up.sum(1))
        
            m.addConstrs( mu_down + ell_down[:,j] + (M.T@pi_down.T)[j,:] >= X[:, all_target_cols[j]]*coef[j] for j in range(len(all_target_cols)))
            m.addConstr( t_down == K_feat*mu_down + ell_down.sum(1))    
            
            # Dual Constraints\old
            #m.addConstrs( np.ones((n_feat))*mu_up[i] + ell_up[i] + M.T@pi_up[i] >= sp.diags(X[i,all_target_cols])@coef for i in range(n_train_obs))
            #m.addConstrs( t_up[i] == K_feat*mu_up[i] + ell_up[i].sum() for i in range(n_train_obs))
        
            #m.addConstrs( np.ones((n_feat))*mu_down[i] + ell_down[i] + M.T@pi_down[i] >= -sp.diags(X[i, all_target_cols])@coef for i in range(n_train_obs))
            #m.addConstrs( t_down[i] == K_feat*mu_down[i] + ell_down[i].sum() for i in range(n_train_obs))    
                    
            print('Time to declare: ', time.time()-start)
            m.addConstr( fitted == X[:,all_target_cols]@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)
            
            #m.addConstrs( loss[i] >= d[i]@d[i] for i in range(n_train_obs))
            
            print('Solving the problem...')
        
            m.addConstr( loss >= self.p_up*(-Y.reshape(-1) + fitted + t_up))
            m.addConstr( loss >= (self.p_down)*(Y.reshape(-1) - fitted + t_down))
        
            # Objective
            m.setObjective((1/n_train_obs)*loss.sum(), gp.GRB.MINIMIZE)                    
            m.optimize()
            coef_fdr = np.append(coef.X, fix_coef.X)
            
            self.objval = m.ObjVal
            self.coef_ = coef_fdr
            self.bias_ = bias.X
            self.cpu_time = m.Runtime
                    
        elif solution == 'affine':
            # Same features deleted at all samples/ approximation with affinely adjustable robust counterpart
            # Create incidence matrix 
            M = []
            for group in group_col:
                for i in range(len(group[:-1])):
                    temp = np.zeros(n_feat)
                    temp[group[i]] = 1
                    temp[group[i+1]] = -1        
                    M.append(temp)
            M = np.array(M)

            #### Variables
            
            # Primal
            d = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'epigraph')
            fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fitted')
            bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
            coef = m.addMVar((n_feat), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
            fix_coef = m.addMVar(len(fix_col), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fixed_coef')
            
            # Linear decision rules for approximation
            p = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            q = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
            ##### Dual variables
            # Sum of absolute values
            z = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            mu = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = 0)
            pi = m.addMVar(M.shape[0], vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
            # positive part of absolute value per sample
            ell_up = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_up = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            pi_up = m.addMVar((n_train_obs, M.shape[0]), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

            # negative part of absolute value per sample
            ell_down = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_down = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            pi_down = m.addMVar((n_train_obs, M.shape[0]), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            a = m.addMVar((n_train_obs,n_feat),vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1) 
            start = time.time()
    
            #### Contraints
            # check to avoid degenerate solutions
            if self.K == n_groups: 
                print("Degeneracy occurred")
                m.addConstr( coef == 0 )
    
            m.addConstr( d >= p.sum() + mu.sum() + K_feat*z )
            m.addConstr( np.ones((n_feat,1))@z + mu + M.T@pi >= sum(q) )
    
            # Dual Constraints to linearize each sample
            m.addConstr( fitted == X[:,all_target_cols]@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)
            
            # Dual constraints/ new            
            m.addConstrs( mu_up + ell_up[:,j] + (M.T@pi_up.T)[j,:] >= X[:,all_target_cols[j]]*coef[j] - (1/self.p_up)*q[:,j] for j in range(len(all_target_cols)))
            m.addConstrs( mu_down + ell_down[:,j] + (M.T@pi_down.T)[j,:] >= -X[:, all_target_cols[j]]*coef[j] - (1/(self.p_down))*q[:,j] for j in range(len(all_target_cols)))
            m.addConstr( p >= self.p_up*(-Y.reshape(-1) + fitted + K_feat*mu_up + ell_up.sum(1)) )
            m.addConstr( p >= self.p_down*(Y.reshape(-1) - fitted + K_feat*mu_down + ell_down.sum(1)) )
            # Dual constraints/ old
            #m.addConstrs( np.ones((n_feat))*mu_up[i] + ell_up[i] + M.T@pi_up[i] >= X[i,all_target_cols]@coef - (1/self.p_up)*q[i] for i in range(n_train_obs))
            #m.addConstrs( np.ones((n_feat))*mu_down[i] + ell_down[i] + M.T@pi_down[i] >= X[i,all_target_cols]@coef - (1/(1-self.p_down))*q[i] for i in range(n_train_obs))            
            #m.addConstrs( p[i] >= self.p_up*(Y[i] - fitted[i] + K_feat*mu_up[i] + ell_up[i].sum()) for i in range(n_train_obs))
            #m.addConstrs( p[i] >= self.p_down*(-Y[i] + fitted[i] + K_feat*mu_down[i] + ell_down[i].sum()) for i in range(n_train_obs))
            # Objective
            m.setObjective((1/n_train_obs)*d.sum(), gp.GRB.MINIMIZE)
            print('Time to declare: ', time.time()-start)

            print('Solving the problem...')        
            m.optimize()
            
            # store output
            self.objval = m.ObjVal
            coef_fdr = np.zeros(total_n_feat)
            for i, col in enumerate(all_target_cols):
                coef_fdr[col] = coef.X[i]
            for i, col in enumerate(fix_col):
                coef_fdr[col] = fix_coef.X[i]
            self.coef_ = coef_fdr
            self.bias_ = bias.X
            self.cpu_time = m.Runtime
            
  def predict(self, X):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.core.series.Series):
        X = X.copy().values        

    predictions = X@self.coef_ + self.bias_
    return np.array(predictions)
  