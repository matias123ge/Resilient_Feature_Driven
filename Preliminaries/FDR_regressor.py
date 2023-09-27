# -*- coding: utf-8 -*-
"""
Feature Deletion Robust regression

"""

#Import Libraries
import numpy as np
import itertools
import gurobipy as gp
import time
import scipy.sparse as sp
import matplotlib.pyplot as plt
from QR_regressor import *
import pandas as pd

class FDR_regressor(object):
  '''Initialize the Feature Deletion Robust Regression
  
  Paremeters:
      quant: estimated quantile
      K: number of features that are missing at each sample/ budget of robustness (integer). Special cases:
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
      
    self.p_up = p_up
    self.p_down = p_down
    self.feat_cluster = feat_cluster
    # For a list of quantiles, declare the mode once and warm-start the solution
    if (type(self.p_up) == list) or (type(self.p_up) == np.ndarray):
        self.solve_multiple = True
    else:
        self.solve_multiple = False
    self.K = K
            
  def fit(self, X, Y, target_col, fix_col, fit_lb = True, verbose = -1, solution = 'reformulation'):

        
    # If data are pandas, transform to numpys
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.core.series.Series):
        X = X.copy().values        

    if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.core.series.Series):
        Y = Y.copy().values        

    # transform to 1D vector
    Y = Y.reshape(-1)

    total_n_feat = X.shape[1]
    n_train_obs = len(Y)
    if fit_lb == True:
        fit_lower_bound = 0
    else:
        fit_lower_bound = -gp.GRB.INFINITY
    #target_col = [np.where(Predictors.columns == c)[0][0] for c in target_pred]
    #fix_col = [np.where(Predictors.columns == c)[0][0] for c in fixed_pred]
    n_feat = len(target_col)

    # loss quantile and robustness budget
    target_quant = self.p_down/(self.p_down+self.p_up)
    K = self.K
    
    ### Create GUROBI model
    m = gp.Model()
    if verbose == -1:
        m.setParam('OutputFlag', 0)
    else:
        m.setParam('OutputFlag', 1)
        
    if (Y==0).all():
        print('Y = 0: skip training')
        if self.solve_multiple == False:
            self.coef_ = np.zeros(X.shape[1])
            self.bias_ = 0
        elif self.solve_multiple == True:
            self.coef_ = []
            self.bias_ = []
            for q in self.quant:
                self.coef_.append(np.zeros(X.shape[1]))
                self.bias_.append(0)

        return

    print('Setting up GUROBI model...')

    if K == 0:
        if self.solve_multiple == False:
            fdr_model = QR_regressor(price_up=self.p_up,price_down=self.p_down)    
            fdr_model.fit(X, Y)
            
            self.coef_ = fdr_model.coef_
            self.bias_ = fdr_model.bias_
            self.cpu_time = fdr_model.cpu_time
        
        elif self.solve_multiple == True:
        
                # Solve for mutliple quantiles w warm start
                self.coef_ = []
                self.bias_ = []        
                for p in self.p_up:
                    print('Quantile: ',(self.p_down/(self.p_down+self.p_up)))

                    fdr_model = QR_regressor(quantile = q)    
                    fdr_model.fit(X, Y)
                    
                    self.coef_.append(fdr_model.coef_)
                    self.bias_.append(fdr_model.bias_)    

        return
    elif K>0:
        if solution == 'reformulation':
            # Different features can be deleted at different samples
            # variables
            fitted = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = fit_lower_bound, name = 'fitted')
            bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
            d = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'residual')
            loss = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')
                
            # Dual variables
            ell_up = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_up = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            t_up = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph_aux')
        
            ell_down = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_down = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            t_down = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'epigraph_aux')
                
            # Linear Decision Rules: different set of coefficients for each group
            coef = m.addMVar(n_feat, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
            fix_coef = m.addMVar(len(fix_col), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fixed_coef')
        
            # check to avoid degenerate solutions
            if K == len(target_col): 
                m.addConstr( coef == 0 )

            start = time.time()
            
            # Dual Constraints-New version
            m.addConstrs( mu_up + ell_up[:,j] >= -X[:,target_col[j]]*coef[j] for j in range(len(target_col)))            
            m.addConstr( t_up == K*mu_up + ell_up.sum(1))
        
            m.addConstrs( mu_down + ell_down[:,j] >= X[:,target_col[j]]*coef[j] for j in range(len(target_col)))
            m.addConstr( t_down == K*mu_down + ell_down.sum(1)) 
                    
            print('Time to declare: ', time.time()-start)
            m.addConstr( fitted == X[:,target_col]@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)
            
            print('Solving the problem...')

            m.addConstr( loss >= self.p_up*(-Y + fitted + t_up))
            m.addConstr( loss >= self.p_down*(Y - fitted + t_down))
        
            # Objective
            m.setObjective((1/n_train_obs)*loss.sum(), gp.GRB.MINIMIZE)                    
            m.optimize()
            
            #coef_fdr = np.append(coef.X, fix_coef.X)
            coef_fdr = np.zeros(total_n_feat)
            for i, col in enumerate(target_col):
                coef_fdr[col] = coef.X[i]
            for i, col in enumerate(fix_col):
                coef_fdr[col] = fix_coef.X[i]

            self.objval = m.ObjVal
            self.coef_ = coef_fdr
            self.bias_ = bias.X
            self.cpu_time = m.Runtime
        
            return 
            
        elif solution == 'affine':
            # Same features deleted at all samples/ approximation with affinely adjustable robust counterpart
            
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
    
            # positive part of absolute value per sample
            ell_up = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_up = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
            # negative part of absolute value per sample
            ell_down = m.addMVar((n_train_obs, n_feat), vtype = gp.GRB.CONTINUOUS, lb = 0)
            mu_down = m.addMVar(n_train_obs, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
            start = time.time()
    
            #### Contraints
            # check to avoid degenerate solutions
            if K == len(target_col): 
                m.addConstr( coef == 0 )
    
            m.addConstr( d >= p.sum() + mu.sum() + K*z )
            m.addConstr( np.ones((n_feat,1))@z + mu >= sum(q) )
            # Dual Constraints to linearize each sample
            m.addConstr( fitted == X[:,target_col]@coef + X[:,fix_col]@fix_coef + np.ones((n_train_obs,1))@bias)
            
            # Dual Constraints/ new version
            m.addConstrs( mu_up + ell_up[:,j] >= X[:,target_col[j]]*coef[j] - (1/self.p_up)*q[:,j] for j in range(len(target_col)))            
            m.addConstrs( mu_down + ell_down[:,j] >= X[:,target_col[j]]*coef[j] - (1/self.p_down)*q[:,j] for j in range(len(target_col)))

            m.addConstr( p >= self.p_up*(-Y + fitted + K*mu_up + ell_up.sum(1)) )
            m.addConstr( p >= self.p_down*(Y - fitted + K*mu_down + ell_down.sum(1)) )

            m.setObjective((1/n_train_obs)*d.sum(), gp.GRB.MINIMIZE)
            print('Time to declare: ', time.time()-start)

            print('Solving the problem...')        
            m.optimize()
            
            # store output
            self.objval = m.ObjVal
            #coef_fdr = np.append(coef.X, fix_coef.X)
            coef_fdr = np.zeros(total_n_feat)
            for i, col in enumerate(target_col):
                coef_fdr[col] = coef.X[i]
            for i, col in enumerate(fix_col):
                coef_fdr[col] = fix_coef.X[i]

            self.coef_ = coef_fdr
            self.bias_ = bias.X
            self.cpu_time = m.Runtime
            
            return

  def predict(self, X):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.core.series.Series):
        X = X.copy()        

    predictions = X@self.coef_ + self.bias_
    return np.array(predictions)
