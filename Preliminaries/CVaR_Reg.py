# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:36:59 2023

"""

#Imports 
import gurobipy as gp
#from gurobipy import GRB
import numpy as np
import pandas as pd

class CVaR_reg(object):
    
    '''Feature-driven predictor for the newsvendor problem
    Description....
    alpha = controls l1 regularization
        '''
        
    def __init__(self, pen_up = 1, pen_down = 1, fit_intercept = True, k = 0.5 , alpha = 0.05 ):
        # define target quantile, penalization term, and whether to include intercept
        self.pen_up = pen_up
        self.pen_down = pen_down 
        self.fit_intercept = fit_intercept
        self.alpha = alpha 
        self.k = k 
    @staticmethod  
    def zero_div(x, y):
        try:
            return x / y
        except ZeroDivisionError:
            return 0

    def fit(self, X, Y, verbose = -1):

        rows, cols = X.shape

        # If data are pandas, transform to numpys
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.core.series.Series):
            X = X.copy().values        
    
        if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.core.series.Series):
            Y = Y.copy().values        
    
        # transform to 1D vector
        Y = Y.reshape(-1)
        
        # Create a new model
        m = gp.Model("Newsvendor")
        if verbose == -1:
            m.setParam('OutputFlag', 0)
        else:
            m.setParam('OutputFlag', 1)
        # Create variables
        u = m.addMVar(rows, vtype = gp.GRB.CONTINUOUS, lb = 0 , ub = gp.GRB.INFINITY)
        beta = m.addMVar(cols, lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY)
        intercept = m.addMVar(1,lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY)
        eta = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY)
        delta = m.addMVar(rows, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = gp.GRB.INFINITY)
        var = m.addMVar(1, lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY)
        m.addConstr( -self.pen_up*(Y-(X@beta+np.ones((rows,1))@intercept)) <= u)
        m.addConstr( self.pen_down * (Y-(X@beta+np.ones((rows,1))@intercept))<= u)
        m.addConstr(0 <= X@beta+np.ones((rows,1))@intercept)
        m.addConstr(  - np.ones((rows,1))@eta + u <= delta) 
        m.addConstr(1 >= X@beta+np.ones((rows,1))@intercept)
        m.addConstr(eta+self.zero_div(1,(1-float(self.alpha)))*(1/rows)*delta.sum()==var)
        m.setObjective((1-self.k)*(1/rows)*u.sum() + self.k*var , gp.GRB.MINIMIZE)
    
        print('Solving the problem...')
    
        m.optimize()
        self.objective_ = m.Objval
        self.VaR = var.X
        self.coef_ = beta.X
        self.bias_ = intercept.X
    
        return 
        
    def predict(self, X):
        
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.core.series.Series):
            X = X.copy().values        

        predictions = X@self.coef_ + self.bias_
        return np.array(predictions)

