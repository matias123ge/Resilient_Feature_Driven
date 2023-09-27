#Imports 
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

class SAAreg(object):
    def __init__(self, price_up, price_down, fit_intercept = True):
        # define target quantile, penalization term, and whether to include intercept
        self.price_up =price_up
        self.price_down = price_down 
        self.fit_intercept = fit_intercept
        
    def fit(self, Data, PowerData , verbose = -1, weights = None):
        ''' weights: if None, then 1/nobs, i.e., SAA
            Else, weights is an array with len(Data), derived from a rf model.
            In all cases: weights >=0, sum(weights) = 1'''
        rows, cols = Data.shape            
        if weights == None:
            weights = (1/rows)
        
        y = PowerData
        x = Data
        psiup = self.price_up
        psidw = self.price_down
        # Create a new model
        m = gp.Model("Newsvendor")
        if verbose == -1:
            m.setParam('OutputFlag', 0)
        else:
            m.setParam('OutputFlag', 1)
        # Create variables
        fitted = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fitted')

        # aux variables
        loss = m.addMVar(rows, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')        
    
        ### Constraints
        # fitted values


        # linearize loss for each sample
        m.addConstr( loss >= -psiup*(y - np.ones((rows,1))@fitted))
        m.addConstr( loss >= psidw*(y - np.ones((rows,1))@fitted))

        ### Objective
        
        m.setObjective(weights*loss.sum(), gp.GRB.MINIMIZE)
    
        print('Solving the problem...')
    
        m.optimize()

        self.coef_ = fitted.X
        return 
        
    def predict(self):
        predictions = self.coef_
        return predictions




