#Imports 
import gurobipy as gp
#from gurobipy import GRB
import numpy as np
import pandas as pd

class MLRreg(object):
    
    '''Feature-driven predictor for the newsvendor problem
    Description....
    alpha = controls l1 regularization
        '''
        
    def __init__(self, pen_up, pen_down, fit_intercept = True, alpha = 0 ):
        # define target quantile, penalization term, and whether to include intercept
        self.pen_up = pen_up
        self.pen_down = pen_down 
        self.fit_intercept = fit_intercept
        self.alpha = alpha 
        
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
        fitted = m.addMVar(rows, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'fitted')
        coef = m.addMVar(cols, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'LDR')
        bias = m.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'bias')
        
        # aux variables
        loss = m.addMVar(rows, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'loss')        
        aux = m.addMVar(cols, vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'aux') # aux for the regularization penalty
        
        ### Constraints
        # fitted values
        m.addConstr( fitted == X@coef + np.ones((rows,1))@bias)

        # linearize loss for each sample
        m.addConstr( loss >= -self.pen_up*(Y - fitted))
        m.addConstr( loss >= self.pen_down*(Y - fitted))
    
        # l1 regularization penalty
        m.addConstr( aux >= coef)
        m.addConstr( aux >= -coef)
        
        #!!!! TBD on this
        #m.addConstr(fitted <=1 )
        #m.addConstr(fitted >= 0)
        
        ### Objective
        m.setObjective((1/rows)*loss.sum()+ self.alpha*aux.sum(), gp.GRB.MINIMIZE)
    
        print('Solving the problem...')
    
        m.optimize()

        self.coef_ = coef.X
        self.bias_ = bias.X
        
        return 
        
    def predict(self, X):
        
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.core.series.Series):
            X = X.copy().values        

        predictions = X@self.coef_ + self.bias_
        return np.array(predictions)

