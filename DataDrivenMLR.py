def DataDrivenNewsVendor(Data, PowerData , Priceup, Pricedown):
    #Imports 
    import gurobipy as gp
    from gurobipy import GRB
    import numpy as np
    import pandas as pd

    rows, cols = Data.shape
    Power = PowerData
    x = Data
    psiup = Priceup
    psidw = Pricedown
    # Create a new model

    linreg = gp.Model("Newsvendor")
    linreg.setParam('OutputFlag',0)
    # Create variables
    u = linreg.addMVar(len(Power), vtype = gp.GRB.CONTINUOUS, lb = 0, ub = gp.GRB.INFINITY)
    fit = linreg.addMVar(len(Power),vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY)
    beta = linreg.addMVar(cols, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY)
    intercept = linreg.addMVar(1, vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY)
    linreg.setObjective((1/len(Power))*u.sum(), GRB.MINIMIZE)
    linreg.addConstr(fit == x@beta+np.ones((rows,1))@intercept)
    #linreg.addConstr(0 <= fit)
    #linreg.addConstr(np.max(Power) >= fit)
    linreg.addConstr(-psiup*(Power-fit) <=u )
    linreg.addConstr(psidw*(Power-fit) <=u)
    linreg.optimize() 
    Objective = linreg.getObjective()
    print("coefficients: {}".format(beta.X))
    return Objective.getValue(), beta.X, intercept.X

