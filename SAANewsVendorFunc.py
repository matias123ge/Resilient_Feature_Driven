def SAANewsVendor(Data, PowerData , Priceup, Pricedown):
    #Imports 
    import gurobipy as gp
    from gurobipy import GRB
    import numpy as np
    import pandas as pd
    Power = PowerData
    # Create a new model
    psiup = Priceup
    psidw = Pricedown
    n = gp.Model("NewsvendorSAA")
    n.setParam("OutputFlag",0)
    # Create variables
    ED =n.addMVar(1,vtype=gp.GRB.CONTINUOUS, lb = 0, ub = 1)
    u = n.addMVar(len(Power), vtype = gp.GRB.CONTINUOUS, lb = 0, ub = gp.GRB.INFINITY)
    v = n.addMVar(len(Power), vtype = gp.GRB.CONTINUOUS, lb = 0, ub = gp.GRB.INFINITY)
    fit = n.addMVar(len(Power), vtype = gp.GRB.CONTINUOUS, lb = 0, ub = gp.GRB.INFINITY)
    #Assuming equiprobable 
    n.addConstr(fit == Power-np.ones((len(Power),1))@ED )
    n.addConstr( -u <= fit )
    n.addConstr(fit <= v)
    n.setObjective((1/len(Power))*((psiup)*u.sum()+(psidw)*v.sum()), GRB.MINIMIZE)
    n.optimize()
    #print(m.getObjective().getValue())

    Objective = n.getObjective()
    return Objective.getValue(), ED.X