# -*- coding: utf-8 -*-
"""
Utility functions (probably not used all)

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.interpolation import shift
from statsmodels.tsa.stattools import pacf, acf
from scipy.ndimage.interpolation import shift

def lagged_predictors_pd(df, col_name, freq, d = 200, thres = .1, intraday = True):
    'Input dataframe, creates lagged predictors of selected column based on PACF'
    PACF = pacf(df[col_name], nlags = d)
    ACF = acf(df[col_name], nlags = d)

    plt.plot(PACF, label = 'PACF')
    plt.plot(ACF, label = 'ACF')
    plt.show()
    
    #Lags = np.argwhere(abs(PACF) > thres) - 1
    Lags = np.where(abs(PACF)>=thres)[0][1:]
    if intraday == False:
        Lags = Lags[Lags> (int(freq*24)-1) ]
    name = col_name+'_'
    name_list = []
    for lag in Lags:
        temp_name = name+str(int(1//freq)*lag)
        df[temp_name] = shift(df[col_name], lag)
        name_list.append(temp_name)
    return df, name_list

def eval_point_pred(predictions, actual, digits = None):
    ''' Evaluates determinstic forecasts
        Outputs: MAPE, RMSE, MAE'''
    predictions = predictions.copy().reshape(-1)
    actual = actual.copy().reshape(-1)
    mape = np.mean(abs( (predictions-actual)/actual) )
    rmse = np.sqrt( np.mean(np.square( predictions-actual) ) )
    mae = np.mean(abs(predictions-actual))
    if digits is None:
        return mape,rmse, mae
    else: 
        return round(mape, digits), round(rmse, digits), round(mae, digits)
    
def pinball(prediction, target, quantiles):
    ''' Evaluates Probabilistic Forecasts, outputs Pinball Loss for specified quantiles'''
    num_quant = len(quantiles)
    pinball_loss = np.maximum( (np.tile(target, (1,num_quant)) - prediction)*quantiles,(prediction - np.tile(target , (1,num_quant) ))*(1-quantiles))
    return pinball_loss  

def CRPS(target, quant_pred, quantiles, digits = None):
    ''' Evaluates Probabilistic Forecasts, outputs CRPS'''
    n = len(quantiles)
    #Conditional prob
    p = 1. * np.arange(n) / (n - 1)
    #Heaviside function
    H = quant_pred > target 
    if digits == None:
        return np.trapz( (H-p)**2, quant_pred).mean()
    else:
        return round(np.trapz( (H-p)**2, quant_pred).mean(), digits)
    
def pit_eval(target, quant_pred, quantiles, plot = False, nbins = 20):
    '''Evaluates Probability Integral Transformation
        returns np.array and plots histogram'''
    #n = len(target)
    #y = np.arange(1, n+1) / n
    y = quantiles
    PIT = [ y[np.where(quant_pred[i,:] >= target[i])[0][0]] if any(quant_pred[i,:] >= target[i]) else 1 for i in range(len(target))]
    PIT = np.asarray(PIT).reshape(len(PIT))
    if plot:
        plt.hist(PIT, bins = nbins)
        plt.show()
    return PIT

def reliability_plot(target, pred, quantiles, boot = 100, label = None):
    ''' Reliability plot with confidence bands'''
    cbands = []
    for j in range(boot):
        #Surgate Observations
        Z = np.random.uniform(0,1,len(pred))
        
        Ind = 1* (Z.reshape(-1,1) < np.tile(quantiles,(len(pred),1)))
        cbands.append(np.mean(Ind, axis = 0))
    
    ave_proportion = np.mean(1*(pred>target), axis = 0)
    cbands = 100*np.sort( np.array(cbands), axis = 0)
    lower = int( .05*boot)
    upper = int( .95*boot)
 
    ave_proportion = np.mean(1*(pred>target), axis = 0)
    plt.vlines(100*quantiles, cbands[lower,:], cbands[upper,:])
    plt.plot(100*quantiles,100*ave_proportion, '-*')
    plt.plot(100*quantiles,100*quantiles, '--')
    plt.legend(['Observed', 'Target'])
    plt.show()
    return

def brier_score(predictions, actual, digits = None):
    ''' Evaluates Brier Score''' 
    if digits == None:
        return np.mean(np.square(predictions-actual))
    else:
        return round(np.mean(np.square(predictions-actual)), digits)
    
def VaR(data, quant = .05, digits = 3):
    ''' Evaluates Value at Risk at quant-level'''
    if digits is None:
        return np.quantile(data, q = quant)
    else:
        return round(np.quantile(data, q = quant), digits)

def CVaR(data, quant = .05, digits = 3):
    ''' Evaluates Conditional Value at Risk at quant-level'''

    VaR = np.quantile(data, q = quant)
    if digits is None:
        return data[data<=VaR].mean()
    else:
        return round(data[data<=VaR].mean(), digits)


