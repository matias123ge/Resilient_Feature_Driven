
#%%

import pickle
import os, sys

cd = os.path.dirname(__file__)  #Current directory
sys.path.append(cd)
#%%
import seaborn as sns
import gurobipy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lags_df = pd.read_csv(f'{cd}\\ID-case\\results\\LagSensitivity.csv', index_col=0)
#%%
models_to_plot = lags_df.columns[3:-1]
sns.set()
# %%
#Plot average penaltý for each percentage of missing data for each lag 
percentages = lags_df['percentage missing'].unique()
for percent in percentages: 
    for models in models_to_plot:   
        plotval = lags_df[lags_df['percentage missing'] == percent] 
        plt.errorbar([1,2,3],plotval.groupby('lag')[models].mean(),yerr = [np.ones(3)*(lags_df[lags_df['percentage missing'] == percent][models].std()),np.ones(3)*lags_df[lags_df['percentage missing'] == percent][models].std()],capzie = 4)
        plt.title("Sensitivity to lags at {} Missing values".format(percent))     
    plt.legend(models_to_plot)
    plt.xlabel("Nr lags ")
    plt.ylabel("Average Penalty €/MWh") 
    plt.savefig(f'{cd}\\ID-case\\figures\\SensitivityPlotPercent({percent}).pdf')
    plt.show()   
    


# %%
