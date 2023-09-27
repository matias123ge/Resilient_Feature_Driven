#%%
import sklearn
from main_DA_trading import eval_dual_predictions
from Feature_driven_reg import * 
import numpy as np
#%%
def crossval(x, y,p_up,p_down,group_col,rvals,K): 
    innererror = []
    HiddenValues = []
    outerror = []
    #Create train and validation 
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.33, random_state=42)
    #Do validation for all instances of rvals across 10 different validation sets.   
    for i in range(len(rvals)):
        model = Feature_driven_reg(p_up,p_down,alpha=rvals[i])
        model.fit(X_train,y_train)
        #Test on 10 variations of randomly deleted features: 
        for k in range(K):
            #Create validation sets with randomized groups missing. 
            randomsize = np.random.randint(0,len(group_col))
            groupchoice = np.random.choice(np.arange(0,len(group_col)), size = randomsize , replace = False)
            if groupchoice.size == 0: 
                miss_X = X_test.copy()
            else:
                getindexes = []
                for i in range(len(groupchoice)): 
                    print(groupchoice)
                    getindexes.append(group_col[groupchoice[i]])
                    print(getindexes)
                    miss_X = X_test.copy()
                for j in range(len(getindexes)):
                    print(getindexes)
                    miss_X[:,getindexes[j]] = 0   
            val = model.predict(miss_X)
            error = eval_dual_predictions(val,y_test,p_up,p_down)
            innererror.append(error)
        
        outerror.append(np.mean(innererror))
    return outerror