"""
Optimal ML Model
"""

import numpy as np
import pandas as pd

from ..gems import prepare
from ..optimal_models import OptimalDTreeRegressor, OptimalKNRegressor, OptimalLinearRegression, OptimalABRegressor, OptimalRFRegressor 


class BaseBoa:
    
    NUM_REGRESSION_MODELS_SUPPORTED = 5
    
    reg_model_index = {0: 'dtree', 
                       1: 'knn', 
                       2: 'linreg',
                       3: 'adaboost',
                       4: 'rforest',
                       }
    
    def __repr__(self):
        return type(self).__name__ # finish implementation!!!
    
    
    def __len__(self):
        return len(self.ladder)

    
    def ladder(self):
        return self.ladder
    
    
    def optimal_model(self, rank=0):
        return self.ladder[rank][0]
    
    
    def model(self, rank=0):
        return self.ladder[rank][0].model
    
    
    def model_score(self, rank=0):
        return self.ladder[rank][1]
    
       
    def get_sets(self, rank=0, save=False):
        return self.ladder[rank][0].get_sets(save=save)
    


class RegressionBoa(BaseBoa):
    
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.ladder = []
    
    
    def hunt(self, X_train=None, X_test=None, y_train=None, y_test=None, data=None, target=None):
        for i in range(BaseBoa.NUM_REGRESSION_MODELS_SUPPORTED):
            model_key = BaseBoa.reg_model_index[i]
            optimal_model = self.find_optimal(regressor=model_key, X_train=X_train, y_train=y_train, data=data, target=target)
            
            if X_test is not None and y_test is not None:
                score = optimal_model.score(X_test, y_test)
            else:
                score = optimal_model.score()
                
            self.ladder.append((optimal_model, score))
            
        self.ladder.sort(key=lambda tup: tup[1], reverse=True)
        
        return self.ladder
    
    
    def find_optimal(self, regressor, X_train=None, y_train=None, data=None, target=None):
        if regressor == 'dtree':
            model = self.find_optimal_dtree(X_train=X_train, y_train=y_train, data=data, target=target)
        elif regressor == 'knn':
            model = self.find_optimal_knn(X_train=X_train, y_train=y_train, data=data, target=target)
        elif regressor == 'linreg':
            model = self.find_optimal_linreg(X_train=X_train, y_train=y_train, data=data, target=target)
        elif regressor == 'adaboost':
            model = self.find_optimal_adaboost(X_train=X_train, y_train=y_train, data=data, target=target)
        elif regressor == 'rforest':
            model = self.find_optimal_rforest(X_train=X_train, y_train=y_train, data=data, target=target)
        else:
            raise ValueError('Requested regressor not found. Please choose either \'dtree\' or \'knn\' or \'linreg\' or \'adaboost\' or \'rforest\'')
            
        return model
    
    
    def find_optimal_dtree(self, X_train=None, y_train=None, data=None, target=None):
        optimal_dtree = OptimalDTreeRegressor(self.random_state)
        optimal_dtree.find(X_train=X_train, y_train=y_train, data=data, target=target)
        
        return optimal_dtree
    
    
    def find_optimal_knn(self, X_train=None, y_train=None, data=None, target=None):
        optimal_knn = OptimalKNRegressor(self.random_state)
        optimal_knn.find(X_train=X_train, y_train=y_train, data=data, target=target)
        
        return optimal_knn
    
    
    def find_optimal_linreg(self, X_train=None, y_train=None, data=None, target=None):
        optimal_linreg = OptimalLinearRegression(self.random_state)
        optimal_linreg.find(X_train=X_train, y_train=y_train, data=data, target=target)
        
        return optimal_linreg
    
    
    def find_optimal_adaboost(self, X_train=None, y_train=None, data=None, target=None):
        optimal_adaboost = OptimalABRegressor(self.random_state)
        optimal_adaboost.find(X_train=X_train, y_train=y_train, data=data, target=target)
        
        return optimal_adaboost
    
    
    def find_optimal_rforest(self, X_train=None, y_train=None, data=None, target=None):
        optimal_rforest = OptimalRFRegressor(self.random_state)
        optimal_rforest.find(X_train=X_train, y_train=y_train, data=data, target=target)
        
        return optimal_rforest
        



# Skeletons below, finish and move them later
class ClassBoa(_BaseBoa):

    def __init__(self, random_state=None):
        self.random_state = random_state
        self.ladder = []


    def hunt(self, X_train=None, X_test=None, y_train=None, y_test=None, data=None, target=None):
        pass



class TreeBoa(_BaseBoa):

    def __init__(self, random_state=None, type_='reg'):
        self.random_state = random_state
        self.ladder = []


    def hunt(self, X_train=None, X_test=None, y_train=None, y_test=None, data=None, target=None):
        pass



