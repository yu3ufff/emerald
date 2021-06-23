"""
Optimal ML Model
"""

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

from ..gems import train_test_sets

class _BaseBoa:
    def digest(self, data, impute='median'):
        data = data
        nums = [col for col in data.columns if data.dtypes[col] == 'int']
        cats = [col for col in data.columns if data.dtypes[col] != 'int']


class RegBoa(_BaseBoa):

    def __init__(self, random_state=None):
        self.random_state = random_state
        self.best = None
        self.second = None
        self.third = None


    def hunt(self, data, target=None, test_size=0.2, strat_var=None, n_splits=1):
        data = data

        # creating the train and test sets
        X_train, X_test, y_train, y_test = train_test_sets(data, target, self.random_state, test_size, strat_var, n_splits)
        


class ClassBoa(_BaseBoa):

    def __init__(self, random_state=None):
        self.random_state = random_state
        self.best = None
        self.second = None
        self.third = None


    def hunt(self, data, target=None, test_size=0.2, strat_var=None, n_splits=1):
        data = data

        # creating the train and test sets
        X_train, X_test, y_train, y_test = train_test_sets(data, target, self.random_state, test_size, strat_var, n_splits)


class TreeBoa(_BaseBoa):

    def __init__(self, random_state=None, type_='reg'):
        self.random_state = random_state
        self.type_ = type_
        self.best = None
        self.second = None
        self.third = None


    def hunt(self, data, target=None, test_size=0.2, strat_var=None, n_splits=1):
        data = data

        # creating the train and test sets
        X_train, X_test, y_train, y_test = train_test_sets(data, target, self.random_state, test_size, strat_var, n_splits)



