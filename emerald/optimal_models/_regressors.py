import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

import statsmodels.api as sm

from ..gems import prepare

class BaseRegressor:

    def __repr__(self):
        return type(self).__name__

    def score(self, X_test=None, y_test=None):
        if not self.model:
            raise ValueError('There is no model to use for scoring')

        if X_test is not None and y_test is not None:
            return self.model.score(X_test, y_test)
        elif self.X_test is not None and self.y_test is not None:
            return self.model.score(self.X_test, self.y_test)
        else:
            raise ValueError('No test sets to use for scoring')

    def get_sets(self, save=False):
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        if not save:
            self.X_train, self.X_test, self.y_train, self.y_test = (None, None, None, None)
        return X_train, X_test, y_train, y_test


class OptimalDTRegressor(BaseRegressor):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.best_params = None

    def find(
            self,
            X_train=None,
            y_train=None,
            data=None,
            target=None,
            param_grid=[{'max_depth': [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 200, None], 'min_samples_split': [2, 5, 10]}]
    ):
        if isinstance(data, pd.DataFrame) and isinstance(target, str):
            X_train, X_test, y_train, y_test = prepare(data, target, random_state=self.random_state)

            # save sets
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        elif X_train is None or y_train is None:
            raise ValueError('Please pass in either your data and target feature or training set and target')
        else:
            # temporarily save sets
            self.X_train = X_train
            self.y_train = y_train

        grid = GridSearchCV(DecisionTreeRegressor(), param_grid, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)

        optimal_depth = grid.best_params_['max_depth']
        optimal_split = grid.best_params_['min_samples_split']

        self.model = DecisionTreeRegressor(max_depth=optimal_depth, min_samples_split=optimal_split, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)

        self.best_params = grid.best_params_

        if self.y_test is None:
            self.X_train, self.y_train = (None, None)

        return self.model


class OptimalKNRegressor(BaseRegressor):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.best_params = None

    def find(
            self,
            X_train=None,
            y_train=None,
            data=None,
            target=None,
            param_grid=[{'n_neighbors': [1, 2, 3, 4, 5, 10, 15, 20], 'p': [1, 2]}]
    ):
        if isinstance(data, pd.DataFrame) and isinstance(target, str):
            X_train, X_test, y_train, y_test = prepare(data, target, random_state=self.random_state)

            # standardize X_train and X_test
            robust_scaler = RobustScaler()
            X_train_scaled = robust_scaler.fit_transform(X_train)
            X_test_scaled = robust_scaler.fit_transform(X_test)
            
            # transform arrays back to DataFrame
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

            # save sets
            self.X_train = X_train_scaled
            self.X_test = X_test_scaled
            self.y_train = y_train
            self.y_test = y_test
        elif X_train is None or y_train is None:
            raise ValueError('Please pass in either your data and target feature or training set and target')
        else:
            # temporarily save sets
            self.X_train = X_train
            self.y_train = y_train

        grid = GridSearchCV(KNeighborsRegressor(), param_grid, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)

        optimal_neighbors = grid.best_params_['n_neighbors']
        optimal_p = grid.best_params_['p']

        self.model = KNeighborsRegressor(n_neighbors=optimal_neighbors, p=optimal_p)
        self.model.fit(self.X_train, self.y_train)

        self.best_params = grid.best_params_

        if self.y_test is None:
            self.X_train, self.y_train = (None, None)

        return self.model


class OptimalLinearRegression(BaseRegressor):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.selected_features = None

    def find(
            self,
            X_train=None,
            y_train=None,
            data=None,
            target=None,
            alpha=0.05
    ):
        if isinstance(data, pd.DataFrame) and isinstance(target, str):
            X_train, X_test, y_train, y_test = prepare(data, target, random_state=self.random_state)
            
            # selecting the features with p-values <= alpha
            X1_train = sm.add_constant(X_train)
            ols = sm.OLS(y_train, X1_train)
            lin_reg = ols.fit()

            selected_features = [col for col in X_train.columns if lin_reg.summary2().tables[1]['P>|t|'][col] <= alpha]

            # updating the sets to only have the selected features
            X_train = X_train[selected_features] 
            X_test = X_test[selected_features] 

            # save sets
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        elif X_train is None or y_train is None:
            raise ValueError('Please pass in either your data and target feature or training set and target')
        else:
            # temporarily save sets
            self.X_train = X_train
            self.y_train = y_train

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        self.selected_features = selected_features

        if self.y_test is None:
            self.X_train, self.y_train = (None, None)
            # print('Warning: The training set has been updated. Please get the new selected features to use for further training and testing.')

        return self.model


    # Ensemble Methods
class OptimalABRegressor(BaseRegressor):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.best_params = None

    def find(
            self,
            X_train=None,
            y_train=None,
            data=None,
            target=None,
            param_grid=[{'n_estimators': [10, 50, 100, 500], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]}]
    ):
        if isinstance(data, pd.DataFrame) and isinstance(target, str):
            X_train, X_test, y_train, y_test = prepare(data, target, random_state=self.random_state)

            # save sets
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        elif X_train is None or y_train is None:
            raise ValueError('Please pass in either your data and target feature or training set and target')
        else:
            # temporarily save sets
            self.X_train = X_train
            self.y_train = y_train
    
        grid = GridSearchCV(AdaBoostRegressor(), param_grid, n_jobs=-1)
        grid.fit(self.X_train, self.y_train.values.ravel())

        optimal_rate = grid.best_params_['learning_rate']
        optimal_estimators = grid.best_params_['n_estimators']

        self.model = AdaBoostRegressor(n_estimators=optimal_estimators, learning_rate=optimal_rate, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train.values.ravel())

        self.best_params = grid.best_params_

        if self.y_test is None:
            self.X_train, self.y_train = (None, None)

        return self.model


class OptimalRFRegressor(BaseRegressor):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.best_params = None

    def find(
            self,
            X_train=None,
            y_train=None,
            data=None,
            target=None,
            param_grid=[{'n_estimators': [10, 50, 100, 500], 'min_samples_split': [2, 5, 10, 25, 50]}]
    ):
        if isinstance(data, pd.DataFrame) and isinstance(target, str):
            X_train, X_test, y_train, y_test = prepare(data, target, random_state=self.random_state)

            # save sets
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        elif X_train is None or y_train is None:
            raise ValueError('Please pass in either your data and target feature or training set and target')
        else:
            # temporarily save sets
            self.X_train = X_train
            self.y_train = y_train

        grid = GridSearchCV(RandomForestRegressor(), param_grid, n_jobs=-1)
        grid.fit(self.X_train, self.y_train.values.ravel())

        optimal_split = grid.best_params_['min_samples_split']
        optimal_estimators = grid.best_params_['n_estimators']

        self.model = RandomForestRegressor(n_estimators=optimal_estimators, min_samples_split=optimal_split, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train.values.ravel())

        self.best_params = grid.best_params_

        if self.y_test is None:
            self.X_train, self.y_train = (None, None)

        return self.model
    