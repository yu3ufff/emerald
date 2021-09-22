import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

import statsmodels.api as sm

from ..gems import prepare

class BaseRegressor:

    def score(self, X_test=None, y_test=None):
        if not self.model:
            raise ValueError('There is no model to use for scoring')

        if self.y_test is not None:
            return self.model.score(self.X_test, self.y_test)
        elif X_test is not None and y_test is not None:
            return self.model.score(X_test, y_test)
        else:
            raise ValueError('No test sets to use for scoring')

    def get_sets(self):
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        self.X_train, self.X_test, self.y_train, self.y_test = (None, None, None, None)
        return X_train, X_test, y_train, y_test


class OptimalDTreeRegressor(BaseRegressor):
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
            param_grid=[{'max_depth': [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 200, None]}]
    ):
        if isinstance(data, pd.DataFrame):
            X_train, X_test, y_train, y_test = prepare(data, target, random_state=self.random_state)

            # save sets
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        elif X_train is None or y_train is None:
            raise ValueError('Please pass in some data')
        else:
            # temporarily save sets
            self.X_train = X_train
            self.y_train = y_train

        grid = GridSearchCV(DecisionTreeRegressor(), param_grid)
        grid.fit(self.X_train, self.y_train)

        optimal_depth = grid.best_params_['max_depth']

        self.model = DecisionTreeRegressor(max_depth=optimal_depth, random_state=self.random_state)
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
        if isinstance(data, pd.DataFrame):
            X_train, X_test, y_train, y_test = prepare(data, target, random_state=self.random_state)

            # standardize X_train and X_test
            robust_scaler = RobustScaler()
            X_train_scaled = robust_scaler.fit_transform(X_train)
            X_test_scaled = robust_scaler.fit_transform(X_test)

            # save sets
            self.X_train = X_train_scaled
            self.X_test = X_test_scaled
            self.y_train = y_train
            self.y_test = y_test
        elif X_train is None or y_train is None:
            raise ValueError('Please pass in some data')
        else:
            # temporarily save sets
            self.X_train = X_train
            self.y_train = y_train

        grid = GridSearchCV(KNeighborsRegressor(), param_grid)
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
        self.best_features = None

    def find(
            self,
            X_train=None,
            y_train=None,
            data=None,
            target=None,
            alpha=0.05
    ):
        if isinstance(data, pd.DataFrame):
            X_train, X_test, y_train, y_test = prepare(data, target, random_state=self.random_state)

            # save sets
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        elif X_train is None or y_train is None:
            raise ValueError('Please pass in some data')
        else:
            # temporarily save sets
            self.X_train = X_train
            self.y_train = y_train

        # selecting the features with p-values <= alpha
        X1_train = sm.add_constant(X_train)
        ols = sm.OLS(self.y_train, X1_train)
        lin_reg = ols.fit()

        selected_features = [col for col in self.X_train.columns if lin_reg.summary2().tables[1]['P>|t|'][col] <= alpha]

        # updating the sets to only have the selected features
        self.X_train = self.X_train[selected_features]
        self.X_test = self.X_test[selected_features] if self.X_test is not None else None

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        self.best_features = selected_features

        if self.y_test is None:
            self.X_train, self.y_train = (None, None)
            print('Warning: The training set has been updated. Please get the new selected features to use for further training and testing.')

        return self.model


class OptimalLinearSVR(BaseRegressor):
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
            param_grid=[{'C': [0.1, 1, 1.5, 3, 10, 50, 100, 1000], 'epsilon': [0, 0.01, 0.1, 0.2, 0.3, 0.5]}]
    ):
        if isinstance(data, pd.DataFrame):
            X_train, X_test, y_train, y_test = prepare(data, target, random_state=self.random_state)

            # standardize X_train and X_test
            robust_scaler = RobustScaler()
            X_train_scaled = robust_scaler.fit_transform(X_train)
            X_test_scaled = robust_scaler.fit_transform(X_test)

            # save sets
            self.X_train = X_train_scaled
            self.X_test = X_test_scaled
            self.y_train = y_train
            self.y_test = y_test
        elif X_train is None or y_train is None:
            raise ValueError('Please pass in some data')
        else:
            # temporarily save sets
            self.X_train = X_train
            self.y_train = y_train
            
        grid = GridSearchCV(LinearSVR(), param_grid)
        grid.fit(self.X_train, self.y_train.values.ravel())

        optimal_C = grid.best_params_['C']
        optimal_epsilon = grid.best_params_['epsilon']

        self.model = LinearSVR(epsilon=optimal_epsilon, C=optimal_C, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train.values.ravel())

        self.best_params = grid.best_params_

        if self.y_test is None:
            self.X_train, self.y_train = (None, None)

        return self.model

    
    
    # Ensemble Methods
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
            param_grid=[{'n_estimators': [10, 50, 100], 'min_samples_split': [5, 10, 25, 50]}]
    ):
        if isinstance(data, pd.DataFrame):
            X_train, X_test, y_train, y_test = prepare(data, target, random_state=self.random_state)

            # save sets
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        elif X_train is None or y_train is None:
            raise ValueError('Please pass in some data')
        else:
            # temporarily save sets
            self.X_train = X_train
            self.y_train = y_train

        grid = GridSearchCV(RandomForestRegressor(), param_grid)
        grid.fit(self.X_train, self.y_train.values.ravel())

        optimal_split = grid.best_params_['min_samples_split']
        optimal_estimators = grid.best_params_['n_estimators']

        self.model = RandomForestRegressor(n_estimators=optimal_estimators, min_samples_split=optimal_split, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train.values.ravel())

        self.best_params = grid.best_params_

        if self.y_test is None:
            self.X_train, self.y_train = (None, None)

        return self.model
