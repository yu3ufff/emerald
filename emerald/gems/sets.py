import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

def train_test_sets(data, target=None, test_size=0.2, random_state=None, strat_var=None, n_splits=1):
        data = data

        # separating the predictors and the target
        if target is None:
            X = np.c_[data.iloc[:, :-1]]
            y = data.iloc[:, -1]
        else:
            predictors = list(data.columns)
            predictors.remove(target)
            X = data[predictors]
            y = np.c_[data[target]]

        # creating the train and test sets
        if strat_var is None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
            for train_index, test_index in splitter.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

        return X_train, X_test, y_train, y_test