import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.base import TransformerMixin

def prepare(data, target, impute=True, test_size=0.2, stratified=False, random_state=None):
        data = data.copy()

        # organizing the features into categories
        # numericals = [col for col in data.columns if (data.dtypes[col] == 'int' or data.dtypes[col] == 'float') and set(data[col].unique()) != {0, 1}]
        categoricals = [col for col in data.columns if (data.dtypes[col] != 'int' and data.dtypes[col] != 'float') 
                        or remove_nan(set(data[col].unique())) == {0, 1} or remove_nan(set(data[col].unique())) == {0., 1.}]
        # numbers = [col for col in data.columns if data.dtypes[col] == 'int' or data.dtypes[col] == 'float']
        non_numbers = [col for col in data.columns if data.dtypes[col] != 'int' and data.dtypes[col] != 'float']
        
        if target in categoricals:
            raise ValueError('Please enter a continuous variable as a target.')

        # remove rows and columns with all nan values
        data.dropna(axis=0, how='all', inplace=True)
        data.dropna(axis=1, how='all', inplace=True)
        
        # impute missing data
        if impute:
            data = PolishedImputer().fit_transform(data)
            # turn the categorical num columns back to type int
            for col in data.columns:
                if col in categoricals and data.dtypes[col] == 'float':
                    data[col] = data[col].astype(np.int64)
        
        
        # creating dummies if needed
        if non_numbers:
            data = pd.get_dummies(data, columns=non_numbers, drop_first=True)
            
        # separating the predictors and the target
        predictors = list(data.columns)
        predictors.remove(target)
        
        # creating the train and test sets
        if stratified:
            # use pd.cut to split target into groups
            target_col = data[target]
            data['target_cat'] = pd.cut(target_col, 
                                        bins=[target_col.min(), target_col.quantile(0.25), target_col.median(), 
                                              target_col.quantile(0.75), target_col.max()], 
                                        include_lowest=True)
            
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            for train_index, test_index in splitter.split(data, data['target_cat']):
                strat_train_set = data.loc[train_index]
                strat_test_set = data.loc[test_index]
                
            # for observation:
            # print(data['target_cat'].value_counts() / len(data)) 
            # print(strat_test_set['target_cat'].value_counts() / len(strat_test_set)) 
                
            # drop categorical feature for target
            for set_ in (strat_train_set, strat_test_set):
                set_.drop('target_cat', axis=1, inplace=True)
                
            # finish splitting sets
            X_train = strat_train_set[predictors]
            X_test = strat_test_set[predictors]
            y_train = pd.DataFrame(np.c_[strat_train_set[target]], columns=[target])
            y_test = pd.DataFrame(np.c_[strat_test_set[target]], columns=[target])
        else:
            X = data[predictors]
            y = pd.DataFrame(np.c_[data[target]], columns=[target])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
        return X_train, X_test, y_train, y_test


class PolishedImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Categorical features are imputed with the most frequent value in the column.

        Continuous features are imputed with mean of column.

        """
    def fit(self, data, y=None):
        categoricals = [col for col in data.columns if (data.dtypes[col] != 'int' and data.dtypes[col] != 'float') 
                        or remove_nan(set(data[col].unique())) == {0, 1} or remove_nan(set(data[col].unique())) == {0., 1.}]
        self.fill = pd.Series([data[col].value_counts().index[0] if col in categoricals else data[col].mean() for col in data],
                              index=data.columns)
        # Note: possibly differentiate between ints and floats for continuous (mean vs median)
        return self

    
    def transform(self, X, y=None):
        return X.fillna(self.fill)


def remove_nan(set_):
    return {x for x in set_ if x==x}