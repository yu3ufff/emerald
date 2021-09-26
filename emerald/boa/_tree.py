import numpy as np
import pandas as pd

from ..gems import prepare

from ._base import BaseBoa

# Skeleton
class TreeBoa(BaseBoa):

    def __init__(self, random_state=None, type_='reg'):
        self.random_state = random_state
        self.ladder = []


    def hunt(self, X_train=None, X_test=None, y_train=None, y_test=None, data=None, target=None):
        pass


    def find_optimal(self, tree, X_train=None, y_train=None, data=None, target=None):
        pass