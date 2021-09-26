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


    def hunt(self, X_train=None, X_test=None, y_train=None, y_test=None, data=None, target=None):
        pass


    def find_optimal(self, model, X_train=None, y_train=None, data=None, target=None):
        pass