''' Import Libraries'''
import pandas as pd
import numpy as np
from discriminants import MultivariateGaussian


class Classifier:
    ''' This is a class prototype for any classifier. It contains two empty methods: predict, fit'''
    def __init__(self):
        self.model_params = {}
        pass
    
    def predict(self, x):
        '''This method takes in x (numpy array) and returns a prediction y'''
        raise NotImplementedError
    
    def fit(self, x, y):
        '''This method is used for fitting a model to data: x, y'''
        raise NotImplementedError

class Prior(Classifier):
    
    def __init__(self):
        self.model_params = {}
        pass
    

    def predict(self, x):
        '''This method takes in x (numpy array) and returns a prediction y'''
        raise NotImplementedError
    
    def fit(self, x, y):
        '''This method is used for fitting a model to data: x (numpy array), y (numpy array)'''
        raise NotImplementedError



''' Create our Discriminant Classifier Class'''    
class DiscriminantClassifier(Classifier):
    ''''''
    def __init__(self):
        ''' Initialize Class Dictionary'''
        self.model_params = {}
        self.classes = {}
        
    def set_classes(self, *discs):
        '''Pass discriminant objects and store them in self.classes
            This class is useful when you have existing discriminant objects'''
        for disc in discs:
            self.classes[disc.name] = disc

            
    def fit(self, dataframe, label_key=['Labels'], default_disc=MultivariateGaussian):
        ''' Calculates model parameters from a dataframe for each discriminant.
            Label_Key specifies the column that contains the class labels. ''' 
        labels = dataframe[label_key].unique()
        for label in labels:
            data = dataframe[dataframe[label_key] == label].drop(columns=[label_key]).to_numpy()
            self.classes[label] = default_disc(data, prior=len(data)/len(dataframe), name=label)
        self.pool_variances()
                
    
    def predict(self, x):
        ''' Returns a Key (class) that corresponds to the highest discriminant value'''
        scores = {}
        for label, disc in self.classes.items():
            scores[label] = disc.calc_discriminant(x)
        fin = max(scores, key=scores.get)
        return fin

    def pool_variances(self):
        ''' Calculates a pooled variance and sets the corresponding model params '''
        totsamp = 0
        poolsigma = None
        for disc in self.classes.values():
            n = disc.params['data'].shape[0]
            totsamp += n - 1
            if poolsigma is None:
                poolsigma = (n - 1) * disc.params['sigma']
            else:
                poolsigma += (n - 1) * disc.params['sigma']
        poolsigma /= totsamp
        for disc in self.classes.values():
            disc.params['sigma'] = poolsigma
        
        
