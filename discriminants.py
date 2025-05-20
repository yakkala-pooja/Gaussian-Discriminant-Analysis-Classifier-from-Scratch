''' Import Libraries'''
import pandas as pd
import numpy as np


class Discriminant:
    ''' Prototype class for Discriminants'''
    def __init__(self):
        self.params = {}
        self.name = ''
        
    def fit(self, data):
        raise NotImplementedError
    
    def calc_discriminant(self, x):
        raise NotImplementedError



class GaussianDiscriminant(Discriminant):
    ''' Assumes a Gaussian Distribution for P(x|C_i)'''
    def __init__(self, data = None, prior=0.5, name = 'Not Defined'):
        '''Initialize pi and model parameters'''
        self.pi = np.pi
        self.params = {'mu':None, 'sigma':None, 'prior':prior}
        if data is not None:
            self.fit(data)
        self.name = name
    
    def fit(self, data):
        ''' Data is a numpy array consisting of data from a single class, where each row is a sample'''
        self.params['mu']    = np.mean(data)
        self.params['sigma'] = np.std(data)
        
        
    def calc_discriminant(self, x):
        '''Returns a discriminant value for a single sample'''
        mu = self.params['mu']
        sigma= self.params['sigma']
        prior = self.params['prior']
        g = - np.log(sigma) - 0.5 * ((x - mu) ** 2 / (sigma ** 2)) + np.log(prior)
        return g


''' Create our MV Discriminant Class'''
class MultivariateGaussian(Discriminant):
    
    def __init__(self, data=None, prior=0.5, name = 'Not Defined'):
        '''Initialize pi and model parameters'''
        self.pi = np.pi
        self.params = {'mu':None, 'sigma':None, 'prior':prior, 'data': None}
        if data is not None:
            self.fit(data)
        self.name = name
        
    def fit(self, data):
        ''' Data is a numpy array consisting of data from a single class, where each row is a sample'''
        self.params['mu']    = np.average(data, axis=0)
        self.params['sigma'] = np.cov(data.T, bias=True)
        self.params['data'] = data
        
    def calc_discriminant(self, x):
        mu, sigma, prior = self.params['mu'], self.params['sigma'], self.params['prior']
        diff = x - mu
        siginv = np.linalg.inv(sigma)
        mdist = np.dot(np.dot(diff.T, siginv), diff)
    
        d = len(x)
        sigdet = np.linalg.det(sigma)
        g = - (d / 2) * np.log(2 * np.pi) - 0.5 * np.log(sigdet) - 0.5 * mdist + np.log(prior)
        return g
