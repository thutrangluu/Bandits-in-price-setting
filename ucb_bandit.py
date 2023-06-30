# -*- coding: utf-8 -*-
# import modules 
import numpy as np 
import matplotlib.pyplot as plt 
import warnings


class ucb_bandit:
    
    def __init__(self, k, steps, mu='random', arms=[], **kwargs ):
        # Number of arms
        self.k = k
        self.arms = arms
        # Number of iterations
        self.steps = steps
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(steps)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
         # Best arm
        self.current_k = 0
        # Confidence level
        self.confidence_level = kwargs.pop('confidence_level', 1.5)
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)
    
   
    def pull(self):
        # uncertainties=np.zeros(self.k)
        # for k in range(self.k):
        #     if self.k_n[k] == 0: 
        #         uncertainties[k] = float('inf') 
        #     else:
        uncertainties = self.confidence_level * (np.sqrt(np.log(self.n) / (self.k_n+ 1e-5)))
        uncertainties[np.isnan(uncertainties)] = np.inf
        
        a = np.argmax(self.k_reward + uncertainties)

        self.current_k = a
        alpha = (self.mu[a] ** 2) * ((1 - self.mu[a]) / self.mu[a])
        beta = alpha * ((1 / self.mu[a]) - 1)
        
        reward = self.arms[a]*np.random.beta(alpha, beta)
        # reward = np.random.normal(self.mu[a], 1)
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
         with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(self.steps):
                self.pull()
                self.reward[i] = self.mean_reward
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(k)
        self.mean_reward = 0
        self.reward = np.zeros(steps)
        self.k_reward = np.zeros(k)
        self.current_k = 0


