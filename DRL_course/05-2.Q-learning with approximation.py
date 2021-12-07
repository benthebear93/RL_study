import random, datetime, gym, os, time, psutil, cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import sklearn.preprocessing
from pyvirtualdisplay import Display

env = gym.make('MountainCar-v0')
eval_env = gym.make('MountainCar-v0')

class LinearApprox:
    def __init__(self):
        self.nA = 3
        self.w = np.zeros((self.nA, 400))
        self.set_featurizer()
        self.epsilon = 0.1
        self.alpha = 0.01
    
    def set_featurizer(self):
        # Get satistics over observation space samples for normalization
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Create radial basis function sampler to convert states to features for nonlinear function approx
        self.featurizer = sklearn.pipeline.FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=100))
                ])
        
        # Fit featurizer to our scaled inputs
        self.featurizer.fit(self.scaler.transform(observation_examples))

    def featurize_state(self, state):
        # Normalize and turn into feature
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized

    def Q(self, state, action):
        value = state.dot(self.w[action])
        return value

    def get_action(self, state, test=False):
        # Epsilon greedy policy
        probs = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action =  np.argmax([self.Q(state, action) for action in range(self.nA)])
        probs[best_action] += (1.0 - self.epsilon)
        if test:
            probs = np.zeros(self.nA, dtype=float)
            probs[best_action] += 1.0
        action = np.random.choice(self.nA, p=probs)
        return action

    def update_epsilon(self, epsilon):
        self.epsilon = np.min([epsilon, 0.1])
        
    def update_alpha(self, alpha):
        self.alpha = np.min([alpha, 0.01]) 