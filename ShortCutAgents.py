#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShortCut environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2022
Template provided by: Thomas Moerland
Implementation by: Josef Hamelink & Ayush Kandhai
"""


import random
import numpy as np

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.Q = np.zeros(shape=(self.n_states, self.n_actions),dtype=np.float64)
        self.N = np.zeros(shape=(self.n_states, self.n_actions),dtype=np.int8)
        pass
        
    def select_action(self, state: int, epsilon: float) -> int:
        s = state                                                   # for more concise notation
        probabilities: np.array = np.zeros(self.n_actions)          # probability that an action is chosen
        for a in range(self.n_actions):
            if (a == np.argmax(self.Q[s])):                         # for action with the highest mean value:
                probabilities[a] = (1-epsilon)                      # assign highest probability score
            else:                                                   # for all other actions:
                probabilities[a] = (epsilon / (self.n_actions-1))   # assign equally divided remaining probability scores
        return np.random.choice(self.n_actions, p=probabilities)    # choose one action based on probabilities
        
    def update(self, state, action, reward) -> None:
        s, a, r = state, action, reward                             # for more concise notation
        self.N[s][a] += 1                                           # increment number of times the action is taken by one
        self.Q[s][a] += (1/self.N[s][a]) * (r - self.Q[s][a])       # update estimated mean reward for the action
        return

class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass
        
    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions)) # Replace this with correct action selection
        return a
        
    def update(self, state, action, reward):
        # TO DO: Add own code
        pass

class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass
        
    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions)) # Replace this with correct action selection
        return a
        
    def update(self, state, action, reward):
        # TO DO: Add own code
        pass