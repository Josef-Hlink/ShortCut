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

class Agent(object):
    """Parent class for type hinting (intellisense) purposes"""
    def __init__(self, n_actions: int = 0, n_states: int = 0, epsilon: float = 0.0):
       # so IDE knows that an Agent  instance should have these properties
        self.sname = ''
        self.lname = ''
        self.Q = np.zeros(shape=(2, 2))
        self.epsilon = epsilon
        pass

    def select_action(self, state: int = 0, epsilon: float = 0.0):
        pass

    def update(self, state, action, reward,  next_state):
        pass

class QLearningAgent(Agent):

    def __init__(self, n_actions: int, n_states: int, epsilon: float, alpha: float):
        self.sname = 'QL'               # short name
        self.lname = 'Q Learning'       # long name
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(shape=(self.n_states, self.n_actions),dtype=np.float64)
        pass
        
    def select_action(self, state: int) -> int:
        rand: float = np.random.rand()
        greedy_a: int = np.argmax(self.Q[state])
        if rand > self.epsilon:
            return greedy_a
        else:
            expl_a = np.random.choice(self.n_actions)
            while expl_a == greedy_a:
                expl_a = np.random.choice(self.n_actions)
            return expl_a
    
    def update(self, state, action, reward, next_state) -> None:
        s, a, r, sp = state, action, reward, next_state                         # for more concise notation
        self.Q[s][a] += self.alpha * (r + np.max(self.Q[sp]) - self.Q[s][a])    # update estimated mean reward for the action
        return

class SARSAAgent(Agent):

    def __init__(self, n_actions: int, n_states: int, epsilon: float, alpha: float):
        self.sname = 'SARSA'    # short name
        self.lname = 'SARSA'    # long name
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(shape=(self.n_states, self.n_actions),dtype=np.float64)
        pass
        
    def select_action(self, state):
        rand: float = np.random.rand()
        greedy_a: int = np.argmax(self.Q[state])
        if rand > self.epsilon:
            return greedy_a
        else:
            expl_a = np.random.choice(self.n_actions)
            while expl_a == greedy_a:
                expl_a = np.random.choice(self.n_actions)
            return expl_a
        
    def update(self, state, action, reward, next_state) -> None:
        s, a, r, sp = state, action, reward, next_state                     # for more concise notation
        ap: int = self.select_action(sp)                                    # (hypothetical) next action according to own policy
        self.Q[s][a] += self.alpha * (r + self.Q[sp][ap] - self.Q[s][a])    # update estimated mean reward for the action
        return

class ExpectedSARSAAgent(Agent):

    def __init__(self, n_actions: int, n_states: int, epsilon: float, alpha: float):
        self.sname = 'ESARSA'           # short name
        self.lname = 'Expected SARSA'   # long name
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(shape=(self.n_states, self.n_actions),dtype=np.float64)
        pass
        
    def select_action(self, state):
        rand: float = np.random.rand()
        greedy_a: int = np.argmax(self.Q[state])
        if rand > self.epsilon:
            return greedy_a
        else:
            expl_a = np.random.choice(self.n_actions)
            while expl_a == greedy_a:
                expl_a = np.random.choice(self.n_actions)
            return expl_a
            
    def update(self, state, action, reward, next_state) -> None:
        if self.epsilon == 0:
            return
        s, a, r, sp = state, action, reward, next_state                         # for more concise notation
        expQ: float = 0.0
        for ap in range(self.n_actions):
            # get weighted Q action values based on probability of the action being chosen and accumulate them
            weightedQ = (1- self.epsilon) * self.Q[sp][ap] if ap == np.argmax(self.Q[sp]) else \
                        (self.epsilon/(self.n_actions-1)) * self.Q[sp][ap]
            expQ += weightedQ
        self.Q[s][a] += self.alpha * (r + expQ - self.Q[s][a])                  # update estimated mean reward for the action
        return
