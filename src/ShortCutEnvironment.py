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
from scipy import rand

class Environment(object):

    def __init__(self):
        # so IDE knows that an Environment instance should have these properties
        self.y = 0
        self.x = 0
        pass

    def reset(self):
        pass
    
    def render(self):
        pass
    
    def step(self, action):
        pass
    
    def possible_actions(self):
        pass
    
    def state(self):
        pass
    
    def state_size(self):
        pass
    
    def action_size(self):
        pass
    
    def done(self):
        pass

class ShortcutEnvironment(Environment):
    def __init__(self, seed=None):
        self.r = 12
        self.c = 12
        self.rng = random.Random(seed)
        s = np.zeros((self.r, self.c+1), dtype=str)
        s[:] = 'X'
        s[:,-1] = '\n'
        s[self.r//3:, self.c//3:2*self.c//3] = 'C'
        s[5*self.r//6-1,:self.c//2] = 'X'
        s[2*self.r//3:5*self.r//6:,self.c//2] = 'X'
        s[2*self.r//3,self.c//2:2*self.c//3] = 'X'
        s[2*self.r//3, 2*self.c//3] = 'G'
        self.s = s
        self.reset()
    
    def reset(self):
        self.x = self.c//6
        rand_number = int(2*self.rng.random())
        #rand_number = 0
        if rand_number:
            self.y = 5*self.r//6 - 1
        else:
            self.y = self.r//6
        self.starty = self.y
        self.isdone = False
        return rand_number
    
    def state(self):
        return self.y*self.c + self.x
    
    def state_size(self):
        return self.c*self.r
    
    def action_size(self):
        return 4
    
    def done(self):
        return self.isdone
    
    def possible_actions(self):
        return [0, 1, 2, 3]
    
    def step(self, action):
        if self.isdone:
            raise ValueError('Environment has to be reset.')
        
        if not action in self.possible_actions():
            raise ValueError(f'Action ({action}) not in set of possible actions.')
        
        if action == 0:
            if self.y>0:
                self.y -= 1         # go up
        elif action == 1:
            if self.y<self.r-1:
                self.y += 1         # go down
        elif action == 2:
            if self.x>0:
                self.x -= 1         # go left
        elif action == 3:
            if self.x<self.c-1:
                self.x += 1         # go right
        
        if self.s[self.y, self.x]=='G': # Goal reached
            self.isdone = True
            return -1
        elif self.s[self.y, self.x]=='C': # Fall off cliff
                self.x = self.c//6
                self.y = self.starty
                return -100
        return -1
    
    
    def render(self):
        s = self.s.copy()
        s[self.y, self.x] = 'p'
        s = np.char.replace(s, 'X', '\033[0;47m \033[0m')
        s = np.char.replace(s, 'C', '\033[0;40m \033[0m')
        s = np.char.replace(s, 'G', '\033[0;42m \033[0m')
        s = np.char.replace(s, 'p', '\033[0;44m \033[0m')
        print(s.tobytes().decode('utf-8'))

class WindyShortcutEnvironment(Environment):
    def __init__(self, seed=None):
        self.r = 12
        self.c = 12
        self.rng = random.Random(seed)
        s = np.zeros((self.r, self.c+1), dtype=str)
        s[:] = 'X'
        s[:,-1] = '\n'
        s[self.r//3:, self.c//3:2*self.c//3] = 'C'
        s[5*self.r//6-1,:self.c//2] = 'X'
        s[2*self.r//3:5*self.r//6:,self.c//2] = 'X'
        s[2*self.r//3,self.c//2:2*self.c//3] = 'X'
        s[2*self.r//3, 2*self.c//3] = 'G'
        self.s = s
        self.reset()
    
    def reset(self):
        self.x = self.c//6
        rand_number = int(2*self.rng.random())
        if rand_number:
            self.y = 5*self.r//6 - 1
        else:
            self.y = self.r//6
        self.starty = self.y
        self.isdone = False
        return rand_number
    
    def state(self):
        return self.y*self.c + self.x
    
    def state_size(self):
        return self.c*self.r
    
    def action_size(self):
        return 4
    
    def done(self):
        return self.isdone
    
    def possible_actions(self):
        return [0, 1, 2, 3]
    
    def step(self, action):
        if self.isdone:
            raise ValueError('Environment has to be reset.')
        
        if not action in self.possible_actions():
            raise ValueError(f'Action ({action}) not in set of possible actions.')
        
        if action == 0:
            if self.y>0:
                self.y -= 1
        elif action == 1:
            if self.y<self.r-1:
                self.y += 1
        elif action == 2:
            if self.x>0:
                self.x -= 1
        elif action == 3:
            if self.x<self.c-1:
                self.x += 1
        
        if self.rng.random()<0.5:
            # Wind!
            if self.y < self.r-1:
                self.y += 1
        
        if self.s[self.y, self.x]=='G': # Goal reached
            self.isdone = True
            return -1
        elif self.s[self.y, self.x]=='C': # Fall off cliff
                self.x = self.c//6
                self.y = self.starty
                return -100
        return -1
    
    
    def render(self):
        s = self.s.copy()
        s[self.y, self.x] = 'p'
        print(s.tobytes().decode('utf-8'))
    

def test():
    env = ShortcutEnvironment()
    validStep = True
    step_nr = 0
    print('starting position')
    env.render()
    while (validStep and not env.done()):
        stateBefore = env.state()
        env.step(1)                     # move one step down
        step_nr += 1
        print(f'step {step_nr}')
        env.render()
        stateAfter = env.state()
        if stateAfter == stateBefore:   # agent moved into wall
            validStep = False

if __name__ == '__main__':
    test()
