#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
Template provided by: Thomas Moerland
Implementation by: Josef Hamelink & Ayush Kandhai
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Timesteps')
        self.ax.set_ylabel('Reward')
        self.ax.set_ylim([0,1.0])
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan'] # color cycle
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self, y, color_index: int, label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y, label=label, color=self.colors[color_index])
        else:
            self.ax.plot(y, color=self.colors[color_index], alpha=0.3)
        
    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

class ComparisonPlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Parameter (exploration)')
        self.ax.set_ylabel('Average reward') 
        self.ax.set_xscale('log')
        self.ax.set_xticks(ticks=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
                           labels=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
        self.ax.set_ylim([0.70,0.90])
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan'] # color cycle
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self, x, y, color_index: int, label=None):
        ''' x: vector of parameter values
        y: vector of associated mean reward for the parameter values in x 
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(x, y, color=self.colors[color_index], label=label)
        else:
            self.ax.plot(x, y, color=self.colors[color_index], alpha=0.3)
        
    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

def close_figures() -> None:
    '''so we don't need to import matplotlib in BanditExperiment'''
    return plt.close('all')

if __name__ == '__main__':
    # Test Learning curve plot
    x = np.arange(100)
    y = 0.01*x + np.random.rand(100) - 0.4 # generate some learning curve y
    LCTest = LearningCurvePlot(title="Test Learning Curve")
    LCTest.add_curve(y,label='method 1')
    LCTest.add_curve(smooth(y,window=35),label='method 1 smoothed')
    LCTest.save(name='learning_curve_test.png')

    # Test Performance plot
    PerfTest = ComparisonPlot(title="Test Comparison")
    PerfTest.add_curve(np.arange(5),np.random.rand(5),label='method 1')
    PerfTest.add_curve(np.arange(5),np.random.rand(5),label='method 2')
    PerfTest.save(name='comparison_test.png') 