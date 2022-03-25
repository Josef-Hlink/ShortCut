#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2022
Template provided by: Thomas Moerland
Implementation by: Josef Hamelink & Ayush Kandhai
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Cumulative reward')
        #self.ax.set_ylim([-5000,0])
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan'] # color cycle
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self, y, color_index: int=0, label=None):
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

class PathPlot:
    def __init__(self, title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('column')
        self.ax.set_ylabel('row')
        labelrange = [i for i in range(12)]
        self.ax.set_xticks(ticks=labelrange, labels=labelrange)
        self.ax.set_yticks(ticks=labelrange, labels=labelrange)
        self.ax.tick_params(axis='both', bottom=False, left=False)
        if title is not None:
            self.ax.set_title(title)
    
    def add_Q_values(self, Qvalues: np.array):
        self.ax.imshow(Qvalues, cmap='bone', interpolation='none')

    def add_starting_positions(self, x: list[int], y: list[list[int]]):
        self.ax.scatter(x, y, marker="o", s=100, c="black", edgecolor='white', label='starting pos.')

    def add_goal_position(self, x: int, y: int):
        self.ax.scatter(x, y, marker="o", s=100, c="white", edgecolor='black', label='goal pos.')

    def add_cliffs(self, x: list[list[int]], y: list[list[int]]):
        self.ax.scatter(x, y, marker="x", s=100, c="black", label='cliffs')
    
    def add_path(self, x: list[int], y: list[int], actions: list[int], path_nr: int = 1):
        x, y = np.array(x), np.array(y)
        actions = np.array(actions)
        get_marker = {0: '^', 1: 'v', 2: '<', 3: '>'}
        unique_actions = np.unique(actions)
        
        for u_a in unique_actions:
            mask = actions == u_a
            l = f'path {path_nr}' if u_a == 3 else ''
            if path_nr == 1:
                self.ax.scatter(x[mask], y[mask], marker=get_marker[u_a], s=100, c='violet', edgecolors='magenta', alpha=0.3, label=l)
            elif path_nr == 2:
                self.ax.scatter(x[mask], y[mask], marker=get_marker[u_a], s=100, c='skyblue', edgecolors='teal', alpha=0.3, label=l)

    def save(self, name: str = 'test.png'):
        self.ax.legend(loc=4, markerscale=0.6, prop={'size': 6})
        self.fig.savefig(name,dpi=300)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

def close_figures() -> None:
    '''so we don't need to import matplotlib in BanditExperiment'''
    return plt.close('all')

def progress(iteration: int, n_iters: int) -> None:
    steps = int(50 * (iteration+1) // n_iters)			        # number of characters that will represent the progress
    percentage = round(100 * (iteration+1) / float(n_iters))    # rounded percentage
    
    # white bg, bold char, blue char, reset after
    format_done = lambda string: f'\033[0;47m\033[1m\033[94m{string}\033[0m'
    # white bg, bold char, lightgrey char, reset after
    format_todo = lambda string: f'\033[0;47m\033[1m\033[2m{string}\033[0m'
    done_char = format_done('━')
    todo_char = format_todo('━')

    frames = ['/', '-', '\\', '|']
    frame = frames[percentage%4]
    spin_char = format_done(frame)
    
    bar = (steps)*done_char + (50-steps)*todo_char		# the actual bar
    suffix = spin_char + ' ' + str(percentage) + '%'    # spinner and percentage
    
    print('\r|' + bar + '| ' + suffix, end='')          # print bar
    if percentage == 100:						    # for last iteration
        print('\r|' + 50*done_char + '| complete')	    # print bar, with an actual endline character
    return

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