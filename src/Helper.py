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

import os, shutil                   # directory management 
import numpy as np                  # arrays
import matplotlib.pyplot as plt     # plots

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

def progress(iteration: int, n_iters: int) -> None:
    """
    Prints a progress bar to show that the script is actually running
    ---
    params:
        - iteration: the iteration that is currently being processed
        - n_iters: the total number of iterations that are to be processed
    """
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

def fix_directories() -> str:
    """
    Make sure the current working directory is `src` and an empty `results` directory is present at the same level as `src`
    ---
    return:
        - path where all results are to be stored
    """
    cwd = os.getcwd()
    if cwd.split(os.sep)[-1] != 'src':
        if not os.path.exists(os.path.join(cwd, 'src')):
            raise FileNotFoundError('Please work from either the parent directory "ShortCut" or directly from "src".')
        os.chdir(os.path.join(cwd, 'src'))
        cwd = os.getcwd()
        print(f'Working directory changed to "{cwd}"')

    resultspath = os.path.join('..', 'results')
     
    if os.path.exists(resultspath):
        shutil.rmtree(resultspath)
    os.mkdir(resultspath)
    return resultspath

def main():
    print('Do not run this file directly')

if __name__ == '__main__':
    main()
