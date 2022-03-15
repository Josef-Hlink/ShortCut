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

import numpy as np
from Helper import LearningCurvePlot
from ShortCutAgents import QLearningAgent
from ShortCutEnvironment import ShortcutEnvironment
#         ^                          ^ wtf is this lmao

class Experiment:
    def __init__(self, n_states: int, n_actions: int, n_episodes: int, n_repetitions: int, smoothing_window: int):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_episodes = n_episodes
        self.n_repetitions = n_repetitions
        self.smoothing_window = smoothing_window
    
    def run_repetitions(self, epsilon: float, alpha: float):
        #all_Rs = np.zeros(shape=(self.n_repetitions, self.n_episodes), dtype=np.int8)
        #dir = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

        for repetition in range(self.n_repetitions):
            env = ShortcutEnvironment()                     # initialize environment | NOTE no params?
            agent = QLearningAgent(self.n_actions, self.n_states, epsilon)     # initialize agent
            #Rs = np.zeros(self.n_episodes)                  # the rewards for all timesteps will be stored here
            s: int = env.state()
            steps_required = [0 for _ in range(self.n_episodes)]

            for episode in range(self.n_episodes):
                if episode % 500 == 0:
                    print(episode)
                env.reset()
                steps = 0
                allStates = []
                while not env.done():
                    a: int = agent.select_action(s, epsilon)    # choose action
                    r: int = env.step(a)                        # get reward by performing action on environment
                    next_state = env.state()
                    agent.update(s, a, r, alpha, next_state)                # update policy
                    if episode == self.n_episodes - 1:
                        print('state', s, '--> state', next_state)
                        allStates.append(s)
                        env.render()
                    s: int = next_state
                    steps += 1
                steps_required[episode] = steps
            #all_Rs[repetition] = Rs                         # store vector of rewards
        for el in steps_required:
            print(el)
        Qperrow = [agent.Q[i:i+11] for i in range(0, 144, 12)]
        Nperrow = [agent.N[i:i+11] for i in range(0, 144, 12)]
        for row in range(12):
            print('Row number:', row)
            print('Q:')
            print(np.round(Qperrow[row], 3))
            print('N:')
            print(Nperrow[row])

        grid = np.zeros((12,12))
        for state in allStates:
            x, y = state%12, state//12
            grid[y][x] += 1
        print(grid)
        
        plot = LearningCurvePlot(title='Q "Learning"')
        plot.add_curve(steps_required, color_index = 0, label='Idiot agent')
        plot.save(name='test.png')
        #return np.average(all_Rs, 0)
    
    def run(self):
        epsilon = 0.1
        alpha = 0.1
        rewards = self.run_repetitions(epsilon, alpha)
        #rewards = np.round(rewards, 2)

def main():
    exp = Experiment(n_states=144, n_actions=4, n_episodes = 1000, n_repetitions=1, smoothing_window=31)
    exp.run()

if __name__ == '__main__':
    main()