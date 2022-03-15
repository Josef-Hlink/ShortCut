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
from matplotlib import pyplot as plt

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
        temp = np.zeros(shape=(self.n_repetitions, self.n_episodes))

        for repetition in range(self.n_repetitions):
            env = ShortcutEnvironment()                     # initialize environment | NOTE no params?
            agent = QLearningAgent(self.n_actions, self.n_states, epsilon)     # initialize agent
            #Rs = np.zeros(self.n_episodes)                  # the rewards for all timesteps will be stored here
            steps_required = [0 for _ in range(self.n_episodes)]
            cum_rarray = np.zeros(self.n_episodes)

            for episode in range(self.n_episodes):
                # if episode % 500 == 0:
                #     print(episode)
                s: int = env.state()
                steps = 0
                allStates = []
                cum_r = 0

                while not env.done():
                    a: int = agent.select_action(s, epsilon)    # choose action
                    r: int = env.step(a)                        # get reward by performing action on environment
                    cum_r += r
                    next_state = env.state()
                    agent.update(s, a, r, alpha, next_state)                # update policy
                    # if episode == self.n_episodes - 1:
                    #     print('state', s, '--> state', next_state)
                    #     print(s%12, ',', s//12, '-->', next_state%12, ',', next_state//12)
                    #     allStates.append(s)
                    #     print('Qs:', agent.Q[s], 'Qsp:', agent.Q[next_state])
                    #     env.render()
                    s: int = next_state
                    steps += 1
                cum_rarray[episode] = cum_r

                env.reset()
                
                #steps_required[episode] = steps
            temp[repetition] = cum_rarray
            #all_Rs[repetition] = Rs                         # store vector of rewards
        res = np.average(temp, axis=0)
        return res
        
        # for el in steps_required:
        #     print(el)
        # Qperrow = [agent.Q[i:i+12] for i in range(0, 144, 12)]
        # Nperrow = [agent.N[i:i+12] for i in range(0, 144, 12)]
        # for row in range(12):
        #     print('Row number:', row)
        #     print('Q:')
        #     print(np.round(Qperrow[row], 3))
        #     print('N:')
        #     print(Nperrow[row])

        # grid = np.zeros((12,12))
        # for state in allStates:
        #     x, y = state%12, state//12
        #     grid[y][x] += 1
        # print(grid)

        maxQs = np.zeros((12,12))
        for row in range(12):
            for col in range(12):
                maxQs[row][col] = np.max(agent.Q[row*12+col]).round(1)
            
            #print(maxQs)

        # x_s = [2, 2]
        # y_s = [[2], [9]]

        # y_c, x_c = np.where(env.s == 'C')
        # y_e, x_e = np.where(env.s == 'G')

        # plt.imshow(maxQs, cmap='hot', interpolation='none')
        # plt.scatter(x_s, y_s, marker="o", s=100, c="white", edgecolor='black', label='starting points')
        # plt.scatter(x_e, y_e, marker="o", s=100, c="black", edgecolor='white', label='end point')

        # plt.scatter(x_c, y_c, marker="x", s=100, c="black", label='cliffs')
        # plt.legend()
        # plt.show()

        # plot = LearningCurvePlot(title='Q "Learning"')
        # plot.add_curve(steps_required, color_index = 0, label='Idiot agent')
        # plot.save(name='test.png')
        #return np.average(all_Rs, 0)
    
    def run(self):
        plot = LearningCurvePlot(title='Q Learning')
        epsilon = 0.1
        alphas = [0.01, 0.1, 0.5, 0.9]
        for index, alpha in enumerate(alphas):
            rewards = self.run_repetitions(epsilon, alpha)
            plot.add_curve(y=rewards, color_index=index, label = f'alpha: {alpha}')
        plot.save(name='QL')

        #rewards = np.round(rewards, 2)

def main():
    exp = Experiment(n_states=144, n_actions=4, n_episodes = 1000, n_repetitions=10, smoothing_window=31)
    exp.run()

if __name__ == '__main__':
    main()
