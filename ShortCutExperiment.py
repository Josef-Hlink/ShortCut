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

import numpy as np                  # arrays
import time                         # calculating runtime
from datetime import datetime       # time that the experiment has started
from Helper import LearningCurvePlot, PathPlot
from ShortCutAgents import Agent, QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import Environment, ShortcutEnvironment

class Experiment:
    def __init__(self, n_states: int = 144, n_actions: int = 4, n_episodes: int = 1000, n_repetitions: int = 1,
                 epsilon: float = 0.01, alpha: float = 0.01):
        self.ns = n_states          # number of possible states in the environment
        self.na = n_actions         # number of actions that the agent can take
        self.ne = n_episodes        # number of episodes that the agent trains for
        self.nreps = n_repetitions  # number of times an experiment is to be repeated
        self.epsilon = epsilon      # tunes the greediness of the agent ( = exploration rate )
        self.alpha = alpha          # tunes how heavy the agent weighs new information ( = learning rate )
    
    def __call__(self, environment_type: object, agent_type: object):
        return self.run_repetitions(environment_type, agent_type)

    def run_repetitions(self, environment_type: object, agent_type: object) -> np.array:

        NewAgent = agent_type
        NewEnvironment = environment_type

        cum_rs_per_repetition = np.zeros(shape=(self.nreps, self.ne))

        for repetition in range(self.nreps):
            if self.nreps > 1:
                progress(repetition+1, self.nreps)
            env = NewEnvironment()                                          # initialize environment
            agent = NewAgent(self.na, self.ns, self.epsilon, self.alpha)    # initialize agent
            cum_rs = np.zeros(self.ne)
            for episode in range(self.ne):
                cum_rs[episode] = self.run_episode(env, agent)

            cum_rs_per_repetition[repetition] = cum_rs
        cum_r_per_episode: np.array = np.average(cum_rs_per_repetition, axis=0)
        
        if self.nreps == 1:     # NOTE this is a bit hacky, but it allows us to pass an otherwise inaccessible property of the agent
            return agent        # which we need to plot the heatmap of max Q values
        
        return cum_r_per_episode    # otherwise we simply return the cum_r's for every episode
    
    def run_episode(self, env: Environment, agent: Agent) -> int:
        env.reset()             # (re)set environment ( = place agent back at starting position )
        cum_r: int = 0          # cumulative reward
        s: int = env.state()    # retrieve current (starting) state
        while not env.done():
            a: int = agent.select_action(s)     # choose action
            r: int = env.step(a)                # get reward by performing action on environment
            cum_r += r                          # update total reward
            next_s = env.state()                # retrieve what state the chosen action will result in
            agent.update(s, a, r, next_s)       # update policy
            s: int = next_s                     # change the current state to be next state
        
        return cum_r


def path_plot(agent: Agent) -> None:    
    env = ShortcutEnvironment()
    Qvalues = agent.Q

    plot = PathPlot(title=f'Path Plot {agent.lname}')

    maxQs = np.amax(Qvalues, axis=1).reshape((12,12))
    plot.add_Q_values(maxQs)

    x_s = [2, 2]
    y_s = [[2], [9]]
    plot.add_starting_positions(x_s, y_s)

    y_e, x_e = np.where(env.s == 'G')
    plot.add_goal_position(x_e, y_e)

    y_c, x_c = np.where(env.s == 'C')
    plot.add_cliffs(x_c, y_c)

    plot.save(f'pp{agent.sname}.png')
    return

def Q_Learning() -> None:
    # Heatmap
    print('Running for pathplot...')
    exp = Experiment(n_states=144, n_actions=4, n_episodes = 10000, n_repetitions = 1, epsilon = 0.1, alpha = 0.1)
    trainedAgent = exp(ShortcutEnvironment, QLearningAgent)
    print('done\n')
    path_plot(trainedAgent)

    # Learning Curve
    rewards_for_alpha: dict[float: np.array] = {0.01: None, 0.1: None, 0.5: None, 0.9: None}
    for alpha in rewards_for_alpha.keys():
        print(f'Running for {alpha = }...')
        exp = Experiment(n_states=144, n_actions=4, n_episodes = 1000, n_repetitions=100, epsilon = 0.1, alpha=alpha)
        rewards_for_alpha[alpha] = exp(ShortcutEnvironment, QLearningAgent)
    
    plot = LearningCurvePlot(title = 'Learning Curve (Q Learning)')
    for index, (alpha, rewards) in enumerate(rewards_for_alpha.items()):
        plot.add_curve(y = rewards, color_index = index, label = f'α = {alpha}')
    plot.save(name = 'lcQL.png')
    return

def SARSA() -> None:
    # Heatmap
    print('Running for pathplot...')
    exp = Experiment(n_states=144, n_actions=4, n_episodes = 10000, n_repetitions = 1, epsilon = 0.1, alpha = 0.1)
    Qvalues = exp(ShortcutEnvironment, SARSAAgent)
    print('done\n')
    path_plot(Qvalues)

    # Learning Curve
    rewards_for_alpha: dict[float: np.array] = {0.01: None, 0.1: None, 0.5: None, 0.9: None}
    for alpha in rewards_for_alpha.keys():
        print(f'Running for {alpha = }...')
        exp = Experiment(n_states=144, n_actions=4, n_episodes = 1000, n_repetitions=100, epsilon = 0.1, alpha=alpha)
        rewards_for_alpha[alpha] = exp(ShortcutEnvironment, SARSAAgent)
    
    plot = LearningCurvePlot(title = 'Learning Curve (SARSA)')
    for index, (alpha, rewards) in enumerate(rewards_for_alpha.items()):
        plot.add_curve(y = rewards, color_index = index, label = f'α = {alpha}')
    plot.save(name = 'lcSARSA.png')
    return

def progress(iteration: int, n_iters: int) -> None:
    step = int(iteration/n_iters * 50)
    fill = '\033[0;47m\033[1m\033[94m-\033[0m'
    empty = '\033[0;47m \033[0m'
    print('\r|' + step*fill + (50-step)*empty + '| ' + str(step*2) + '% done', end='')
    if iteration == n_iters:
        print('')
    return

def main():
    start: float = time.perf_counter()              # <-- timer start
    print(f'\nStarting experiment at {datetime.now().strftime("%H:%M:%S")}\n')

    print('\033[1m---Q Learning---\033[0m')
    Q_Learning()

    print('\n\033[1m---SARSA---\033[0m')
    SARSA()
    
    end: float = time.perf_counter()                # <-- timer end
    minutes: int = int((end-start) // 60)
    seconds: float = round((end-start) % 60, 1)
    stringtime: str = f'{minutes}:{str(seconds).zfill(4)} min' if minutes else f'{seconds} sec'
    print(f'\nExperiment finished in {stringtime}\n')

if __name__ == '__main__':
    main()
