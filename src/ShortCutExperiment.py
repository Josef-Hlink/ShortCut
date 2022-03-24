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

import os, shutil                   # directory management 
import numpy as np                  # arrays
import time                         # calculating runtime
from datetime import datetime       # time that the experiment has started
from Helper import LearningCurvePlot, PathPlot, progress
from ShortCutAgents import Agent, QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import Environment, ShortcutEnvironment, WindyShortcutEnvironment

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
                progress(repetition, self.nreps)
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
    
def run_greedy_episode(env: Environment, agent: Agent, start_at: tuple[int] = (2,2)) -> tuple[list]:
    """
    Run one greedy episode based on an agent that has already trained
    ---
    params:
        - env: an Environment instance the agent is to interact with
        - agent: a greedy Agent instance that has already been trained
        - start_at: the agent's starting position
    
    returns:
        - y_p: [p for path] list containing the agent's y position per step
        - x_p: [p for path] list containing the agent's x position per step
    """
    y_p, x_p = [], []
    env.y, env.x = start_at[0], start_at[1] # place the agent in the environment at desired starting position
    
    s: int = env.state()                    # retrieve current (starting) state
    while not env.done():
        a: int = agent.select_action(s)     # choose action
        env.step(a)                         # perform action
        s = env.state()                     # update state
        y_p.append(env.y)                   # store new position
        x_p.append(env.x)
    y_p.pop()   # remove endpoint state
    x_p.pop()
    return y_p, x_p

def path_plot(agent: Agent, path1, path2, windy=False) -> None:    
    env = ShortcutEnvironment()
    Qvalues = agent.Q

    kind = 'Windy' if windy else ''
    plot = PathPlot(title=f'{kind} Path Plot ({agent.lname})')

    maxQs = np.amax(Qvalues, axis=1).reshape((12,12))
    plot.add_Q_values(maxQs)

    x_s = [2, 2]
    y_s = [[2], [9]]
    plot.add_starting_positions(x_s, y_s)

    y_e, x_e = np.where(env.s == 'G')
    plot.add_goal_position(x_e, y_e)

    y_c, x_c = np.where(env.s == 'C')
    plot.add_cliffs(x_c, y_c)

    plot.add_path(path1[1], path1[0], path_nr = 1)
    plot.add_path(path2[1], path2[0], path_nr = 2)

    char = 'w' if windy else ''
    plot.save(name = os.path.join(RESULTSPATH, f'{char}pp{agent.sname}.png'))
    return

def Q_Learning() -> None:
    print('Running for pathplots...')
    
    # Heatmap Normal Environment
    exp = Experiment(n_states=144, n_actions=4, n_episodes = 10000, n_repetitions = 1, epsilon = 0.1, alpha = 0.1)
    trainedAgent: Agent = exp(ShortcutEnvironment, QLearningAgent)
    trainedAgent.epsilon = 0.0          # make the agent perform a greedy run
    path1 = run_greedy_episode(ShortcutEnvironment(), trainedAgent, start_at=(2,2))
    path2 = run_greedy_episode(ShortcutEnvironment(), trainedAgent, start_at=(9,2))

    path_plot(trainedAgent, path1, path2)
    progress(0, 2)

    # Heatmap Windy Environment
    exp = Experiment(n_states=144, n_actions=4, n_episodes = 10000, n_repetitions = 1, epsilon = 0.1, alpha = 0.1)
    trainedAgent: Agent = exp(WindyShortcutEnvironment, QLearningAgent)
    trainedAgent.epsilon = 0.0          # make the agent perform a greedy run
    path1 = run_greedy_episode(WindyShortcutEnvironment(), trainedAgent, start_at=(2,2))
    path2 = run_greedy_episode(WindyShortcutEnvironment(), trainedAgent, start_at=(9,2))

    path_plot(trainedAgent, path1, path2, windy=True)
    progress(1, 2)

    # Learning Curve
    rewards_for_alpha: dict[float: np.array] = {0.01: None, 0.1: None, 0.5: None, 0.9: None}
    for alpha in rewards_for_alpha.keys():
        print(f'Running 100 repetitions for {alpha = }...')
        exp = Experiment(n_states=144, n_actions=4, n_episodes = 1000, n_repetitions=100, epsilon = 0.1, alpha=alpha)
        rewards_for_alpha[alpha] = exp(ShortcutEnvironment, QLearningAgent)
    
    plot = LearningCurvePlot(title = 'Learning Curve (Q Learning)')
    for index, (alpha, rewards) in enumerate(rewards_for_alpha.items()):
        plot.add_curve(y = rewards, color_index = index, label = f'α = {alpha}')
    plot.save(name = os.path.join(RESULTSPATH, 'lcQL.png'))
    return

def SARSA() -> None:
    print('Running for pathplots...')

    # Heatmap Normal Environment
    exp = Experiment(n_states=144, n_actions=4, n_episodes = 10000, n_repetitions = 1, epsilon = 0.1, alpha = 0.1)
    trainedAgent = exp(ShortcutEnvironment, SARSAAgent)
    trainedAgent.epsilon = 0.0          # make the agent perform a greedy run
    path1 = run_greedy_episode(ShortcutEnvironment(), trainedAgent, start_at=(2,2))
    path2 = run_greedy_episode(ShortcutEnvironment(), trainedAgent, start_at=(9,2))

    path_plot(trainedAgent, path1, path2)
    progress(0, 2)

    # Heatmap Windy Environment
    exp = Experiment(n_states=144, n_actions=4, n_episodes = 10000, n_repetitions = 1, epsilon = 0.1, alpha = 0.1)
    trainedAgent: Agent = exp(WindyShortcutEnvironment, SARSAAgent)
    trainedAgent.epsilon = 0.0          # make the agent perform a greedy run
    path1 = run_greedy_episode(WindyShortcutEnvironment(), trainedAgent, start_at=(2,2))
    path2 = run_greedy_episode(WindyShortcutEnvironment(), trainedAgent, start_at=(9,2))

    path_plot(trainedAgent, path1, path2, windy=True)
    progress(1, 2)

    # Learning Curve
    rewards_for_alpha: dict[float: np.array] = {0.01: None, 0.1: None, 0.5: None, 0.9: None}
    for alpha in rewards_for_alpha.keys():
        print(f'Running 100 repetitions for {alpha = }...')
        exp = Experiment(n_states=144, n_actions=4, n_episodes = 1000, n_repetitions=100, epsilon = 0.1, alpha=alpha)
        rewards_for_alpha[alpha] = exp(ShortcutEnvironment, SARSAAgent)
    
    plot = LearningCurvePlot(title = 'Learning Curve (SARSA)')
    for index, (alpha, rewards) in enumerate(rewards_for_alpha.items()):
        plot.add_curve(y = rewards, color_index = index, label = f'α = {alpha}')
    plot.save(name = os.path.join(RESULTSPATH, 'lcSARSA.png'))
    return

def ESARSA() -> None:
    print('Running for pathplots...')

    # Heatmap Normal Environment
    exp = Experiment(n_states=144, n_actions=4, n_episodes = 10000, n_repetitions = 1, epsilon = 0.1, alpha = 0.1)
    trainedAgent = exp(ShortcutEnvironment, ExpectedSARSAAgent)
    trainedAgent.epsilon = 0.0          # make the agent perform a greedy run
    path1 = run_greedy_episode(ShortcutEnvironment(), trainedAgent, start_at=(2,2))
    path2 = run_greedy_episode(ShortcutEnvironment(), trainedAgent, start_at=(9,2))

    path_plot(trainedAgent, path1, path2)
    progress(0, 2)

    # Heatmap Windy Environment
    exp = Experiment(n_states=144, n_actions=4, n_episodes = 10000, n_repetitions = 1, epsilon = 0.1, alpha = 0.1)
    trainedAgent: Agent = exp(WindyShortcutEnvironment, ExpectedSARSAAgent)
    trainedAgent.epsilon = 0.0          # make the agent perform a greedy run
    path1 = run_greedy_episode(WindyShortcutEnvironment(), trainedAgent, start_at=(2,2))
    path2 = run_greedy_episode(WindyShortcutEnvironment(), trainedAgent, start_at=(9,2))

    path_plot(trainedAgent, path1, path2, windy=True)
    progress(1, 2)

    # Learning Curve
    rewards_for_alpha: dict[float: np.array] = {0.01: None, 0.1: None, 0.5: None, 0.9: None}
    for alpha in rewards_for_alpha.keys():
        print(f'Running 100 repetitions for {alpha = }...')
        exp = Experiment(n_states=144, n_actions=4, n_episodes = 1000, n_repetitions=100, epsilon = 0.1, alpha=alpha)
        rewards_for_alpha[alpha] = exp(ShortcutEnvironment, ExpectedSARSAAgent)
    
    plot = LearningCurvePlot(title = 'Learning Curve (Expected SARSA)')
    for index, (alpha, rewards) in enumerate(rewards_for_alpha.items()):
        plot.add_curve(y = rewards, color_index = index, label = f'α = {alpha}')
    plot.save(name = os.path.join(RESULTSPATH, 'lcESARSA.png'))
    return

def main():
    cwd = os.getcwd()
    if cwd.split(os.sep)[-1] != 'src':
        if not os.path.exists(os.path.join(cwd, 'src')):
            raise FileNotFoundError('Please work from either the parent directory "ShortCut" or directly from "src".')
        os.chdir(os.path.join(cwd, 'src'))
        cwd = os.getcwd()
        print(f'Working directory changed to "{cwd}"')

    global RESULTSPATH
    RESULTSPATH = os.path.join('..', 'results')
     
    if os.path.exists(RESULTSPATH):
        shutil.rmtree(RESULTSPATH)
        os.mkdir(RESULTSPATH)

    start: float = time.perf_counter()              # <-- timer start
    print(f'\nStarting experiment at {datetime.now().strftime("%H:%M:%S")}\n')

    print('\033[1m---Q Learning---\033[0m')
    Q_Learning()

    print('\n\033[1m---SARSA---\033[0m')
    SARSA()

    print('\n\033[1m---Expected SARSA---\033[0m')
    ESARSA()
    
    end: float = time.perf_counter()                # <-- timer end
    minutes: int = int((end-start) // 60)
    seconds: float = round((end-start) % 60, 1)
    stringtime: str = f'{minutes}:{str(seconds).zfill(4)} min' if minutes else f'{seconds} sec'
    print(f'\nExperiment finished in {stringtime}\n')

if __name__ == '__main__':
    main()
