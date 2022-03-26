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

import os                           # directory management 
import numpy as np                  # arrays
import time                         # calculating runtime
from datetime import datetime       # time that the experiment has started
from Helper import LearningCurvePlot, PathPlot, fix_directories, progress
from ShortCutAgents import Agent, QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import Environment, ShortcutEnvironment, WindyShortcutEnvironment

def main():
    global RESULTSPATH
    RESULTSPATH = fix_directories()

    start: float = time.perf_counter()              # <-- timer start
    print(f'\nStarting experiment at {datetime.now().strftime("%H:%M:%S")}')

    run_experiments_for_agent(QLearningAgent)
    run_experiments_for_agent(SARSAAgent)
    run_experiments_for_agent(ExpectedSARSAAgent)

    end: float = time.perf_counter()                # <-- timer end
    minutes: int = int((end-start) // 60)
    seconds: float = round((end-start) % 60, 1)
    stringtime: str = f'{minutes}:{str(seconds).zfill(4)} min' if minutes else f'{seconds} sec'
    print(f'\nExperiment finished in {stringtime}\n')

class Experiment:
    def __init__(self, n_states: int = 144, n_actions: int = 4, n_episodes: int = 1000, n_repetitions: int = 1,
                 epsilon: float = 0.01, alpha: float = 0.01):
        """
        Initialize an experiment where a number of repetitions are ran
        ---
        params:
            - n_states: number of possible states in the environment
            - n_actions: number of possible actions the agent can choose from
            - n_episodes: number of episodes the agent trains for
            - n_repetitions: number of times a single experiment is to be repeated
            - epsilon: tunes the greediness of the agent ( = exploration rate )
            - alpha: tunes how heavy the agent weighs new information ( = learning rate )
        ---
        NOTE: Agent and Environment types are not specified yet, this is done when the experiment is actually called.
        """
        self.ns = n_states
        self.na = n_actions
        self.ne = n_episodes
        self.nreps = n_repetitions
        self.epsilon = epsilon
        self.alpha = alpha
    
    def __call__(self, environment_type: object, agent_type: object):
        """Wrapper for run_repetitions"""
        return self.run_repetitions(environment_type, agent_type)

    def run_repetitions(self, environment_type: object, agent_type: object) -> np.array:
        """
        Run the experiment with a specified environment and agent
        ---
        params:
            - environment_type: an Environment class, NOT an instance of Environment
            - agent_type: an Agent class, NOT an instance of Agent
        
        returns:
            - a trained instance of some Agent (if number of repetitions is 1)
            - the average cumulative reward obtained per timestep (otherwise)
        """
        # rename parameters for grammatical reasons
        NewAgent = agent_type
        NewEnvironment = environment_type

        # 2D array where every single cumulative reward will be stored
        cum_rs_per_repetition = np.zeros(shape=(self.nreps, self.ne))

        for repetition in range(self.nreps):
            if self.nreps > 1:
                progress(repetition, self.nreps)                            # print progress bar
            env = NewEnvironment()                                          # initialize environment
            agent = NewAgent(self.na, self.ns, self.epsilon, self.alpha)    # initialize agent
            cum_rs = np.zeros(self.ne)
            for episode in range(self.ne):
                cum_rs[episode] = self.run_episode(env, agent)              # <-- run episode and store cumulative reward
            cum_rs_per_repetition[repetition] = cum_rs
        
        # average out the repetitions to get reward per episode
        cum_r_per_episode: np.array = np.average(cum_rs_per_repetition, axis=0)
        
        if self.nreps == 1:         # NOTE this is a bit hacky, but it allows us to pass an otherwise inaccessible property of the agent
            return agent            # which we need to plot the heatmap of max Q values
        
        return cum_r_per_episode    # otherwise we simply return the cum_r's for every episode
    
    def run_episode(self, env: Environment, agent: Agent) -> int:
        """
        Run a single training episode with a specified environment and agent
        ---
        params:
            - environment_type: an initialized Environment instance
            - agent_type: an initialized Agent instance
        
        returns:
            - cumulative (total) reward obtained during the episode
        """
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
    Run one greedy episode with an agent that has already trained
    ---
    params:
        - env: an Environment instance the agent is to interact with
        - agent: a greedy Agent instance that has already been trained
        - start_at: the agent's starting position
    
    returns:
        - y_p: [p for path] list containing the agent's y position per step
        - x_p: [p for path] list containing the agent's x position per step
        - a_p: [p for path] list containing the action chosen by the agent per step
    """
    # place the agent in the environment at desired starting position
    env.y, env.x = start_at[0], start_at[1]
    
    x_p, y_p, a_p = [], [], []
    s: int = env.state()                    # retrieve current (starting) state
    while not env.done():
        a: int = agent.select_action(s)     # choose action
        env.step(a)                         # perform action
        s = env.state()                     # update state
        y_p.append(env.y)                   # store new position
        x_p.append(env.x)                   # ""
        a_p.append(a)                       # and chosen action
    y_p.pop()   # remove endpoint state
    x_p.pop()   # ""
    a_p.pop(0)  # and first action
    return x_p, y_p, a_p

def run_experiments_for_agent(agent_type: object) -> None:
    """
    Main function that handles all three experiments for every one of the three agent types
    ---
    param:
        - agent_type: the type of agent that will be used for the three experiments
    """
    # rename parameter for grammatical reasons
    newAgent = agent_type
    agent: Agent = newAgent()

    # print formatted string to indicate what agent is being experimented with
    print(f'\n\033[1m---{agent.lname}---\033[0m')

    print('Running for pathplots...')
    # Path Plots
    environments = [ShortcutEnvironment, WindyShortcutEnvironment]
    for index, environment in enumerate(environments):
        exp = Experiment(n_states=144, n_actions=4, n_episodes = 10_000, n_repetitions = 1, epsilon = 0.1, alpha = 0.1)
        trainedAgent: Agent = exp(environment, newAgent)
        trainedAgent.epsilon = 0.0          # make the agent perform a greedy run
        path1 = run_greedy_episode(environment(), trainedAgent, start_at=(2,2))
        path2 = run_greedy_episode(environment(), trainedAgent, start_at=(9,2))

        path_plot(trainedAgent, path1, path2, windy=index)  # param windy takes a bool, so index 0 will amount to False and 1 to True
        progress(index, 2)

    # Learning Curves
    rewards_for_alpha: dict[float: np.array] = {0.01: None, 0.1: None, 0.5: None, 0.9: None}
    for alpha in rewards_for_alpha.keys():
        print(f'Running 100 repetitions for {alpha = }...')
        exp = Experiment(n_states=144, n_actions=4, n_episodes = 1000, n_repetitions=100, epsilon = 0.1, alpha=alpha)
        rewards_for_alpha[alpha] = exp(ShortcutEnvironment, newAgent)
    
    plot = LearningCurvePlot(title = f'Learning Curve ({agent.lname})')
    for index, (alpha, rewards) in enumerate(rewards_for_alpha.items()):
        plot.add_curve(y = rewards, color_index = index, label = f'α = {alpha}')
    plot.save(name = os.path.join(RESULTSPATH, f'lc{agent.sname}.png'))
    return

def path_plot(agent: Agent, path1, path2, windy=False) -> None:
    """
    Get all relevant data and create an instance of  Helper.py's `PathPlot`
    ---
    params:
        - agent: the trained Agent instance of which we want to visualize its Q gradient
        - path1: tuple containing three lists (y & x coordinates of the path, and the actions chosen in each position)
        - path2: tuple containing three lists (y & x coordinates of the path, and the actions chosen in each position)
        - windy: boolean that indicates the environment's weather (for title and filename purposes)
    """
    env = ShortcutEnvironment()
    Qvalues = agent.Q

    kind = 'Windy' if windy else ''
    plot = PathPlot(title=f'{kind} Path Plot ({agent.lname})')

    # store only the maximal Q value for each position, as the agent is greedy
    maxQs = np.amax(Qvalues, axis=1).reshape((12,12))
    plot.add_Q_values(maxQs)

    # send starting positions to plot
    x_s = [2, 2]
    y_s = [[2], [9]]
    plot.add_starting_positions(x_s, y_s)

    # send goal position to plot
    y_e, x_e = np.where(env.s == 'G')
    plot.add_goal_position(x_e, y_e)

    # send cliff positions to plot
    y_c, x_c = np.where(env.s == 'C')
    plot.add_cliffs(x_c, y_c)

    # add actual paths to plot
    plot.add_path(x=path1[0], y=path1[1], actions=path1[2], path_nr=1)
    plot.add_path(x=path2[0], y=path2[1], actions=path2[2], path_nr=2)

    char = 'w' if windy else ''
    plot.save(name = os.path.join(RESULTSPATH, f'{char}pp{agent.sname}.png'))
    return

if __name__ == '__main__':
    main()
