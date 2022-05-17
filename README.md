# RL Agents on simple 2D Environment with Shortcut

## Preface

This project was done as the 2<sup>nd</sup> programming assignment of Leiden University's 3<sup>rd</sup> year Computer Science (AI flavoured) course.
The templates (most notably ShortCutEnvironment) were provided by [Aske Plaat](https://www.universiteitleiden.nl/en/staffmembers/aske-plaat#tab-1), [Thomas Moerland](https://www.universiteitleiden.nl/medewerkers/thomas-moerland#tab-1) & [Daan Pelt](https://www.universiteitleiden.nl/en/staffmembers/daan-pelt#tab-1).

## Usage

1. Clone the project
2. In your terminal of choice, `cd` into "ShortCut"
3. Type the following commands to set up project

```
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

4. Run the experiment

```
(venv) $ cd src
(venv) $ chmod +x ShortCutExperiment.py
(venv) $ ./ShortCutExperiment.py
```

## Project Description

In this project we explore three agents with distinct policies; **Q-learning**, **SARSA**, and **Expected SARSA**.
The environment the agents are to interact with is a two-dimensional grid with dimensions 12x12.
In the picture below you can see the two starting positions (one of which is near the entrance of the *shortcut*), as well as a goal position.

<img src='https://github.com/Josef-Hlink/ShortCut/blob/main/supplementary/env.png' alt='Environment' width='240' height='240'>

The `X`'s represent cliffs (with very negative rewards associated with them), the black squares represent the starting positions, and the white square represents the goal position.
All three policies will choose actions by means of an ε-greedy approach.
For more info on what ε-greedy means, please visit my repo on exploration vs. exploitation [Bandit](https://github.com/Josef-Hlink/Bandit).
The only functional difference between the agents is how they update their Q-values.
While Q-Learning is _off-policy_, both SARSA and Expected SARSA (ESARSA) are examples of _on-policy_ models.

### Q-Learning

<img src='https://github.com/Josef-Hlink/ShortCut/blob/main/supplementary/QL-pseudo.png' alt='Pseudo code for Q Learning algorithm' height='240'></img>

In the image above you can see the pseudocode for a general Q-Learning algorithm.
This image was taken from Sutton & Barto's 2018 book: _Reinforcement Learning: An Introduction_.
An incomplete, but still extremely nice free version can be found at http://incompleteideas.net/book/the-book.html.
Q-Learning is called an off-policy model because its update function does not depend on any existing policy.

### SARSA
Lorem Ipsum
