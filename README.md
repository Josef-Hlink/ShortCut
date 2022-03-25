# RL Agents on simple 2D Environment with Shortcut

## Preface
This project was done as the 2<sup>nd</sup> programming assignment of Leiden University's 3<sup>rd</sup> year Computer Science (AI flavoured) course.
The templates (mostly `ShortCutEnvironment`) were provided by [Aske Plaat](https://www.universiteitleiden.nl/en/staffmembers/aske-plaat#tab-1), [Thomas Moerland](https://www.universiteitleiden.nl/medewerkers/thomas-moerland#tab-1) & [Daan Pelt](https://www.universiteitleiden.nl/en/staffmembers/daan-pelt#tab-1).

## Project Description
In this project we explore three agents with distinct policies; **Q-learning**, **SARSA**, and **Expected SARSA**.
The environment the agents are to interact with is a two-dimensional grid of dimensions 12x12.
In the picture below you can see the two starting positions (one of which is near the entrance of the *shortcut*), along with the goal position.

<img src='https://github.com/Josef-Hlink/ShortCut/blob/main/supplementary/env.png' alt='Environment' width='240' height='240'>

The `X`'s represent cliffs (with very negative rewards associated with them), the black squares represent the starting positions, and the white square represents the goal position.
