# Trick Or Retreat

In this project, my goal is to develop a reinforcement learning model where a trick-or-treater must escape a haunted mansion, avoiding ghosts and collecting treats along the way. This is my first project using reinforcement learning and I’m excited to see where it takes me!

## Game Introduction
I’ve designed a 5x5 grid where each square is either empty or contains a piece of candy, a ghost, the exit door or the trick-or-treater.

The trick-or-treater can move in four directions: up, down, left and right and is limited to moving one square at a time. The game ends when they find the exit door of the haunted mansion.

The goal is for the trick-or-treater to find the optimal path to escape, collecting as many candies as they can while avoiding the ghosts haunting the mansion!

## Notebooks Overview
### 01_practice
To practice using Open AI's Gymnasium and Stable Baselines 3 using the Lunar Lander pre-defined environment from Open AI.

#### 01-Lunar-Lander.ipynb
- Set up Lunar-Lander Environment. 
- Use random actions to see the agent in action. 
- Train the agent using PPO from Stable Baselines 3.
- Evaluate the performance of the agent.

### 02_custom_env_setup
Notebooks to set up a custom environment and train an agent using Open AI's Gymnasium and Stable Baselines 3.

#### simple_env.py
- Building a simple custom environment class with a single reward.

#### 01-simple-env.ipynb
- Training the agent for simple_env using PPO from Stable Baselines 3.
- Evaluating the training and performance of the agent.

#### intermediate_env.py
- Building an intermediate custom environment class.
- Multiple rewards and penalties (candies and ghosts).

#### 02-intermediate-env.ipynb
- Training the agent for simple_env using PPO and Q-learning.
- Evaluating the training and performance of the agent.
- Comparison between training methods.

#### final_env.py
- Building the final custom environment class where the ghosts(penalties) move every few timesteps.

#### 03-final_env.ipynb
- Training the agent using DQN in order to deal with moving rewards and exploding state space.
- Evaluating the training and performance of the agent.


## Blog Posts
For more information or explanations please visit my blog posts on the project where I dive into the theory and explain my code:

- [Introduction to Reinforcement Learning](https://simrenbasra.github.io/simys-blog/2024/10/21/trick_or_retreat_part_1.html)
- [Getting Started with Open AI and Stable Baselines 3 using Lunar Lander](https://simrenbasra.github.io/simys-blog/2024/10/31/trick_or_retreat_part2.html)

- [Custom Environment Set Up: Phase 1 (Simple)](https://simrenbasra.github.io/simys-blog/2024/11/14/trick_or_retreat_part3.html)
- [Custom Environment Set Up: Phase 2 (Intermediate)](https://simrenbasra.github.io/simys-blog/2024/11/20/trick_or_retreat_part4.html)
- [Custom Environment Set Up: Phase 3 (Final)](https://simrenbasra.github.io/simys-blog/2024/12/07/trick_or_retreat_part5.html)

## How to run
1. Clone the Repository:

`git clone https://github.com/simrenbasra/Trick_Or_ReTreat.git`

2. Install dependencies:

`pip install -r requirements.txt`

Requires env to be set up and activated as well as pip to be installed.

Run the notebooks Open the notebooks in the notebooks/directory and follow the instructions