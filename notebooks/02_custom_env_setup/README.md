# Trick Or Retreat: Custom Environment Set Up 

Before starting this project, my knowledge of reinforcement learning (RL) was purely theoretical - I had never attempted to implement any of the concepts. To avoid feeling overwhelmed by the complexity of building custom environments, I decided to break the project into three phases:

1.	**PHASE 1**: Building a Single Reward Environment

2.	**PHASE 2**: Adding Extra Rewards and Penalties

3.	**PHASE 3**: Adding Dynamic Movements

## Simple Environment
`simple_env.py`

Python file with custom environment class  (Simple Haunted Mansion), contains all environment methods.

`01-simple_env.ipynb`

Python notebook where I test the rendering of the environment and train the agent using PPO from Stable Baselines 3.

#### Agents Objective - Simple Environment
In this environment, the trick-or-treater 🏃 is the agent, and the exit door 🚪 is the target. 

The agent is placed randomly in the haunted mansion and must find the exit door to escape!

- There are 4 actions the agent can take: up, down, left and right.
- The game terminates when the agent has reached the exit door of the haunted mansion.

#### Reward Structure - Simple Environment
- A reward of 1 is given when the trick-or-treater finds the door.
- No penalty is given for the number of timesteps taken to find the door.

#### Results for Phase 1
Testing trained agent over 15 episodes, here are the results:

| Episode | Score |
|---------|-------|
| 1       | 1     |
| 2       | 1     |
| 3       | 1     |
| 4       | 1     |
| 5       | 1     |
| 6       | 1     |
| 7       | 1     |
| 8       | 1     |
| 9       | 1     |
| 10      | 1     |
| 11      | 1     |
| 12      | 1     |
| 13      | 1     |
| 14      | 1     |
| 15      | 1     |

After seeing the agent's actions via pygame, its clear the agent doesn't always take the most optimal path, to get the agent to do this I can add a penalty to the number of timesteps taken to reach the target. This is something I will implement by adding a penalty of 0.1 for every action which did not result in finding the target location.

To do this I will add a else case into the if statement in the step function for reward.


## Intermediate Environment
`intermediate_env.py`

Python file with custom environment class (Intermediate Haunted Mansion), contains all environment methods.

`01-intermediate_env.ipynb`

Python notebook where I train the agent first using PPO and then Q-learning, comapring the effecitveness of both methods.

#### Agents Objective - Intermediate Environment
In Phase 1, the agent's task was simple; find and reach the exit door. 

But now, with the addition of ghosts and candies, the agent not only needs to find and reach the exit, but also:

-	Avoid the ghosts
-	Collect as many candies as possible

**New Environment Features:**
-	Both candies and penalties remained static throughout this phase.
  
-	There are 4 actions the agent can take: up, down, left and right.
  
-	The game terminates when the agent has reached the exit door of the haunted mansion.
  
-	Small penalty is applied for each step taken that does not result in termination (reaching the exit door).
  
#### Reward Structure - Intermediate Environment
The reward logic is now more complex:

-	**Exit Door:** Reward of + 20 for reaching the exit door and completing the task.
  
-	**Candies:** Reward of + 15 for each candy collected.
  
-	**Ghosts:** Penalty of -25 is given if the agent contacts a ghost. This high penalty ensures that avoiding ghosts is prioritised in Q-learning, where future rewards are considered. As a result, the penalty had to be higher than the target reward +20.(More on Q-learning later in the post).
  
-	**Step Penalty:** A small penalty of 0.01 for each action that doesn’t lead to the target. This penalty is set low to avoid discouraging exploration, which is needed during the early stages of Q-learning. In PPO this penalty was set to 0.1 since PPO is more stable and so can handle a larger step penalty.


#### PPO Results 
Results after testing the trained PPO model across 15 episodes:

| Episode | Score |
|---------|-------|
| 1       | 34.0  |
| 2       | 19.8  |
| 3       | 33.4  |
| 4       | 48.2  |
| 5       | 47.4  |
| 6       | 48.0  |
| 7       | 34.0  |
| 8       | 48.2  |
| 9       | 48.0  |
| 10      | 33.6  |
| 11      | 19.4  |
| 12      | 33.2  |
| 13      | 33.4  |
| 14      | 48.2  |
| 15      | 33.2  |


Overall the agent shows a strong performance with all scores greater than 30 meaning the agent successfully reached the exit door while avoiding ghosts and collecting at least one candy. Also can see that agent's starting position affects whether it decides to collect more candies.

Training with PPO was sufficient in this instance, could be due to the relative simplicity of the environment still. All rewads/penalites in the environemnt are static, only the agent moves. 

#### Q-learning Results 

| Episode | Score  |
|---------|--------|
| 1       | 19.99  |
| 2       | 34.94  |
| 3       | 34.95  |
| 4       | 34.94  |
| 5       | 19.96  |
| 6       | 34.95  |
| 7       | 19.98  |
| 8       | 34.94  |
| 9       | 19.97  |
| 10      | 19.94  |
| 11      | 34.93  |
| 12      | 19.95  |
| 13      | 34.94  |
| 14      | 19.98  |
| 15      | 34.95  |
| 16      | 19.98  |
| 17      | 34.94  |
| 18      | 34.94  |
| 19      | 19.94  |
| 20      | 19.94  |

By comparing the Q-learning test results to the PPO test results it is clear PPO has outperformed Q-learning. This could be because:

- PPO updates the policy based on what the agent observes over an episode, gradually adjustiong the poilicy using clipping to avoid drastic changes in policy. This is why we see the learning is more smooth with PPO.

- Q-learning does not have a policy to update, it updates values which represent estimates of 'goodness' an action is depending on states of environment. These updates occur after each action which is probably why the results from training are so volatile and learning is less stable.

- Q-learning needs more episodes in order to explore all possible state-action pairs in the Q-table, PPO does not need to explore as much.

- Q-learning also requires epsilon-greedy methods for action selection which can be quite noisy, PPO uses probabilities for updates which are less nosiy.

To see more on Q-table and what the agent learnt, look at the comments in the notebook or go to my post about [Phase 2](https://simrenbasra.github.io/simys-blog/2024/11/20/trick_or_retreat_part4.html).


## Final Environment 
`final_env.py`

Python file with custom environment class (Final Haunted Mansion), contains all environment methods.

`01-final_env.ipynb`

Python notebook where I train the agent using DQN from Stable Baselines 3, I also show my attempt at implementing DQN using Keras/Tensorflow. Unfortunately I ran out of time for this project and was not able to spend more time debugging my attempt at DQN. This is something I will need to revisit in the future. 

#### Agents Objective - Intermediate Environment
The agent's objective is the same as Phase 2:

-	Avoid the ghosts
-	Collect as many candies as possible

But now the agent faces an additional added complexity as the ghosts will be placed randomly on the grid at the start of each episode.

**New Environment Features:**

-	Ghosts will be randomly placed on the grid at the start of each episode.
  
#### Reward Structure - Intermediate Environment
The reward logic is now more complex:

-	**Exit Door:** Reward of + 10 for reaching the exit door and completing the task.
  
-	**Candies:** Reward of + 3 to incentivise the agent to collect the candies but only if they are on route to target. 
  
-	**Ghosts:** Penalty of -7 is given if the agent contacts a ghost. If I was to set this any higher, the agent gets stuck.
  
-	**Step Penalty:** A small penalty of 0.75 - 1.0 for each action that doesn’t lead to the target. This penalty is set higher than Q-learning. This could be beacuse DQN provides more stable learning due to Experience Replay and Target Networks (see blog post). 

#### Deep Q-Network Results 

| Episode | Score  |
|---------|--------|
| 1       | 10.0   |
| 2       | 11.2   |
| 3       | -5.6   |
| 4       | 8.2    |
| 5       | 10.0   |
| 6       | 10.6   |
| 7       | 10.6   |
| 8       | 10.0   |
| 9       | 10.6   |
| 10      | -5.0   |
| 11      | 10.0   |
| 12      | 10.0   |
| 13      | 10.0   |
| 14      | 10.0   |
| 15      | 11.2   |
| 16      | 9.4    |
| 17      | 10.0   |
| 18      | 11.2   |
| 19      | 9.4    |
| 20      | 10.6   |

- High Scores (9.4 and above): 

    - The agent performs consistently well in most episodes, showing it effectively avoids ghosts and maximises rewards.
    - High scores indicate that the agent has learned to prioritise reaching the door and if sensible, collects candies along the way.

- Poor Performance (Episode 3 and 10): 

    - In Episodes 3 and 10, the agent encountered a ghost on the way to the door. This might be due to the remaining 5% exploration rate, causing the agent to take random actions occasionally and follow suboptimal paths.
    - Could also be the total rewards gathered during the episodes outweighs the penalty of a ghost. 
- Potentially worth exploring with ghost penalties in the future, but also must be cautious not to increase too much as this may cause the agent to become stuck.

Overall, the scores reflect a strong understanding of the environment, with a few lapses in performance due to exploration randomness since epsilon decays to 0.05.

## Summary
There is a clear progression in complexity throughout each Phase of my custom environment. This project has been enjoyable, I have learnt so much about Reinforcement Learning and am sure there is a lot left to learn! 

I feel more comfortable in my knowledge of RL, specifically when it comes to Q-learning and Deep Q-Networks. To read more about the project and concepts I used in these notebooks, please look at my [blog posts](https://simrenbasra.github.io/simys-blog/) on the Trick Or ReTreat project.
