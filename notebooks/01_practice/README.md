# Lunar Lander

To get familiar with Gymnasium, I began by experimenting with one of its pre-defined environment - Lunar Lander.

This approach helped me gain a solid understanding before diving into creating a custom environment for **Trick or ReTreat**. 

## About Lunar Lander

The Lunar Lander environment is a classic control reinforcement learning task where an agent must successfully land on a landing pad while managing its speed, angle and engine thrust.

#### **Action Space**

The agent can take any of the following four discrete actions:

-	**0:** Do nothing

-	**1:** Fire the left engine

-	**2:** Fire the main engine

-	**3:** Fire the right engine

#### **Observation Space**

The observation space is defined as a box with the following bounds:

-	**Lower Bound:** [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
  
-	**Upper Bound:** [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]
  
-	**Shape:** 8-dimensional vector

The observation vector contains the following elements:

**1.	X Position:** Horizontal position of the lander.

**2.	Y Position:** Vertical position of the lander.

**3.	X Velocity:** Velocity of the lander along the x-axis.

**4.	Y Velocity:** Velocity of the lander along the y-axis.

**5.	Angle of Lander:** The current angle of the lander.

**6.	Angular Velocity:** The rotation speed of the lander.

**7.	Left Leg Contact:** 1 if the left leg is in contact with the ground, 0 if not.

**8.	Right Leg Contact:** 1 if the right leg is in contact with the ground, 0 if not.

#### **Reward Structure**

The goal of the agent is to land between the two flags. Rewards are given based on the following criteria:

- The closer the lander is to the landing pad, the more points are awarded.
  
-	Points are also awarded for reducing the lander's speed.
  
-	The reward decreases if the lander is more tilted.
  
-	Each leg in contact with the ground awards an additional 10 points.
  
-	Firing the side engines incurs a penalty of -0.03 points each time (indicated by red dots in the rendering).
  
-	Firing the main engine incurs a larger penalty of -0.3 points each time.

-	An additional reward of +100 points is given for a safe landing, while crashing results in a penalty of -100 points.
  
-	A reward above 200 points indicates good landing and performance of the agent.

#### **Episode End Conditions**

An episode can end in two ways:

-	**Truncation:** The episode is truncated when the agent scores 200 points.

-	**Termination:** The episode terminates if the lander crashes, goes out of bounds or becomes asleep.

-----

## **Random Action Selection**

To start, I explored the effects of the agent randomly selecting actions from its action space to see how the agent performs. This helped me familiarise myself with using basic concepts of both libraries. I limited the number of steps to 1000 to avoid lengthy runtimes.

#### **Results**

I collected the total rewards per episode (capped at 1000), below are the results:

| Episode | Score               |
|---------|---------------------|
| 1       | -238.99             |
| 2       | -151.35             |
| 3       | -293.43             |
| 4       | -498.66             |
| 5       | -94.48              |

It's clear that the performance was poor, as all scores are negative. The agent fails to achieve the task through random actions, highlighting the need for training to better understand its environment.

## **Proximal Policy Optimisation Algorithm** 

After exploring the Lunar Lander environment through random action selection, I wanted to improve the agent's performance through training. 

Stable Baselines3 documentation is very thorough and provides a list of all available RL algorithms for training. In the end, I chose Proximal Policy Optimisation (PPO) because it is relatively simple to understand and quite stable. 


## **Testing Results**

After training the model, I tested its performance over 10 episodes. Here are the results:

| Episode        | Score            |
|----------------|------------------|
| 1              | 17.01            |
| 2              | 262.80           |
| 3              | -28.20           |
| 4              | 266.47           |
| 5              | 293.32           |
| 6              | 242.91           |
| 7              | 278.50           |
| 8              | 250.77           |
| 9              | 241.46           |
| 10             | 280.66           |

Overall, agent shows a much stronger performance with most scores above 200.

There are some low and even negative scores which may suggest the agent struggled in certain scenarios. 

To know more about the PPO please navigate to my Blog Post on this project: (Link)[https://simrenbasra.github.io/simys-blog/2024/10/31/trick_or_retreat_part2.html]
