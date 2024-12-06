import numpy as np
import gymnasium as gym
import pygame
import sys

class Final_Haunted_Mansion(gym.Env):

    # Defining metadata (render_modes/render_fps)
    metadata = {'render_modes' : ['human'], 'render_fps': 1}

    ##########################################################################
    # Init
    ##########################################################################

    def __init__(self, size: int = 5, render_mode = 'human', step_penalty = 0.1):
        ''' 
        Description:
            Initialises the environment

        Inputs:
            size: int 
                The grid size, 5 by 5 for default 

            render_mode: str
                For visualisation, default render mode set to 'human'

        Outputs:
            size : int
                The size of the grid, which will be a square of `size x size`.
                
            render_mode : str
                The rendering mode used by the environment.
                
            num_rows : int
                Number of rows in the grid, equal to `size`.
                
            num_cols : int
                Number of columns in the grid, equal to `size`.
                
            agent_location : array
                Initial location of the agent as a numpy array.
                
            target_location : array
                Fixed location of the target/door as a numpy array, set to [4, 4].

            ghosts_location: array
                Fixed locations of the ghosts, locations passed in as a list.

            candies_location: array
                Fixed locations of the candies, locations passed in as a list.
            
            step_penalty: float
                Controls the penalty for each step taken that does not result in termination
            
            observation_space : gym.spaces.Dict
                Observation space for the environment, containing the agent's and target's grid positions.
                
            action_space : gym.spaces.Discrete
                Action space with four discrete actions: right, up, left, down.
                
            action_to_direction : dict
                Dictionary that maps actions.
                
            screen_size : int 
                The pixel size of the screen for displaying the grid; defaults to 800.
                
            cell_size : int 
                The pixel size of each cell in the grid.
        '''

        # Setting size of grid to size input parameter
        self.size = size   

        # Setting render mode to render_mode input parameter    
        self.render_mode = render_mode

        # Setting number of rows and columns of grid using size
        self.num_rows, self.num_cols = self.size, self.size

        # Placeholder value for agent location, the agent is out of bounds and is randomly set on the grid during reset() function
        self.agent_location = np.array([-1, -1], dtype=np.int64)   

        # Setting position of the target_location (exit door), the door is static
        self.target_location = np.array([4, 4], dtype=np.int64)

        # Setting positions of ghosts (using nested array as more than one ghost)
        # self.ghosts_location = np.array([[0, 0],[4, 2],[2, 4]])
        self.ghosts_location = np.array([[-1, -1],[-1, -1],[-1, -1]])
        
        # Setting positions of candies (using nested array as more than one candy)
        self.candies_location = np.array([[2, 2],[3, 0]])

        # Setting penalty for each step the action takes where target location is not reached
        self.step_penalty = step_penalty

        self.timestep = 0

        # Observations are represented as dictionaries with the agent's and the target's location.
        self.observation_space = gym.spaces.Dict(
            {
                'agent': gym.spaces.Box(0, size - 1, shape=(2,), dtype = np.int64),
                'target': gym.spaces.Box(0, size - 1, shape=(2,), dtype = np.int64),
                # shape for ghosts/candies to (no ghosts/candies, 2), where each has [x, y] coordinates
                'ghosts': gym.spaces.Box(0, size - 1, shape=(self.ghosts_location.shape[0], 2), dtype = np.int64),
                # Setting lower bound to -1 as once candies are collected they are placed out of bounds (see step())
                'candies': gym.spaces.Box(-1, size - 1, shape=(self.candies_location.shape[0], 2), dtype = np.int64)
            }
        )

        # We have 4 actions: right, up, left, down
        self.action_space = gym.spaces.Discrete(4)

        # Dictionary to map the actions to directions on the grid
        # (0,0) top left corner , (4,4) bottom right up and down are reveresed 
        self.action_to_direction = {            
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # up
        }

        # Initialise Pygame if render_mode is 'human'

        if self.render_mode == 'human':
            pygame.init()
            self.screen_size = 800
            self.cell_size = self.screen_size // self.size
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))

            pygame.display.set_caption('Trick or ReTreat: Escape the Mansion!')


    ##########################################################################
    # Returning Observations
    ##########################################################################

    def _get_obs(self):
        ''' 
        Description:
            Returns environment observations based on agents location. 

        Outputs:
            observations: dict
                Returns location of the agent, target, ghosts and candies.
        '''
        observation = {
            'agent': self.agent_location, 
            'target': self.target_location, 
            'ghosts': self.ghosts_location,
            'candies': self.candies_location
        }
        
        return observation


    ##########################################################################
    # Returning Distance (between the agent and door)
    ##########################################################################

    def _get_info(self):
        ''' 
        Description:
            Returns environment information based on agents location (door). 

        Outputs:
            information: 
                Returns distance between agent and target location (door).
        '''
        return {
            'distance': np.linalg.norm(
                self.agent_location - self.target_location, ord=1
            )
        }

    ##########################################################################
    # Resetting the Environment
    ##########################################################################

    def reset(self, seed:int = None, options: dict = None):
        ''' 
        Description:
            Resets environment to an initial state.

        Inputs:
            seed: int 
                Control randomness, set to None as default.

        Outputs:
            information: 
                Returns initial observation and info of environmend based on agent's starting location
        '''
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Setting the agents starting location randomly on the grid
        self.agent_location = self.np_random.integers(0, self.size, size=2, dtype= np.int64)
        # Reset candies on grid
        self.candies_location = np.array([[2, 2],[3, 0]])

        for i in range(len(self.ghosts_location)):
            # Generate a random position for the ghost
            ghost_pos = self.np_random.integers(0, self.size, size=2, dtype=np.int64)
            
            # Check if the ghost position is valid (not overlapping with agent, candies, or other ghosts)
            while (np.array_equal(ghost_pos, self.agent_location) or
                any(np.array_equal(ghost_pos, candy) for candy in np.array([[2, 2], [3, 0]])) or
                np.array_equal(ghost_pos, self.target_location) or
                any(np.array_equal(ghost_pos, other_ghost) for j, other_ghost in enumerate(self.ghosts_location) if j != i)):
                # If there's a clash, generate a new random position
                ghost_pos = self.np_random.integers(0, self.size, size=2, dtype=np.int64)
                
            # Set the updated ghost location
            self.ghosts_location[i] = ghost_pos
        
        # Getting initial observations and info based on starting agent position
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()
        
        return observation, info

    ##########################################################################
    # Step
    ##########################################################################  
    
    def step(self, action):

        ''' 
        Description:
            To get observation, reward, terminated, truncated and info once agent has taken an action.

        Inputs:
            action: int 
                Control randomness, set to None as default.

        Outputs:
            observation:
                Returns observation (agent, target location) of environment based on action agent has taken.

            reward:
                Points received when agent reaches the target.
            
            terminated:
                Boolean flag, set to True only if the the agent reaches the target(door).

            truncated:
                Boolean flag, set to False always (simple environment).

            info:
                Returns info of environment based on action agent has taken.
        '''

        # self.timestep += 1 

        # Converting action to int
        if isinstance(action, np.ndarray):
            action = np.int64(action.item()) 

        direction = self.action_to_direction[action]
        
        # We use np.clip to make sure we don't leave the grid bounds
        self.agent_location = np.clip(
            self.agent_location + direction, 0, self.size - 1
        )

        truncated = False

        # Terminated only when reward is reached (agent same location as door)
        terminated =np.array_equal(self.agent_location, self.target_location)

        # Initialising reward to 0
        reward = 0
  
        if terminated:
            reward += 10
        else:
            # Check if the agent encounters a ghost
            if any(np.array_equal(self.agent_location, ghost) for ghost in self.ghosts_location):
                # Penalty of - 7
                reward -= 7
            
            for candy in self.candies_location:
                if np.array_equal(self.agent_location, candy) and not np.array_equal(candy, [-1, -1]):
                    # Reward of +3
                    reward += 3
                    # Removing candy from grid after agnet has collected it (by setting it out of bounds)
                    candy[:] = [-1, -1]  

        # Adding penalty for every step agent takes, default = 0.1 to avoid discouraging exploration
        reward -= self.step_penalty

        # Get observation and info after taking an action
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    ##########################################################################
    # Render
    ##########################################################################  
   
    def render(self):
        ''' 
        Description:
            To visualise the environment and agent's actions, only rendering mode is for human.

        Outputs:
            pygame display window depicting grid, agent's movement and target location.
        '''
                
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Set background to all white
        self.screen.fill((255, 255, 255))

        # Looping through rows and columns to draw rectanges (representing the grid)
        for row in range(self.size):
            for col in range(self.size):
                # Calculates x and y positions of each cell in the grid by multiplying the col/row index by pixel cell size
                cell_x = col * self.cell_size
                cell_y = row * self.cell_size
                # Drawing white rectangle to represent each cell in the grid
                pygame.draw.rect(self.screen, (0, 0, 0), (cell_x, cell_y, self.cell_size, self.cell_size), 1)

        # To calculate the offset to ensure images are placed in centre of cells
        offset = self.cell_size * 0.1

        # Transform target grid cooridinates into pixel coordinates 
        door_pos = self.target_location * self.cell_size
        
        # Representing the door as an image from Canva
        door_img = pygame.image.load('images/Door.png')
        # Scaling the image of the Door to be smaller than size of the cell
        door_img = pygame.transform.scale(door_img,(self.cell_size * 0.8,self.cell_size * 0.8))
        # Drawing the image to the grid, adding offset to ensure img is in the middle
        self.screen.blit(door_img, (door_pos[0] + offset , door_pos[1] + offset))
        
        # Iterate over each ghost in the grid
        for ghost in self.ghosts_location:
            # Calculate the position of the ghost in terms of pixels
            ghost_pos = ghost * self.cell_size
            # Load the image of the ghost
            ghost_img = pygame.image.load('images/Ghost.png')
            # Scale the image to fit within the cell
            ghost_img = pygame.transform.scale(ghost_img, (self.cell_size * 0.8, self.cell_size * 0.8))
            # Add offset to center the image in the cell and render it at the calculated position
            self.screen.blit(ghost_img, (ghost_pos[0] + offset, ghost_pos[1] + offset))
        
        # Iterate over each candy in the grid
        for candy in self.candies_location:
            # Calculate the position of the candy in terms of pixels
            candy_pos = candy * self.cell_size
            # Load the image of the candy
            candy_img = pygame.image.load('images/Candy.png')
            # Scale the image to fit within the cell
            candy_img = pygame.transform.scale(candy_img, (self.cell_size * 0.8, self.cell_size * 0.8))
            # Add offset to center the image in the cell and render it at the calculated position
            self.screen.blit(candy_img, (candy_pos[0] + offset, candy_pos[1] + offset))

        agent_pos = self.agent_location * self.cell_size
        # Representing the agent as an image from Canva
        agent_img = pygame.image.load('images/Agent.png')
        agent_img = pygame.transform.scale(agent_img,(self.cell_size * 0.8,self.cell_size * 0.8))
        self.screen.blit(agent_img, (agent_pos[0] + offset, agent_pos[1] + offset))

        # To keep updating the display after each action
        pygame.display.update()  

    ##########################################################################
    # Close
    ##########################################################################  

    def close(self):
        ''' 
        Description:
            To close environment.

        Outputs:
            Quit all pygame windows after the environment is no longer in use.
        '''
        
        if self.render_mode == 'human':
            pygame.quit()  # Close the pygame window