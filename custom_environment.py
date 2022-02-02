import numpy as np
import gym
from gym.space import Discrete, Box
from gym imort Env

class enviroment(Env):
    def __init__(self):
        #Allowable actions
        self.action_space = Discrete(3)
        #Allowable states
        self.observation_spce = Box(low = np.array([0]), high = np.array([100]))
        #Initial state
        self.state = 38 + np.random.randn(1)
        #Length of each episode
        self.episode_length = 60

    def step(self, action):
        """Function takes a step in the environment"""
        self.state += action - 1
        self.episode_length -= 1

        #compute reward

        info = {}

        return self.state, reward, done, info

    def render(self):
        """Funciton renders the environment"""
        pass
    def reset(self):
        """Function reset the env after each episode"""
        self.state = 38 + np.random.randn(1)
        #Length of each episode
        self.episode_length = 60

        return self.state




