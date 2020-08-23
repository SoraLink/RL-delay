import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
from Algorithm.Util.StateActionPair import StateActionPair
import numpy as np
from rllab.envs.env_spec import EnvSpec

class EnvRegistry():

    def __init__(self, task, transmit_delay=1, receive_delay=1, num_of_episode=100):
        self.env = gym.make(task)
        self.env.render("human")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.transmit_delay = transmit_delay
        self.receive_delay = receive_delay
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            self.zero_action = np.zeros(1)
        else:
            self.zero_action = np.zeros(self.env.action_space.shape)
        self.trans_queue = []
        self.rcv_queue = [self.zero_action*self.receive_delay]
        # self.spec.action_space = self.env.action_space
        # self.spec.observation_space = self.env.observation_space

    def run(self):
        if len(self.action_queue) == self.transmit_delay+1:
            observation, reward, done, info = self.env.step(self.rcv_queue.pop(-1))
            self.trans_queue.append((observation, reward, done))
        else:
            observation, reward, done, info = self.env.step(self.zero_action)
            self.trans_queue.append((observation, reward, done))

    def reset(self):
        observation = self.env.reset()
        self.trans_queue = []
        self.rcv_queue = [self.zero_action*self.receive_delay]
        while(self.trans_queue<self.transmit_delay):
            self.run()
        return observation

    def step(self, actions):
        self.rcv_queue.append(actions[0])
        self.run()
        return self.trans_queue.pop(-1)