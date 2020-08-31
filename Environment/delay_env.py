import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import numpy as np

class Env():
    def run(self):
        raise NotImplementedError
    def step(self):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError

class DelayEnv(Env):
    def __init__(self, task, transmit_delay=1, receive_delay=1):
        self.env = gym.make(task)
        self.env.render("human")
        # self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        # self.reward_range = self.env.reward_range
        self.transmit_delay = transmit_delay
        self.receive_delay = receive_delay
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.zero_action = np.zeros(1)
        else:
            self.zero_action = np.zeros(self.env.action_space.shape)
        self.trans_queue = []
        self.rcv_queue = [self.zero_action]*self.receive_delay
        for k,v in self.env.__dict__.items():
            self.__setattr__(k,v)

    def run(self):
        if len(self.rcv_queue) == self.receive_delay+1:
            observation, reward, done, info = self.env.step(self.rcv_queue.pop(0))
            self.trans_queue.append((observation, reward, done, info))
        else:
            observation, reward, done, info = self.env.step(self.zero_action)
            self.trans_queue.append((observation, reward, done, info))

    def reset(self):
        observation = self.env.reset()
        self.trans_queue = []
        self.rcv_queue = [self.zero_action]*self.receive_delay
        while(len(self.trans_queue)<self.transmit_delay):
            self.run()
        return observation

    def step(self, actions):
        self.rcv_queue.append(actions)
        self.run()
        return self.trans_queue.pop(0)