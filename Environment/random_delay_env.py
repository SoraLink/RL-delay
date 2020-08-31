from Environment.delay_env import Env
import numpy as np
from Util.math_util.random_num_generater_factory import NumberFactory
import gym
import pybullet

class RandomDelayEnv(Env):

    def __init__(self, task, distributiuon='normal', **kwargs):
        # super().__init__(task, transmit_delay=transmit_delay, receive_delay=receive_delay, num_of_episode=num_of_episode)
        self.ran_num_gen = NumberFactory.get_num_generater(distributiuon, **kwargs)
        self.env = gym.make(task)
        self.env.render("human")
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.zero_action = np.zeros(1)
        else:
            self.zero_action = np.zeros(self.env.action_space.shape)
        for k,v in self.env.__dict__.items():
            self.__setattr__(k,v)

    def run(self):
        observation, reward, done, info = self.env.step(self.rcv_queue.pop(0))
        return (observation, reward, done, info)
    
    def change_delay(self):
        delay = self.ran_num_gen.get_random_num()
        while delay<len(self.rcv_queue):
            self.rcv_queue.pop(0)
        while delay>len(self.rcv_queue):
            self.rcv_queue.append(self.zero_action)
    
    def step(self, action):
        self.rcv_queue.append(action)
        res = self.run()
        self.change_delay()
        return res

    def reset(self):
        delay = self.ran_num_gen.get_random_num()
        self.rcv_queue = [self.zero_action] * delay
        observation = self.env.reset()
        return observation
        
    
    

