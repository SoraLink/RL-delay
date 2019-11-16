import gym
import pybullet_envs
from Algorithm.Util.StateActionPair import StateActionPair
import numpy as np

class EnvRegistry():

    def __init__(self, task, transmit_delay=1, receive_delay=1, num_of_episode=100):
        self.env = gym.make(task)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.if_stop = False
        self.if_pause = False
        self.transmit_delay = transmit_delay
        self.receive_delay = receive_delay
        self.action_queue = []
        self.action_and_state = []
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            self.zero_action = 0
        else:
            self.zeor_action = np.zeros(self.env.action_space.shape)
        self.complete_data = None
        # self.spec.action_space = self.env.action_space
        # self.spec.observation_space = self.env.observation_space

    def run(self):
        # print(len(self.action_and_state))
        self.env.render()
        # self.observation, self.reward, self.done, self.info = self.env.step(self.action)
        if len(self.action_queue) > self.transmit_delay+1:
            raise Exception("length of action queue error")
        elif len(self.action_queue) == self.transmit_delay+1:
            observation, reward, done, info = self.env.step(self.action_queue[0].predicted_action)
            # self.action_queue[0].set_label(self.last_observation)
            pair = StateActionPair(observation, get_list_actions(self.action_queue[1:]))
            self.action_queue[0].set_info(reward, self.last_observation, done)
            self.action_and_state.append(pair)
            self.complete_data = self.action_queue.pop(0)
            self.last_observation = observation
        else:
            observation, reward, done, info = self.env.step(self.zero_action)
            pair = StateActionPair(observation, get_list_actions(self.action_queue))
            self.action_and_state.append(pair)
            self.fill_zeors()
            self.last_observation = observation
    # time.sleep(0.01)

    def reset(self):
        self.env.reset()
        self.action_queue = []
        self.action_and_state = []
        while len(self.action_and_state) < self.receive_delay:
            self.run()
        return self.action_and_state.pop(0)

    def step(self, pair):
        self.action_queue.append(pair)
        self.run()
        return self.action_and_state.pop(0)

    def fill_zeors(self):
        while len(self.action_and_state[-1].actions) < self.transmit_delay:
            if isinstance(self.action_space, gym.spaces.discrete.Discrete):
                self.action_and_state[-1].actions.insert(0, 0)
            else:
                self.action_and_state[-1].actions.insert(0, np.zeros(self.env.action_space.shape))
        assert(len(self.action_and_state[-1].actions)==self.transmit_delay)
            # self.action_and_state[-1].actions.insert(0, np.zeros((self.env.action_space,)))


def get_list_actions(pairs):
    reslist = []
    for pair in pairs:
        reslist.append(pair.predicted_action)
    return reslist


# Example:
# env_thread = EnvRegistry("task name")
# env_thread.start()
# env_thread.step(action="0.01")
