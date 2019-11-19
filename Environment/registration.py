import gym
import pybullet_envs
from Algorithm.Util.StateActionPair import StateActionPair
import numpy as np
from rllab.envs.env_spec import EnvSpec


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
            self.zero_action = np.zeros(self.env.action_space.shape)
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
            self.last_observation = observation
    # time.sleep(0.01)


    def append_action(self, action):
        for pair in self.action_and_state:
            pair.actions.append(action)

    def reset(self):
        self.env.reset()
        self.action_queue = []
        self.action_and_state = []
        while len(self.action_and_state) <= self.receive_delay:
            self.run()
        self.fill_zeors()
        self.assert_test()
        return self.action_and_state.pop(0)

    def step(self, pair):
        self.action_queue.append(pair)
        self.append_action(pair.predicted_action)
        self.run()
        self.fill_zeors()
        self.assert_test()
        return self.action_and_state.pop(0)

    def fill_zeors(self):
        while len(self.action_and_state[0].actions) < self.transmit_delay+self.receive_delay:
            if isinstance(self.action_space, gym.spaces.discrete.Discrete):
                self.action_and_state[0].actions.insert(0, 0)
            else:
                self.action_and_state[0].actions.insert(0, np.zeros(self.env.action_space.shape))
            # self.action_and_state[-1].actions.insert(0, np.zeros((self.env.action_space,)))

    def assert_test(self):
        assert (len(self.action_and_state[0].actions) == self.transmit_delay + self.receive_delay)

    @property
    def spec(self):
        return EnvSpec(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )


def get_list_actions(pairs):
    reslist = []
    for pair in pairs:
        reslist.append(pair.predicted_action)
    return reslist


# Example:
# env_thread = EnvRegistry("task name")
# env_thread.start()
# env_thread.step(action="0.01")
