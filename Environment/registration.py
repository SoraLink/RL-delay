import gym
import threading
import time
import tensorflow as tf
import copy
from Algorithm.Util.StateActionPair import StateActionPair

class EnvRegistry(threading.Thread):

    def __init__(self, task, transmit_delay=20, receive_delay=20, num_of_episode=100):
        threading.Thread.__init__(self)
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
        self.num_of_episode = num_of_episode
        self.last_action = tf.zeros(self.env.action_space)
        self.complete_data = []
        self.spec = self.env.spec

    def run(self):
        self.env.reset()
        for i_episode in range(self.num_of_episode):
            last_observation = self.env.reset()
            while True:
                print(len(self.action_and_state))
                self.env.render()
                if self.if_stop:
                    break
                # self.observation, self.reward, self.done, self.info = self.env.step(self.action)
                if len(self.action_space) > self.transmit_delay:
                    raise Exception("length of action queue error")
                if len(self.action_space) == self.transmit_delay:
                    observation, reward, done, info = self.env.step(self.action_queue[0].predicted_action)
                    self.last_action = self.action_queue[0].predicted_action
                    self.action_queue[0].set_label(observation)
                    pair = StateActionPair(observation, get_list_actions(self.action_queue))
                    self.action_queue[0].set_info(reward,last_observation,done)
                    self.action_and_state.append(pair)
                    self.complete_data.append(self.action_queue.pop())
                    last_observation = observation
                else:
                    observation, reward, done, info = self.env.step(self.last_action)
                    pair = StateActionPair(observation, get_list_actions(self.action_queue))
                    self.action_and_state.append(pair)
                    self.fill_zeors()
                    last_observation = observation
                # time.sleep(0.01)
                if done:
                    self.if_pause = True
                    self.sleep()
                    break

    def restart(self):
        self.action_queue = []
        self.action_and_state = []
        self.if_pause = False

    def stop(self):
        self.if_stop = True

    def sleep(self, pause_time=0.01):
        print("sleeping...")
        while self.if_pause:
            time.sleep(pause_time)

    def step(self, pair):
        self.action_queue.append(pair)
        if len(self.action_and_state) < self.receive_delay:
            return None
        else:
            pair = self.action_and_state.pop()
            return pair

    def fill_zeors(self):
        while len(self.action_and_state[-1].actions) < self.transmit_delay:
            self.action_and_state[-1].actions.insert(0, tf.zeros((self.env.action_space,)))


def get_list_actions(pairs):
    reslist = []
    for pair in pairs:
        reslist.append(pair.predicted_action)
    return reslist


# Example:
# env_thread = EnvRegistry("task name")
# env_thread.start()
# env_thread.step(action="0.01")
