import gym
import threading
import time
import tensorflow as tf

class EnvRegistry(threading.Thread):

    def __init__(self, task, transmit_delay=20, receive_delay=20):
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

    def run(self):
        self.env.reset()
        for i_episode in range():
            self.env.reset()
            while True:
                self.env.render()
                if self.if_stop:
                    break
                # self.observation, self.reward, self.done, self.info = self.env.step(self.action)
                if len(self.action_queue) > self.transmit_delay:
                    observation, reward, done, info = self.env.step(self.action_queue[0])
                    self.action_and_state.append([observation, reward, done, info, self.action_queue.pop()])
                    if done:
                        self.if_pause = True
                else:
                    observation, reward, done, info = self.env.step(self.action_space.sample())
                    self.action_and_state.append([observation, reward, done, info, self.action_queue.pop()])
                    print(len(self.action_and_state))
                    if done:
                        self.if_pause = True
                if self.if_pause:
                    self.sleep()

    def stop(self):
        self.if_stop = True

    def sleep(self, pause_time=0.01):
        while self.if_pause:
            time.sleep(pause_time)

    def step(self, action):
        self.action_queue.append(action)
        if len(self.action_and_state) < self.receive_delay:
            return None
        else:
            pair = self.action_and_state.pop()
            return pair



# Example:
# env_thread = EnvRegistry("task name")
# env_thread.start()
# env_thread.step(action="0.01")
