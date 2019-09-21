import gym
import threading
import time

class EnvRegistry(threading.Thread):

    def __init__(self, task):
        threading.Thread.__init__(self)
        self.env = gym.make(task)
        self.action_space = self.env.action_space
        self.action = self.env.action_space.sample()
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.if_stop = False
        self.if_pause = False

    def run(self):
        self.env.reset()
        while True:
            if self.if_stop:
                break
            self.sleep()
            self.observation, self.reward, self.done, self.info = self.env.step(self.action)
            if self.done:
                self.pause = True
                self.sleep()

    def restart(self):
        self.pause = False

    def pause(self):
        self.pause =True

    def stop(self):
        self.if_stop = True

    def sleep(self, pause_time=0.01):
        while self.if_pause:
            time.sleep(pause_time)



