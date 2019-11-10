from Environment.registration import EnvRegistry

class PolicyTest():
    def __init__(self, task):
        self.env = EnvRegistry(task=task)

    def step(self, action):
        while True:
            pair = self.env.step(action)
            if pair:
                #observation, reward, done
                return [pair.observation, pair.reward, pair.done]

    def reset(self):
        if self.env.if_pause is True:
            self.env.restart()
        while True:
            observation = self.env.reset
            if observation:
                return observation