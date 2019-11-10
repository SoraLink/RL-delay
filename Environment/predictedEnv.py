from Environment.registration import EnvRegistry

class PredictedEnv:
    def __init__(self, task):
        self.env = EnvRegistry(task)

    def step(self, action):
        pair = self._env.step(action)

    def reset(self):
        