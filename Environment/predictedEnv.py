class PredictedEnv:
    def __init__(self, env, model):
        self._env = env
        self._model = model
        self.spec = self._env.spec

    def step(self, action):
        pair = self._env.step(action)

    def reset(self):
        