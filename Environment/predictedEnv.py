from Environment.registration import EnvRegistry
from Algorithm.Model.Model import Model
from rllab.envs.env_spec import EnvSpec
from Algorithm.Util.Dataset import Dataset

class PredictedEnv:
    def __init__(self, task, t_delay, r_delay, sess):
        self.env = EnvRegistry(task, transmit_delay=t_delay, receive_delay=r_delay)
        self.predict_model = Model(sess= sess,rnn_unit=32,nn_unit=32, delay=t_delay+r_delay,
                                   observation_space=self.env.observation_space.shape,
                                   action_space=self.env.action_space.shape, scope="model",
                                   mask_value=0.00001)
        self.data_set = Dataset(3000)

    def step(self, pair):
        pair = self.env.step(pair)
        pair = self.predict_model.run(pair)
        if self.env.complete_data is not None:
            self.data_set.add_instance(self.env.complete_data)
        return pair

    def reset(self):
        pair = self.env.reset()
        pair = self.predict_model.run(pair)
        return pair

    @property
    def spec(self):
        return EnvSpec(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )

    def train_model(self):
        pairs = self.data_set.get_instance_randomly(64)
        self.predict_model.train(pairs)