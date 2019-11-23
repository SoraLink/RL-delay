from Environment.registration import EnvRegistry
from Model.NeuralNetwork.Model import Model
from rllab.envs.env_spec import EnvSpec
from Algorithm.Util.Dataset import Dataset

class PredictedEnv:
    _initialised = False

    def __init__(self, task, t_delay, r_delay, sess):
        self.env = EnvRegistry(task, transmit_delay=t_delay, receive_delay=r_delay)
        # print("..................",self.env.observation_space.shape)
        self.predict_model = Model(sess= sess,rnn_unit=32,nn_unit=32, delay=t_delay+r_delay,
                                   observation_space=self.env.observation_space.shape[0],
                                   action_space=1, scope="model",
                                   mask_value=0.00001)
        self.data_set = Dataset(3000)
        adapted_methods = ['observation_space', 'action_space']
        for value in adapted_methods:
            func = getattr(self.minion, value)
            self.__setattr__(value, func)

        self._initialised = True


    def step(self, action):
        self.pair.set_predicted_action(action)
        self.pair = self.env.step(self.pair)
        self.pair = self.predict_model.run(self.pair)
        if self.env.complete_data is not None:
            self.data_set.add_instance(self.env.complete_data)
        return self.pair.state

    def reset(self):
        pair = self.env.reset()
        self.pair = self.predict_model.run(pair)
        return self.pair.state

    @property
    def spec(self):
        return EnvSpec(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )

    def train_model(self):
        pairs = self.data_set.get_instance_randomly(64)
        self.predict_model.train(pairs)

    def __getattr__(self, attr):
        """Attributes not in Adapter are delegated to the minion"""
        return getattr(self.minion, attr)

    def __setattr__(self, key, value):
        """Set attributes normally during initialisation"""
        # if not self._initialised:
        #     super().__setattr__(key, value)
        # else:
        #     """Set attributes on minion after initialisation"""
        #     setattr(self.minion, key, value)
        super().__setattr__(key, value)



class MinionAdapter:
    _initialised = False

    def __init__(self, minion, **adapted_methods):
        self.minion = minion
#         print(self._initialised)
        for key, value in adapted_methods.items():
            func = getattr(self.minion, value)
            self.__setattr__(key, func)

        self._initialised = True

    def __getattr__(self, attr):
        """Attributes not in Adapter are delegated to the minion"""
        return getattr(self.minion, attr)

    def __setattr__(self, key, value):
        """Set attributes normally during initialisation"""
        if not self._initialised:
            super().__setattr__(key, value)
        else:
            """Set attributes on minion after initialisation"""
            setattr(self.minion, key, value)