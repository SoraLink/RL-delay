from Environment.registration import EnvRegistry
from Model.NeuralNetwork.Model import Model
from rllab.envs.env_spec import EnvSpec
from Algorithm.Util.Dataset import Dataset
import numpy as np

class PredictedEnv:
    _initialised = False

    def __init__(self, task, t_delay, r_delay, sess):
        self.a = 1
        self.env = EnvRegistry(task, transmit_delay=t_delay, receive_delay=r_delay)
        # print("..................",self.env.observation_space.shape)
        self.predict_model = Model(sess= sess,rnn_unit=32,nn_unit=32, delay=t_delay+r_delay,
                                   observation_space=self.env.observation_space.shape[0],
                                   action_space=self.env.action_space.shape[0], scope="model",
                                   mask_value=0.00001)
        self.data_set = Dataset(3000)
        adapted_methods = ['observation_space', 'action_space']
        for value in adapted_methods:
            func = getattr(self.env.env, value)
            self.__setattr__(value, func)

        self._initialised = True
        self.name = task

    def load_pool(self, algo):
        self.pool_isInit = True
        self.pool = SimpleReplayPool(
            observation_dim=algo.policy.state_dim,
            action_dim=algo.policy.action_dim,
            lam=algo.lam,
            gamma=algo.gamma
        )


    def step(self, action, value):
        #TODO use self.pool.add_sample(value, terminal, observation, action, reward)
        self.pair.set_predicted_action(action)
        self.pair.value = value
        self.pair, done = self.env.step(self.pair)
        self.pair = self.predict_model.run(self.pair)
        if self.env.complete_data is not None:
            self.data_set.add_instance(self.env.complete_data)

        return self.pair.predicted_state, self.pair.reward, done, {}

    def reset(self):
        pair, done = self.env.reset()
        self.pair = self.predict_model.run(pair)
        return self.pair.predicted_state, done

    @property
    def spec(self):
        return EnvSpec(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )

    def train_model(self):
        # print(len(self.data_set.pairs))
        pairs = self.data_set.get_instance_randomly(64)
        self.predict_model.train(pairs)

    def get_pool(self):
        
        pairs = self.data_set.pairs
        for pair in pairs:
            self.pool.add_sample(pair.value, pair.done, pair.predicted_state.reshape(-1,), pair.predicted_action, pair.reward) 
        return self.pool

    # def __getattr__(self, attr):
    #     """Attributes not in Adapter are delegated to the minion"""
    #     return getattr(self.minion, attr)

    # def __setattr__(self, key, value):
    #     """Set attributes normally during initialisation"""
    #     # if not self._initialised:
    #     #     super().__setattr__(key, value)
    #     # else:
    #     #     """Set attributes on minion after initialisation"""
    #     #     setattr(self.minion, key, value)
    #     super().__setattr__(key, value)


class SimpleReplayPool():
    def __init__(
            self, observation_dim, action_dim, lam, gamma):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._A = []
        self._observations = []
        self._actions = []
        self._target_values = []
        self.lam = lam
        self.gama = gamma
        self._terminal = []
        self._rewards = []

    def reset(self):
        self._observations = []
        self._actions = []
        self._A = []
        self._target_values = []
        self._terminal = []
        self._rewards = []

    def add_sample(self, value, terminal, observation=None, action=None, reward=None, ):
        self._observations.append(observation)
        self._target_values.append(value)
        self._terminal.append(terminal)
        self._actions.append(action)
        self._rewards.append(reward)

    def _compute_A(self, nextvpred):
        self._observations = np.array(self._observations, dtype=np.float32)
        self._actions = np.array(self._actions, dtype=np.float32)
        self._target_values = np.array(self._target_values, dtype=np.float32)
        new = np.append(self._terminal, 0)
        vpred = np.append(self._target_values, nextvpred)
        print(vpred.shape)
        T = len(self._rewards)

        self._advantage = gaelam = np.empty(T, 'float32')
        rew = self._rewards
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            delta = rew[t] + self.gama * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + self.gama * self.lam * nonterminal * lastgaelam
        self._target_values = self._advantage + self._target_values