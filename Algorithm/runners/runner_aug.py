import numpy as np
import tensorflow as tf
from baselines.common.runners import AbstractEnvRunner


class Runner:

    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner
    run():
    - Make a mini batch
    """

    def __init__(self, *, env, model, nsteps, gamma, lam, p, ):
        self.p = p
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv * nsteps,) + env.observation_space.shape
        self.ob_space_shape = env.observation_space.shape[0] + env.action_space.shape[0] * p
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs = env.reset()
        self.resize()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        self.env = env
        # Discount rate
        self.gamma = gamma
        self.key = 0
        self.start_key = 0
        self.pre_action_storage = dict()

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            self.pre_action_storage[self.key] = actions
            mb_obs.append([self.obs.copy()])
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append([self.dones])

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs, rewards, self.dones, infos, re_key = self.env.step(actions[0], self.key)
            self.key += 1
            if re_key is not None:
                self.start_key = re_key + 1
                while re_key in self.pre_action_storage:
                    self.pre_action_storage.pop(re_key)
                    re_key -= 1
            obs_element = list(self.obs)
            for ac_key in range(self.start_key, self.key):
                obs_element.extend(self.pre_action_storage[ac_key][0])
            self.obs = np.asarray(obs_element)
            # for info in infos:
            #     maybeepinfo = info.get('episode')
            #     if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append([rewards])
            if self.dones:
                self.obs = self.env.reset()
            self.resize()
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        counter = 1
        for t in range(self.nsteps):
            if (mb_dones[t]):
                counter += 1
        performance = sum(mb_rewards) / counter
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, performance)

    def resize(self):
        self.obs = list(self.obs)
        self.obs = self.obs[:self.ob_space_shape]
        while len(self.obs) < self.ob_space_shape:
            self.obs.append(0)
        self.obs = np.asarray(self.obs)


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
