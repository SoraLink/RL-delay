import numpy as np
from abc import ABC, abstractmethod
from Algorithm.Util.StateActionPair import StateActionPair

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        tmp, done = self.env.reset()
        self.obs[:] = np.array([tmp])
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, delay):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.delay = delay

    def run(self):
        assert self.env.pool_isInit
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps+self.delay):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            # mb_obs.append(self.obs.copy())
            # mb_actions.append(actions)
            # mb_values.append(values)
            # mb_neglogpacs.append(neglogpacs)
            # mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            print(actions, actions.shape)
            tmp = []
            for idx, action in enumerate(actions):
                next_ob, _, next_terminal, _ = self.env.step(action, values[idx], neglogpacs[idx])
                tmp.append((next_ob, next_terminal))
            tmp = list(zip(*tmp))
            self.obs[:], self.dones = tuple(tmp)
            if True in self.dones:
                tmp, done = self.env.reset()
                self.obs[:] = np.array([tmp])
            self.obs = np.array(self.obs)
            # rewards = np.array(rewards)
            self.dones = np.array(self.dones)
            print(self.obs.shape)
            print(self.dones)

            # for info in infos:
            #     maybeepinfo = info.get('episode')
            #     if maybeepinfo: epinfos.append(maybeepinfo)
            # mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        pool = self.env.get_pool()
        mb_obs = np.asarray(pool._observations, dtype=self.obs.dtype)
        mb_rewards = np.asarray(pool._rewards, dtype=np.float32)
        mb_actions = np.asarray(pool._actions)
        mb_values = np.asarray(pool._target_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(pool._neglogaction, dtype=np.float32)
        # pool._terminal.append([False])
        mb_dones = np.asarray(pool._terminal, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.env.nsteps)):
            if t == self.env.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


