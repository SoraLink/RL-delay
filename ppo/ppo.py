import os, sys

lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

import numpy as np
import tensorflow as tf
from .RLutils.dataset import Dataset
from .RLutils import logger 
import joblib


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

class PPO:
    def __init__(self,
                 env,
                 policy,
                 old_policy,
                 session,
                 c1,
                 c2,
                 lam,
                 gamma,
                 max_local_step,
                 batch_size=25,
                 epsilon=0.2,
                 n_epochs=1000000,
                 epoch_length=100,
                 min_pool_size=20,
                 replay_pool_size=1000,
                 discount=0.99,
                 max_path_length=250,
                 policy_weight_decay=0,
                 policy_learning_rate=3e-4,
                 n_updates=10,
                 restore_fd=None):
        self.sess = session
        self.env = env
        self.policy = policy
        self.old_policy = old_policy
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_poolsize = min_pool_size
        self.max_local_step = max_local_step
        self.pool_size = replay_pool_size
        self.disc = discount
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.lam = lam
        self.gamma = gamma
        self.max_path_length = max_path_length
        self.policy_weight_decay = policy_weight_decay
        self.policy_learning_rate = policy_learning_rate
        self.n_updates = n_updates
        self.policy_update_method = tf.train.AdamOptimizer(
            policy_learning_rate,
        )
        # self.policy_update_method = tf.train.AdagradOptimizer(
        #     policy_learning_rate,
        # )
        self.opt_info = None
        logger.task_name_scope = self.env.name
        self.restore_fd = restore_fd

    def train(self, itr = None):

        assert self.env.pool_isInit

        self.init_opt()
        epoch = 0
        if self.restore_fd:
            info = logger.init(restore=True, dir_name=self.restore_fd, session=self.sess, itr=itr)
            epoch = info['epoch']
        else:
            logger.init()

        observation, done = self.env.reset()
        terminal = True
        steps = []
        step = 0
        while True:
            epoch += 1
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("%s started" % epoch)
            path_return = []
            path_value = []
            for i in range(self.max_local_step):
                action, value = self.policy.get_a_v(observation)
                next_ob, _, next_terminal, _ = self.env.step(action, value)
                step += 1
                observation = next_ob
                terminal = next_terminal
                # print(terminal)
                if terminal:
                    # print(i)
                    steps.append(step)
                    step = 0
                    observation, done = self.env.reset()
                    # print(ob)
            path_return = np.array(path_return)
            path_value = np.array(path_value)
            logger.record_tabular('epoch', epoch)
            # logger.record_tabular('path reward', path_return.sum())
            # logger.record_tabular('path reward (mean)', path_return.mean())
            # logger.record_tabular('path reward (var)', path_return.var())
            # logger.record_tabular('path value (sum)', path_value.sum())
            # logger.record_tabular('path value (max)', path_value.max())
            # logger.record_tabular('path value (min)', path_value.min())
            # logger.record_tabular('path value (mean)', path_value.mean())
            # logger.record_tabular('path value (var)', path_value.var())
            joblib.dump(steps, './steps.pkl', compress=3)

            pool = self.env.get_pool()
            pool._compute_A(value)
            
            logger.log('computing!!')
            ob, ac, atarg, tdlamret = pool._observations, pool._actions, pool._advantage, pool._target_values
            logger.log('Obs shape: {0}, Action shape: {1}, Advantage shape: {2}, Tgt_value shape: {3}'.format(
                ob.shape, ac.shape, atarg.shape, tdlamret.shape))
            atarg = (atarg - atarg.mean()) / atarg.std()
            logger.log('Pause for training.')
            optim_batchsize = self.batch_size or ob.shape[0]
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=True)
            if hasattr(self.policy, "ob_rms"): self.policy.ob_rms.update(ob)
            assign_old_eq_new = self.opt_info['update_old_policy']
            assign_old_eq_new()
            acc = []
            for i in range(self.n_updates):
                for batch in d.iterate_once(optim_batchsize):
                    pass
                    acc.append(self.do_training(batch, 0))

            self.opt_info['update_old_policy']()
            self.env.pool.reset()
            total_loss = np.array(list(zip(*acc))[0])
            surrogate = np.array(list(zip(*acc))[1])
            value_loss = np.array(list(zip(*acc))[2])
            entrpy_loss = np.array(list(zip(*acc))[3])
            logger.record_tabular('total loss (mean)', total_loss.mean())
            logger.record_tabular('total loss (var)', surrogate.var())
            logger.record_tabular('surrogate (mean)', surrogate.mean())
            logger.record_tabular('surrogate (var)', surrogate.var())
            logger.record_tabular('value loss (mean)', value_loss.mean())
            logger.record_tabular('value loss (var)', value_loss.var())
            logger.record_tabular('entropy loss (mean)', entrpy_loss.mean())
            logger.record_tabular('entropy loss (var)', entrpy_loss.var())
            logger.log('Training finished.')
            logger.save_itr_params(epoch, self.sess, self.get_epoch_snapshot(epoch))
            logger.dump_tabular(with_prefix=True)
            logger.pop_prefix()

    def run(self, itr, max_epoch = 50):


        self.init_opt()

        epoch = 0
        assert self.restore_fd
        info = logger.init(restore=True, dir_name=self.restore_fd, session=self.sess, itr=itr)

        observation = self.env.reset()
        terminal = True
        act = []
        steps = []
        step = 0
        while True:
            epoch += 1
            logger.push_prefix('Running: epoch #%d | ' % epoch)
            logger.log("%s started" % epoch)
            path_return = []
            path_value = []
            for i in range(self.max_local_step):
                # self.env.render()
                # print('len',len(observation))
                # observation = observation[:12] + observation[13:]

                action, value = self.policy.get_a_v(observation)
                # action = np.array([0,0,-10])
                act.append(action)
                next_ob, reward, next_terminal, _ = self.env.step(action)
                step += 1
                # print(reward)
                path_return.append(reward)
                path_value.append(value)
                # pool.add_sample(value, terminal, observation, action, reward)
                observation = next_ob
                terminal = next_terminal
                # print(terminal)
                if terminal:
                    # print(i)
                    steps.append(step)
                    step = 0
                    # observation = self.env.reset()
                    break
            if epoch > max_epoch:
                self.env.save_to_csv(epoch=epoch)
                break
            # break
                    # print(ob)
        return act

    def load(self, itr):
        self.init_opt()
        if self.restore_fd:
            info = logger.init(restore=True, dir_name=self.restore_fd, session=self.sess, itr=itr)
        else:
            logger.init()


    def init_opt(self):
        advantage = tf.placeholder(tf.float32, [None])
        target_value = tf.placeholder(tf.float32, [None])
        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
        ep = self.epsilon * lrmult
        action = tf.placeholder(tf.float32, [None] + [self.policy.action_dim])
        ratio = tf.exp(-self.policy.neglogp(action) + self.old_policy.neglogp(action))
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - ep, 1.0 + ep)
        surrogate = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))
        value = self.policy.value
        value_loss = tf.reduce_mean(tf.square(value - target_value))
        entropy_loss = - 0.5 * tf.log(2 * np.pi * np.e * tf.reduce_sum(self.policy.sigma))
        total_loss = surrogate + self.c1 * value_loss + self.c2 * entropy_loss
        var_list = self.policy.get_trainable_param()
        assign_old_eq_new = \
            [tf.assign(oldv, newv) for (newv, oldv) in
             self.zipsame(self.policy.get_param(), self.old_policy.get_param())]
        policy_updt = self.policy_update_method.minimize(total_loss, var_list=var_list)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        def policy_update(obs, act, adv, t_v, step):
            cur_lrmult = 1
            return self.sess.run((policy_updt, total_loss, surrogate, value_loss, entropy_loss)
                                 , feed_dict={self.policy.state: obs,
                                              self.old_policy.state: obs,
                                              action: act,
                                              advantage: adv,
                                              target_value: t_v,
                                              lrmult: cur_lrmult})

        def update_old_policy():
            self.sess.run(assign_old_eq_new)

        self.opt_info = dict(
            policy_update=policy_update,
            update_old_policy=update_old_policy
        )

        print('opt done')

    def do_training(self, batch, step):

        obss = batch['ob']
        actions = batch['ac']
        advantages = batch['atarg']
        target_values = batch['vtarg']
        policy_update = self.opt_info['policy_update']
        _, total_loss, surrogate, value_loss, entropy_loss = policy_update(obss, actions, advantages, target_values,
                                                                           step)
        self.env.train_model()
        logger.log('total_loss: %s' % total_loss)
        logger.log('surrogate: %s' % surrogate)
        logger.log('value_loss: %s' % value_loss)
        logger.log('entropy loss: %s' % entropy_loss)
        return total_loss, surrogate, value_loss, entropy_loss

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
        )

    def zipsame(self, *seqs):
        L = len(seqs[0])
        assert all(len(seq) == L for seq in seqs[1:])
        return zip(*seqs)