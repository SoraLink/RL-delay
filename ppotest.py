import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)
import shift
from ppo import PPO
from SHIFT_env_4 import SHIFT_env as Env
from mlp_stoch_policy import Policy

import numpy as np
import tensorflow as tf

symbol = 'AAPL'
seed = 13

trader = shift.Trader("test001")
trader.disconnect()
trader.connect("initiator.cfg", "password")
trader.sub_all_order_book()

env = Env(trader = trader,
          t = 2,
          nTimeStep=5,
          ODBK_range=5,
          symbol=symbol)

sess = tf.Session()
# tf.set_random_seed(seed)
policy = Policy(sess, env.obs_space(), env.action_space(), 'ppo', hidden_units=(64, 64))

old_policy = Policy(sess, env.obs_space(), env.action_space(), 'oldppo', hidden_units=(64, 64))

dspg = PPO(env=env,
                policy=policy,
                old_policy=old_policy,
                session=sess,
            # restore_fd='2019-04-21_19-14-17',
            policy_learning_rate = 0.001,
            epoch_length = 500,
           c1 = 1,
           c2 = 0.5,
           lam = 0.95,
           gamma = 0.99,
           max_local_step = 50,
            )
dspg.train()




