import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

from ppo.ppo import PPO
from ppo.mlp_stoch_policy import Policy
import tensorflow as tf
from Environment.predictedEnv import PredictedEnv
from sac.misc.tf_utils import get_default_session

import numpy as np
import tensorflow as tf

def main():
    sess = get_default_session()
    env = PredictedEnv("HopperPyBulletEnv-v0", 2, 2, sess)

    init = tf.global_variables_initializer()
    sess.run(init)

    act_dim = env.action_space.shape
    obs_dim = env.observation.shape
    # tf.set_random_seed(seed)
    policy = Policy(sess, obs_dim, act_dim, 'ppo', hidden_units=(64, 64))

    old_policy = Policy(sess, obs_dim, act_dim, 'oldppo', hidden_units=(64, 64))

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


if __name__ == "__main__":
    main()

