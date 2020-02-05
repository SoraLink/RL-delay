import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

# from ppo_nm.ppo import PPO
from ppo_nm.mlp_stoch_policy import Policy
import tensorflow as tf
from Environment.registration import EnvRegistry
from sac.misc.tf_utils import get_default_session

import numpy as np
import tensorflow as tf

def main():
    sess = get_default_session()
    env = EnvRegistry("HopperPyBulletEnv-v0", 0, 0)
    env.name = "HopperPyBulletEnv-v0"
    # init = tf.global_variables_initializer()
    # sess.run(init)
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    # print("act_dim: ", act_dim)
    # print("obs_dim: ", obs_dim)
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
            c1 = 0.5,
            c2 = 0.01,
            lam = 0.95,
            gamma = 0.99,
            max_local_step = 2000,
                )


    dspg.train()

def main2():
    print('running..................')
    import gym
    from ppo_nm.ppo_regular import PPO
    sess = get_default_session()
    env = gym.make("HopperPyBulletEnv-v0")
    env.render()
    env.name = "HopperPyBulletEnv-v0"
    # init = tf.global_variables_initializer()
    # sess.run(init)
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    # print("act_dim: ", act_dim)
    # print("obs_dim: ", obs_dim)
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
            c1 = 0.5,
            c2 = 0.01,
            lam = 0.95,
            gamma = 0.99,
            max_local_step = 2000,
                )


    dspg.train()


if __name__ == "__main__":
    main2()



