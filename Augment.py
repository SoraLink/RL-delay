import os, sys
import tensorflow as tf
from Environment.random_delay_env import RandomDelayEnv
import pybullet
from Algorithm.Algo.ppo_aug import learn
from baselines import logger

lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)


def main(p=0):
    with tf.Session() as sess:
        logger.configure(dir='E:/Papers/RL-delay/Data/augment_uniform/{}'.format(p))
        init = tf.global_variables_initializer()
        sess.run(init)
        env = RandomDelayEnv("HopperPyBulletEnv-v0", distributiuon='threeAverage', average=p, p=p)
        per_env = RandomDelayEnv("HopperPyBulletEnv-v0", distributiuon='threeAverage', average=p, p=p)
        model = learn(
            network='mlp',
            env=env,
            per_env=per_env,
            seed=None,
            total_timesteps=10000000,
            p=p
        )


if __name__ == '__main__':
    main(10)

