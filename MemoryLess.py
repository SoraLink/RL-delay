import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
import tensorflow as tf
from Environment.random_delay_env import RandomDelayEnv
import pybullet
from Algorithm.Algo.ppo_m import learn

def main(p=0):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        env = RandomDelayEnv("HopperPyBulletEnv-v0", distributiuon='threeAverage',p=p)
        per_env = RandomDelayEnv("HopperPyBulletEnv-v0", distributiuon='threeAverage',p=p)
        model = learn(
            network='mlp',
            env = env,
            per_env = per_env,
            seed=None,
            total_timesteps=10000000
        )

if __name__ == '__main__':
    main(5)