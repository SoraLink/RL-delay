import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from ppo.ppo import PPO
from ppo.mlp_stoch_policy import Policy
import tensorflow as tf
from Environment.predictedEnvFNN import PredictedEnv
from Environment.registration import EnvRegistry
from sac.misc.tf_utils import get_default_session

from Algorithm.Algo.ppo import learn

import numpy as np
import tensorflow as tf

def main():
    sess = get_default_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    env = PredictedEnv("HopperPyBulletEnv-v0", 8, 8, sess)
    model = learn(
        network='mlp',
        env = env,
        seed=None,
        total_timesteps=100000000
    )
    

if __name__ == '__main__':
    main()