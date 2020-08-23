import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from ppo.ppo import PPO
from ppo.mlp_stoch_policy import Policy
import tensorflow as tf
from Environment.registration_m import EnvRegistry
from Environment.registration import EnvRegistry
from sac.misc.tf_utils import get_default_session

from ppo2.ppo2_code import learn

def main():
    sess = get_default_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    env = EnvRegistry("HopperPyBulletEnv-v0", 8, 8)
    model = learn(
        network='mlp',
        env = env,
        seed=None,
        total_timesteps=100000000
    )
    

if __name__ == '__main__':
    main()