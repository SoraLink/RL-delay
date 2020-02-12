import os, sys
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)
print(lib_path)

import gym
import pybulletgym
from ppo2_code import learn

def main():
    env = gym.make("HopperPyBulletEnv-v0")
    env.render()
    learn(network = 'mlp',
          env = env,
          total_timesteps = 1e6)



if __name__ == "__main__":
    main()