import os, sys
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)
print(lib_path)

import gym
import pybulletgym
from ppo2.ppo2_code import learn
from Environment.registration import EnvRegistry

def main():
    env = gym.make("HopperPyBulletEnv-v0")
    env.render()
    learn(network = 'mlp',
          env = env,
          total_timesteps = 1e6)

def main2():
    print('running..................')
    name = "HopperPyBulletEnv-v0"
    env = EnvRegistry(name, 0, 0)
    env.name = name
    learn(network = 'mlp',
          env = env,
          total_timesteps = 1e8)


if __name__ == "__main__":
    main2()