import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym

env = gym.make('HopperPyBulletEnv-v0')
env.render("human")
# env.render() # call this before env.reset, if you want a window showing the environment
observation = env.reset()  # should return a state vector if everything worked
while True:
    env.render(mode = "human")
    action = env.action_space.sample()
    observation = env.step(action)
