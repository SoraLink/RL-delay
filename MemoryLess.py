import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
import tensorflow as tf
from Environment.random_delay_env import RandomDelayEnv
import pybullet
from Algorithm.Algo.ppo_m import learn

def main():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        env = RandomDelayEnv("HopperPyBulletEnv-v0", distributiuon='possion',lam=2)
        model = learn(
            network='mlp',
            env = env,
            seed=None,
            total_timesteps=100000000
        )

# def main2():
#     import gym
#     sess = get_default_session()
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     env = gym.make("HopperPyBulletEnv-v0")
#     env.render()
#     model = learn(
#         network='mlp',
#         env = env,
#         seed=None,
#         total_timesteps=100000000
#     )

# def test_microbatches():
        
#     import gym
#     import tensorflow as tf
#     import numpy as np
#     from functools import partial

#     from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
#     from baselines.common.tf_util import make_session
#     from baselines.ppo2.ppo2 import learn

#     from baselines.ppo2.microbatched_model import MicrobatchedModel
#     import pybulletgym  # register PyBullet enviroments with open ai gym
#     def env_fn():
#         env = gym.make('HopperPyBulletEnv-v0')
#         env.render()
#         env.seed(0)
#         return env

#     learn_fn = partial(learn, network='mlp', nsteps=32, total_timesteps=32, seed=0)

#     env_ref = DummyVecEnv([env_fn])
#     sess_ref = make_session(make_default=True, graph=tf.Graph())
#     learn_fn(env=env_ref)
#     vars_ref = {v.name: sess_ref.run(v) for v in tf.trainable_variables()}

    # env_test = DummyVecEnv([env_fn])
    # sess_test = make_session(make_default=True, graph=tf.Graph())
    # learn_fn(env=env_test, model_fn=partial(MicrobatchedModel, microbatch_size=2))
    # # learn_fn(env=env_test)
    # vars_test = {v.name: sess_test.run(v) for v in tf.trainable_variables()}

    # for v in vars_ref:
    #     np.testing.assert_allclose(vars_ref[v], vars_test[v], atol=3e-3)


if __name__ == '__main__':
    main()