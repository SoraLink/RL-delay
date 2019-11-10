from Environment.registration import EnvRegistry
import time
from sac.algos.sac import SAC
from rllab.envs.normalized_env import normalize
from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp, unflatten
from sac.policies import GaussianPolicy, LatentSpacePolicy, GMMPolicy, UniformPolicy
from sac.misc.sampler import SimpleSampler
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.preprocessors import MLPPreprocessor
from variants import parse_domain_and_task, get_variants



def main():
    M = 64
    variant_generator = get_variants(domain=domain, task=task, policy=args.policy)
    variants = variant_generator.variants()
    variants = [unflatten(variant, separator='.') for variant in variants]
    variant = variants[0] #TODO
    env_params = variant['env_params']
    policy_params = variant['policy_params']
    value_fn_params = variant['value_fn_params']
    algorithm_params = variant['algorithm_params']
    replay_buffer_params = variant['replay_buffer_params']
    sampler_params = variant['sampler_params']

    task = variant['task']
    domain = variant['domain']

    env = EnvRegistry("CartPole-v1",2,2)
    env.spec = env.env.spec

    pool = SimpleReplayBuffer(env_spec=env.spec, **replay_buffer_params)

    sampler = SimpleSampler(**sampler_params)

    base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)
    policy = GaussianPolicy(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
        reparameterize=policy_params['reparameterize'],
        reg=1e-3,
    )
    initial_exploration_policy = UniformPolicy(env_spec=env.spec)
    qf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf1')
    qf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf2')
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M))

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        initial_exploration_policy=initial_exploration_policy,
        pool=pool,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        lr=algorithm_params['lr'],
        scale_reward=algorithm_params['scale_reward'],
        discount=algorithm_params['discount'],
        tau=algorithm_params['tau'],
        reparameterize=algorithm_params['reparameterize'],
        target_update_interval=algorithm_params['target_update_interval'],
        action_prior=policy_params['action_prior'],
        save_full_state=False,
    )




    env.start()
    while True:
        action = env.action_space.sample()
        feedback = env.step(action)
        if feedback is None:
            continue
        else:
            observation, reward, done, info, action = feedback
            print("Observation: ", observation)
            print("reward: ", reward)
            print("done: ", done)
            print("info: ", info)
            print("action: ", action)
            time.sleep(0.01)
            if done:
                env.restart()


if __name__ == '__main__':
    main()