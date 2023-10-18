from os.path import exists
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from baselines.util import get_args, change_env
from baselines.poke_red_gym import make_env


def baseline_parallel():
    args = get_args()

    env_config = change_env(args)

    num_cpu = args.cpu_count  # Also sets the number of episodes per training iteration
    ep_length = args.max_steps

    env = SubprocVecEnv([make_env(env_config, i) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length,
        save_path=args.session_path,
        name_prefix='poke')

    # env_checker.check_env(env)
    learn_steps = 1
    file_name = 'session_e41c9eff/poke_38207488_steps'  # 'session_e41c9eff/poke_250871808_steps'

    # 'session_bfdca25a/poke_42532864_steps' #'session_d3033abb/poke_47579136_steps' #'session_a17cc1f5/poke_33546240_steps' #'session_e4bdca71/poke_8945664_steps' #'session_eb21989e/poke_40255488_steps' #'session_80f70ab4/poke_58982400_steps'
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=512, n_epochs=1, gamma=0.999)

    for i in range(learn_steps):
        model.learn(total_timesteps=ep_length * num_cpu * 1000, callback=checkpoint_callback)


if __name__ == '__main__':
    baseline_parallel()
