from os.path import exists
from stable_baselines3 import PPO

from baselines.util import get_args, change_env
from baselines.poke_red_gym import make_env


def pretrained_interactive():
    args = get_args()

    # interactive
    args.headless = False

    env_config = change_env(args)

    num_cpu = args.cpu_count  # Also sets the number of episodes per training iteration
    ep_length = args.max_steps
    env = make_env(env_config)()  # SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    # env_checker.check_env(env)
    file_name = 'pretrained_sessions/session_4da05e87_main_good/poke_439746560_steps'

    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=512, n_epochs=1, gamma=0.999)

    # keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    while True:
        action = 7  # pass action
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except IOError:
            agent_enabled = False

        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()


if __name__ == '__main__':
    pretrained_interactive()
