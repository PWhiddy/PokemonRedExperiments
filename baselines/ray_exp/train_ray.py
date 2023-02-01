import uuid
from pathlib import Path
import ray
from ray.rllib.algorithms import ppo
from red_gym_env_ray import RedGymEnv

ep_length = 2048 # 2048 * 8
sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

env_config = {
            'headless': True, 'save_final_state': True, 'early_stop': False,
            'action_freq': 24, 'init_state': '../../has_pokedex_nballs.state', 'max_steps': ep_length, 
            'print_rewards': False, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
            'gb_path': '../../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 500_000.0
        }

ray.init(num_gpus=1)

#algo = ppo.PPO(env=RedGymEnv, config={
#    "num_gpus": 1,
#    "model_config": {
#        "use_lstm":True
#    },
#    "framework": "torch",
#    "env_config": env_config,  # config to pass to env class
#})

# Create the Algorithm from a config object.
config = (
    ppo.PPOConfig()
    .environment(RedGymEnv, env_config=env_config)
    .framework("torch")
    .resources(num_gpus=4)
    .rollouts(num_rollout_workers=48)
    .training(
        model={
            "grayscale": True,
            "framestack": True,
            #  model with an LSTM.
       #     "use_lstm": True,
            # To further customize the LSTM auto-wrapper.
       #     "lstm_cell_size": 64
            # Specify our custom model from above.
            #"custom_model": "my_torch_model",
            # Extra kwargs to be passed to your model's c'tor.
           # "custom_model_config": {},
        },
        gamma=0.98,
        train_batch_size=512
    )
)
algo = config.build()
for _ in range(2000):
  results = algo.train()
  print(results)
#algo.train()
#algo.stop()
