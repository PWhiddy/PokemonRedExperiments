from red_env import RedEnv
from rnn_agent import RNNAgent
from agent_play_env import agents_play_envs

def make_env():
    return RedEnv(headless=True, rollout_dir='rnn_rollouts_reward_11')

def make_agent():
    return RNNAgent('checkpoints/ff65b8c9_e0_b384.pt', 11, temp=0.25)

if __name__ == '__main__':
    
    agent_env_pairs = [
        (make_agent, make_env)
    ]

    agents_play_envs(agent_env_pairs, 20*5, max_runs_per_emulator=500) #10000)