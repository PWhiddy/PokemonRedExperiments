from red_env import RedEnv
from random_agent import RandomAgent
from agent_play_env import agents_play_envs

def make_env():
    return RedEnv(headless=True, rollout_dir='random_rollouts')

if __name__ == '__main__':
    
    agent_env_pairs = [
        (RandomAgent, make_env)
    ]

    agents_play_envs(agent_env_pairs, 20*5) #10000)