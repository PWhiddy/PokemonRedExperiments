
from typing_extensions import Required
from red_game import RedEnv
from random_agent import RandomAgent

if __name__ == '__main__':
    sim = RedEnv(headless=True)
    rand_agent = RandomAgent()

    count = 0
    total_reward = 0
    max_reward = 0
    min_reward = 999999
    while True:
        reward = sim.play_episode(rand_agent, 10000)
        count += 1
        total_reward += reward
        min_reward = min(min_reward, reward)
        max_reward = max(max_reward, reward)
        print(f'round {count}, reward: {reward}, min: {min_reward}, max: {max_reward}, average: {total_reward/count}')