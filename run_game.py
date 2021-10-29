
from red_game import RedEnv
from random_agent import RandomAgent

if __name__ == '__main__':
    sim = RedEnv(headless=True)
    rand_agent = RandomAgent()

    count = 0
    total_reward = 0
    while True:
        reward = sim.play_episode(rand_agent, 10000)
        count += 1
        total_reward += reward
        print(f'round {count}, reward: {reward}, average: {total_reward/count}')