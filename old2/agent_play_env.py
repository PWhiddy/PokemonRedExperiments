from multiprocessing import Process

def agents_play_envs(
    agent_env_pairs, max_env_steps,
    max_runs_per_emulator=100000
    ):

    procs = len(agent_env_pairs)

    print(f'Running {procs} emulator{"s in parallel" if procs > 1 else ""}')
    processes = [
        Process(target=run_random_sims, 
        args=(create_agent, create_env, max_env_steps, max_runs_per_emulator))
        for (create_agent, create_env) in agent_env_pairs
    ]
    for p in processes:
        p.daemon = True
        p.start()
    for p in processes:
        p.join()

def run_random_sims(create_agent_func, create_env_func, max_steps, max_runs):
    env = create_env_func()
    em_id = env.instance_id
    print(f'Initialized emulator {em_id}')
    #agent = create_agent_func()

    count = 0
    total_reward = 0
    max_reward = 0
    min_reward = 99999999
    while True and count < max_runs:
        agent = create_agent_func()
        reward = env.play_episode(agent, max_steps)
        count += 1
        total_reward += reward
        min_reward = min(min_reward, reward)
        max_reward = max(max_reward, reward)
        print(
            f'{em_id} round {count}, reward: {reward}, '
            f'min: {min_reward}, max: {max_reward}, average: {total_reward/count}'
        )