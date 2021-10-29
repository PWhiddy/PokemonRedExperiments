from pathlib import Path
import numpy as np
import mediapy as media
import json

class Rollout:

    def __init__(self, name, agent_name, basepath='rollouts'):

        self.path = Path(basepath) / Path(name)
        self.actions = []
        self.frames = []
        self.rewards = []
        self.agent_name = agent_name
        Path(basepath).mkdir(exist_ok=True)

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_state_action_pair(self, frame, action):
        self.frames.append(frame)
        self.actions.append(action)

    def get_state_action_pair(self, index):
        return (self.frames[index], self.actions[index])

    def get_state_action_pairs(self):
        return zip(self.frames, self.actions)

    def save_to_file(self):
        # save frames as video
        out_frames = np.array(self.frames)
        media.write_video(self.path.with_suffix('.mp4'), out_frames, fps=30)
        # save actions and metadata as json
        with self.path.with_suffix('.json').open('w') as f:
            json.dump({'actions': self.actions, 'rewards': self.rewards, 'agent': self.agent_name}, f)

    ### TODO make these instances load from files for training data!
    