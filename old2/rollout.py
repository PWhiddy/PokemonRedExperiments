from pathlib import Path
import json
from itertools import islice
import numpy as np
import mediapy as media

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
        with self.path.with_suffix('.json').open('w') as file:
            json.dump({
                'actions': self.actions, 
                'rewards': self.rewards, 
                'agent': self.agent_name}, 
                file
            )

    @classmethod
    def from_saved_path(cls, path, basepath, limit=None):
        with path.open('r') as f:
            data = json.load(f)
        new_instance = cls(path.stem, data['agent'], basepath=basepath)
        new_instance.actions = data['actions']
        new_instance.rewards = data['rewards']
        if limit is not None:
            new_instance.actions = new_instance.actions[:limit]
            new_instance.rewards = new_instance.rewards[:limit]
        with media.VideoReader(path.with_suffix('.mp4')) as reader:
            new_instance.frames = np.array(tuple(islice(reader, limit)))
        return new_instance
    
def load_rollouts(dir_path):
    for p in Path(dir_path).glob('*'):
        if p.suffix == '.json':
            yield Rollout.from_saved_path(p, dir_path)