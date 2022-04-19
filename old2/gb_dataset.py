from pathlib import Path
import torch
from torch.utils.data import Dataset
from pyboy import WindowEvent
from rollout import Rollout

class GBDataset(Dataset):

    def __init__(self, dir_path, limit_steps=None):     
        self.dir_path = dir_path
        self.all_paths = [p for p in Path(dir_path).glob('*.json')]
        self.limit_steps = limit_steps
    
    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        roll = Rollout.from_saved_path(
            self.all_paths[idx], self.dir_path, self.limit_steps)
        target_actions = roll.actions
        previous_actions = [WindowEvent.PASS] + actions[:-1]
        final_reward = roll.rewards[-1]
        reward_to_go = [final_reward -  r for r in roll.rewards]

        return (
            torch.tensor(roll.frames, dtype=torch.float) / 255.0, 
            torch.tensor(reward_to_go, dtype=torch.float),
            torch.tensor(previous_actions, dtype=torch.long),
            torch.tensor(target_actions, dtype=torch.long),

        )